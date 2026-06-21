#!/usr/bin/env python3
"""LLM-as-a-judge evaluation harness for Vet-eye CareBot.

For every (question, gold answer) pair in the validation workbook it:
  1. runs the *real* RAG pipeline (rag.answer → retrieve + generate), then
  2. scores the system answer with an LLM judge against the gold answer and the
     retrieved context, and
  3. records deterministic retrieval metrics that need no judge.

Metrics
-------
LLM-judged (per question):
  - correctness   1-5  technical/semantic agreement with the gold answer
  - groundedness  1-5  every claim supported by retrieved context (anti-hallucination)
  - completeness  1-5  covers the key steps/facts present in the gold answer
  - cites_source  bool answer references a manual/source as the prompt requires
  - behavior_ok   bool refusal/escalation behaviour matches what the gold shows
Deterministic:
  - retrieval_hit bool the expected device's manual appears in the retrieved docs
  - top1_hit      bool the top-ranked retrieved doc is the expected manual
  - latency_s          wall-clock time for the RAG answer

A question counts as PASS when correctness >= 4 and groundedness >= 4 and behavior_ok.

Usage
-----
    python evaluate.py                        # full run (all 210 questions)
    python evaluate.py --per-category 5       # random 5 per category (stratified)
    python evaluate.py --per-category 5 --seed 7   # reproducible sample
    python evaluate.py --limit 20             # first 20 questions (quick smoke)
    python evaluate.py --model "Vet Pro 70"   # only one device
    python evaluate.py --workers 6 --out results

Writes <out>.jsonl (per-question detail), <out>.csv (flat table) and
<out>_summary.json, and prints an aggregate summary broken down by model and
category.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

import openpyxl

import rag
from clients import build_context, detect_device_source, make_openai_client, make_search_client
from config import get_settings

# Categories whose gold answer is a deliberate refusal or hand-off to L2; the
# judge is told to expect that behaviour rather than a technical procedure.
REFUSAL_CATEGORIES = {"RAG - Odmowa", "Fallback / L2"}

PASS_CORRECTNESS = 4
PASS_GROUNDEDNESS = 4


@dataclass
class Item:
    lp: int
    model: str
    category: str
    question: str
    gold: str
    expected_source: str | None  # the manual we expect retrieval to hit


@dataclass
class Result:
    lp: int
    model: str
    category: str
    question: str
    gold: str
    answer: str = ""
    retrieved_sources: list[str] = field(default_factory=list)
    # deterministic
    retrieval_hit: bool = False
    top1_hit: bool = False
    latency_s: float = 0.0
    # judge
    correctness: int = 0
    groundedness: int = 0
    completeness: int = 0
    cites_source: bool = False
    behavior_ok: bool = False
    rationale: str = ""
    passed: bool = False
    error: str = ""


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> list[Item]:
    """Parse the validation workbook. Each question is a block: the first row
    carries Lp./Model/Category/Question and the first answer line; subsequent
    rows (blank Lp.) carry continuation answer lines until the next Lp."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]

    items: list[Item] = []
    cur: dict | None = None
    for row in ws.iter_rows(values_only=True):
        lp, model, cat, q, a = (list(row) + [None] * 5)[:5]
        if isinstance(lp, int):  # start of a new question block
            if cur:
                items.append(_finalize(cur))
            cur = {
                "lp": lp,
                "model": (model or "").strip(),
                "cat": (cat or "").strip(),
                "q": (q or "").strip(),
                "ans": [str(a).strip()] if a else [],
            }
        elif cur is not None and a:  # continuation answer line
            cur["ans"].append(str(a).strip())
    if cur:
        items.append(_finalize(cur))
    return [it for it in items if it.question]


def _finalize(cur: dict) -> Item:
    return Item(
        lp=cur["lp"],
        model=cur["model"],
        category=cur["cat"],
        question=cur["q"],
        gold="\n".join(cur["ans"]).strip(),
        # Reuse the production alias detector so "Vet Pro-key 75" resolves to the
        # same source filename the retriever would filter on.
        expected_source=detect_device_source(cur["model"]),
    )


def sample_per_category(items: list[Item], n: int, seed: int) -> list[Item]:
    """Randomly draw up to `n` questions from each category (stratified sample)."""
    rng = random.Random(seed)
    buckets: dict[str, list[Item]] = {}
    for it in items:
        buckets.setdefault(it.category, []).append(it)
    picked: list[Item] = []
    for cat in sorted(buckets):
        group = buckets[cat]
        picked.extend(rng.sample(group, min(n, len(group))))
    picked.sort(key=lambda it: it.lp)
    return picked


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You are a strict, fair evaluator of a Polish-language technical \
support assistant (Vet-eye CareBot) for veterinary ultrasound devices. You compare the \
ASSISTANT ANSWER against a GOLD reference answer and the RETRIEVED CONTEXT that was \
available to the assistant. You do not use outside knowledge about the devices; the \
RETRIEVED CONTEXT is the only ground truth for what is supported.

Return ONLY a JSON object with these keys:
  "correctness":   integer 1-5  (5 = technically equivalent to GOLD; 1 = wrong/contradictory)
  "groundedness":  integer 1-5  (5 = every claim is supported by RETRIEVED CONTEXT; 1 = mostly fabricated)
  "completeness":  integer 1-5  (5 = covers all key steps/facts in GOLD; 1 = misses the essentials)
  "cites_source":  boolean      (true if the answer references a manual/source, e.g. "Źródło" or [1])
  "behavior_ok":   boolean      (does the answer's behaviour match GOLD? see below)
  "rationale":     string       (one or two concise sentences, English)

behavior_ok rules:
  - If REFUSAL EXPECTED is true: the answer is correct only if it refuses / escalates to a \
human (does NOT invent a procedure). Set behavior_ok true only then.
  - Otherwise: behavior_ok is true if the answer attempts to help with the technical issue \
consistent with GOLD (it need not be verbatim).
Be conservative: if the answer adds technical claims absent from RETRIEVED CONTEXT, lower \
groundedness even if they happen to match GOLD."""


def build_judge_prompt(it: Item, answer: str, context: str) -> str:
    refusal = it.category in REFUSAL_CATEGORIES
    return (
        f"CATEGORY: {it.category}\n"
        f"REFUSAL EXPECTED: {str(refusal).lower()}\n\n"
        f"USER QUESTION:\n{it.question}\n\n"
        f"GOLD REFERENCE ANSWER:\n{it.gold}\n\n"
        f"RETRIEVED CONTEXT (what the assistant could ground on):\n{context or '(none)'}\n\n"
        f"ASSISTANT ANSWER:\n{answer}\n\n"
        "Score now. Return only the JSON object."
    )


def _parse_judge_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Salvage the outermost {...} if the model wrapped it in prose/fences.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def judge(client, judge_model: str, it: Item, answer: str, context: str) -> dict:
    resp = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_prompt(it, answer, context)},
        ],
        max_completion_tokens=700,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    return _parse_judge_json(resp.choices[0].message.content or "{}")


# --------------------------------------------------------------------------- #
# Per-item evaluation
# --------------------------------------------------------------------------- #
def evaluate_item(it: Item, s, openai_client, search_client, judge_model: str) -> Result:
    r = Result(lp=it.lp, model=it.model, category=it.category,
               question=it.question, gold=it.gold)
    try:
        t0 = time.perf_counter()
        # Pin retrieval to the test case's device, mirroring production where the
        # device is known from the widget / sticky conversation rather than re-
        # detected from each question (many questions never name the model).
        answer, docs = rag.answer(
            openai_client, search_client, s, it.question,
            device_source=it.expected_source,
        )
        r.latency_s = round(time.perf_counter() - t0, 2)
        r.answer = answer

        sources = [d.get("source", "") for d in docs]
        r.retrieved_sources = sources
        if it.expected_source:
            r.retrieval_hit = it.expected_source in sources
            r.top1_hit = bool(sources) and sources[0] == it.expected_source

        # Rebuild the same context string the judge should see.
        context = build_context(docs)

        v = judge(openai_client, judge_model, it, answer, context)
        r.correctness = int(v.get("correctness", 0))
        r.groundedness = int(v.get("groundedness", 0))
        r.completeness = int(v.get("completeness", 0))
        r.cites_source = bool(v.get("cites_source", False))
        r.behavior_ok = bool(v.get("behavior_ok", False))
        r.rationale = str(v.get("rationale", ""))[:500]
        r.passed = (
            r.correctness >= PASS_CORRECTNESS
            and r.groundedness >= PASS_GROUNDEDNESS
            and r.behavior_ok
        )
    except Exception as e:  # one bad item must not kill the whole run
        r.error = f"{type(e).__name__}: {e}"
    return r


# --------------------------------------------------------------------------- #
# Aggregation & reporting
# --------------------------------------------------------------------------- #
def _avg(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


def summarize(results: list[Result]) -> dict:
    ok = [r for r in results if not r.error]

    def block(rs: list[Result]) -> dict:
        return {
            "n": len(rs),
            "pass_rate": _avg([1.0 if r.passed else 0.0 for r in rs]),
            "correctness": _avg([r.correctness for r in rs]),
            "groundedness": _avg([r.groundedness for r in rs]),
            "completeness": _avg([r.completeness for r in rs]),
            "cites_source": _avg([1.0 if r.cites_source else 0.0 for r in rs]),
            "behavior_ok": _avg([1.0 if r.behavior_ok else 0.0 for r in rs]),
            "retrieval_hit": _avg([1.0 if r.retrieval_hit else 0.0 for r in rs]),
            "top1_hit": _avg([1.0 if r.top1_hit else 0.0 for r in rs]),
            "avg_latency_s": _avg([r.latency_s for r in rs]),
        }

    by_model, by_cat = {}, {}
    for r in ok:
        by_model.setdefault(r.model, []).append(r)
        by_cat.setdefault(r.category, []).append(r)

    return {
        "overall": block(ok),
        "errors": len(results) - len(ok),
        "by_model": {k: block(v) for k, v in sorted(by_model.items())},
        "by_category": {k: block(v) for k, v in sorted(by_cat.items())},
    }


def print_summary(summary: dict) -> None:
    o = summary["overall"]
    print("\n" + "=" * 64)
    print(f"OVERALL  (n={o['n']}, errors={summary['errors']})")
    print("=" * 64)
    print(f"  PASS RATE     {o['pass_rate']*100:5.1f}%")
    print(f"  correctness   {o['correctness']:.2f} / 5")
    print(f"  groundedness  {o['groundedness']:.2f} / 5")
    print(f"  completeness  {o['completeness']:.2f} / 5")
    print(f"  cites_source  {o['cites_source']*100:5.1f}%")
    print(f"  behavior_ok   {o['behavior_ok']*100:5.1f}%")
    print(f"  retrieval_hit {o['retrieval_hit']*100:5.1f}%   (top1 {o['top1_hit']*100:.1f}%)")
    print(f"  avg latency   {o['avg_latency_s']:.2f}s")

    def table(title: str, d: dict) -> None:
        print(f"\n{title}")
        print(f"  {'key':<22}{'n':>4}{'pass':>7}{'corr':>6}{'grnd':>6}{'retr':>6}")
        for k, b in d.items():
            print(f"  {k:<22}{b['n']:>4}{b['pass_rate']*100:>6.0f}%"
                  f"{b['correctness']:>6.1f}{b['groundedness']:>6.1f}{b['retrieval_hit']*100:>5.0f}%")

    table("BY MODEL", summary["by_model"])
    table("BY CATEGORY", summary["by_category"])
    print("=" * 64)


def write_outputs(results: list[Result], out: str) -> None:
    with open(f"{out}.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    cols = ["lp", "model", "category", "passed", "correctness", "groundedness",
            "completeness", "cites_source", "behavior_ok", "retrieval_hit",
            "top1_hit", "latency_s", "question", "rationale", "error"]
    with open(f"{out}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-as-judge eval for Vet-eye CareBot.")
    ap.add_argument("--dataset", default=None, help="Path to the validation .xlsx.")
    ap.add_argument("--per-category", type=int, default=0,
                    help="Randomly sample up to N questions from each category (stratified).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for --per-category sampling (reproducible).")
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions.")
    ap.add_argument("--model", default=None, help="Filter to one device, e.g. 'Vet Pro 70'.")
    ap.add_argument("--workers", type=int, default=4, help="Concurrent questions.")
    ap.add_argument("--out", default="eval_results", help="Output file prefix.")
    args = ap.parse_args()

    dataset = args.dataset or next(iter(glob.glob("Dataset*.xlsx")), None)
    if not dataset or not os.path.exists(dataset):
        print("Validation dataset (.xlsx) not found.", file=sys.stderr)
        return 1

    s = get_settings()
    openai_client = make_openai_client(s)
    search_client = make_search_client(s)
    judge_model = os.getenv("JUDGE_DEPLOYMENT", s.chat_deployment)

    items = load_dataset(dataset)
    if args.model:
        items = [it for it in items if it.model == args.model]
    if args.per_category:
        items = sample_per_category(items, args.per_category, args.seed)
    if args.limit:
        items = items[: args.limit]

    sample_note = (f"  |  sample: {args.per_category}/category (seed {args.seed})"
                   if args.per_category else "")
    print(f"Dataset: {dataset}  |  questions: {len(items)}  |  "
          f"answer model: {s.chat_deployment}  |  judge: {judge_model}  |  "
          f"workers: {args.workers}{sample_note}")

    results: list[Result] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(evaluate_item, it, s, openai_client, search_client, judge_model): it
            for it in items
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            flag = "ERR " if r.error else ("PASS" if r.passed else "fail")
            print(f"[{done:>3}/{len(items)}] #{r.lp:<3} {flag}  "
                  f"c={r.correctness} g={r.groundedness}  {r.model:<16} "
                  f"{r.category[:18]:<18} {r.error[:40]}")

    results.sort(key=lambda r: r.lp)
    summary = summarize(results)
    print_summary(summary)
    write_outputs(results, args.out)
    with open(f"{args.out}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}.jsonl, {args.out}.csv, {args.out}_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
