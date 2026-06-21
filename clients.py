from __future__ import annotations

from typing import Iterable

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

from config import Settings


def make_openai_client(s: Settings) -> OpenAI:
    return OpenAI(
        base_url=s.aoai_endpoint,
        api_key=s.aoai_api_key,
        max_retries=3,
    )


def make_search_client(s: Settings) -> SearchClient:
    return SearchClient(
        endpoint=s.search_endpoint,
        index_name=s.search_index,
        credential=AzureKeyCredential(s.search_api_key),
    )


def make_index_client(s: Settings) -> SearchIndexClient:
    return SearchIndexClient(
        endpoint=s.search_endpoint,
        credential=AzureKeyCredential(s.search_api_key),
    )


def embed(client: OpenAI, s: Settings, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single request."""
    resp = client.embeddings.create(
        model=s.embedding_deployment,
        input=texts,
        dimensions=s.embedding_dimensions,
    )
    # API preserves input order
    return [item.embedding for item in resp.data]


# Maps each manual's indexed `source` (the PDF filename) to the aliases a user
# might type. Note the third manual is indexed as "Vet pro-key 76_instrukcja.pdf"
# (the file on disk), even though its commercial name is "iScan 2 multi" / "75".
_DEVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "Vet portable 15_instrukcja.pdf": ("iscan mini", "vet portable 15", "portable 15"),
    "Vet pro 70_instrukcja.pdf": ("blue vet", "bluevet", "vet pro 70", "pro 70"),
    "Vet pro-key 76_instrukcja.pdf": (
        "iscan 2 multi", "iscan 2", "vet pro-key 75", "vet pro-key 76",
        "pro-key 75", "pro-key 76", "pro key 75", "pro key 76",
    ),
}


def detect_device_source(query: str) -> str | None:
    """Return the manual `source` to restrict to, if the query names exactly one
    device. Returns None when zero or multiple devices match, so retrieval falls
    back to searching across all manuals."""
    q = query.lower()
    matched = {src for src, aliases in _DEVICE_ALIASES.items() if any(a in q for a in aliases)}
    return next(iter(matched)) if len(matched) == 1 else None


def retrieve(
    search_client: SearchClient,
    openai_client: OpenAI,
    s: Settings,
    query: str,
    device_source: str | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: vector similarity + BM25 keyword, fused and re-ranked by
    AI Search's semantic ranker. This is the fastest path to high-quality
    grounding because the vector recall and lexical precision complement each
    other, and the semantic ranker reorders the top candidates.

    If a device is known, retrieval is hard-filtered to that device's manual so
    grounding can never leak in from another product. `device_source` lets the
    caller pin the device explicitly (e.g. a conversation-sticky selection); when
    omitted it is auto-detected from the query text.
    """
    query_vector = embed(openai_client, s, [query])[0]
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=s.top_k,
        fields="content_vector",
    )

    device_source = device_source or detect_device_source(query)
    # OData filter; applies to both the vector and keyword legs of the hybrid.
    source_filter = f"source eq '{device_source}'" if device_source else None

    results = search_client.search(
        search_text=query,                      # BM25 keyword leg of the hybrid
        vector_queries=[vector_query],          # vector leg of the hybrid
        filter=source_filter,                   # restrict to one manual if named
        query_type="semantic",                  # semantic re-ranking
        semantic_configuration_name="carebot-semantic",
        top=s.top_k,
        select=["id", "content", "source", "page"],
    )

    docs: list[dict] = []
    for r in results:
        docs.append(
            {
                "id": r["id"],
                "content": r["content"],
                "source": r.get("source", "unknown"),
                "page": r.get("page"),
                "score": r.get("@search.reranker_score") or r.get("@search.score"),
            }
        )

    if s.neighbor_expansion:
        docs = _expand_neighbors(search_client, docs)
    return docs


def _expand_neighbors(search_client: SearchClient, docs: list[dict]) -> list[dict]:
    """Append the page-before/page-after chunks of the retrieved hits.

    Procedures are chunked per page (see ingest.py), so a hit can carry the
    intro of a procedure while the remaining steps live on the adjacent page.
    Pulling page±1 keeps those steps in-context, which reduces both false
    refusals and incomplete answers. The neighbours are appended *after* the
    ranked hits (never reordering them), in one extra filtered lookup.

    Only runs when every hit shares one `source`; with a mix of manuals the page
    numbers are ambiguous, so we skip rather than risk pulling the wrong device.
    """
    sources = {d["source"] for d in docs}
    if len(sources) != 1:
        return docs
    source = next(iter(sources))

    have = {d["id"] for d in docs}
    pages = {d["page"] for d in docs if d.get("page") is not None}
    want = sorted(p for p in {p + d for p in pages for d in (-1, 1)} if p >= 1 and p not in pages)
    if not want:
        return docs

    page_filter = " or ".join(f"page eq {p}" for p in want)
    neighbours = search_client.search(
        search_text="*",
        filter=f"source eq '{source}' and ({page_filter})",
        top=len(want) * 3,  # a page may be more than one chunk
        select=["id", "content", "source", "page"],
    )
    for r in neighbours:
        if r["id"] in have:
            continue
        have.add(r["id"])
        docs.append(
            {
                "id": r["id"],
                "content": r["content"],
                "source": r.get("source", "unknown"),
                "page": r.get("page"),
                "score": None,  # not semantically ranked; pulled as adjacent context
            }
        )
    return docs


def build_context(docs: Iterable[dict]) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        page = f", p.{d['page']}" if d.get("page") is not None else ""
        blocks.append(f"[{i}] (source: {d['source']}{page})\n{d['content']}")
    return "\n\n".join(blocks)
