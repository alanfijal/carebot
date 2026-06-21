#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import tiktoken
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from pypdf import PdfReader

from clients import embed, make_index_client, make_openai_client, make_search_client
from config import get_settings

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 80
EMBED_BATCH = 64       # chunks embedded per OpenAI request
UPLOAD_BATCH = 500     # docs uploaded per AI Search request

_ENC = tiktoken.get_encoding("cl100k_base")


def ensure_index(recreate: bool = False) -> None:
    s = get_settings()
    index_client = make_index_client(s)

    if recreate:
        try:
            index_client.delete_index(s.search_index)
            print(f"Deleted existing index '{s.search_index}'.")
        except ResourceNotFoundError:
            pass

    try:
        index_client.get_index(s.search_index)
        print(f"Index '{s.search_index}' already exists.")
        return
    except ResourceNotFoundError:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(
            name="content", type=SearchFieldDataType.String, analyzer_name="standard.lucene"
        ),
        SimpleField(
            name="source", type=SearchFieldDataType.String, filterable=True, facetable=True
        ),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=s.embedding_dimensions,
            vector_search_profile_name="carebot-hnsw",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="carebot-hnsw-algo",
                parameters=HnswParameters(m=4, ef_construction=400, ef_search=500),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="carebot-hnsw", algorithm_configuration_name="carebot-hnsw-algo"
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="carebot-semantic",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="content")],
                ),
            )
        ]
    )

    index = SearchIndex(
        name=s.search_index,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )
    index_client.create_index(index)
    print(f"Created index '{s.search_index}'.")



def chunk_text(text: str) -> list[str]:
    """Token-aware splitting with overlap, so chunks respect the embedder."""
    tokens = _ENC.encode(text)
    if not tokens:
        return []
    chunks = []
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_TOKENS]
        chunk = _ENC.decode(window).strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def pdf_to_records(pdf_path: Path) -> list[dict]:
    """Extract per-page text and split into chunk records."""
    reader = PdfReader(str(pdf_path))
    source = pdf_path.name
    records: list[dict] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for chunk in chunk_text(text):
            # Deterministic id -> re-running ingest updates instead of duplicating.
            digest = hashlib.sha1(
                f"{source}:{page_num}:{chunk[:64]}".encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "id": digest,
                    "content": chunk,
                    "source": source,
                    "page": page_num,
                }
            )
    return records


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest_records(records: list[dict]) -> None:
    s = get_settings()
    openai_client = make_openai_client(s)
    search_client = make_search_client(s)

    # 1) Embed in batches.
    print(f"Embedding {len(records)} chunks...")
    for batch in _batched(records, EMBED_BATCH):
        vectors = embed(openai_client, s, [r["content"] for r in batch])
        for r, v in zip(batch, vectors):
            r["content_vector"] = v

    # 2) Upload in batches.
    print(f"Uploading {len(records)} chunks to '{s.search_index}'...")
    uploaded = 0
    for batch in _batched(records, UPLOAD_BATCH):
        results = search_client.merge_or_upload_documents(documents=batch)
        uploaded += sum(1 for r in results if r.succeeded)
    print(f"Done. {uploaded}/{len(records)} chunks indexed.")



def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest manuals into AI Search.")
    parser.add_argument("--path", default="manuals", help="Folder with PDF manuals.")
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and rebuild the index first."
    )
    args = parser.parse_args()

    folder = Path(args.path)
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    pdfs = sorted(folder.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {folder}/. Drop your manuals there first.")
        return 1

    ensure_index(recreate=args.recreate)

    all_records: list[dict] = []
    for pdf in pdfs:
        recs = pdf_to_records(pdf)
        print(f"  {pdf.name}: {len(recs)} chunks")
        all_records.extend(recs)

    if not all_records:
        print("No extractable text found (are these scanned PDFs?).")
        return 1

    ingest_records(all_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
