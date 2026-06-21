# 🩺 Vet-eye CareBot — PoC

A first-line (L1) technical-support assistant for **Vet-eye S.A.** veterinary
ultrasound (USG) equipment. It answers operator questions in **Polish and
English**, grounded in the official manuals via **RAG**, and cites its sources.

This is a **proof of concept** that intentionally uses only three Azure
resources (the full design also includes Azure Function, Key Vault, Content
Safety, Entra SSO and Application Insights):

| Layer        | Service                                                        |
|--------------|----------------------------------------------------------------|
| Presentation | **Azure Web App** (App Service) hosting this Streamlit app      |
| AI           | **Microsoft Foundry** — `gpt-5-mini` + `text-embedding-3-large` |
| Data / Search| **Azure AI Search** — hybrid (vector + keyword) + semantic rank |

## How it works

```
                ┌──────────── ingest.py (offline) ────────────┐
   manuals/*.pdf │  extract → token-chunk → embed → upload     │ →  AI Search index
                └─────────────────────────────────────────────┘

   user question ─► embed ─► hybrid+semantic search (AI Search)
                              │ top-k manual chunks
                              ▼
                 gpt-5-mini (grounded, streamed) ─► cited answer
```

**Speed/quality techniques used**
- **Hybrid retrieval + semantic reranking** — best grounding in one round-trip.
- **Token-aware chunking** with overlap (500/80 tokens) tuned for manuals.
- **Streamed responses** (`st.write_stream`) so answers appear immediately.
- **Cached Azure clients** (`st.cache_resource`) — no per-rerun reconnects.
- **Batched embeddings** (64/req) + deterministic chunk IDs for fast, idempotent
  re-ingestion.
- **Bounded chat history** + `reasoning_effort=low` to keep latency low.

## Setup

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure
cp .env.example .env        # then fill in your Azure values

# 3. Add manuals and build the knowledge base
cp /path/to/*.pdf manuals/
python ingest.py            # add --recreate to rebuild the index from scratch

# 4. Run the chat app
streamlit run app.py
```

Open http://localhost:8501.

## Files

| File              | Purpose                                                     |
|-------------------|-------------------------------------------------------------|
| `app.py`          | Streamlit chat UI + RAG orchestration (retrieve → generate) |
| `ingest.py`       | Vectorisation script: PDFs → chunks → embeddings → AI Search |
| `clients.py`      | Shared Azure client factories + retrieval helpers           |
| `config.py`       | Env-based settings                                          |
| `.env.example`    | Configuration placeholder                                   |

## Deploying to Azure Web App

Set the `.env` values as **Application Settings** on the Web App, then use this
**startup command**:

```bash
python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Run `python ingest.py` (locally or from a job) whenever the manuals change.

> **PoC scope:** no auth, no Content Safety moderation, no Key Vault. Add Entra
> SSO, Content Safety and Key Vault before any production use.
