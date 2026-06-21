from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return val


@dataclass(frozen=True)
class Settings:
    # --- Microsoft Foundry / Azure OpenAI ---
    aoai_endpoint: str
    aoai_api_key: str
    aoai_api_version: str
    chat_deployment: str
    embedding_deployment: str
    embedding_dimensions: int

    # --- Azure AI Search ---
    search_endpoint: str
    search_api_key: str
    search_index: str

    # --- Retrieval / generation tuning ---
    top_k: int               # how many chunks to retrieve
    neighbor_expansion: bool  # also pull page±1 of hits (procedures span pages)
    reasoning_effort: str     # gpt-5 family: minimal | low | medium | high
    max_completion_tokens: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        aoai_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
        aoai_api_key=_require("AZURE_OPENAI_API_KEY"),
        aoai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5-mini"),
        embedding_deployment=os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        ),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "3072")),
        search_endpoint=_require("AZURE_SEARCH_ENDPOINT"),
        search_api_key=_require("AZURE_SEARCH_API_KEY"),
        search_index=os.getenv("AZURE_SEARCH_INDEX", "carebot-manuals"),
        top_k=int(os.getenv("RAG_TOP_K", "8")),
        neighbor_expansion=os.getenv("RAG_NEIGHBOR_EXPANSION", "true").lower()
        not in ("0", "false", "no"),
        reasoning_effort=os.getenv("RAG_REASONING_EFFORT", "low"),
        max_completion_tokens=int(os.getenv("RAG_MAX_COMPLETION_TOKENS", "2000")),
    )
