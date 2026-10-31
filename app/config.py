"""Runtime settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    # Where the knowledge base comes from (JSONL produced by scripts/ingest.py).
    corpus_path: Path = field(default_factory=lambda: Path(_env("CORPUS_PATH", str(ROOT / "data" / "sample_corpus.jsonl"))))

    # Retrieval: "embedding" (sentence-transformers) or "tfidf" (offline, no model download).
    retriever: str = field(default_factory=lambda: _env("RETRIEVER", "embedding"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))

    # Answer generation: "anthropic", "openai", "extractive" (BERTurk QA, offline) or "retrieval" (sources only).
    generator: str = field(default_factory=lambda: _env("GENERATOR", "anthropic"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    extractive_model: str = field(default_factory=lambda: _env("EXTRACTIVE_MODEL", "savasy/bert-base-turkish-squad"))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "900")))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "5000")))
