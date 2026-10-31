"""Glue: retrieve → generate → return the answer with its sources."""

from __future__ import annotations

from typing import Sequence

from .generator import Generator
from .retriever import Retriever

MODES = ("mevzuat", "ictihat")


class LegalAssistant:
    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 4, min_relative_score: float = 0.4):
        self.retriever = retriever
        self.min_relative_score = min_relative_score
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str, mode: str = "mevzuat", history: Sequence[dict] = ()) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        # Case-law mode pulls case-law summaries first, then fills up with legislation.
        if mode == "ictihat":
            hits = self.retriever.search(question, k=max(1, self.top_k // 2), kind="ictihat")
            seen = {h.passage.id for h in hits}
            hits += [h for h in self.retriever.search(question, k=self.top_k, kind="mevzuat") if h.passage.id not in seen]
            hits = hits[: self.top_k]
        else:
            hits = self.retriever.search(question, k=self.top_k)

        # Drop weak matches so the model is not fed unrelated passages.
        if hits:
            best = max(h.score for h in hits)
            hits = [h for h in hits if h.score >= self.min_relative_score * best]

        gen = self.generator.generate(question, hits, mode, history)
        return {
            "answer": gen.text,
            "mode": mode,
            "sources": [
                {
                    "n": n,
                    "id": h.passage.id,
                    "kind": h.passage.kind,
                    "title": h.passage.title,
                    "ref": h.passage.ref,
                    "url": h.passage.url,
                    "snippet": h.passage.text[:280] + ("…" if len(h.passage.text) > 280 else ""),
                    "score": round(h.score, 3),
                }
                for n, h in enumerate(hits, 1)
            ],
            "models": {"retriever": self.retriever.name, "generator": self.generator.name},
        }

    def info(self) -> dict:
        return {"retriever": self.retriever.name, "generator": self.generator.name, "top_k": self.top_k}
