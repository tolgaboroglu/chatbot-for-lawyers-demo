"""Answer generators. Every backend takes the question plus retrieved passages
and returns Markdown that cites sources as [1], [2] …

Backends
--------
anthropic   Claude via the Anthropic API (default, best answers)
openai      OpenAI chat models via the OpenAI API
extractive  BERTurk fine-tuned for Turkish extractive QA; runs locally, no API key
retrieval   No model: shows the best-matching passages; useful offline and for testing
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from .prompts import SYSTEM_PROMPT, build_user_message
from .retriever import Hit

# No IGNORECASE on the letters: Python folds ı/İ onto i/I, which would match English text.
_TR_LETTERS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TR_WORDS = re.compile(r"\b(ne|nasil|hangi|mi|mu|kac|nedir)\b")


def is_turkish(text: str) -> bool:
    return bool(_TR_LETTERS.search(text) or _TR_WORDS.search(text.lower()))


@dataclass
class Generation:
    text: str
    model: str


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: Sequence[Hit], mode: str, history: Sequence[dict] = ()) -> Generation: ...


def _clean_history(history: Sequence[dict], limit: int = 6) -> list[dict]:
    turns = [
        {"role": t["role"], "content": str(t["content"])[:4000]}
        for t in history
        if t.get("role") in ("user", "assistant") and t.get("content")
    ][-limit:]
    while turns and turns[0]["role"] != "user":  # APIs expect the first turn to be the user's
        turns.pop(0)
    return turns


class AnthropicGenerator:
    def __init__(self, model: str, max_tokens: int = 900):
        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"Claude ({model})"

    def generate(self, question, hits, mode, history=()):
        messages = _clean_history(history) + [
            {"role": "user", "content": build_user_message(question, hits, mode)}
        ]
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT, messages=messages
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return Generation(text.strip(), self.model)


class OpenAIGenerator:
    def __init__(self, model: str, max_tokens: int = 900):
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"OpenAI ({model})"

    def generate(self, question, hits, mode, history=()):
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + _clean_history(history)
            + [{"role": "user", "content": build_user_message(question, hits, mode)}]
        )
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens, messages=messages
        )
        return Generation((resp.choices[0].message.content or "").strip(), self.model)


class ExtractiveGenerator:
    """Finds the answer span inside the retrieved passages with a Turkish QA model.

    Unlike v1, the answer is never padded with unrelated context: it returns the
    span, the sentence it came from, and the source it was found in.
    """

    def __init__(self, model: str):
        from transformers import pipeline

        self.qa = pipeline("question-answering", model=model, tokenizer=model)
        self.model = model
        self.name = f"Extractive QA ({model})"

    def generate(self, question, hits, mode, history=()):
        tr = is_turkish(question)
        best = None
        for n, hit in enumerate(hits, 1):
            out = self.qa(question=question, context=hit.passage.text, max_answer_len=80)
            if best is None or out["score"] > best[0]:
                best = (out["score"], n, hit, out)
        if best is None or best[0] < 0.05:
            msg = "Kaynaklarda bu soruya net bir cevap bulunamadı." if tr else "The sources do not answer this question clearly."
            return Generation(msg, self.model)

        score, n, hit, out = best
        text = hit.passage.text
        start = text.rfind(".", 0, out["start"]) + 1
        end = text.find(".", out["end"])
        sentence = text[start: end + 1 if end != -1 else len(text)].strip()
        h_short, h_ctx = ("Kısa cevap", "Kaynaktaki ifade") if tr else ("Short answer", "Context")
        body = (
            f"### {h_short}\n{out['answer'].strip()} [{n}]\n\n"
            f"### {h_ctx}\n{sentence} [{n}]\n\n"
            f"Model confidence: {score:.0%}"
        )
        return Generation(body, self.model)


class RetrievalOnlyGenerator:
    """No language model: lists the best-matching passages with citations."""

    name = "Retrieval only (no LLM)"
    model = "none"

    def generate(self, question, hits, mode, history=()):
        tr = is_turkish(question)
        if not hits:
            msg = "Bilgi tabanında bu soruyla eşleşen bir kaynak bulunamadı." if tr else "No matching source in the knowledge base."
            return Generation(msg, self.model)
        heading = "En ilgili kaynaklar" if tr else "Most relevant sources"
        lines = [f"### {heading}"]
        for n, hit in enumerate(hits, 1):
            p = hit.passage
            first = re.split(r"(?<=[.!?])\s+", p.text)[:2]
            lines.append(f"- **{p.title}** `{p.ref}`: {' '.join(first)} [{n}]")
        return Generation("\n".join(lines), self.model)


def build_generator(kind: str, settings) -> Generator:
    if kind == "anthropic":
        return AnthropicGenerator(settings.anthropic_model, settings.max_tokens)
    if kind == "openai":
        return OpenAIGenerator(settings.openai_model, settings.max_tokens)
    if kind == "extractive":
        return ExtractiveGenerator(settings.extractive_model)
    if kind == "retrieval":
        return RetrievalOnlyGenerator()
    raise ValueError(f"Unknown GENERATOR {kind!r}; use anthropic, openai, extractive or retrieval")
