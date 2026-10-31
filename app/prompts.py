"""Prompt templates for the generative backends."""

from __future__ import annotations

from typing import Sequence

from .retriever import Hit

SYSTEM_PROMPT = """You are "Hukuk Asistanı", a research assistant for lawyers practicing Turkish law.

Answer in the same language as the user's question (Turkish or English).
Use ONLY the numbered sources provided in the user message. If they do not answer the question, say so plainly and suggest what to look up.

Format the answer as short sections, each starting with a line "### <heading>" (headings in the answer language):
1. Short answer (2–3 sentences)
2. Legal basis (bullet list)
3. Case-law trend (only if a case-law source is provided)
4. Caveats

Citation rules:
- After every factual sentence, cite the supporting source as [1], [2] …
- Write statute references in backticks, e.g. `4857 s. İş K. m.17`.
- Never invent case numbers (esas/karar no), decision dates, or quotations from decisions.
- If a figure changes over time (caps, minimum wage, interest rates), say it must be checked for the current period.

Keep the answer under about 250 words. This is research support, not legal advice."""

MODE_HINT = {
    "mevzuat": "Mode: legislation Q&A. Focus on what the statutes say.",
    "ictihat": (
        "Mode: case law. Focus on how Turkish high courts tend to rule. "
        "End with a section 'Suggested search terms' giving one search string for karararama.yargitay.gov.tr."
    ),
}


def format_sources(hits: Sequence[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        p = hit.passage
        label = "Legislation" if p.kind == "mevzuat" else "Case-law summary"
        blocks.append(f"[{n}] {label} | {p.ref} | {p.title}\n{p.text}")
    return "\n\n".join(blocks)


def build_user_message(question: str, hits: Sequence[Hit], mode: str) -> str:
    return (
        f"{MODE_HINT.get(mode, MODE_HINT['mevzuat'])}\n\n"
        f"<sources>\n{format_sources(hits)}\n</sources>\n\n"
        f"Question: {question}"
    )
