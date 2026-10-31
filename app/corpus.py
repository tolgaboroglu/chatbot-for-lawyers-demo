"""Loading and chunking the legal knowledge base."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

KINDS = ("mevzuat", "ictihat")


@dataclass(frozen=True)
class Passage:
    """One retrievable piece of text, with enough metadata to cite it."""

    id: str
    kind: str  # "mevzuat" (legislation) or "ictihat" (case law)
    title: str
    ref: str  # citation shown to the user, e.g. "4857 s. İş K. m.17"
    text: str
    url: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_jsonl(path: str | Path) -> list[Passage]:
    passages: list[Passage] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            kind = row.get("kind", "mevzuat")
            if kind not in KINDS:
                raise ValueError(f"{path}:{line_no}: kind must be one of {KINDS}, got {kind!r}")
            passages.append(
                Passage(
                    id=str(row["id"]),
                    kind=kind,
                    title=row.get("title", ""),
                    ref=row.get("ref", ""),
                    text=row["text"].strip(),
                    url=row.get("url"),
                )
            )
    if not passages:
        raise ValueError(f"{path} contains no passages")
    return passages


def save_jsonl(passages: Iterable[Passage], path: str | Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> Iterator[str]:
    """Split long text into sentence-aligned chunks of at most ``max_chars``.

    Consecutive chunks share roughly ``overlap`` characters of context so an
    answer that spans a boundary can still be retrieved.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        if text:
            yield text
        return

    sentences = _SENTENCE_END.split(text)
    current = ""
    for sentence in sentences:
        while len(sentence) > max_chars:  # a single very long sentence
            head, sentence = sentence[:max_chars], sentence[max_chars - overlap:]
            if current:
                yield current
                current = ""
            yield head
        if len(current) + len(sentence) + 1 > max_chars and current:
            yield current
            tail = current[-overlap:]
            cut = tail.find(" ")
            current = (tail[cut + 1:] if cut != -1 else tail) + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        yield current
