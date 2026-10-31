"""Retrievers: dense embeddings (default) or TF-IDF (offline fallback)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .corpus import Passage


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 4, kind: str | None = None) -> list[Hit]: ...


def tr_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless I rules (str.lower gets İ/I wrong)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def _top_k(scores: np.ndarray, passages: Sequence[Passage], k: int, kind: str | None) -> list[Hit]:
    order = np.argsort(-scores)
    hits: list[Hit] = []
    for idx in order:
        p = passages[int(idx)]
        if kind and p.kind != kind:
            continue
        if scores[idx] <= 0:
            break
        hits.append(Hit(p, float(scores[idx])))
        if len(hits) == k:
            break
    return hits


_WORD = re.compile(r"\w+", re.UNICODE)

# Question words and fillers that carry no legal meaning. "en fazla" (at most) is
# dropped as a phrase so it does not match "fazla çalışma" (overtime).
_STOP_PHRASES = re.compile(r"\b(en fazla|en az|ne kadar|ne zaman)\b")
_STOPWORDS = {
    "ve", "ile", "bir", "bu", "şu", "da", "de", "ki", "mi", "mı", "mu", "mü", "midir", "mıdır",
    "ne", "nedir", "nasıl", "hangi", "kaç", "için", "gibi", "olarak", "olabilir", "olur", "var", "yok",
    "the", "is", "a", "an", "of", "for", "in", "what", "how", "to", "and",
}


def clean_query(text: str) -> str:
    return _STOP_PHRASES.sub(" ", tr_lower(text))


def f5_tokens(text: str) -> list[str]:
    """Turkish "F5" stemming: keep the first five letters of each word.

    A simple, well-studied baseline for agglutinative Turkish
    (kiracının → kirac, kirasına → kiras) that needs no dictionary.
    """
    return [w[:5] for w in _WORD.findall(tr_lower(text)) if len(w) > 1 and w not in _STOPWORDS]


class TfidfRetriever:
    """Hybrid TF-IDF: F5-stemmed words + character n-grams. Works fully offline."""

    def __init__(self, passages: Sequence[Passage], word_weight: float = 0.6):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.name = "TF-IDF (F5 stems + char n-grams)"
        self.passages = list(passages)
        self.word_weight = word_weight
        docs = [f"{p.title} {p.title} {p.ref} {p.text}" for p in self.passages]  # title counted twice
        self.words = TfidfVectorizer(analyzer=f5_tokens, sublinear_tf=True)
        self.chars = TfidfVectorizer(preprocessor=tr_lower, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.word_matrix = self.words.fit_transform(docs)
        self.char_matrix = self.chars.fit_transform(docs)

    def search(self, query: str, k: int = 4, kind: str | None = None) -> list[Hit]:
        query = clean_query(query)
        w = (self.word_matrix @ self.words.transform([query]).T).toarray().ravel()
        c = (self.char_matrix @ self.chars.transform([query]).T).toarray().ravel()
        scores = self.word_weight * w + (1 - self.word_weight) * c
        return _top_k(scores, self.passages, k, kind)


class EmbeddingRetriever:
    """Dense retrieval with a sentence-transformers model.

    E5 models expect "query: " / "passage: " prefixes; they are added
    automatically when the model name contains "e5".
    """

    def __init__(self, passages: Sequence[Passage], model_name: str, encoder=None):
        self.name = model_name
        self.passages = list(passages)
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name)
        self.encoder = encoder
        self._e5 = "e5" in model_name.lower()
        docs = [self._prefix(f"{p.title}. {p.text}", "passage") for p in self.passages]
        self.matrix = self._encode(docs)

    def _prefix(self, text: str, role: str) -> str:
        return f"{role}: {text}" if self._e5 else text

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return vecs

    def search(self, query: str, k: int = 4, kind: str | None = None) -> list[Hit]:
        q = self._encode([self._prefix(query, "query")])[0]
        scores = self.matrix @ q
        return _top_k(scores, self.passages, k, kind)


def build_retriever(kind: str, passages: Sequence[Passage], embedding_model: str) -> Retriever:
    if kind == "tfidf":
        return TfidfRetriever(passages)
    if kind == "embedding":
        return EmbeddingRetriever(passages, embedding_model)
    raise ValueError(f"Unknown RETRIEVER {kind!r}; use 'embedding' or 'tfidf'")
