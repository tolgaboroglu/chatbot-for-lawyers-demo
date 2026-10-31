"""Offline tests: TF-IDF retrieval + retrieval-only generator, no model downloads or API keys."""

from pathlib import Path

import numpy as np
import pytest

from app.corpus import Passage, chunk_text, load_jsonl
from app.generator import RetrievalOnlyGenerator, _clean_history, is_turkish
from app.pipeline import LegalAssistant
from app.prompts import build_user_message
from app.retriever import EmbeddingRetriever, TfidfRetriever, tr_lower
from app.server import create_app

CORPUS = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.jsonl"


@pytest.fixture(scope="module")
def passages():
    return load_jsonl(CORPUS)


@pytest.fixture(scope="module")
def assistant(passages):
    return LegalAssistant(TfidfRetriever(passages), RetrievalOnlyGenerator(), top_k=4)


def test_corpus_loads_both_kinds(passages):
    kinds = {p.kind for p in passages}
    assert kinds == {"mevzuat", "ictihat"}
    assert len({p.id for p in passages}) == len(passages), "ids must be unique"


def test_turkish_lowercase():
    assert tr_lower("İŞ KANUNU") == "iş kanunu"
    assert tr_lower("ISPAT") == "ıspat"


@pytest.mark.parametrize(
    "question, expected_id",
    [
        ("Kıdem tazminatına hangi durumlarda hak kazanılır?", "is-k-1475-m14"),
        ("İhbar süresi kaç hafta?", "is-k-4857-m17"),
        ("Deneme süresi en fazla kaç ay?", "is-k-4857-m15"),
        ("Kira artışı TÜFE sınırı", "tbk-m344"),
        ("Yıllık izin kaç gün?", "is-k-4857-m53"),
        ("Konut kirasına yapılacak artış en fazla ne kadar olabilir?", "tbk-m344"),
        ("Kiraya veren gereksinim nedeniyle tahliye isteyebilir mi?", "tbk-m350"),
        ("Alacak kaç yılda zamanaşımına uğrar?", "tbk-m146-147"),
    ],
)
def test_tfidf_finds_the_right_article(assistant, question, expected_id):
    hits = assistant.retriever.search(question, k=3)
    assert expected_id in [h.passage.id for h in hits]


def test_f5_stemming():
    from app.retriever import f5_tokens
    assert f5_tokens("Kiracının KİRASINA") == ["kirac", "kiras"]


def test_case_law_mode_puts_case_law_first(assistant):
    result = assistant.ask("İstifa eden işçi kıdem tazminatı alabilir mi?", mode="ictihat")
    assert result["sources"][0]["kind"] == "ictihat"
    assert result["sources"][0]["id"] == "ict-istifa-kidem"


def test_answer_cites_every_source(assistant):
    result = assistant.ask("Fazla mesai nasıl ispatlanır?", mode="ictihat")
    for src in result["sources"]:
        assert f"[{src['n']}]" in result["answer"]


def test_rejects_bad_input(assistant):
    with pytest.raises(ValueError):
        assistant.ask("   ")
    with pytest.raises(ValueError):
        assistant.ask("soru", mode="other")


def test_chunking_respects_limit():
    text = " ".join(f"Bu {i}. cümledir ve biraz uzundur." for i in range(200))
    chunks = list(chunk_text(text, max_chars=300, overlap=60))
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_embedding_retriever_with_fake_encoder():
    class FakeEncoder:  # bag-of-letters vectors, enough to exercise the code path
        def encode(self, texts, normalize_embeddings=True):
            vecs = np.array([[t.lower().count(ch) for ch in "abcdefghijklmnopqrstuvwxyz"] for t in texts], dtype=float)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    docs = [Passage("a", "mevzuat", "kira", "x", "kira kira kira"), Passage("b", "mevzuat", "izin", "y", "izin izin izin")]
    r = EmbeddingRetriever(docs, "intfloat/multilingual-e5-base", encoder=FakeEncoder())
    assert r.search("kira", k=1)[0].passage.id == "a"


def test_prompt_numbers_sources(assistant):
    hits = assistant.retriever.search("kira artışı", k=2)
    msg = build_user_message("Kira artışı ne kadar?", hits, "mevzuat")
    assert "[1]" in msg and "[2]" in msg and "Question:" in msg


def test_history_starts_with_user():
    turns = _clean_history([{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}])
    assert turns[0]["role"] == "user"


def test_language_hint():
    assert is_turkish("Kira artışı ne kadar?")
    assert not is_turkish("What is the limitation period?")


def test_api(assistant):
    client = create_app(assistant).test_client()
    assert client.get("/api/health").json["status"] == "ok"
    ok = client.post("/api/ask", json={"question": "Yıllık izin kaç gün?", "mode": "mevzuat"})
    assert ok.status_code == 200 and ok.json["sources"]
    assert client.post("/api/ask", json={"question": ""}).status_code == 400
    assert client.post("/api/ask", json={"question": "x", "mode": "nope"}).status_code == 400
    assert client.get("/").status_code == 200
