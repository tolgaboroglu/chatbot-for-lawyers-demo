"""Flask web app and JSON API.

Run:  python -m app.server            (settings from .env / environment)
      python -m app.server --share    (also open a public ngrok tunnel)
"""

from __future__ import annotations

import argparse
import logging

from flask import Flask, jsonify, render_template, request

from .config import Settings
from .corpus import load_jsonl
from .generator import build_generator
from .pipeline import MODES, LegalAssistant
from .retriever import build_retriever

log = logging.getLogger("hukuk-asistani")
MAX_QUESTION_CHARS = 1500


def build_assistant(settings: Settings) -> LegalAssistant:
    passages = load_jsonl(settings.corpus_path)
    retriever = build_retriever(settings.retriever, passages, settings.embedding_model)
    generator = build_generator(settings.generator, settings)
    log.info("Loaded %d passages | retriever=%s | generator=%s", len(passages), retriever.name, generator.name)
    return LegalAssistant(retriever, generator, settings.top_k)


def create_app(assistant: LegalAssistant | None = None, settings: Settings | None = None) -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    assistant = assistant or build_assistant(settings or Settings())

    @app.get("/")
    def index():
        return render_template("index.html", info=assistant.info())

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", **assistant.info())

    @app.post("/api/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        mode = data.get("mode", "mevzuat")
        history = data.get("history") or []
        if not question:
            return jsonify(error="Write a question first."), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify(error=f"Questions are limited to {MAX_QUESTION_CHARS} characters."), 400
        if mode not in MODES:
            return jsonify(error=f"mode must be one of {list(MODES)}"), 400
        if not isinstance(history, list):
            history = []
        try:
            return jsonify(assistant.ask(question, mode, history))
        except Exception:  # keep the UI alive and the details in the server log
            log.exception("ask failed")
            return jsonify(error="The model call failed. Check the server log and your API key."), 502

    return app


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Hukuk Asistanı web app")
    parser.add_argument("--share", action="store_true", help="expose the app with an ngrok tunnel")
    args = parser.parse_args()

    settings = Settings()
    app = create_app(settings=settings)
    if args.share:
        from pyngrok import ngrok

        print("Public URL:", ngrok.connect(settings.port).public_url)
    app.run(host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
