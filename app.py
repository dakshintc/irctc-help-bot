"""Flask app for the unofficial IRCTC help chatbot.

Routes:
    GET  /         chat UI
    POST /chat     {"message": "...", "lang": "auto|en|hi", "history": []}
    GET  /health   index and model status
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from flask import Flask, jsonify, render_template, request

import config
from guard import check_message, detect_language
from rag import RAGEngine

VERSION = "1.0.0"


def create_app(engine: RAGEngine | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    bot = engine or RAGEngine()
    app.extensions["rag"] = bot

    # Simple in-process per-IP rate limit (good enough for a single worker).
    hits: dict[str, deque[float]] = defaultdict(deque)
    lock = threading.Lock()

    def rate_limited(ip: str) -> bool:
        now = time.time()
        window = now - 60
        with lock:
            bucket = hits[ip]
            while bucket and bucket[0] < window:
                bucket.popleft()
            if len(bucket) >= config.RATE_LIMIT_PER_MINUTE:
                return True
            bucket.append(now)
            return False

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.path.startswith("/chat"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            disclaimer=config.DISCLAIMER,
            disclaimer_hi=config.DISCLAIMER_HI,
            version=VERSION,
        )

    @app.get("/health")
    def health():
        info = bot.health()
        status = "ok" if info["chunks"] > 0 else "degraded"
        return (
            jsonify(
                {
                    "status": status,
                    "version": VERSION,
                    "chunks": info["chunks"],
                    "embedding_model": info["embedding_model"],
                    "collection": info["collection"],
                    "backend": info.get("backend", "faiss"),
                    "llm": info["llm"],
                    "disclaimer": config.DISCLAIMER,
                }
            ),
            200 if status == "ok" else 503,
        )

    @app.post("/chat")
    def chat():
        ip = request.headers.get("X-Forwarded-For", request.remote_addr or "local").split(",")[0]
        if rate_limited(ip):
            return (
                jsonify(
                    {
                        "error": "rate_limited",
                        "message": "Too many messages. Please wait a minute and try again.",
                    }
                ),
                429,
            )

        payload: Any = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "bad_request", "message": "Send a JSON body."}), 400

        message = payload.get("message", "")
        if not isinstance(message, str):
            return jsonify({"error": "bad_request", "message": "message must be text."}), 400
        message = message.strip()
        if not message:
            return jsonify({"error": "empty_message", "message": "Please type a question."}), 400
        if len(message) > config.MAX_INPUT_CHARS:
            return (
                jsonify(
                    {
                        "error": "too_long",
                        "message": f"Please keep the question under {config.MAX_INPUT_CHARS} characters.",
                    }
                ),
                400,
            )

        requested_lang = payload.get("lang", "auto")
        if requested_lang not in ("auto", "en", "hi"):
            requested_lang = "auto"
        language = detect_language(message) if requested_lang == "auto" else requested_lang

        history = payload.get("history") or []
        if not isinstance(history, list):
            history = []
        clean_history = []
        for turn in history[-6:]:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                clean_history.append({"role": role, "content": content[:1200]})

        # Reject credentials before anything is embedded, logged or sent to an API.
        guard = check_message(message, language)
        if guard.kind == "sensitive":
            app.logger.warning("Blocked sensitive input (%s) from %s", guard.reason, ip)
            return (
                jsonify(
                    {
                        "error": "sensitive_data",
                        "reason": guard.reason,
                        "message": guard.message,
                        "language": language,
                        "blocked": True,
                        "disclaimer": config.DISCLAIMER_HI
                        if language == "hi"
                        else config.DISCLAIMER,
                    }
                ),
                422,
            )

        if guard.kind == "blocked_intent":
            chunks = bot.retrieve(message, lang=language)
            return jsonify(
                {
                    "answer": guard.message,
                    "language": language,
                    "sources": RAGEngine._sources(chunks, 2),
                    "mode": "guard",
                    "blocked": True,
                    "confident": True,
                    "disclaimer": config.DISCLAIMER_HI
                    if language == "hi"
                    else config.DISCLAIMER,
                }
            )

        try:
            result = bot.answer(message, lang=language, history=clean_history)
        except FileNotFoundError:
            return (
                jsonify(
                    {
                        "error": "index_missing",
                        "message": "Knowledge base not built. Run: python ingest.py",
                    }
                ),
                503,
            )
        except Exception as exc:  # noqa: BLE001 - never leak internals to the browser
            app.logger.exception("chat failed: %s", exc)
            return (
                jsonify(
                    {
                        "error": "server_error",
                        "message": "Something went wrong. Please try again.",
                    }
                ),
                500,
            )

        payload_out = dict(result)
        payload_out["disclaimer"] = (
            config.DISCLAIMER_HI if result["language"] == "hi" else config.DISCLAIMER
        )
        payload_out["blocked"] = False
        return jsonify(payload_out)

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/chat"):
            return jsonify({"error": "not_found"}), 404
        return render_template("index.html", disclaimer=config.DISCLAIMER,
                               disclaimer_hi=config.DISCLAIMER_HI, version=VERSION), 404

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "too_large", "message": "Request body is too large."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    print(f"IRCTC help chatbot v{VERSION}")
    print(f"Embedding model: {config.EMBED_MODEL}")
    print(
        "Answer mode: Claude if the key is valid, otherwise offline fallback"
        if config.LLM_ENABLED
        else "Answer mode: offline (RAG_OFFLINE=1 or no API key)"
    )
    print("Open http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
