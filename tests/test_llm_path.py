"""Tests for the Claude answer path, using a mocked Anthropic client.

The environment used for the build did not have a valid API key, so these tests
exercise everything except the real network call: request shape, grounded system
prompt, language directive, source plumbing, and the offline fallback on failure.
"""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import pytest

import config
from rag import RAGEngine


def _install_fake_client(monkeypatch, text="- Mocked answer from context.", fail=None):
    captured = {"calls": 0, "kwargs": None}

    class FakeMessages:
        def create(self, **kwargs):
            captured["calls"] += 1
            captured["kwargs"] = kwargs
            if fail:
                raise fail
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                model="mock-model",
            )

    class FakeClient:
        def __init__(self, api_key=None):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    return captured


@pytest.fixture
def llm_engine(monkeypatch):
    """An engine that believes Claude is configured, but uses a fake client."""
    monkeypatch.setattr(config, "LLM_ENABLED", True)
    monkeypatch.setattr(config, "OFFLINE", False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key-not-real")
    return RAGEngine(offline=False)


def test_claude_path_returns_mode_and_sources(llm_engine, monkeypatch):
    captured = _install_fake_client(monkeypatch)
    result = llm_engine.answer(
        "What are the cancellation charges for 3A?",
        lang="en",
        force_offline=False,
    )

    assert result["mode"] == "claude"
    assert result["answer"] == "- Mocked answer from context."
    assert result["language"] == "en"
    assert result["sources"], "Claude answers must still carry source links"
    assert captured["calls"] == 1

    kwargs = captured["kwargs"]
    assert kwargs["model"] == config.CLAUDE_MODEL
    assert kwargs["max_tokens"] == 500
    system = kwargs["system"]
    assert "Use ONLY the CONTEXT" in system
    assert "Never ask for or repeat passwords" in system
    assert "Never claim to book" in system
    assert "English" in system

    last_message = kwargs["messages"][-1]["content"]
    assert last_message.startswith("CONTEXT:")
    assert "QUESTION: What are the cancellation charges for 3A?" in last_message
    # Grounding: retrieved chunks must be present in the prompt.
    assert "₹180" in last_message or "3 Tier" in last_message


def test_claude_prompt_switches_to_hindi(llm_engine, monkeypatch):
    captured = _install_fake_client(monkeypatch, text="- क्लाउड उत्तर")
    result = llm_engine.answer("तत्काल बुकिंग का समय क्या है?", force_offline=False)

    assert result["mode"] == "claude"
    assert result["language"] == "hi"
    assert "Hindi (Devanagari)" in captured["kwargs"]["system"]
    assert "तत्काल" in captured["kwargs"]["messages"][-1]["content"]


def test_claude_failure_falls_back_and_is_not_retried(llm_engine, monkeypatch):
    captured = _install_fake_client(monkeypatch, fail=RuntimeError("invalid api key"))

    first = llm_engine.answer(
        "What are the cancellation charges for 3A?", force_offline=False
    )
    assert first["mode"] == "offline-fallback"
    assert "180" in first["answer"]
    assert llm_engine._llm_failed is True

    second = llm_engine.answer(
        "What are the cancellation charges for 2A?", force_offline=False
    )
    assert second["mode"] == "offline-fallback"
    assert "200" in second["answer"]
    # The failing API must be attempted only once per process.
    assert captured["calls"] == 1


def test_history_is_forwarded_to_claude(llm_engine, monkeypatch):
    captured = _install_fake_client(monkeypatch)
    llm_engine.answer(
        "What is the cancellation charge for 2A?",
        lang="en",
        history=[
            {"role": "user", "content": "What are the cancellation charges for 3A?"},
            {"role": "assistant", "content": "- ₹180 plus GST."},
            {"role": "system", "content": "should be ignored"},
        ],
        force_offline=False,
    )
    roles = [m["role"] for m in captured["kwargs"]["messages"]]
    assert roles == ["user", "assistant", "user"]
