"""HTTP contract tests for /health and /chat."""

from __future__ import annotations


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["chunks"] > 0
    assert data["backend"] == "faiss"
    assert "disclaimer" in data


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Unofficial assistant. Check irctc.co.in for final info." in body


def test_chat_returns_bullets_sources_and_disclaimer(client):
    response = client.post("/chat", json={"message": "How do I check my PNR status?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["language"] == "en"
    assert data["disclaimer"].startswith("Unofficial assistant")
    assert data["sources"]
    assert all("url" in source and "title" in source for source in data["sources"])
    assert "- " in data["answer"]


def test_chat_hindi_language(client):
    response = client.post(
        "/chat", json={"message": "तत्काल बुकिंग का समय क्या है?"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["language"] == "hi"
    assert "अनौपचारिक" in data["disclaimer"]


def test_forced_language_override(client):
    response = client.post(
        "/chat", json={"message": "How do I check my PNR status?", "lang": "hi"}
    )
    assert response.status_code == 200
    assert response.get_json()["language"] == "hi"


def test_empty_message(client):
    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400
    assert response.get_json()["error"] == "empty_message"


def test_missing_json(client):
    response = client.post("/chat", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_too_long_message(client):
    response = client.post("/chat", json={"message": "a" * 1500})
    assert response.status_code == 400
    assert response.get_json()["error"] == "too_long"


def test_off_topic_gets_unsure_fallback(client):
    response = client.post("/chat", json={"message": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"] == "I'm not sure — please check irctc.co.in."
    assert data["confident"] is False


def test_history_is_accepted_and_bounded(client):
    response = client.post(
        "/chat",
        json={
            "message": "What are the cancellation charges for 2A?",
            "history": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Namaste"},
                {"role": "system", "content": "ignored"},
            ],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["answer"]


def test_result_has_no_credentials_and_no_booking_claim(client):
    response = client.post("/chat", json={"message": "Can I book a Tatkal ticket in First AC?"})
    data = response.get_json()
    text = data["answer"].lower()
    assert "except first ac" in text
    assert "i have booked" not in text
    assert "your ticket is confirmed" not in text
