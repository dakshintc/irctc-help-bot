"""Sensitive-input and policy tests."""

from __future__ import annotations

import pytest

from app import create_app
from guard import check_message, detect_language, wants_live_data
from rag import RAGEngine


@pytest.mark.parametrize(
    "message,reason",
    [
        ("My card number is 4111 1111 1111 1111", "card_number"),
        ("My Aadhaar is 1234 5678 9012", "aadhaar"),
        ("My OTP is 483920", "otp"),
        ("cvv: 123", "cvv"),
        ("My UPI id is rahul@okhdfcbank", "upi_id"),
        ("password: hunter2", "password"),
        ("ATM pin 4321", "pin"),
    ],
)
def test_sensitive_patterns_blocked(message, reason):
    result = check_message(message)
    assert result.kind == "sensitive"
    assert result.reason == reason
    assert result.message


@pytest.mark.parametrize(
    "message",
    [
        "How do I reset my password?",
        "What is the cancellation charge for 3A?",
        "Which ID proof should I carry?",
        "Is Tatkal available in sleeper?",
    ],
)
def test_normal_questions_not_blocked(message):
    assert check_message(message).kind == "ok"


def test_language_detection():
    assert detect_language("How do I cancel my ticket?") == "en"
    assert detect_language("टिकट कैसे रद्द करूँ?") == "hi"


def test_live_data_detection():
    assert wants_live_data("Check my PNR status right now")
    assert not wants_live_data("How do I check my PNR status?")


def test_booking_request_is_refused_but_helpful(client):
    response = client.post("/chat", json={"message": "Please book a ticket for me"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["blocked"] is True
    assert data["mode"] == "guard"
    assert "can't book" in data["answer"].lower()
    assert data["sources"]


def test_booking_howto_is_answered(client):
    response = client.post("/chat", json={"message": "How do I book an e-ticket?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["blocked"] is False
    assert "Plan My Journey".lower() in data["answer"].lower()


def test_sensitive_input_rejected_by_api(client):
    response = client.post(
        "/chat", json={"message": "My card is 4111 1111 1111 1111, is it saved?"}
    )
    assert response.status_code == 422
    data = response.get_json()
    assert data["error"] == "sensitive_data"
    assert "card" in data["message"].lower()
    # The secret must never be echoed back.
    assert "4111" not in response.get_data(as_text=True)


def test_sensitive_input_hindi(client):
    response = client.post("/chat", json={"message": "मेरा OTP 483920 है, बुक कर दो"})
    assert response.status_code == 422
    data = response.get_json()
    assert data["language"] == "hi"
    assert "ओटीपी" in data["message"] or "OTP" in data["message"]


def test_rate_limit():
    app = create_app(RAGEngine(offline=True))
    app.config.update(TESTING=True)
    import config

    old = config.RATE_LIMIT_PER_MINUTE
    config.RATE_LIMIT_PER_MINUTE = 2
    try:
        with app.test_client() as client:
            assert client.post("/chat", json={"message": "hello"}).status_code != 429
            assert client.post("/chat", json={"message": "hello"}).status_code != 429
            assert client.post("/chat", json={"message": "hello"}).status_code == 429
    finally:
        config.RATE_LIMIT_PER_MINUTE = old
