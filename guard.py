"""Input safety checks.

Blocks messages that contain credentials or payment secrets before they reach the
retriever, the LLM or the logs. Also detects "do it for me" intents that the bot
must refuse (booking, cancelling).

The checks are deliberately conservative: a false positive is a polite request to
rephrase, while a false negative could leak a real secret. Questions that ask *how*
something works are always treated as informational.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Lang = Literal["en", "hi"]


DEVANAGARI = re.compile(r"[\u0900-\u097F]")

HOWTO = re.compile(
    r"\b(how|what|where|when|which|why|steps?|process|procedure|rules?|charges?|fees?|"
    r"documents?|list|explain|guide|meaning|difference)\b"
    r"|(कैसे|क्या|कहाँ|कब|कौन|क्यों|नियम|चार्ज|शुल्क|प्रक्रिया|दस्तावेज़|मतलब|अंतर|जानकारी|बताइए|बताओ)",
    re.I,
)


def detect_language(text: str) -> Lang:
    """Return 'hi' when the text contains Devanagari, else 'en'."""
    return "hi" if DEVANAGARI.search(text or "") else "en"


@dataclass(frozen=True)
class GuardResult:
    kind: Literal["ok", "sensitive", "blocked_intent"]
    reason: str = ""
    message: str = ""


SENSITIVE_MESSAGES = {
    "en": (
        "I can't accept card numbers, CVV/PIN, OTPs, passwords, UPI IDs or Aadhaar "
        "numbers. Please remove that from your message and ask again. Never share "
        "these with anyone, including this assistant."
    ),
    "hi": (
        "मैं कार्ड नंबर, CVV/PIN, OTP, पासवर्ड, UPI आईडी या आधार नंबर नहीं ले सकता। "
        "कृपया उसे संदेश से हटाकर दोबारा पूछें। ये जानकारी किसी को भी न दें, "
        "इस सहायक को भी नहीं।"
    ),
}

SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 13-19 digits, optionally separated by spaces/dashes (card-like).
    ("card_number", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")),
    # Aadhaar: 12 digits, often grouped 4-4-4.
    ("aadhaar", re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")),
    ("otp", re.compile(r"\b(?:otp|o\.t\.p\.?|ओटीपी)\b[^\d]{0,25}\d{4,8}\b", re.I)),
    (
        "otp_number",
        re.compile(r"\b\d{4,8}\b\s*(?:is|है)?\s*(?:my|mera|मेरा)?\s*(?:otp|ओटीपी)\b", re.I),
    ),
    ("cvv", re.compile(r"\b(?:cvv|cvc|सीवीवी)\b\s*[:=\-]?\s*\d{3,4}\b", re.I)),
    (
        "upi_id",
        re.compile(
            r"\b[\w.\-]{2,}@(?:oksbi|okhdfcbank|okicici|okaxis|paytm|ybl|ibl|axl|upi|apl|jupiteraxis)\b",
            re.I,
        ),
    ),
    (
        "password",
        re.compile(
            r"\b(?:my\s+)?(?:password|passwd|pwd|पासवर्ड)\b\s*(?:is|है|:|=|\-)\s*\S+", re.I
        ),
    ),
    ("pin", re.compile(r"\b(?:atm\s*)?pin\b\s*[:=\-]?\s*\d{4,6}\b", re.I)),
    (
        "card_expiry",
        re.compile(
            r"\b(?:exp(?:iry)?|valid\s*(?:thru|till))\b\s*[:=\-]?\s*\d{1,2}\s*/\s*\d{2,4}\b",
            re.I,
        ),
    ),
]

# "Do it for me" intents: not dangerous, but outside what the bot can do.
BOOKING_REQUEST = re.compile(
    r"\b(?:please|kindly)\s+book\b"
    r"|\bbook\b[^.?!]{0,25}\bfor me\b"
    r"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?book\b"
    r"|\bbook\s+my\s+(?:ticket|seat|berth)\b"
    r"|\bi\s+want\s+(?:you\s+)?to\s+book\b"
    r"|मेरे\s*लिए\s*(?:टिकट|सीट|बर्थ)?\s*(?:बुक|आरक्षित)\s*(?:करो|कर\s*दो|कीजिए|करें)",
    re.I,
)
CANCEL_REQUEST = re.compile(
    r"\b(?:please|kindly)\s+cancel\b"
    r"|\bcancel\b[^.?!]{0,20}\bfor me\b"
    r"|\b(?:can|could|would|will)\s+you\s+(?:please\s+)?cancel\b"
    r"|\bcancel\s+my\s+(?:ticket|booking|reservation)\b"
    r"|मेरा\s*(?:टिकट|आरक्षण)\s*(?:रद्द|कैंसिल)\s*(?:करो|कर\s*दो|कीजिए|करें)",
    re.I,
)

LIVE_DATA = re.compile(
    r"\b(?:live|current|right now|today'?s?)\b[^.?!]{0,40}\b(?:pnr|seat|availability|fare|status)\b"
    r"|\b(?:check|show|tell)\s+(?:me\s+)?(?:the\s+)?(?:my\s+)?(?:pnr|seat availability|live status)\b"
    r"|(?:मेरा|मेरी)\s*(?:pnr|सीट|किराया)\s*(?:चेक|देख|बता)"
    r"|\b(?:seat availability|live train status)\b",
    re.I,
)


def wants_live_data(message: str) -> bool:
    """True when the user is asking for data this bot cannot provide.

    "How do I check my PNR?" is a help question, not a live lookup.
    """
    text = message or ""
    return bool(LIVE_DATA.search(text)) and not bool(HOWTO.search(text))


def check_message(message: str, lang: Lang | None = None) -> GuardResult:
    """Screen one user message. Returns ok / sensitive / blocked_intent."""
    text = (message or "").strip()
    lang = lang or detect_language(text)
    informational = bool(HOWTO.search(text))

    for name, pattern in SENSITIVE_PATTERNS:
        if pattern.search(text):
            return GuardResult(
                kind="sensitive",
                reason=name,
                message=SENSITIVE_MESSAGES[lang],
            )

    # Explicit request to perform a real transaction (but not "how do I ...").
    if not informational and (BOOKING_REQUEST.search(text) or CANCEL_REQUEST.search(text)):
        if lang == "hi":
            msg = (
                "मैं टिकट बुक, बदल या रद्द नहीं कर सकता — मैं सिर्फ जानकारी देने वाला "
                "अनौपचारिक सहायक हूँ। बुकिंग/कैंसिलेशन के लिए irctc.co.in या IRCTC Rail "
                "Connect ऐप इस्तेमाल करें। नीचे जानकारी और सही पेज का लिंक दिया है।"
            )
        else:
            msg = (
                "I can't book, change or cancel a ticket — I'm an unofficial information "
                "assistant only. Please use irctc.co.in or the IRCTC Rail Connect app for "
                "the actual transaction. Here is the guidance and the right official page."
            )
        return GuardResult(kind="blocked_intent", reason="transaction", message=msg)

    return GuardResult(kind="ok")
