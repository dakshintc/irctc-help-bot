"""Configuration for the unofficial IRCTC help chatbot.

All secrets come from the environment (optionally loaded from a local .env file).
Nothing sensitive is ever written into source code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
FAQ_DIR = DATA_DIR / "faqs"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
INDEX_NAME = os.getenv("INDEX_NAME", "irctc_help")

TOP_K = int(os.getenv("TOP_K", "4"))
MAX_ANSWER_BULLETS = int(os.getenv("MAX_ANSWER_BULLETS", "8"))

# Minimum cosine similarity for a chunk to count as relevant.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.28"))

# Offline mode forces the extractive answerer and never calls the API.
OFFLINE = os.getenv("RAG_OFFLINE", "0").strip().lower() in {"1", "true", "yes", "on"}

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
LLM_ENABLED = bool(ANTHROPIC_API_KEY) and not OFFLINE

DISCLAIMER = "Unofficial assistant. Check irctc.co.in for final info."
DISCLAIMER_HI = "अनौपचारिक सहायक। अंतिम जानकारी के लिए irctc.co.in देखें।"

UNSURE = "I'm not sure — please check irctc.co.in."
UNSURE_HI = "मुझे पक्का नहीं पता — कृपया irctc.co.in पर देखें।"

# Politeness for the scraper (seconds between requests).
SCRAPE_DELAY = float(os.getenv("SCRAPE_DELAY", "2.0"))
USER_AGENT = os.getenv(
    "SCRAPER_USER_AGENT",
    "IRCTCHelpBot/1.0 (+educational, unofficial; contact: local-admin)",
)
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "40"))
