# IRCTC Help — Unofficial Assistant

A small, self-hosted chatbot that answers common **IRCTC / Indian Railways help questions** in short, point-form **English or Hindi**. It retrieves answers from a local knowledge base built only from public IRCTC and Indian Railways help pages, cites the official source pages, and refuses anything outside that scope.

> **Unofficial assistant. Check irctc.co.in for final info.**
> This project is not affiliated with IRCTC or Indian Railways. It cannot book, cancel or change tickets, and it does not show live PNR, seat, fare or train-running data.

---

## What it can answer

- E-ticket booking steps, registration, payment options, e-ticket vs I-ticket
- Cancellation steps, cancellation charges, RAC/waitlist refunds, TDR
- Refunds when a train is cancelled or running more than 3 hours late
- Tatkal timings (10:00 / 11:00), charges, class rules, passenger limits
- Accepted photo ID proofs and what happens if ID is missing
- PNR: what it is and where to check it (no live lookup)
- Travel classes: 1A, 2A, 3A, 3E, CC, EC, SL, 2S, FC
- Quotas: GN, LD, TQ, PT, SS, HP, FT, DF, YU, RAC and more
- Account safety: passwords, OTPs, card/UPI fraud warnings

**Hard boundaries:** no passwords/OTPs/card/UPI/Aadhaar collection, no booking, no live data, no answers invented outside the retrieved corpus.

---

## How it works

```
Browser (HTML/CSS/JS)
        │  POST /chat
        ▼
Flask (app.py)
        ├── guard.py ....... sensitive-data + transaction-intent checks
        ├── rag.py ......... language detect → hybrid retrieve → answer
        │        ├── FAISS index (cosine)
        │        ├── sentence-transformers multilingual MiniLM
        │        └── Claude API (optional) / offline extractive fallback
        └── GET /health .... index + model status
```

- **Chunking:** one `##` section per chunk (~850 chars), split on sentence boundaries with overlap.
- **Retrieval:** sentence embeddings + keyword matches. Claude gets the **top 4 chunks**; the offline answerer looks at a few more so it can pick a coherent FAQ section.
- **Answering:**
  - With a valid `ANTHROPIC_API_KEY`, Claude answers strictly from the retrieved context.
  - Without a key (or if the API rejects it), a deterministic extractive answerer returns bullets from the same chunks, so the app always works offline.
- **Sources:** every confident answer includes 1–3 official links; "not sure" replies show no citation rather than an irrelevant one.
- **Unsure:** weak retrieval returns exactly `I'm not sure — please check irctc.co.in.` (Hindi: `मुझे पक्का नहीं पता — कृपया irctc.co.in पर देखें।`).

### Why FAISS instead of Chroma?

Chroma's persistent client uses SQLite. On network-mounted filesystems (S3/FUSE, some Docker/NFS setups) SQLite fails with `disk I/O error`. FAISS stores a plain byte index, which works on both local disk and network mounts. Chroma was the first choice in the brief; FAISS is the explicitly allowed alternative and is what this build uses.

---

## Requirements

- Python **3.10+**
- ~2 GB free RAM (the embedding model is small, but torch loads it)
- ~600 MB disk for the model on first run (cached by Hugging Face)
- No GPU required
- Optional: an Anthropic API key for Claude-generated wording (the app works without one)

---

## Quick start

```bash
cd irctc-help-bot

# 1. Virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the environment (optional Claude key)
cp .env.example .env
# edit .env and paste your key after ANTHROPIC_API_KEY=

# 4. Build the knowledge base
python ingest.py

# 5. Run the app
python app.py
```

Open **http://localhost:5000**.

Shortcuts with `make` (optional):

```bash
make install   # pip install -r requirements.txt
make ingest    # python ingest.py
make run       # python app.py
make offline   # RAG_OFFLINE=1 python app.py
make test      # pytest
```

The first `ingest.py` run downloads the embedding model (~470 MB) and prints progress. Later runs use the cache.

### Prebuilt package

The delivered `irctc-help-bot.zip` includes a prebuilt `data/index/` and the scraped
`data/raw/` text, so after `pip install -r requirements.txt` you can start the app
immediately. Run `python ingest.py` after adding or editing FAQ files. If you clone the
source without the index, `python ingest.py` is required before `python app.py`.

### Minimal offline run (no API key)

```bash
pip install -r requirements.txt
python ingest.py
RAG_OFFLINE=1 python app.py
```

`RAG_OFFLINE=1` forces the extractive answerer, so no network or API key is needed at runtime.

---

## `.env` configuration

Copy `.env.example` to `.env`. **Never commit `.env`.**

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Claude API key. Leave empty to use the offline answerer. |
| `CLAUDE_MODEL` | `claude-sonnet-4-5-20250929` | Model used for grounded answers. |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Must stay the same between ingest and runtime. |
| `TOP_K` | `4` | Chunks sent to Claude. |
| `MAX_ANSWER_BULLETS` | `8` | Upper bound for offline bullets. |
| `MIN_SCORE` | `0.28` | Minimum cosine similarity to treat a chunk as relevant. |
| `RAG_OFFLINE` | `0` | `1` forces offline answers (no API calls). |
| `MAX_INPUT_CHARS` | `1000` | Maximum accepted question length. |
| `RATE_LIMIT_PER_MINUTE` | `40` | Per-IP request limit for `/chat`. |
| `SCRAPE_DELAY` | `2.0` | Seconds between scraper requests. |
| `SCRAPER_USER_AGENT` | `IRCTCHelpBot/1.0 ...` | Identifying User-Agent; set a contact address if you deploy this. |
| `REQUEST_TIMEOUT` | `30` | HTTP timeout for scraping, in seconds. |

Example `.env`:

```dotenv
ANTHROPIC_API_KEY=your_key_here
CLAUDE_MODEL=claude-sonnet-4-5-20250929
RAG_OFFLINE=0
```

If the key is missing or rejected (HTTP 401), the app logs it once, falls back to the offline answerer, and keeps serving requests.

---

## Scraper

```bash
python scraper.py              # fetch uncached sources
python scraper.py --force      # re-fetch everything
python scraper.py --offline    # report what is cached, no network
python scraper.py --list       # show configured URLs
python scraper.py --only tatkal
```

What it does:

- Checks `robots.txt` per host before fetching.
  - `contents.irctc.co.in` publishes no restrictions → allowed.
  - `www.irctc.co.in` returns **HTTP 403** for robots and blocks non-browser clients, so those URLs are **skipped**.
- Sends an identifying User-Agent, waits `SCRAPE_DELAY` seconds between requests, retries up to 3 times.
- Extracts text from HTML and PDFs, strips navigation/footer boilerplate, and writes cleaned text plus provenance to `data/raw/<slug>.txt`.
- Writes `data/raw/manifest.json` with status for every source.

```bash
python scraper.py
python ingest.py
```

The scraper is optional: the bundled `data/faqs/` corpus is enough to build and run the bot.

### Adding your own FAQ files

If IRCTC changes or blocks a page, drop your own text into the corpus:

```
data/faqs/en/my-topic.md
data/faqs/hi/my-topic.md
```

Front matter is optional but recommended:

```markdown
---
title: My IRCTC topic
source: https://www.irctc.co.in/some-page
source_title: IRCTC official page
lang: en
category: booking
---

## A heading that becomes a chunk

- One fact per bullet.
- Keep numbers, class codes and rupee amounts exact.
```

Then rebuild:

```bash
python ingest.py
```

`ingest.py` reads `.md` and `.txt` files under `data/faqs/`. It also supports `--include-raw` (ingest scraped `data/raw` text too) and `--source <path>`.

---

## Running the app

```bash
python app.py
```

- UI: <http://localhost:5000>
- Health: <http://localhost:5000/health>

For a different port or host, use Flask's CLI:

```bash
flask --app app run --host 0.0.0.0 --port 8000
```

For production, put it behind gunicorn/waitress:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```

The in-memory rate limiter is per process; use a shared store if you run multiple workers behind a load balancer.

### API

`POST /chat`

```bash
curl -s http://localhost:5000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What are the cancellation charges for 3A?","lang":"auto"}'
```

```json
{
  "answer": "Cancellation charges for confirmed tickets:\n- AC Chair Car / AC 3 Tier / AC 3 Economy: ₹180 plus GST.\n- ...",
  "language": "en",
  "sources": [
    {"title": "IRCTC E-Ticket Cancel / Refund Rules w.e.f. 12-Nov-2015", "url": "https://contents.irctc.co.in/en/Eticket_new_cancel.html"}
  ],
  "mode": "offline",
  "confident": true,
  "blocked": false,
  "disclaimer": "Unofficial assistant. Check irctc.co.in for final info."
}
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `message` | string | Required, 1–1000 characters. |
| `lang` | `"auto"` \| `"en"` \| `"hi"` | Defaults to auto (Devanagari → Hindi). |
| `history` | array | Optional previous turns, max 6 used. |

Status codes:

- `200` answer (including policy refusals, which have `blocked: true`)
- `400` empty / too long / bad JSON
- `422` `sensitive_data` — card, OTP, CVV, UPI, password, Aadhaar
- `429` rate limited
- `503` index missing (run `python ingest.py`)

`GET /health`

```bash
curl -s http://localhost:5000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "chunks": 67,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "collection": "irctc_help",
  "backend": "faiss",
  "llm": "offline",
  "disclaimer": "Unofficial assistant. Check irctc.co.in for final info."
}
```

---

## Testing

```bash
pytest
```

50 tests covering:

- 10 English sample questions with expected facts (plus the 8/10 aggregate check)
- 5 Hindi questions with expected facts
- Sensitive-input blocking (card, Aadhaar, OTP, CVV, UPI, password, PIN)
- Policy refusal for "book/cancel for me", while "how do I book?" is answered
- `/health`, `/chat`, validation, language override, sources, disclaimer
- Off-topic fallback and bullet-length limits
- The Claude path with a mocked Anthropic client: prompt/grounding, Hindi directive,
  history forwarding, source plumbing, and the one-time fallback on API failure

Tests force `RAG_OFFLINE=1`, so they never call the API or need a key. If the index is missing, the fixture builds it once. The real network call is the only thing a mocked test cannot cover — run one live `/chat` request after adding a valid key to confirm it.

Useful variants:

```bash
pytest -v
pytest tests/test_sample_questions.py
pytest -k "hindi"
```

---

## Project structure

```
irctc-help-bot/
├── PLAN.md
├── README.md
├── requirements.txt
├── .env.example
├── pytest.ini
├── Makefile
├── config.py               # env loading + constants
├── guard.py                # sensitive input + transaction-intent checks
├── scraper.py              # polite scraper with robots.txt handling
├── ingest.py               # clean → chunk → embed → FAISS
├── vectorstore.py          # FAISS save/load (single-file index)
├── rag.py                  # retrieval + Claude/offline answering
├── app.py                  # Flask routes
├── templates/index.html    # chat page
├── static/
│   ├── style.css
│   └── chat.js
├── data/
│   ├── faqs/en/*.md        # curated English corpus
│   ├── faqs/hi/*.md        # curated Hindi corpus
│   ├── raw/                # scraped text + manifest (generated)
│   └── index/              # vectors.faiss + chunks.jsonl (generated)
└── tests/
    ├── conftest.py
    ├── test_guard.py
    ├── test_api.py
    ├── test_llm_path.py       # mocked Claude client
    └── test_sample_questions.py
```

Generated and ignored: `data/index/`, `data/raw/*`, `data/build_info.json`, `.env`, `__pycache__/`.

---

## Data sources

Only public help/FAQ content:

- `https://contents.irctc.co.in/en/etktfaq.html`
- `.../TatkalFaq.html`, `.../TatkalBooking.html`
- `.../Eticket_new_cancel.html`, `.../CancellationProcedure.html`
- `.../Waitlisted_E-Ticket.html`, `.../Registration.html`, `.../ITicket.html`
- `.../REFUND RULES wef 12-Nov-15.pdf`
- `.../Chatbot_Cancellation_Refund_English.pdf`
- `.../CC_68_2012_id_proof.pdf`
- `https://www.indianrail.gov.in/StaticContents/quota_Code.html`
- `.../tatkal_Scheme.html`, `.../refund_Rules.html`

The curated files in `data/faqs/` summarise these pages (and their Hindi counterparts) with the source URL preserved in front matter. Facts such as charges, timings and class codes are copied from those official pages; no external or invented rules are added.

Content is © IRCTC / Indian Railways. This project is for educational/help-desk use; check the official pages before acting.

---

## Troubleshooting

### `disk I/O error` / SQLite errors

You are probably on a network-mounted filesystem. This build uses FAISS specifically to avoid Chroma's SQLite dependency. If you see this from another tool, move the project to local disk or set the working directory to a local path.

### `Vector index not found. Run python ingest.py first.`

Run:

```bash
python ingest.py
```

Check that `data/index/vectors.faiss` and `data/index/chunks.jsonl` exist. `/health` returns `503 degraded` until the index is built.

### `python ingest.py` is slow or stalls at model download

The embedding model is ~470 MB and downloads on first use. Check your connection, or pre-download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

For a fully offline install, download the model on another machine and set `EMBED_MODEL` to the local folder path. Keep the same model for ingest and runtime, otherwise the index will not match queries.

### Claude answers are not used / `401 MISSING_AUTH` / `AuthenticationError`

- Check `.env` contains a real `ANTHROPIC_API_KEY` and no quotes/spaces.
- Verify the key with a small request outside this app.
- If the key is invalid, the app logs one warning and uses the offline answerer (`mode: "offline-fallback"`). This is expected and safe.
- Set `RAG_OFFLINE=1` to skip API attempts entirely.

### Answers say "I'm not sure"

- The question may be outside the corpus (live PNR, seat availability, fares, booking).
- Or the wording is too far from the FAQ text. Rephrase with IRCTC terms ("Tatkal", "3A", "cancellation charges").
- Add a FAQ file under `data/faqs/en/` or `data/faqs/hi/`, then re-run `python ingest.py`.

### Hindi answers are missing or partly English

- Add Hindi documents under `data/faqs/hi/` with `lang: hi` in front matter.
- Rebuild the index: `python ingest.py`.
- The retriever prefers Hindi chunks; if a topic has only English source text, it answers in English with a short Hindi lead explaining that.

### Scraper skips `www.irctc.co.in`

That host returns **HTTP 403** for `robots.txt` and blocks non-browser clients. The scraper is being polite and skips it. Use `contents.irctc.co.in` sources, or add FAQ text manually under `data/faqs/`.

### Scraper reports "empty" for a PDF

Some PDFs are scanned images without a text layer. Extract them with OCR yourself and save as `.txt`/`.md` under `data/faqs/`, or skip them — the curated corpus already covers the key facts.

### `Address already in use` on port 5000

```bash
flask --app app run --port 8000
# or
python -c "import app; app.app.run(port=8000)"
```

### `pytest` cache warnings on a network mount

`pytest.ini` already points `cache_dir` to `/tmp`. If your environment still complains, run `pytest -p no:cacheprovider`.

### Rate limited (HTTP 429)

The default is 40 requests/minute per IP. Wait a minute, or raise `RATE_LIMIT_PER_MINUTE` in `.env`. The limiter is in-memory and resets when the process restarts.

### Sensitive input rejected (HTTP 422)

This is intentional. Remove card numbers, Aadhaar numbers, OTPs, CVV/PIN, UPI IDs and passwords from the message. The bot never needs them, and the blocked message is not stored in history.

---

## Limitations

- Not affiliated with IRCTC; no live data.
- Cannot book, cancel, modify or check a real ticket.
- The corpus is a snapshot and may lag official rule changes (especially charges and timings).
- Offline answers are extractive bullets, not conversational prose; Claude mode is smoother but needs a valid key.
- English and Hindi only.
- The rate limiter is per process and in-memory.
- No user accounts, database, analytics or chat persistence; conversations live in the browser tab.

---

## License / attribution

Code in this repository is provided as-is for educational use. IRCTC and Indian Railways content, names and marks belong to their respective owners. Always confirm final rules on [irctc.co.in](https://www.irctc.co.in).
