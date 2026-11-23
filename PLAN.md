# PLAN — Unofficial IRCTC Help Chatbot

> **Unofficial assistant. Check irctc.co.in for final info.**
> This project is not affiliated with IRCTC or Indian Railways.

## 1. Goal

A small, self-hosted chatbot that answers common IRCTC/Indian Railways help questions
(booking, cancellation, refunds, Tatkal, ID rules, PNR, travel classes, quotas) in short
point-form English or Hindi, points users to the right official IRCTC page, and refuses to
handle sensitive data or pretend to book tickets.

**Out of scope (hard boundaries):**

- No ticket booking, no modification of real bookings.
- No live PNR status, seat availability, fare or train-running data.
- No collection of passwords, OTPs, card numbers, CVV, UPI IDs or account credentials.
- No answers invented outside the retrieved corpus.

## 2. Success criteria

| Check | Target |
|---|---|
| `pip install -r requirements.txt` | succeeds |
| `python ingest.py` | builds `data/index/` (FAISS) and prints chunk count |
| `python app.py` | chat UI on <http://localhost:5000> |
| `pytest` | all tests pass |
| Answer quality | ≥ 8 of 10 sample questions contain the expected facts |
| Boundaries | sensitive input is rejected; off-topic questions get the "not sure" reply |

## 3. Architecture

```
Browser (HTML/CSS/JS)
        │  POST /chat  {message, lang, history}
        ▼
Flask app (app.py)
        │
        ├── guard.py ......... PII/sensitive-input detection, policy intents
        ├── rag.py ........... language detect → retrieve top 4 → answer
        │        ├── FAISS index (cosine)
        │        ├── sentence-transformers multilingual MiniLM
        │        └── Claude API (optional; offline extractive fallback)
        └── GET /health ...... index + model status
```

### Data flow

1. `scraper.py` fetches public IRCTC help/FAQ pages and PDFs, honouring `robots.txt`
   and a request delay, and writes cleaned text + provenance to `data/raw/`.
2. Curated FAQ files in `data/faqs/<lang>/` are derived from those official pages
   (same facts, readable Q&A form, with source URLs in front matter).
3. `ingest.py` cleans, chunks by heading (~850 chars), embeds, and persists to a FAISS index.
4. `rag.py` retrieves the top 4 chunks, then either asks Claude (strict grounded prompt)
   or produces an extractive bullet answer when no valid key is configured.

## 4. Source policy

Only public IRCTC / Indian Railways help pages:

- `https://contents.irctc.co.in/en/*.html` — IRCTC's public help/FAQ host
  (e-ticket FAQ, Tatkal FAQ/guide, cancellation procedure, registration, I-ticket,
  waitlisted e-ticket, ID proof circular PDF, refund rules PDF).
- `https://www.indianrail.gov.in/StaticContents/*.html` — official quota-code and
  refund/Tatkal rule pages linked from IRCTC's own "Rules and Policies" page.

`https://www.irctc.co.in/robots.txt` itself returns HTTP 403 from Akamai and the main
site blocks non-browser clients, so the scraper records that and skips those URLs.
`contents.irctc.co.in` returns 404 for `/robots.txt` (no restrictions published), and is
fetched politely: identifying User-Agent, 2 s delay, 3 retries, local cache.

If the network is unavailable, ingestion still works from `data/faqs/`. The user can drop
their own `.md` / `.txt` FAQ files into `data/faqs/en/` or `data/faqs/hi/` and re-run
`python ingest.py`.

## 5. Chunking and retrieval

- Chunk = one `##` section of a FAQ document, capped at ~800 characters with sentence-boundary
  splitting and ~120 characters overlap.
- Metadata per chunk: `source` (URL), `source_title`, `title`, `heading`, `lang`, `category`.
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, EN+HI).
- Vector store: FAISS `IndexFlatIP` over normalized vectors (cosine), stored as
  `data/index/vectors.faiss` plus `data/index/chunks.jsonl`. FAISS is used instead of
  Chroma because Chroma's persistent backend needs SQLite, which fails with
  `disk I/O error` on network-mounted workspaces. FAISS is an allowed option in the brief.
- Retrieval: top 4; query language is used as a soft filter (falls back to all languages
  when fewer than 2 matches exist).

## 6. Answer rules

1. Only from retrieved context. If context is weak → exact fallback:
   - EN: “I'm not sure — please check irctc.co.in.”
   - HI: “मुझे पक्का नहीं पता — कृपया irctc.co.in पर देखें।”
2. Short point form, max 6 bullets, one idea per bullet, simple words.
3. Always attach 1–3 source links from the retrieved chunks.
4. Never ask for or echo passwords/OTP/card/UPI details.
5. Never claim to book, cancel or check live data.
6. Reply in the language of the user (Devanagari → Hindi; UI selector overrides).

## 7. API

`POST /chat`

```json
{ "message": "How do I cancel my ticket?", "lang": "auto", "history": [] }
```

```json
{
  "answer": "- ...",
  "language": "en",
  "sources": [{"title": "...", "url": "..."}],
  "mode": "offline",
  "disclaimer": "Unofficial assistant. Check irctc.co.in for final info."
}
```

Errors: `400` for empty/oversized input; `422` with `error: "sensitive_data"` for PII;
`429` when rate-limited.

`GET /health` → `{"status":"ok","chunks":N,"embedding_model":"...","llm":"configured|offline"}`

## 8. UI

Single page: header + disclaimer banner, language selector (Auto / English / हिंदी),
message list with user/bot bubbles, animated typing indicator, source chips, suggestion
chips, textarea with Enter-to-send, and a clear-chat button. All model output is inserted
with `textContent` (no HTML injection).

## 9. Build order

1. `PLAN.md` (this file)
2. Folder structure
3. `scraper.py` + `data/faqs/` corpus
4. `ingest.py`
5. `app.py` + `rag.py` + `guard.py`
6. Chat UI
7. `tests/` (guard, API, 10 sample questions)
8. `README.md`
9. Verify the definition of done end to end.
