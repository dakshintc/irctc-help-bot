"""Polite scraper for public IRCTC / Indian Railways help pages.

- Checks robots.txt before every host and skips disallowed URLs.
- Uses an identifying User-Agent, a 2 s delay, retries and a local cache.
- Extracts readable text from HTML and PDFs, removes navigation boilerplate,
  and saves it under data/raw/ with provenance.

The main irctc.co.in site currently blocks non-browser clients (HTTP 403 from
Akamai), including its robots.txt. Those URLs are skipped and the bundled FAQ
corpus in data/faqs/ is used instead. Re-run this script any time the pages are
reachable, then run `python ingest.py`.

Usage:
    python scraper.py                 # fetch uncached sources
    python scraper.py --force         # re-fetch everything
    python scraper.py --offline       # only report what is cached
    python scraper.py --list          # show sources
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

import config

@dataclass(frozen=True)
class Source:
    url: str
    slug: str
    title: str
    lang: str
    category: str
    kind: str = "html"


# Only public help/FAQ pages. `www.irctc.co.in` pages are listed to document the
# intent, but robots.txt there returns 403 and the scraper will skip them.
SOURCES: list[Source] = [
    Source(
        "https://contents.irctc.co.in/en/etktfaq.html",
        "irctc-eticket-faq",
        "IRCTC e-Ticket FAQ",
        "en",
        "booking",
    ),
    Source(
        "https://contents.irctc.co.in/en/ITicket.html",
        "irctc-iticket-guide",
        "IRCTC I-Ticket booking guide",
        "en",
        "booking",
    ),
    Source(
        "https://contents.irctc.co.in/en/Registration.html",
        "irctc-registration-guide",
        "IRCTC registration guide",
        "en",
        "booking",
    ),
    Source(
        "https://contents.irctc.co.in/en/TatkalFaq.html",
        "irctc-tatkal-faq",
        "IRCTC Tatkal FAQ",
        "en",
        "tatkal",
    ),
    Source(
        "https://contents.irctc.co.in/en/TatkalBooking.html",
        "irctc-tatkal-guide",
        "IRCTC Tatkal booking guide",
        "en",
        "tatkal",
    ),
    Source(
        "https://contents.irctc.co.in/en/Eticket_new_cancel.html",
        "irctc-eticket-cancel",
        "IRCTC e-ticket cancellation and ID rules",
        "en",
        "refund",
    ),
    Source(
        "https://contents.irctc.co.in/en/CancellationProcedure.html",
        "irctc-cancellation-procedure",
        "IRCTC I-ticket cancellation procedure",
        "en",
        "refund",
    ),
    Source(
        "https://contents.irctc.co.in/en/Waitlisted_E-Ticket.html",
        "irctc-waitlisted-eticket",
        "IRCTC waitlisted e-ticket scheme",
        "en",
        "waitlist",
    ),
    Source(
        "https://contents.irctc.co.in/en/FAQs.html",
        "irctc-faq-index",
        "IRCTC FAQ index",
        "en",
        "index",
    ),
    Source(
        "https://contents.irctc.co.in/en/IRRules.html",
        "irctc-rules-index",
        "IRCTC rules and policies index",
        "en",
        "index",
    ),
    Source(
        "https://contents.irctc.co.in/en/ContactUsEn.html",
        "irctc-contact",
        "IRCTC contact information",
        "en",
        "safety",
    ),
    Source(
        "https://contents.irctc.co.in/en/Chatbot_Cancellation_Refund_English.pdf",
        "irctc-chatbot-refund.pdf",
        "IRCTC Chatbot ticket cancellation and refund process",
        "en",
        "refund",
        "pdf",
    ),
    Source(
        "https://contents.irctc.co.in/en/REFUND%20RULES%20wef%2012-Nov-15.pdf",
        "irctc-refund-rules-2015.pdf",
        "IRCTC refund rules w.e.f. 12-Nov-2015",
        "en",
        "refund",
        "pdf",
    ),
    Source(
        "https://contents.irctc.co.in/en/CC_68_2012_id_proof.pdf",
        "irctc-id-proof-circular.pdf",
        "IRCTC/Indian Railways ID proof circular",
        "en",
        "id-rules",
        "pdf",
    ),
    Source(
        "https://contents.irctc.co.in/hi/FAQs.html",
        "irctc-faq-index-hi",
        "IRCTC FAQ index (Hindi)",
        "hi",
        "index",
    ),
    Source(
        "https://contents.irctc.co.in/hi/CancellationProcedure.html",
        "irctc-cancellation-procedure-hi",
        "IRCTC I-ticket cancellation procedure (Hindi)",
        "hi",
        "refund",
    ),
    # Official Indian Railways rule pages linked from IRCTC's "Rules and Policies".
    Source(
        "https://www.indianrail.gov.in/StaticContents/quota_Code.html",
        "indianrail-quota-codes",
        "Indian Railways quota codes",
        "en",
        "classes-quotas",
    ),
    Source(
        "https://www.indianrail.gov.in/StaticContents/tatkal_Scheme.html",
        "indianrail-tatkal-scheme",
        "Indian Railways Tatkal scheme and charges",
        "en",
        "tatkal",
    ),
    Source(
        "https://www.indianrail.gov.in/StaticContents/refund_Rules.html",
        "indianrail-refund-rules",
        "Indian Railways revised refund rules w.e.f. 12.11.2015",
        "en",
        "refund",
    ),
    # Documented but blocked by the site's Akamai edge. Kept so the skip is visible.
    Source(
        "https://www.irctc.co.in/nget/train-search",
        "irctc-train-search",
        "IRCTC train search and PNR status",
        "en",
        "pnr",
    ),
    Source(
        "https://www.irctc.co.in/nget/refundrules",
        "irctc-refund-rules",
        "IRCTC refund rules page",
        "en",
        "refund",
    ),
]

BOILERPLATE_LINES = {
    "home",
    "contact us",
    "top",
    "back to top",
    "copyright",
    "compatible browser",
    "frequently asked questions",
    "read more",
    "irctc mobile",
    "irctc",
}


# --------------------------------------------------------------------------- #
# robots.txt
# --------------------------------------------------------------------------- #
_robots_cache: dict[str, tuple[RobotFileParser | None, str]] = {}


def robots_for(url: str) -> tuple[RobotFileParser | None, str]:
    """Fetch and parse robots.txt once per host.

    Returns (parser, note). A None parser means the host has no usable rules,
    and the caller decides based on the note (allow for 404, skip otherwise).
    """
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin in _robots_cache:
        return _robots_cache[origin]

    robots_url = f"{origin}/robots.txt"
    try:
        resp = requests.get(
            robots_url,
            headers={"User-Agent": config.USER_AGENT},
            timeout=15,
        )
    except requests.RequestException as exc:
        note = f"robots.txt unreachable ({exc.__class__.__name__}); skipped host out of caution"
        _robots_cache[origin] = (None, note)
        return _robots_cache[origin]

    if resp.status_code in (401, 403):
        note = f"robots.txt returned HTTP {resp.status_code}; host skipped"
        _robots_cache[origin] = (None, note)
        return _robots_cache[origin]
    if resp.status_code >= 500:
        note = f"robots.txt returned HTTP {resp.status_code}; host skipped out of caution"
        _robots_cache[origin] = (None, note)
        return _robots_cache[origin]
    if resp.status_code >= 400:
        note = f"robots.txt HTTP {resp.status_code} (no rules published); allow"
        parser = RobotFileParser()
        parser.parse([])
        _robots_cache[origin] = (parser, note)
        return _robots_cache[origin]

    parser = RobotFileParser()
    parser.parse(resp.text.splitlines())
    note = "robots.txt ok"
    _robots_cache[origin] = (parser, note)
    return _robots_cache[origin]


def allowed(url: str) -> tuple[bool, str]:
    parser, note = robots_for(url)
    if parser is None:
        return False, note
    # urllib treats an empty rule set as allow-all.
    return parser.can_fetch(config.USER_AGENT, url) or parser.can_fetch("*", url), note


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _html_text(raw: bytes) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()
    for tag in soup.find_all(["iframe", "svg"]):
        tag.decompose()

    text = soup.get_text("\n")
    return _clean_text(text)


def _pdf_text(raw: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(raw))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(f"## Page {i}\n{page.extract_text() or ''}")
    return _clean_text("\n".join(pages))


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            continue
        if len(line) <= 3 and line.lower() in BOILERPLATE_LINES:
            continue
        if line.lower() in BOILERPLATE_LINES:
            continue
        if line.startswith("Copyright ©") or line.startswith("Copyright (c)"):
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    # Join wrapped lines that clearly continue a sentence.
    merged: list[str] = []
    for line in lines:
        if merged and merged[-1] and merged[-1][-1] not in ".?!:;।" and len(merged[-1]) > 40:
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    return "\n".join(merged).strip()


def write_raw(src: Source, text: str) -> Path:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RAW_DIR / f"{src.slug}.txt"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"# source: {src.url}\n"
        f"# source_title: {src.title}\n"
        f"# lang: {src.lang}\n"
        f"# category: {src.category}\n"
        f"# fetched_at: {now}\n\n"
    )
    path.write_text(header + text + "\n", encoding="utf-8")
    return path


def fetch_one(src: Source, force: bool = False) -> dict:
    path = config.RAW_DIR / f"{src.slug}.txt"
    if path.exists() and not force:
        return {
            "slug": src.slug,
            "url": src.url,
            "status": "cached",
            "path": str(path),
            "chars": len(path.read_text(encoding="utf-8", errors="ignore")),
        }

    ok, note = allowed(src.url)
    if not ok:
        return {"slug": src.slug, "url": src.url, "status": "skipped", "note": note}

    last_error = ""
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                src.url,
                headers={
                    "User-Agent": config.USER_AGENT,
                    "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-IN,en;q=0.8,hi;q=0.6",
                },
                timeout=config.REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                text = _pdf_text(resp.content) if src.kind == "pdf" else _html_text(resp.content)
                if len(text) < 80:
                    return {
                        "slug": src.slug,
                        "url": src.url,
                        "status": "empty",
                        "note": f"extracted only {len(text)} characters",
                    }
                out = write_raw(src, text)
                return {
                    "slug": src.slug,
                    "url": src.url,
                    "status": "fetched",
                    "path": str(out),
                    "chars": len(text),
                }
            last_error = f"HTTP {resp.status_code}"
            if resp.status_code in (401, 403, 404):
                break
        except requests.RequestException as exc:
            last_error = exc.__class__.__name__
        time.sleep(config.SCRAPE_DELAY * attempt)

    return {
        "slug": src.slug,
        "url": src.url,
        "status": "failed",
        "note": last_error or "unknown error",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scrape public IRCTC help pages")
    parser.add_argument("--force", action="store_true", help="re-fetch cached pages")
    parser.add_argument("--offline", action="store_true", help="use cache only, no network")
    parser.add_argument("--list", action="store_true", help="list configured sources")
    parser.add_argument("--only", help="only fetch sources whose slug contains this text")
    args = parser.parse_args(argv)

    if args.list:
        for src in SOURCES:
            print(f"{src.slug:38} {src.lang}  {src.url}")
        return 0

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, src in enumerate(SOURCES, start=1):
        if args.only and args.only not in src.slug:
            continue
        if args.offline:
            path = config.RAW_DIR / f"{src.slug}.txt"
            status = "cached" if path.exists() else "missing"
            results.append({"slug": src.slug, "url": src.url, "status": status})
            print(f"[{i}/{len(SOURCES)}] {src.slug}: {status}")
            continue

        print(f"[{i}/{len(SOURCES)}] {src.slug} ...", flush=True)
        result = fetch_one(src, force=args.force)
        results.append(result)
        msg = result["status"]
        if result.get("note"):
            msg += f" ({result['note']})"
        print(f"    -> {msg}", flush=True)
        # Be polite even between a skip and the next request.
        time.sleep(config.SCRAPE_DELAY)

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_agent": config.USER_AGENT,
        "delay_seconds": config.SCRAPE_DELAY,
        "results": results,
    }
    (config.RAW_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fetched = sum(1 for r in results if r["status"] == "fetched")
    cached = sum(1 for r in results if r["status"] == "cached")
    skipped = sum(1 for r in results if r["status"] in {"skipped", "failed", "empty"})
    print(f"\nDone: {fetched} fetched, {cached} cached, {skipped} skipped/failed.")
    print(f"Raw text: {config.RAW_DIR}")
    print("Next: python ingest.py")
    if skipped and not (fetched or cached):
        print(
            "\nNo source was reachable. The curated FAQ corpus in data/faqs/ still works:\n"
            "  python ingest.py",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
