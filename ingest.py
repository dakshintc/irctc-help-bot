"""Build the FAISS vector store from the FAQ corpus.

Pipeline: load → clean → chunk by heading → embed → store.

Usage:
    python ingest.py                  # build from data/faqs/
    python ingest.py --include-raw    # also ingest scraped text from data/raw/
    python ingest.py --source data/faqs/en
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

import config
import vectorstore
from vectorstore import StoredChunk

MAX_CHARS = 850
MIN_CHARS = 40
OVERLAP_CHARS = 120
EMBED_BATCH = 32


@dataclass
class Doc:
    path: Path
    title: str
    source: str
    source_title: str
    lang: str
    category: str
    body: str


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def parse_front_matter(raw: str, path: Path) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:
        print(f"  ! bad front matter in {path.name}: {exc}", file=sys.stderr)
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].lstrip("\n")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    # Normalise common bullet glyphs to "- ".
    text = re.sub(r"^[•●▪◦]\s*", "- ", text, flags=re.M)
    # Collapse 3+ blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_docs(paths: list[Path], lang_default: str = "en") -> list[Doc]:
    docs: list[Doc] = []
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
            files.extend(sorted(p.rglob("*.txt")))
        elif p.exists():
            files.append(p)

    for path in files:
        if path.name == "manifest.json":
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = parse_front_matter(raw, path)
        body = clean_text(body)
        if len(body) < MIN_CHARS:
            continue

        rel = path.relative_to(config.BASE_DIR)
        lang = str(meta.get("lang") or lang_default)
        if "faqs/hi/" in rel.as_posix():
            lang = "hi"
        elif "faqs/en/" in rel.as_posix():
            lang = "en"

        docs.append(
            Doc(
                path=path,
                title=str(meta.get("title") or path.stem.replace("-", " ").title()),
                source=str(meta.get("source") or "https://www.irctc.co.in"),
                source_title=str(meta.get("source_title") or meta.get("title") or path.stem),
                lang=lang,
                category=str(meta.get("category") or "general"),
                body=body,
            )
        )
    return docs


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split a document into (heading, section_body) pairs."""
    lines = body.split("\n")
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split on sentence/bullet boundaries with a small overlap."""
    if len(text) <= max_chars:
        return [text]

    units = re.split(r"(?<=[.!?।])\s+|\n+", text)
    units = [u.strip() for u in units if u.strip()]
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            # Carry a short tail for context, without cutting mid-word.
            tail = current[-OVERLAP_CHARS:]
            cut = tail.find(" ")
            if 0 < cut < len(tail) - 20:
                tail = tail[cut + 1 :]
            current = f"{tail}\n{unit}".strip()
        else:
            # A single very long sentence: hard-wrap it.
            for i in range(0, len(unit), max_chars):
                pieces.append(unit[i : i + max_chars])
            current = ""
    if current:
        pieces.append(current)
    return [p for p in pieces if len(p) >= MIN_CHARS // 2]


def build_chunks(docs: list[Doc]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for doc in docs:
        for heading, section in split_sections(doc.body):
            for idx, piece in enumerate(split_long(section)):
                text = f"{doc.title} — {heading}\n{piece}".strip()
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                if digest in seen:
                    continue
                seen.add(digest)
                slug = re.sub(r"[^a-zA-Z0-9]+", "-", doc.path.stem).strip("-")
                chunks.append(
                    Chunk(
                        id=f"{slug}-{digest}-{idx}",
                        text=text,
                        metadata={
                            "source": doc.source,
                            "source_title": doc.source_title,
                            "title": doc.title,
                            "heading": heading,
                            "lang": doc.lang,
                            "category": doc.category,
                        },
                    )
                )
    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the IRCTC help vector index")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="extra file/directory to ingest (can repeat)",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="also ingest scraped data/raw/*.txt (noisy; off by default)",
    )
    args = parser.parse_args(argv)

    paths: list[Path] = [config.FAQ_DIR]
    if args.include_raw:
        paths.append(config.RAW_DIR)
    if args.source:
        paths.extend(Path(s) for s in args.source)

    print(f"Loading FAQ documents from: {', '.join(str(p) for p in paths)}")
    docs = load_docs(paths)
    if not docs:
        print(
            "No FAQ documents found. Add .md files under data/faqs/en/ or data/faqs/hi/.",
            file=sys.stderr,
        )
        return 1

    chunk_list = build_chunks(docs)
    print(f"Loaded {len(docs)} documents → {len(chunk_list)} chunks")

    print(f"Loading embedding model: {config.EMBED_MODEL}")
    print("(First run downloads the model, about 470 MB. Later runs use the cache.)")
    model = SentenceTransformer(config.EMBED_MODEL)

    started = time.time()
    all_vectors: list[np.ndarray] = []
    for start in range(0, len(chunk_list), EMBED_BATCH):
        batch = chunk_list[start : start + EMBED_BATCH]
        embeddings = model.encode(
            [c.text for c in batch],
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_vectors.extend(np.asarray(embeddings, dtype="float32"))
        done = min(start + EMBED_BATCH, len(chunk_list))
        print(f"  embedded {done}/{len(chunk_list)} chunks", flush=True)

    elapsed = time.time() - started
    vectors = np.vstack(all_vectors)
    info = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "documents": len(docs),
        "chunks": len(chunk_list),
        "dimensions": int(vectors.shape[1]),
        "embedding_model": config.EMBED_MODEL,
        "index_name": config.INDEX_NAME,
        "backend": "faiss",
        "seconds": round(elapsed, 1),
        "languages": sorted({d.lang for d in docs}),
        "sources": sorted({d.source for d in docs}),
    }

    stored = [
        StoredChunk(id=c.id, text=c.text, metadata=c.metadata) for c in chunk_list
    ]
    vectorstore.save_index(vectors, stored, info)
    (config.DATA_DIR / "build_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\nDone. {len(chunk_list)} chunks stored in {vectorstore.directory()} "
        f"in {elapsed:.1f}s."
    )
    print("Next: python app.py  →  http://localhost:5000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
