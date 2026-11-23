"""FAISS-backed vector store (single-file index + JSONL chunks).

Why FAISS instead of Chroma: Chroma's persistent client uses SQLite, and this
project may live on a network-mounted filesystem where SQLite locking fails with
"disk I/O error". FAISS indexes are plain bytes, so they work on local disk and
on network mounts alike. The interface is small: save once during ingest, load
once at startup, cosine search via normalized inner product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

import config


@dataclass
class StoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]


def index_exists() -> bool:
    return config.INDEX_DIR.joinpath("vectors.faiss").exists() and config.INDEX_DIR.joinpath(
        "chunks.jsonl"
    ).exists()


def save_index(vectors: np.ndarray, chunks: list[StoredChunk], build_meta: dict[str, Any]) -> None:
    """Write the FAISS index and chunk metadata.

    The index is serialised to bytes and written in one call, avoiding seeks that
    some network filesystems do not support.
    """
    if len(vectors) != len(chunks):
        raise ValueError("vectors and chunks must have the same length")
    if vectors.size == 0:
        raise ValueError("cannot save an empty index")

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    matrix = np.ascontiguousarray(vectors.astype("float32"))
    faiss.normalize_L2(matrix)  # cosine similarity via inner product
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    payload = faiss.serialize_index(index).tobytes()
    (config.INDEX_DIR / "vectors.faiss").write_bytes(payload)

    lines = []
    for chunk in chunks:
        lines.append(
            json.dumps(
                {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata},
                ensure_ascii=False,
            )
        )
    (config.INDEX_DIR / "chunks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (config.INDEX_DIR / "build_meta.json").write_text(
        json.dumps(build_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_index() -> tuple[faiss.Index, list[StoredChunk]]:
    if not index_exists():
        raise FileNotFoundError(
            "Vector index not found. Run `python ingest.py` first."
        )
    raw = (config.INDEX_DIR / "vectors.faiss").read_bytes()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    index = faiss.deserialize_index(buffer)

    chunks: list[StoredChunk] = []
    with (config.INDEX_DIR / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            chunks.append(
                StoredChunk(
                    id=str(item["id"]),
                    text=str(item["text"]),
                    metadata=item.get("metadata") or {},
                )
            )

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index/chunk mismatch: {index.ntotal} vectors vs {len(chunks)} chunks"
        )
    return index, chunks


def build_meta() -> dict[str, Any]:
    path = config.INDEX_DIR / "build_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def directory() -> Path:
    return config.INDEX_DIR
