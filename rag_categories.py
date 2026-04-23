"""RAG layer over categories.json for mood classification.

On module load: embeds every example snippet in categories.json via Gemini
text-embedding-004 and caches the vectors to .cache/embeddings.npz keyed by a
hash of (file content, embed model, task type). Subsequent imports reuse the
cache unless any of those inputs change.

retrieve_examples(lyrics, k=3) embeds the query and returns the top-k most
similar snippets by cosine similarity, each tagged with its mood label so the
agent can use them as few-shot prompts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import List

import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("playlistchaos.rag")

_EMBED_MODEL = "models/text-embedding-004"
_TASK_TYPE = "semantic_similarity"
_ROOT = Path(__file__).parent
_CATEGORIES_PATH = _ROOT / "categories.json"
_CACHE_DIR = _ROOT / ".cache"
_CACHE_PATH = _CACHE_DIR / "embeddings.npz"
_PLACEHOLDERS = {"", "your_gemini_key_here", "changeme"}


def _load_api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key in _PLACEHOLDERS:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add a real key "
            "from https://aistudio.google.com/app/apikey."
        )
    return key


genai.configure(api_key=_load_api_key())


def _read_categories() -> tuple[list[str], list[str], bytes]:
    raw = _CATEGORIES_PATH.read_bytes()
    data = json.loads(raw)
    snippets: list[str] = []
    labels: list[str] = []
    for mood in data["moods"]:
        name = mood["name"]
        for snippet in mood["examples"]:
            snippets.append(snippet)
            labels.append(name)
    return snippets, labels, raw


def _cache_key(raw: bytes) -> str:
    h = hashlib.sha256()
    h.update(raw)
    h.update(b"\0")
    h.update(_EMBED_MODEL.encode())
    h.update(b"\0")
    h.update(_TASK_TYPE.encode())
    return h.hexdigest()


def _embed_batch(texts: list[str]) -> np.ndarray:
    result = genai.embed_content(
        model=_EMBED_MODEL,
        content=texts,
        task_type=_TASK_TYPE,
    )
    return np.asarray(result["embedding"], dtype=np.float32)


def _embed_one(text: str) -> np.ndarray:
    result = genai.embed_content(
        model=_EMBED_MODEL,
        content=text,
        task_type=_TASK_TYPE,
    )
    return np.asarray(result["embedding"], dtype=np.float32)


def _load_or_build_index() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snippets, labels, raw = _read_categories()
    key = _cache_key(raw)

    if _CACHE_PATH.exists():
        try:
            cached = np.load(_CACHE_PATH, allow_pickle=False)
            cached_key = cached["source_hash"].item()
            if cached_key == key:
                logger.info(
                    "rag.cache hit path=%s n=%d key=%s",
                    _CACHE_PATH, len(cached["mood_labels"]), key[:12],
                )
                return cached["embeddings"], cached["mood_labels"], cached["snippets"]
            logger.info(
                "rag.cache stale expected=%s got=%s — rebuilding",
                key[:12], cached_key[:12],
            )
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("rag.cache read_error path=%s error=%s — rebuilding", _CACHE_PATH, exc)

    started = time.monotonic()
    vectors = _embed_batch(snippets)
    elapsed = time.monotonic() - started
    logger.info(
        "rag.embed built n=%d dim=%d elapsed=%.2fs",
        vectors.shape[0], vectors.shape[1], elapsed,
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _CACHE_PATH,
        embeddings=vectors,
        mood_labels=np.asarray(labels),
        snippets=np.asarray(snippets),
        source_hash=np.asarray(key),
    )
    logger.info("rag.cache wrote path=%s key=%s", _CACHE_PATH, key[:12])
    return vectors, np.asarray(labels), np.asarray(snippets)


_EMBEDDINGS, _MOOD_LABELS, _SNIPPETS = _load_or_build_index()
_NORMS = np.linalg.norm(_EMBEDDINGS, axis=1)


def retrieve_examples(lyrics: str, k: int = 3) -> List[dict]:
    """Return the top-k category examples most similar to the query lyrics."""
    started = time.monotonic()
    query = _embed_one(lyrics)
    embed_elapsed = time.monotonic() - started

    query_norm = float(np.linalg.norm(query)) or 1.0
    sims = (_EMBEDDINGS @ query) / (_NORMS * query_norm + 1e-12)

    k = max(1, min(k, len(_SNIPPETS)))
    top_idx = np.argsort(-sims)[:k]
    results = [
        {
            "mood": str(_MOOD_LABELS[i]),
            "snippet": str(_SNIPPETS[i]),
            "similarity": float(sims[i]),
        }
        for i in top_idx
    ]

    logger.info(
        "rag.retrieve k=%d query_embed=%.3fs top=%s",
        k, embed_elapsed,
        [(r["mood"], round(r["similarity"], 3)) for r in results],
    )
    return results
