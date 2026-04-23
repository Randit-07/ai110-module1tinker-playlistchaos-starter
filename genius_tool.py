"""Genius API wrapper for fetching song lyrics.

Fails fast at import if GENIUS_ACCESS_TOKEN is not configured — the rest of the
agent pipeline assumes this tool is usable, so we surface the misconfiguration
immediately rather than at first call.

Also provides `fallback_lyrics()` that reads from data/fallback_lyrics.json so
demos stay functional when the live Genius API is unreachable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from lyricsgenius import Genius

load_dotenv()

logger = logging.getLogger("playlistchaos.genius")

_TIMEOUT_SECONDS = 10
_FALLBACK_PATH = Path(__file__).parent / "data" / "fallback_lyrics.json"
_PLACEHOLDER_TOKENS = {"", "your_genius_token_here", "changeme"}


def _load_token() -> str:
    token = (os.getenv("GENIUS_ACCESS_TOKEN") or "").strip()
    if token in _PLACEHOLDER_TOKENS:
        raise RuntimeError(
            "GENIUS_ACCESS_TOKEN is not set. Copy .env.example to .env and add a real "
            "token from https://genius.com/api-clients."
        )
    return token


_genius = Genius(
    _load_token(),
    timeout=_TIMEOUT_SECONDS,
    remove_section_headers=True,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    retries=1,
    verbose=False,
)


def fetch_lyrics(title: str, artist: str) -> Optional[str]:
    """Fetch lyrics for (title, artist) from Genius. Returns None on any failure."""
    started = time.monotonic()
    logger.info("genius.fetch_lyrics request title=%r artist=%r", title, artist)

    try:
        song = _genius.search_song(title=title, artist=artist)
    except Exception as exc:
        # lyricsgenius wraps requests and can raise requests.Timeout, HTTPError,
        # JSONDecodeError, or its own exceptions. For demo reliability we treat
        # every failure the same: log, return None, let the agent fall back.
        elapsed = time.monotonic() - started
        logger.warning(
            "genius.fetch_lyrics error title=%r artist=%r elapsed=%.2fs error=%s",
            title, artist, elapsed, exc,
        )
        return None

    elapsed = time.monotonic() - started
    lyrics = getattr(song, "lyrics", None) if song is not None else None
    if not lyrics:
        logger.info(
            "genius.fetch_lyrics no_match title=%r artist=%r elapsed=%.2fs",
            title, artist, elapsed,
        )
        return None

    logger.info(
        "genius.fetch_lyrics ok title=%r artist=%r chars=%d elapsed=%.2fs",
        title, artist, len(lyrics), elapsed,
    )
    return lyrics


def fallback_lyrics(title: str, artist: str) -> Optional[str]:
    """Return lyrics from the offline demo dataset, or None if no match."""
    if not _FALLBACK_PATH.exists():
        logger.warning("fallback_lyrics missing_file path=%s", _FALLBACK_PATH)
        return None

    try:
        with _FALLBACK_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("fallback_lyrics read_error path=%s error=%s", _FALLBACK_PATH, exc)
        return None

    songs = data.get("songs", []) if isinstance(data, dict) else data
    key_title = title.strip().lower()
    key_artist = artist.strip().lower()

    for entry in songs:
        if (
            entry.get("title", "").strip().lower() == key_title
            and entry.get("artist", "").strip().lower() == key_artist
        ):
            lyrics = entry.get("lyrics")
            if lyrics:
                logger.info(
                    "fallback_lyrics hit title=%r artist=%r chars=%d",
                    title, artist, len(lyrics),
                )
                return lyrics

    logger.info("fallback_lyrics miss title=%r artist=%r", title, artist)
    return None
