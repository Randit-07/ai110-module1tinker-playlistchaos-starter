"""Agent orchestrator — fetches lyrics, retrieves few-shot examples, asks Gemini
to classify, and self-critiques low-confidence answers.

`classify_from_title_artist(title, artist)` is the sole entry point. It never
raises on expected failure modes (missing lyrics, malformed LLM output, low
confidence): each failure collapses to an EnrichedSong with safe defaults so
the caller can always render a result.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import List, TypedDict

import google.generativeai as genai
from dotenv import load_dotenv

import genius_tool
import rag_categories

load_dotenv()

logger = logging.getLogger("playlistchaos.agent")

_MODEL_NAME = "gemini-2.0-flash"
_MAX_LYRICS_CHARS = 1500
_CONFIDENCE_THRESHOLD = 0.6
_VALID_MOODS = {"Hype", "Chill", "Sad", "Angry", "Romantic", "Mixed"}
_REQUIRED_KEYS = {"energy", "mood_hint", "tags", "acoustic", "confidence", "reasoning"}
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
_model = genai.GenerativeModel(_MODEL_NAME)


class EnrichedSong(TypedDict):
    title: str
    artist: str
    energy: int
    mood_hint: str
    tags: List[str]
    acoustic: bool
    confidence: float
    reasoning: str
    sources: dict


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _format_examples(examples: list) -> str:
    if not examples:
        return "(no retrieved examples)"
    blocks = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i} (mood={ex['mood']}, similarity={ex['similarity']:.2f}):\n"
            f'"""\n{ex["snippet"]}\n"""'
        )
    return "\n\n".join(blocks)


def _build_prompt(
    title: str,
    artist: str,
    lyrics: str,
    examples: list,
    *,
    stricter: bool = False,
    self_critique: bool = False,
) -> str:
    header = (
        "You are a music mood classifier. Classify a song's lyrics into ONE of the moods "
        "below and return a single JSON object — no prose, no markdown fences.\n\n"
        f"Allowed moods: {sorted(_VALID_MOODS)}\n"
    )
    schema = (
        "JSON schema (all keys required):\n"
        "- energy: integer 1-10 (1=barely moving, 10=max intensity)\n"
        "- mood_hint: one of the allowed moods above\n"
        '- tags: array of 2-5 short lowercase tags (e.g. "breakup", "nostalgic", "party")\n'
        "- acoustic: boolean (true if lyrics/style suggest acoustic/unplugged)\n"
        "- confidence: float 0.0-1.0 — your honest self-rating\n"
        "- reasoning: one or two sentences justifying the classification\n"
    )
    body = (
        f"{header}\n{schema}\n"
        "Reference examples retrieved by semantic similarity (use these as anchors):\n\n"
        f"{_format_examples(examples)}\n\n"
        f"Song: {title!r} by {artist!r}\n"
        f'Lyrics (may be truncated to {_MAX_LYRICS_CHARS} chars):\n"""\n{lyrics}\n"""\n'
    )
    if stricter:
        body += (
            "\nIMPORTANT: Your previous response was not valid JSON. Return ONLY a raw "
            "JSON object with the exact keys listed above. No markdown, no commentary, "
            "no wrapper.\n"
        )
    if self_critique:
        body += (
            "\nYour previous answer had low confidence. Re-examine the lyrics and the "
            "reference examples carefully. Either justify a higher confidence or "
            'explicitly classify mood_hint as "Mixed" if the lyrics genuinely straddle '
            "multiple moods.\n"
        )
    return body


def _call_llm(prompt: str, label: str) -> str:
    prompt_hash = _hash_prompt(prompt)
    logger.info(
        "agent.llm request label=%s hash=%s chars=%d",
        label, prompt_hash, len(prompt),
    )
    started = time.monotonic()
    response = _model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
        ),
    )
    text = response.text
    elapsed = time.monotonic() - started
    logger.info(
        "agent.llm response label=%s hash=%s chars=%d elapsed=%.2fs",
        label, prompt_hash, len(text or ""), elapsed,
    )
    return text


def _parse(text: str) -> dict:
    if not text:
        raise ValueError("empty response")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"missing keys: {sorted(missing)}")
    return data


def _coerce(data: dict) -> dict:
    """Clamp each field to its declared type/range. LLMs occasionally stringify
    booleans or drift outside 1-10 even with response_mime_type=json."""
    try:
        energy = int(round(float(data["energy"])))
    except (TypeError, ValueError):
        energy = 5
    energy = max(1, min(10, energy))

    mood = str(data.get("mood_hint", "")).strip()
    if mood not in _VALID_MOODS:
        mood = "Mixed"

    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t).strip().lower() for t in tags if str(t).strip()][:5]

    raw_acoustic = data.get("acoustic", False)
    if isinstance(raw_acoustic, str):
        acoustic = raw_acoustic.strip().lower() in {"true", "1", "yes"}
    else:
        acoustic = bool(raw_acoustic)

    try:
        confidence = float(data["confidence"])
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "energy": energy,
        "mood_hint": mood,
        "tags": tags,
        "acoustic": acoustic,
        "confidence": confidence,
        "reasoning": str(data.get("reasoning", "")).strip(),
    }


def _fallback_enrichment(
    title: str,
    artist: str,
    reason: str,
    sources: dict,
) -> EnrichedSong:
    return {
        "title": title,
        "artist": artist,
        "energy": 5,
        "mood_hint": "Mixed",
        "tags": [],
        "acoustic": False,
        "confidence": 0.0,
        "reasoning": reason,
        "sources": sources,
    }


def classify_from_title_artist(title: str, artist: str) -> EnrichedSong:
    """Fetch lyrics, retrieve few-shot examples, classify, self-critique if confidence<0.6."""
    run_start = time.monotonic()
    logger.info("agent.classify request title=%r artist=%r", title, artist)

    lyrics = genius_tool.fetch_lyrics(title, artist)
    lyrics_source = "genius"
    if not lyrics:
        lyrics = genius_tool.fallback_lyrics(title, artist)
        lyrics_source = "fallback" if lyrics else "none"

    if not lyrics:
        result = _fallback_enrichment(
            title, artist,
            "No lyrics available from Genius or offline fallback; defaulting to Mixed.",
            {"lyrics_source": "none", "rag_examples": []},
        )
        logger.info(
            "agent.classify done title=%r mood=%s confidence=%.2f source=none elapsed=%.2fs",
            title, result["mood_hint"], result["confidence"],
            time.monotonic() - run_start,
        )
        return result

    lyrics_input = lyrics[:_MAX_LYRICS_CHARS]
    examples = rag_categories.retrieve_examples(lyrics_input, k=3)

    parsed: dict | None = None
    for label, stricter in (("classify", False), ("classify_retry", True)):
        attempt_prompt = _build_prompt(
            title, artist, lyrics_input, examples, stricter=stricter
        )
        try:
            raw = _call_llm(attempt_prompt, label=label)
            parsed = _parse(raw)
            break
        except Exception as exc:
            logger.warning(
                "agent.llm parse_error label=%s error=%s", label, exc,
            )

    if parsed is None:
        result = _fallback_enrichment(
            title, artist,
            "LLM returned malformed JSON twice; defaulting to Mixed.",
            {"lyrics_source": lyrics_source, "rag_examples": examples},
        )
        logger.info(
            "agent.classify done title=%r mood=%s confidence=%.2f source=%s elapsed=%.2fs status=malformed",
            title, result["mood_hint"], result["confidence"], lyrics_source,
            time.monotonic() - run_start,
        )
        return result

    coerced = _coerce(parsed)

    if coerced["confidence"] < _CONFIDENCE_THRESHOLD:
        logger.info(
            "agent.self_critique triggered confidence=%.2f threshold=%.2f",
            coerced["confidence"], _CONFIDENCE_THRESHOLD,
        )
        critique_prompt = _build_prompt(
            title, artist, lyrics_input, examples, self_critique=True
        )
        try:
            critique_raw = _call_llm(critique_prompt, label="self_critique")
            coerced = _coerce(_parse(critique_raw))
        except Exception as exc:
            logger.warning(
                "agent.self_critique failed error=%s — keeping first response", exc,
            )

    result: EnrichedSong = {
        "title": title,
        "artist": artist,
        **coerced,
        "sources": {
            "lyrics_source": lyrics_source,
            "rag_examples": examples,
        },
    }
    logger.info(
        "agent.classify done title=%r mood=%s confidence=%.2f source=%s elapsed=%.2fs",
        title, result["mood_hint"], result["confidence"], lyrics_source,
        time.monotonic() - run_start,
    )
    return result
