"""Reliability harness — runs the agent on a hand-labeled eval set and reports
per-mood accuracy, overall accuracy, and a 5x5 confusion matrix.

Writes a timestamped run to logs/eval_YYYYMMDD_HHMMSS.json. If
data/eval_baseline.json exists, any per-mood accuracy drop vs baseline is
flagged in stdout. Exits 1 when overall accuracy falls below ACCURACY_THRESHOLD
so CI or a shell pipeline can gate on reliability.

Usage:
    python eval.py

Each classification hits the live Genius and Gemini APIs, so a full run can
take 30-90 seconds and consumes API quota.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from logging_config import setup_logging

setup_logging()

import agent  # noqa: E402 — import after logging setup so agent's imports inherit handlers

_ROOT = Path(__file__).parent
_EVAL_PATH = _ROOT / "data" / "eval_songs.json"
_BASELINE_PATH = _ROOT / "data" / "eval_baseline.json"
_LOGS_DIR = _ROOT / "logs"

ACCURACY_THRESHOLD = 0.6


def _categories_moods() -> list[str]:
    """Mood names from categories.json so new moods auto-flow into the matrix."""
    with (_ROOT / "categories.json").open(encoding="utf-8") as fh:
        return [m["name"] for m in json.load(fh)["moods"]]


MOODS = ["Hype", "Chill", "Sad", "Angry", "Romantic"]  # expected labels in eval_songs.json
ALL_PRED_COLS = _categories_moods() + ["Mixed", "ERROR"]


def load_eval_set() -> list[dict]:
    with _EVAL_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["songs"] if isinstance(data, dict) else data


def run(entries: list[dict]) -> list[dict]:
    results = []
    for i, entry in enumerate(entries, 1):
        title = entry["title"]
        artist = entry["artist"]
        expected = entry["expected_mood"]
        print(
            f"[{i}/{len(entries)}] {title} by {artist} (expected {expected})...",
            flush=True,
        )
        started = time.monotonic()
        try:
            enriched = agent.classify_from_title_artist(title, artist)
            predicted = enriched["mood_hint"]
            confidence = float(enriched["confidence"])
            reasoning = enriched["reasoning"]
            lyrics_source = enriched["sources"].get("lyrics_source", "unknown")
            error = None
        except Exception as exc:
            predicted = "ERROR"
            confidence = 0.0
            reasoning = ""
            lyrics_source = "error"
            error = repr(exc)
        elapsed = time.monotonic() - started
        correct = predicted == expected
        print(
            f"    -> predicted {predicted} (conf {confidence:.2f}, "
            f"src {lyrics_source}, {elapsed:.1f}s) "
            f"{'ok' if correct else 'miss'}",
            flush=True,
        )
        results.append({
            "title": title,
            "artist": artist,
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "reasoning": reasoning,
            "lyrics_source": lyrics_source,
            "correct": correct,
            "elapsed_s": round(elapsed, 2),
            "error": error,
        })
    return results


def per_mood_accuracy(results: list[dict]) -> dict:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for r in results:
        totals[r["expected"]] += 1
        if r["correct"]:
            correct[r["expected"]] += 1
    return {
        m: {
            "correct": correct[m],
            "total": totals[m],
            "accuracy": (correct[m] / totals[m]) if totals[m] else 0.0,
        }
        for m in MOODS
    }


def confusion_matrix(results: list[dict]) -> dict:
    cm = {exp: {col: 0 for col in ALL_PRED_COLS} for exp in MOODS}
    for r in results:
        exp = r["expected"]
        pred = r["predicted"] if r["predicted"] in ALL_PRED_COLS else "ERROR"
        cm[exp][pred] += 1
    return cm


def print_matrix(cm: dict) -> None:
    col_width = max(6, max(len(c) for c in ALL_PRED_COLS))
    print("\nConfusion matrix (rows=expected, cols=predicted):")
    header = "expected".ljust(12) + "".join(c.rjust(col_width + 1) for c in ALL_PRED_COLS)
    print(header)
    for exp in MOODS:
        row = exp.ljust(12)
        for col in ALL_PRED_COLS:
            row += str(cm[exp][col]).rjust(col_width + 1)
        print(row)


def regression_check(per_mood: dict, baseline_path: Path) -> list[str]:
    if not baseline_path.exists():
        return []
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"(baseline unreadable: {exc})")
        return []
    regressions = []
    base_per_mood = baseline.get("per_mood", {})
    for mood, stats in per_mood.items():
        base_acc = base_per_mood.get(mood, {}).get("accuracy")
        if base_acc is not None and stats["accuracy"] < base_acc:
            regressions.append(
                f"{mood}: {stats['accuracy']:.2%} < baseline {base_acc:.2%}"
            )
    return regressions


def main() -> int:
    entries = load_eval_set()
    if len(entries) < 5:
        print(
            f"Eval set has only {len(entries)} entries — expected 10+.",
            file=sys.stderr,
        )
        return 2

    run_started = time.monotonic()
    results = run(entries)
    total_elapsed = time.monotonic() - run_started

    per_mood = per_mood_accuracy(results)
    total = len(results)
    correct_total = sum(1 for r in results if r["correct"])
    overall = correct_total / total if total else 0.0
    cm = confusion_matrix(results)

    print(f"\nOverall accuracy: {correct_total}/{total} = {overall:.2%}")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print("\nPer-mood accuracy:")
    for mood in MOODS:
        stats = per_mood[mood]
        print(
            f"  {mood:10} {stats['correct']}/{stats['total']} "
            f"= {stats['accuracy']:.2%}"
        )

    print_matrix(cm)

    regressions = regression_check(per_mood, _BASELINE_PATH)
    if regressions:
        print("\nRegressions vs baseline:")
        for line in regressions:
            print(f"  - {line}")
    elif _BASELINE_PATH.exists():
        print("\nNo per-mood regressions vs baseline.")

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _LOGS_DIR / f"eval_{stamp}.json"
    out = {
        "timestamp": stamp,
        "total_elapsed_s": round(total_elapsed, 2),
        "threshold": ACCURACY_THRESHOLD,
        "overall": {
            "correct": correct_total,
            "total": total,
            "accuracy": overall,
        },
        "per_mood": per_mood,
        "confusion_matrix": cm,
        "regressions": regressions,
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote results to {out_path}")

    if overall < ACCURACY_THRESHOLD:
        print(
            f"\nFAIL: overall accuracy {overall:.2%} < threshold "
            f"{ACCURACY_THRESHOLD:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
