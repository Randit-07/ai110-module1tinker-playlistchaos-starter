# Playlist Chaos — Agentic Lyrics Classifier

A Streamlit playlist builder with an AI agent on the side. You can add songs manually as before, or enter just a **title + artist** and let the agent fetch the lyrics, retrieve similar mood examples via RAG, ask Gemini to classify the song, and self-critique low-confidence answers — all before the song lands in your playlist.

Built on top of a classroom "debug this starter" exercise; the agent pipeline is the additive final-project layer.

---

## Setup

1. Python 3.10+ is recommended.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the env template and fill in real tokens:
   ```bash
   cp .env.example .env
   ```
   - `GENIUS_ACCESS_TOKEN` — get one at https://genius.com/api-clients
   - `GEMINI_API_KEY` — get one at https://aistudio.google.com/app/apikey

Missing or placeholder tokens cause [genius_tool.py](genius_tool.py), [rag_categories.py](rag_categories.py), and [agent.py](agent.py) to raise a clear `RuntimeError` at import rather than fail halfway through a user-facing request.

---

## Run

Start the app:

```bash
streamlit run app.py
```

Run the reliability harness (hits live Genius + Gemini APIs, takes ~30–90 seconds):

```bash
python eval.py
```

Logs are written to `logs/agent.log` (rotating, 1 MB × 3). Eval runs are dumped as timestamped JSON at `logs/eval_YYYYMMDD_HHMMSS.json`.

---

## How the AI decides

When you type a song title + artist and click **Analyze lyrics** in the sidebar:

```
  Title + Artist
        │
        ▼
  [genius_tool.fetch_lyrics]  ── fail ──►  [genius_tool.fallback_lyrics]  (data/fallback_lyrics.json)
        │                                           │
        ▼                                           ▼
      lyrics (truncated to 1500 chars) ── or ── "no lyrics → Mixed, confidence 0.0"
        │
        ▼
  [rag_categories.retrieve_examples]   ── Gemini text-embedding-004, cosine sim against cached mood snippets
        │   returns top-3 {mood, snippet, similarity}
        ▼
  [agent._build_prompt]                 ── few-shot examples + lyrics + JSON schema
        │
        ▼
  [agent._call_llm]                     ── gemini-2.0-flash, response_mime_type=application/json
        │   parse + validate; if malformed, retry once with stricter prompt
        ▼
  if confidence < 0.6  ──► [self-critique]  "re-examine carefully or classify as Mixed"
        │
        ▼
  EnrichedSong { energy, mood_hint, tags, acoustic, confidence, reasoning, sources }
        │
        ▼
  [playlist_logic.normalize_song]       ── preserves optional agent fields
        │
        ▼
  [playlist_logic.classify_song]        ── unchanged Hype/Chill/Mixed bucket logic
        │
        ▼
  Song appears in its playlist tab with a colored confidence badge and hover-reasoning.
```

The design intentionally keeps the existing `classify_song` decision at the end: the agent only fills in better inputs (`energy`, `tags`, `mood_hint`), and the coarse Hype/Chill/Mixed bucketing stays in one place.

---

## Why you can trust the output

Each guardrail protects a specific failure mode:

| # | Guardrail | What it protects against |
|---|-----------|---------------------------|
| 1 | **Fail-fast token checks** at import of [genius_tool.py](genius_tool.py), [rag_categories.py](rag_categories.py), [agent.py](agent.py) | A misconfigured environment surfacing only when a user hits "Analyze lyrics" |
| 2 | **10-second timeout** on the Genius client | A hung HTTP request freezing the UI |
| 3 | **Uniform None return** on every Genius/requests failure in `fetch_lyrics` | Caller branching per exception type; keeps the downstream fallback path trivial |
| 4 | **Offline fallback dataset** in [data/fallback_lyrics.json](data/fallback_lyrics.json) (public-domain / traditional songs only) | Demo breaking when Genius is rate-limited or unreachable |
| 5 | **Embedding cache with content-hash key** at `.cache/embeddings.npz` | Silent drift when `categories.json` changes but the cache is stale |
| 6 | **Lyric truncation to 1500 chars** before the LLM call | Prompt bloat and context noise on long songs |
| 7 | **`response_mime_type=application/json`** on `gemini-2.0-flash` | Markdown-wrapped or prose-prefixed responses |
| 8 | **Schema retry** with stricter prompt on malformed JSON (once) | A one-off formatting blip failing the whole request |
| 9 | **Self-critique loop** when confidence < 0.6 (once) | Model underthinking ambiguous lyrics instead of honestly saying "Mixed" |
| 10 | **`_coerce()` output clamping** — energy 1–10, mood ∈ allowed set, tags capped at 5 | Out-of-range or invented fields corrupting downstream state |
| 11 | **Never-raise classification** — every expected failure collapses to a safe EnrichedSong with confidence 0.0 | A single bad song killing the whole UI flow |
| 12 | **Confidence badge + hover reasoning** in the playlist view | Hiding the model's self-doubt behind a single label |
| 13 | **Per-call structured logging** (prompt hash, char count, elapsed time) to `logs/agent.log` | Drift going unnoticed between runs |
| 14 | **Reliability harness with exit-1 gating** at [eval.py](eval.py), threshold 0.6 | Quality regressions slipping into a PR or deployment |

---

## Known limitations

- **Genius scraping is brittle.** `lyricsgenius` depends on Genius's public endpoints and HTML structure; rate limits, geographic restrictions, or layout changes silently reduce to "no lyrics found" and a confidence-0.0 fallback. Mitigated but not eliminated by the offline fallback dataset.
- **Confidence is self-reported.** The LLM rates its own answer. A confident wrong answer is still possible — the self-critique loop helps, but doesn't turn self-report into ground truth. Treat the badge as a prompt for skepticism, not a guarantee.
- **English-only.** Both the embedding model and the classifier prompt assume English lyrics. Non-English songs will produce lower-quality retrievals and classifications without raising a visible error.
- **Small eval set.** 10 hand-labeled songs across 5 moods means each correct/incorrect classification moves per-mood accuracy by 50 percentage points. Treat eval numbers directionally, not absolutely. Grow the set to ~50 songs before reading too much into single-digit point differences.
- **Mood taxonomy is hand-crafted, English-centric, and opinionated.** The five moods in [categories.json](categories.json) are a reasonable starting point for Western pop/rock, not a universal mood theory. Songs that blend moods land in "Mixed" — which is a design choice, not a failure.

---

## Eval results

Run `python eval.py` and paste the summary here. Example template:

```
Overall accuracy: N/10 = NN.NN%
Total elapsed: NN.Ns

Per-mood accuracy:
  Hype       N/2 = NN.NN%
  Chill      N/2 = NN.NN%
  Sad        N/2 = NN.NN%
  Angry      N/2 = NN.NN%
  Romantic   N/2 = NN.NN%
```

The full per-row breakdown, confusion matrix, and timings are in `logs/eval_YYYYMMDD_HHMMSS.json` after each run. Promote a trusted run to `data/eval_baseline.json` to enable automatic regression flagging on subsequent runs.

---

## File layout

| File | Purpose |
|------|---------|
| [app.py](app.py) | Streamlit UI — manual entry + auto-classify sidebar, playlist rendering with confidence badges |
| [playlist_logic.py](playlist_logic.py) | `normalize_song`, `classify_song` (unchanged bucket logic), playlist building, search, stats |
| [genius_tool.py](genius_tool.py) | Genius wrapper with timeout, fail-fast tokens, offline fallback |
| [rag_categories.py](rag_categories.py) | Embeds mood snippets once, caches to `.cache/embeddings.npz`, retrieves top-k by cosine |
| [agent.py](agent.py) | Orchestrator — fetch → RAG → Gemini → JSON retry → self-critique |
| [eval.py](eval.py) | Reliability harness with per-mood accuracy, confusion matrix, exit-1 gating |
| [logging_config.py](logging_config.py) | Rotating file logger at `logs/agent.log`, stderr at WARNING+ |
| [categories.json](categories.json) | Mood taxonomy with handwritten example snippets (avoids reproducing copyrighted lyrics) |
| [data/fallback_lyrics.json](data/fallback_lyrics.json) | Public-domain / traditional songs for offline demo reliability |
| [data/eval_songs.json](data/eval_songs.json) | 10 hand-labeled eval songs, 2 per mood |
| [.env.example](.env.example) | Token placeholders — copy to `.env` and fill in |
