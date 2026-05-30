# Agentic Movie Recommender

## Approach

### Candidate Retrieval (Multi-Signal Scoring)
We build a TF-IDF index over **genres, keywords, and production countries** (not overview or title) at load time using `scikit-learn`. Excluding the overview prevents false matches — e.g. Parasite ranking #1 for "food movie" because its overview mentions a cooking scene.

At query time, each movie gets a combined score from four signals:

1. **Keyword + country exact match** (strongest signal — 10 pts per matching word): if the user says "food", movies with "food", "chef", or "restaurant" in their keywords field are heavily boosted; if the user says "Iranian" or "Korean", production country is matched
2. **TF-IDF cosine similarity** (3x weight): semantic match between the enriched query and genres/keywords corpus
3. **Popularity-adjusted vote score**: `vote_average × min(vote_count / 10000, 1.0) × 0.5` — halved so mood signals dominate over raw popularity
4. **Genre boost/penalty**:
   - **Standard mood match**: +1.5 for matching genre (e.g. "funny" → Comedy)
   - **Emotional query mode** (triggered by words like "cry", "sad", "heartbreaking", "grief"): +4.0 for Drama match; −5.0 penalty for Action/Western/Comedy/Horror so a tearjerker beats a blockbuster that merely carries a secondary Drama tag
    - −3.0 for dark genres (Thriller, Crime, Horror) when a light mood is detected
    - −10.0 hard penalty for explicitly excluded genres (e.g. "no thriller")

**Hard Filtering**:
- **Runtime**: Constraints like "under 90 mins" or "at least 2 hours" are parsed and applied as a hard filter.
- **Nationality**: If a specific country is requested (e.g., "Iranian"), the pool is restricted to films from that country (fallback only if 0 results).

The query is also enriched with genres and keywords from the user's watch history, so taste signals from past movies bias retrieval toward similar films.

**Negation parsing**: phrases like "no thriller", "without horror", "don't want romance" are detected, the negated words are removed from the TF-IDF query (so they don't accidentally boost the wrong movies), and the corresponding genres receive the hard −10 penalty.

**Nationality expansion**: adjectives like "Korean", "Japanese", "French" are mapped to their country names ("korea", "japan", "france") so country-based queries correctly match the `production_countries` field.

The top **15 candidates** are retrieved; the top **2** are sent to the LLM (optimized for context limits).

### Surprise / Gibberish Detection
If the user's input contains no recognisable movie-related words (gibberish, random characters, or a completely vague prompt), the system enters **surprise mode**: the 15 candidates are shuffled and a random 5 are shown to the LLM, which is told "Surprise me — pick any movie you think is worth watching tonight." The description is also prefixed with a phrase like "Since you're not sure what to watch —" so the user understands why the pick feels unexpected.

The known-preference vocabulary covers mood words (`cry`, `shock`, `heartbreaking`, `twist`, …), nationality adjectives, genre names, and common descriptive terms. This prevents natural phrasings like "something that will shock me" or "makes me cry" from accidentally triggering surprise mode.

### TMDB API Enrichment
For the top 8 candidates, we fetch **director, top 3 cast members, and tagline** from the TMDB API in parallel (8 threads, 4s timeout each). This gives the LLM richer context to write specific, compelling descriptions that reference real actors and directors. The enrichment runs concurrently with the LLM call to avoid adding latency.

### LLM Selection & Description
`gemma4:31b-cloud` via Ollama Cloud receives the top 2 candidates with their title, year, and genres. The prompt:
- Asks the LLM to pick the best match and write a description that makes the user *need* to watch it tonight
- Instructs it to mention the movie title naturally, by name
- Tells it to interpret the user's mood rather than echoing it back robotically
- Asks for one vivid scene, feeling, or character moment — not a plot summary
- Enforces a "text a friend" style: direct and personal, no "critically acclaimed" or review-site clichés
- Bans meta-commentary and reasoning leakage outside the JSON
- **Match Awareness**: If the library lacks a perfect match (e.g., no "Iranian Comedy" exists), the LLM is nudged to acknowledge the gap naturally in its pitch.

JSON is extracted via regex from the response (`format="json"` causes empty responses with this model). The `reasoning` key is stripped if present.

### Safety & Reliability
- Watch history titles are resolved to TMDB IDs at the start of `get_recommendation` (case-insensitive title lookup), then excluded from the candidate pool before scoring; the LLM never recommends a film the user has already seen
- After LLM responds, `tmdb_id` is validated against the candidate set; fallback to top-scored candidate if invalid or null
- **Transient Error Resilience**: Includes a 3-attempt retry loop for the LLM call to handle "prompt too long" or timeout errors from the server.
- **20-Second Guarantee**: Hard 17-second timeout for the LLM future ensures the entire pipeline finishes within the 20s requirement.
- **Safety Net**: If the LLM misses a requested runtime or fails to acknowledge a poor match, a post-processing layer automatically prepends the caveat or appends the duration.
- **Fallback Content**: If the LLM returns no description, a template fallback is generated locally using the movie's tagline and overview.

---

## Evaluation Strategy

We use **LLM-as-a-judge** via `evaluate.py`: after generating each recommendation, a second LLM call scores it on two dimensions (1–5):
1. **Relevance** — does the movie actually match what the user asked for?
2. **Persuasiveness** — would the description make you want to watch it?

We ran the recommender against 5 diverse preference prompts (sci-fi twist, feel-good comedy, dark thriller, epic fantasy, family drama) and iterated on prompt wording, scoring weights, and description style based on the scores. We also manually verified edge cases: negations ("no thriller"), vague requests ("something good"), gibberish input, empty watch history, and genre-specific preferences.

```bash
python evaluate.py   # runs all test cases and prints scores
```

---

## Code Guide

| File | Purpose |
|---|---|
| `llm.py` | Main implementation — multi-signal scoring, TMDB enrichment, `get_recommendation()`, prompt |
| `evaluate.py` | LLM-as-a-judge eval script — scores recommendations on relevance + persuasion |
| `test_retrieval.py` | Fast retrieval regression tests — 16 cases, no API calls needed |
| `test.py` | Grader test suite — run before submitting |
| `requirements.txt` | Python dependencies |
| `tmdb_top1000_movies.csv` | Movie database |

### Running locally

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

```bash
export OLLAMA_API_KEY=your_key_here
export TMDB_API_KEY=your_tmdb_key_here   # optional but improves descriptions

python llm.py --preferences "I want a funny feel-good movie" --history "The Dark Knight"
python test_retrieval.py   # fast scoring tests, no API needed
python test.py             # grader compliance check
python evaluate.py         # LLM-as-a-judge quality scores
```

### Environment variables

- `OLLAMA_API_KEY` — injected by grader at run time (required)
- `TMDB_API_KEY` — TMDB API key for director/cast/tagline enrichment (optional; degrades gracefully if absent)
