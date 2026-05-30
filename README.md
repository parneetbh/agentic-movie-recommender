# 🎬 Agentic Movie Recommender

**AI-Powered Personalized Movie Recommendations via Multi-Signal Retrieval + LLM**

A movie recommendation agent that combines TF-IDF retrieval, multi-signal scoring, and an LLM (Gemma 4 via Ollama Cloud) to suggest the perfect film based on your mood, preferences, and watch history. Built to respond in under 20 seconds with a compelling, personalized pitch.

## 🚀 How It Works

### 1. Candidate Retrieval (Multi-Signal Scoring)
A TF-IDF index is built over **genres, keywords, and production countries** at load time. Each movie is scored by four signals:

- **Keyword & country exact match** — 10 pts per matching word; production country is matched separately to prevent Hollywood films shot abroad from outranking genuine national productions
- **TF-IDF cosine similarity** — 3× weight; semantic match between the enriched query and the corpus
- **Popularity-adjusted vote score** — `vote_average × min(vote_count / 10000, 1.0) × 0.5`, halved so mood signals dominate
- **Genre boost/penalty** — standard mood match (+1.5), emotional tearjerker mode (+4.0 Drama, −5.0 Action/Comedy/Horror), light mood (−3.0 for dark genres), hard negation penalty (−10.0)

**Hard filters:**
- **Runtime** — "under 90 mins", "at least 2 hours", etc. are parsed and applied before scoring
- **Nationality** — "Korean movie", "Iranian film", etc. restricts the pool to matching productions

**Negation parsing** detects "no thriller", "without horror", "don't want romance" and applies both query cleanup and genre penalties.

The top **15 candidates** are retrieved; the top **2** are sent to the LLM.

### 2. TMDB Enrichment
Director, top 3 cast members, and tagline are fetched from the TMDB API in parallel (8 threads, 4s timeout) for the top 8 candidates, giving the LLM richer context for more specific descriptions.

### 3. LLM Selection & Description
`gemma4:31b-cloud` via Ollama Cloud receives the top 2 candidates and writes a pitch in "text a friend" style — vivid, direct, and personal. The LLM is instructed to mention the title naturally, reference one specific scene or feeling, and avoid review-site clichés.

JSON is extracted via regex (the `format="json"` parameter produces empty responses with this model).

### 4. Surprise Mode
If the input contains no recognizable movie-related words (gibberish, vague non-query), the 15 candidates are shuffled and 2 random ones are sent to the LLM with the prompt "Surprise me". The description is prefixed with a phrase like "Not sure what you're in the mood for?" so the recommendation feels intentional.

### 5. Safety & Reliability
- Watch history is excluded from candidates before scoring
- `tmdb_id` is validated against the candidate set; falls back to top-scored candidate if invalid
- 3-attempt retry loop for transient LLM errors
- Hard 17-second timeout on the LLM call (fits within the 20s requirement)
- Post-processing appends runtime or match caveat if the LLM omits them
- Fallback description generated locally if the LLM returns nothing

## 📁 Project Structure

| File | Purpose |
|---|---|
| `llm.py` | Core implementation — retrieval, scoring, TMDB enrichment, `get_recommendation()` |
| `evaluate.py` | LLM-as-a-judge eval — scores recommendations on relevance + persuasion (1–5) |
| `test_retrieval.py` | Retrieval regression tests — 16 cases, no API calls needed |
| `test.py` | Grader compliance tests — run before submitting |
| `tmdb_top1000_movies.csv` | Movie database (TMDB top 1000) |
| `requirements.txt` | Python dependencies |

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_API_KEY` | ✅ Yes | Ollama Cloud API key |
| `TMDB_API_KEY` | ⬜ Optional | TMDB API key for director/cast/tagline enrichment (degrades gracefully if absent) |

Copy `.env.example` to `.env` and fill in your keys, or export them directly:

```bash
export OLLAMA_API_KEY=your_key_here
export TMDB_API_KEY=your_tmdb_key_here
```

## 🎬 Usage

```bash
# Interactive CLI
python llm.py

# With arguments
python llm.py --preferences "I want a funny feel-good movie" --history "The Dark Knight"

# Fast retrieval tests (no API needed)
python test_retrieval.py

# Grader compliance check
python test.py

# LLM-as-a-judge quality evaluation
python evaluate.py
```

## 🧪 Evaluation

`evaluate.py` runs `get_recommendation()` against 5 diverse prompts and uses the same LLM to score each result on:

1. **Relevance** (1–5) — does the movie match what the user asked for?
2. **Persuasiveness** (1–5) — would the description make you want to watch it?

Test cases cover: sci-fi with twists, feel-good comedy, dark psychological thriller (with watch history), epic fantasy, and family drama.

## 🛠 Tech Stack

- Python
- pandas, scikit-learn
- Ollama (`gemma4:31b-cloud`)
- TMDB API
- python-dotenv
