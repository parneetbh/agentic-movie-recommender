"""
Environment variables required:
  OLLAMA_API_KEY — Ollama Cloud API key (injected by grader at run time)
  TMDB_API_KEY   — TMDB API key for enriched metadata (optional; degrades gracefully if absent)
"""

import concurrent.futures
import json
import os
import random
import re
import time
import argparse

from dotenv import load_dotenv
load_dotenv()

import requests
import ollama
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
MOVIES_DF = pd.read_csv(DATA_PATH)
TOP_MOVIES = MOVIES_DF

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"

# Build TF-IDF index once at load time (genres + keywords + countries — not overview, to avoid false matches)
_corpus = (
    MOVIES_DF['genres'].fillna('') + ' ' +
    MOVIES_DF['keywords'].fillna('') + ' ' +
    MOVIES_DF['production_countries'].fillna('')
).str.lower()

_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
_tfidf_matrix = _vectorizer.fit_transform(_corpus)

# Mood word → genre name
_MOOD_GENRES = {
    "funny": "Comedy", "comedy": "Comedy", "laugh": "Comedy", "light": "Comedy",
    "hilarious": "Comedy", "humor": "Comedy", "humorous": "Comedy",
    "scary": "Horror", "horror": "Horror", "creepy": "Horror",
    "romantic": "Romance", "romance": "Romance",
    "action": "Action", "thrilling": "Thriller", "thriller": "Thriller",
    "tense": "Thriller", "suspense": "Thriller",
    "shock": "Thriller", "shocking": "Thriller", "disturbing": "Thriller",
    "twist": "Thriller", "mindbending": "Thriller", "mindblowing": "Thriller",
    "unexpected": "Thriller", "surprising": "Thriller", "dark": "Thriller",
    "animated": "Animation", "cartoon": "Animation", "family": "Family",
    "sci-fi": "Science Fiction", "scifi": "Science Fiction", "space": "Science Fiction",
    "fantasy": "Fantasy", "adventure": "Adventure",
    "drama": "Drama", "emotional": "Drama", "heartwarming": "Drama",
    "sad": "Drama", "cry": "Drama", "crying": "Drama", "touching": "Drama",
    "happy": "Comedy", "bored": "Comedy", "uplifting": "Comedy",
    "tired": "Comedy",
}

_LIGHT_WORDS = {"funny", "comedy", "light", "easy", "relax", "background", "cooking",
                "casual", "chill", "laugh", "hilarious", "fun", "feel-good"}

# Words that signal the user wants a tearjerker / emotionally intense film.
# These trigger a much stronger genre boost and penalise action-heavy films.
_EMOTIONAL_WORDS = {
    "cry", "crying", "cries", "sad", "tears", "tearjerker", "emotional",
    "heartbreaking", "heartbreak", "devastating", "weep", "weeping",
    "touching", "moving", "bittersweet", "grief", "loss",
}

# Genres that clash with tearjerker intent — penalised when is_emotional is True
_ANTI_EMOTIONAL_GENRES = {"Action", "Western", "Comedy", "Horror"}

_DARK_GENRES = {"Thriller", "Crime", "Horror", "Mystery"}

# Nationality adjective -> country keywords to add to query
_NATIONALITY_MAP = {
    "korean": "korea", "japanese": "japan", "french": "france",
    "italian": "italy", "spanish": "spain", "german": "germany",
    "british": "kingdom", "iranian": "iran", "chinese": "china",
    "indian": "india", "australian": "australia", "mexican": "mexico",
    "russian": "russia", "danish": "denmark", "swedish": "sweden",
    "thai": "thailand", "turkish": "turkey", "arabic": "arabia",
}

# Country indicator words — matched against production_countries ONLY, never keywords.
# Prevents e.g. "Mission Impossible" (keywords contain "india" because it was filmed there)
# from outranking actual Indian productions.
_COUNTRY_WORDS = set(_NATIONALITY_MAP.values())

# Matches "not X", "not too X", "not very X", "no X", "without X", etc.
_NEGATION_RE = re.compile(
    r'\b(?:no|not|without|avoid|hate|dislike|don\'t\s+want|dont\s+want)(?:\s+(?:too|very|that|super|really))?\s+(\w+)',
    re.IGNORECASE
)

# Matches runtime upper-bound constraints such as:
#   "under 100 mins", "less than 1.5 hours", "90 minutes or less",
#   "no longer than 2 hours", "at most 100 min", "shorter than 2 hours"
_RUNTIME_UPPER_RE = re.compile(
    r'\b(?:under|less\s+than|shorter\s+than|no\s+longer\s+than|at\s+most|within|below|not\s+over|not\s+more\s+than)'
    r'\s+(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|hr?s?|hours?)',
    re.IGNORECASE
)
# Matches "90 minutes or less", "100 min or under", "1.5 hours or less"
_RUNTIME_UPPER_RE2 = re.compile(
    r'(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|hr?s?|hours?)\s+or\s+(?:less|under|fewer|shorter)',
    re.IGNORECASE
)
# Matches runtime lower-bound constraints:
#   "over 2 hours", "more than 2 hours", "at least 120 minutes"
# NOTE: "no longer than" is an upper bound handled by _RUNTIME_UPPER_RE above.
_RUNTIME_LOWER_RE = re.compile(
    r'\b(?:over|more\s+than|(?<![nN][oO]\s)longer\s+than|at\s+least|minimum\s+of)'
    r'\s+(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|hr?s?|hours?)',
    re.IGNORECASE
)


def _parse_runtime(preferences: str) -> tuple[int | None, int | None]:
    """Return (max_minutes, min_minutes) extracted from the preference string.
    Returns None for a bound that is not specified."""
    def _to_minutes(value: str, unit: str) -> int:
        minutes = float(value)
        if re.match(r'hr?s?|hours?', unit, re.IGNORECASE):
            minutes *= 60
        return int(round(minutes))

    max_min: int | None = None
    min_min: int | None = None

    for m in _RUNTIME_UPPER_RE.finditer(preferences):
        val = _to_minutes(m.group(1), m.group(2))
        max_min = val if max_min is None else min(max_min, val)
    for m in _RUNTIME_UPPER_RE2.finditer(preferences):
        val = _to_minutes(m.group(1), m.group(2))
        max_min = val if max_min is None else min(max_min, val)
    for m in _RUNTIME_LOWER_RE.finditer(preferences):
        val = _to_minutes(m.group(1), m.group(2))
        min_min = val if min_min is None else max(min_min, val)

    return max_min, min_min

# Words that map to dark genres when negated (e.g. "not depressing" → penalise Thriller/Horror/Crime)
_DARK_MOOD_WORDS = {
    "depressing": {"Thriller", "Horror", "Crime", "Mystery"},
    "dark": {"Thriller", "Horror", "Crime"},
    "scary": {"Horror"},
    "violent": {"Thriller", "Crime", "Action"},
    "heavy": {"Thriller", "Drama", "Crime"},
    "intense": {"Thriller", "Horror", "Crime"},
    "sad": {"Drama"},
}


def _parse_preferences(preferences: str) -> tuple[str, set, set]:
    """Return (cleaned_query, excluded_genres, boosted_genres)."""
    pref_lower = preferences.lower()
    excluded_genres = set()

    for match in _NEGATION_RE.finditer(pref_lower):
        word = match.group(1)
        if word in _MOOD_GENRES:
            excluded_genres.add(_MOOD_GENRES[word])
        if word.title() in ["Thriller", "Horror", "Crime", "Romance", "Comedy",
                             "Action", "Drama", "Animation", "Documentary", "Fantasy", "Mystery"]:
            excluded_genres.add(word.title())
        if word in _DARK_MOOD_WORDS:
            excluded_genres |= _DARK_MOOD_WORDS[word]

    cleaned = _NEGATION_RE.sub('', pref_lower).strip()
    boosted_genres = {genre for word, genre in _MOOD_GENRES.items() if word in cleaned}
    # For multi-genre requests, require ALL matched genres to be present for full boost
    require_all_genres = len(boosted_genres) > 1
    return cleaned, excluded_genres, boosted_genres


def _score_row(genres: str, keywords: str, countries: str, pref_words: set,
               excluded_genres: set, boosted_genres: set,
               is_light: bool, require_all_genres: bool,
               is_emotional: bool = False) -> float:
    if any(g in genres for g in excluded_genres):
        return -10.0

    score = 0.0

    # Topic keyword match (10 pts per word) — country indicator words excluded here
    # to prevent Hollywood films shot in India/Japan/etc. from outranking national productions.
    topic_pref = pref_words - _COUNTRY_WORDS
    if topic_pref:
        keyword_words = set(re.findall(r'\w+', keywords.lower()))
        score += len(topic_pref & keyword_words) * 10.0

    # Country match against production_countries only.
    # Full 10pts if the matched country is the PRIMARY (first-listed) producer.
    # Only 1pt if it appears as a minor co-production country later in the list.
    # If the user specified a country and this film has NO match at all, penalise −8
    # so that e.g. La La Land (has "dance" keywords) doesn't beat actual Indian dance films.
    country_pref = pref_words & _COUNTRY_WORDS
    if country_pref:
        countries_list = [c.strip() for c in countries.split(',') if c.strip()]
        primary_words = set(re.findall(r'\w+', countries_list[0].lower())) if countries_list else set()
        all_country_words = set(re.findall(r'\w+', countries.lower()))
        country_matches = country_pref & all_country_words
        if country_matches:
            primary_matches = country_matches & primary_words
            secondary_matches = country_matches - primary_matches
            score += len(primary_matches) * 10.0
            score += len(secondary_matches) * 1.0
        else:
            score -= 8.0  # country mismatch penalty

    # For multi-genre requests, only boost if ALL requested genres are present.
    # Emotional queries get a much stronger boost so genuine tearjerkers beat
    # blockbusters that merely carry a secondary Drama tag.
    genre_boost = 6.0 if is_emotional else 3.0
    partial_boost = 4.0 if is_emotional else 1.5
    if require_all_genres:
        if all(g in genres for g in boosted_genres):
            score += genre_boost
    elif any(g in genres for g in boosted_genres):
        score += partial_boost

    # Penalise action/western/comedy films for tearjerker queries so that e.g.
    # Django Unchained (Drama+Action+Western) doesn't beat Schindler's List.
    if is_emotional and any(g in genres for g in _ANTI_EMOTIONAL_GENRES):
        score -= 5.0

    if is_light and any(g in genres for g in _DARK_GENRES):
        score -= 3.0

    return score


def _history_context(history_ids: list) -> str:
    if not history_ids:
        return ''
    watched = MOVIES_DF[MOVIES_DF['tmdb_id'].isin(history_ids)]
    return (' '.join(watched['genres'].fillna('').tolist()) + ' ' +
            ' '.join(watched['keywords'].fillna('').tolist()))


def get_candidates(preferences: str, history_ids: list, n: int = 20) -> pd.DataFrame:
    cleaned, excluded_genres, boosted_genres = _parse_preferences(preferences)
    require_all_genres = len(boosted_genres) > 1
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    pref_words = set(re.findall(r'\w+', cleaned)) - set(ENGLISH_STOP_WORDS) - {
        'movie', 'film', 'something', 'want', 'watch', 'like', 'good', 'great',
        'want', 'really', 'make', 'blends', 'blend', 'mix', 'combines', 'bit'
    }
    # Expand nationality adjectives to country keywords (e.g. "korean" -> "korea")
    pref_words |= {_NATIONALITY_MAP[w] for w in pref_words if w in _NATIONALITY_MAP}
    is_light = bool(set(cleaned.split()) & _LIGHT_WORDS)
    is_emotional = bool(set(re.findall(r'\w+', cleaned)) & _EMOTIONAL_WORDS)

    enriched_query = (cleaned + ' ' + _history_context(history_ids)).lower()
    query_vec = _vectorizer.transform([enriched_query])

    # --- Runtime filtering ---
    max_runtime, min_runtime = _parse_runtime(preferences)
    mask = ~MOVIES_DF['tmdb_id'].isin(history_ids)
    if max_runtime is not None or min_runtime is not None:
        runtime_col = pd.to_numeric(MOVIES_DF['runtime_min'], errors='coerce')
        runtime_mask = pd.Series(True, index=MOVIES_DF.index)
        if max_runtime is not None:
            runtime_mask &= (runtime_col <= max_runtime) | runtime_col.isna()
        if min_runtime is not None:
            runtime_mask &= (runtime_col >= min_runtime) | runtime_col.isna()
        combined_mask = mask & runtime_mask
        # Fall back to history-only mask if the runtime filter leaves fewer than 3 results
        if combined_mask.sum() >= 3:
            mask = combined_mask

    # --- Nationality hard-filter ---
    # When the user explicitly names a country/nationality, restrict candidates to films
    # actually produced there. A soft penalty isn't enough — popular non-matching films
    # (Green Book, etc.) can still outscore genuine matches via vote_score.
    country_pref = pref_words & _COUNTRY_WORDS
    if country_pref:
        country_mask = MOVIES_DF['production_countries'].fillna('').str.lower().apply(
            lambda c: any(cp in c for cp in country_pref)
        )
        combined_country = mask & country_mask
        if combined_country.sum() >= 1:
            mask = combined_country

    tfidf_sims = cosine_similarity(query_vec, _tfidf_matrix[mask.values]).flatten()

    df = MOVIES_DF[mask].copy()
    df['_tfidf'] = tfidf_sims

    # Popularity score (0–10 range, normalized)
    vote_score = df['vote_average'] * (df['vote_count'].clip(upper=10000) / 10000)

    # Per-row custom score
    custom = df.apply(
        lambda r: _score_row(
            str(r['genres']), str(r['keywords']), str(r['production_countries']),
            pref_words, excluded_genres, boosted_genres, is_light, require_all_genres,
            is_emotional
        ), axis=1
    )

    # Combine: keyword/genre signals dominate when present; TF-IDF + popularity as tiebreaker.
    # vote_score is halved so a highly-rated blockbuster can't override a precise mood match.
    df['_score'] = custom + df['_tfidf'] * 3 + vote_score * 0.5
    return df.nlargest(n, '_score')


def _fetch_tmdb_details(tmdb_id: int) -> dict:
    if not TMDB_API_KEY:
        return {}
    try:
        params = {"api_key": TMDB_API_KEY}
        details = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}", params=params, timeout=3).json()
        credits = requests.get(f"{TMDB_BASE}/movie/{tmdb_id}/credits", params=params, timeout=3).json()
        director = next((p["name"] for p in credits.get("crew", []) if p["job"] == "Director"), "")
        cast = ", ".join(p["name"] for p in credits.get("cast", [])[:3])
        tagline = details.get("tagline", "")
        return {"director": director, "cast": cast, "tagline": tagline}
    except Exception:
        return {}


def _enrich_candidates(candidates: pd.DataFrame) -> dict:
    top_ids = [int(row.tmdb_id) for row in candidates.head(8).itertuples()]
    results = {tid: {} for tid in top_ids}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_tmdb_details, tid): tid for tid in top_ids}
        done, _ = concurrent.futures.wait(futures, timeout=4)
        for f in done:
            tid = futures[f]
            try:
                results[tid] = f.result()
            except Exception:
                pass
    return results


def _make_fallback(preferences: str, row, tagline: str, is_surprise: bool = False) -> str:
    """Generate a compelling description without the LLM by mirroring the user's words."""
    cleaned, _, _ = _parse_preferences(preferences)
    title = row.title.encode("ascii", "ignore").decode("ascii").strip()
    overview = str(row.overview).encode("ascii", "ignore").decode("ascii").strip()
    tagline = tagline.encode("ascii", "ignore").decode("ascii").strip() if tagline else ""
    snippet = overview[:220] if overview else ""

    if is_surprise:
        opener = random.choice([
            "Since you're not sure what you're in the mood for",
            "Since you want something unexpected",
            "Not sure what to watch? Here's a surprise pick",
            "Going with something unexpected tonight",
        ])
        if tagline:
            return f"{opener} — {title}. {tagline} {snippet}"
        else:
            return f"{opener} — {title}. {snippet}"

    # Detect positive mood words from the cleaned (post-negation) query only
    hook_parts = []
    for word, label in [
        ("fun", "fun"), ("funny", "funny"), ("laugh", "something to laugh at"),
        ("romantic", "romance"), ("love", "a love story"), ("action", "action"),
        ("animated", "animation"), ("family", "something family-friendly"),
        ("sci-fi", "sci-fi"), ("fantasy", "fantasy"), ("drama", "drama"),
        ("chill", "something relaxing"), ("cooking", "easy background watching"),
        ("relax", "something relaxing"), ("light", "something light"),
    ]:
        if word in cleaned:
            hook_parts.append(label)
    for adj, country in _NATIONALITY_MAP.items():
        if adj in cleaned or country in cleaned:
            hook_parts.append(f"{adj.capitalize()} cinema")
            break

    if hook_parts:
        hook = f"You wanted {' and '.join(hook_parts[:2])}"
        if tagline:
            return f"{hook} — {title} delivers. {tagline} {snippet}"
        else:
            return f"{hook} — watch {title}. {snippet}"
    else:
        if tagline:
            return f"{title}: {tagline} {snippet}"
        else:
            return f"{title}: {snippet}"


def get_recommendation(preferences: str, history: list, history_ids: list = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    # Resolve title-based history to IDs so watched films are excluded from candidates.
    # history_ids (explicit IDs) takes priority; titles in history are a fallback.
    if history and not history_ids:
        history_ids = MOVIES_DF[
            MOVIES_DF['title'].str.lower().isin([h.lower() for h in history])
        ]['tmdb_id'].tolist()
    candidates = get_candidates(preferences, history_ids, n=15)

    def _clean(text: str) -> str:
        return text.encode("ascii", "ignore").decode("ascii").strip()

    # Detect gibberish/empty input: a query is meaningful only if at least one word appears
    # in the TF-IDF vocabulary (real movie terms) or in known mood/preference words.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _cleaned_pref, _excl_genres, _boosted_genres = _parse_preferences(preferences)
    _skip = set(ENGLISH_STOP_WORDS) | {'movie', 'film', 'something', 'want', 'watch', 'like', 'good', 'great', 'really'}
    _pref_words = set(re.findall(r'\w+', _cleaned_pref)) - _skip
    _pref_words |= {_NATIONALITY_MAP[w] for w in _pref_words if w in _NATIONALITY_MAP}
    _known_pref = (
        set(_MOOD_GENRES.keys()) | set(_NATIONALITY_MAP.keys()) | set(_NATIONALITY_MAP.values()) |
        _EMOTIONAL_WORDS | {
            'tired', 'bored', 'sad', 'happy', 'relaxed', 'stressed', 'excited',
            'emotional', 'intense', 'dark', 'light', 'background', 'chill', 'cooking',
            'classic', 'recent', 'popular', 'short', 'long', 'weird', 'epic',
            'true', 'story', 'historical', 'foreign', 'subtitles', 'based',
            'makes', 'make', 'feel', 'recommend', 'suggest',
            'shock', 'shocking', 'shocked', 'disturbing', 'disturb',
            'twist', 'mindbending', 'mindblowing', 'unexpected', 'surprising', 'surprise',
            # Runtime constraint words — prevents mis-classification as "surprise me"
            'min', 'mins', 'minute', 'minutes', 'hour', 'hours', 'hr', 'hrs',
            'under', 'shorter', 'longer', 'runtime', 'length', 'quick', 'brief',
        }
    )
    # Compute runtime constraint once; used for is_surprise override, prompt, and post-processing.
    _max_rt, _min_rt = _parse_runtime(preferences)
    has_runtime_constraint = _max_rt is not None or _min_rt is not None
    _vocab = set(_vectorizer.get_feature_names_out())
    is_surprise = not bool(_pref_words & (_vocab | _known_pref))
    if has_runtime_constraint:
        is_surprise = False

    if is_surprise:
        # Shuffle the whole pool so every run gives a different pick
        surprise_pool = candidates.sample(frac=1, random_state=None).head(2)
        llm_preferences = "Surprise me — pick any movie you think is worth watching tonight"
    else:
        surprise_pool = candidates.head(2)
        llm_preferences = preferences

    def _runtime_str(row) -> str:
        """Return a human-friendly runtime string, e.g. '106 min', or '' if unknown."""
        try:
            rt = int(float(row.runtime_min))
            return f"{rt} min"
        except (ValueError, TypeError):
            return ""

    def _genre_short(genres_str: str) -> str:
        """Return at most the first two genres to keep the prompt compact."""
        parts = [g.strip() for g in str(genres_str).split(',') if g.strip()]
        return ', '.join(parts[:2])

    movie_lines = "\n".join(
        f"{i+1}. tmdb_id={int(r.tmdb_id)} | \"{_clean(r.title)}\" ({int(r.year)})"
        f" | {_genre_short(r.genres)}"
        + (f" | {_runtime_str(r)}" if _runtime_str(r) else "")
        for i, r in enumerate(surprise_pool.itertuples())
    )

    runtime_note = " Include how long it is (e.g. 'just 95 min')." if has_runtime_constraint else ""

    # --- Match quality: detect genre / nationality mismatch before calling LLM ---
    def _match_caveat(row) -> str:
        """Return a caveat string if the picked movie misses key requested attributes."""
        genres = str(row.genres)
        countries = str(row.production_countries).lower()
        missing = []

        # Genre mismatch
        if _boosted_genres and not any(g in genres for g in _boosted_genres):
            genre_label = '/'.join(sorted(g.lower() for g in _boosted_genres))
            missing.append(genre_label)

        # Nationality mismatch
        country_pref = _pref_words & _COUNTRY_WORDS
        if country_pref and not any(c in countries for c in country_pref):
            nat = next((adj for adj, c in _NATIONALITY_MAP.items() if c in country_pref), None)
            missing.append(f"{nat} film" if nat else '/'.join(country_pref))

        if missing:
            what = ' '.join(missing)
            return f"Couldn't find an exact {what} match, but you might enjoy this —"
        return ""

    # Only tell the LLM to acknowledge a mismatch if the top candidate actually misses
    # the requested genre/nationality — avoids bloating the prompt on well-matched queries.
    _top_row = candidates.iloc[0]
    poor_match_note = (
        " Note: no perfect match found — briefly acknowledge that, then still pitch it."
        if (not is_surprise and _match_caveat(_top_row))
        else ""
    )

    system_msg = (
        "You are a passionate movie recommender. "
        "Reply with valid JSON only: {\"tmdb_id\": <integer>, \"description\": \"<string>\"}. "
        "No markdown, no extra text."
    )
    user_msg = (
        f'User wants: "{llm_preferences}"\n\n'
        f'Candidates:\n{movie_lines}\n\n'
        f'Pick the best match. Pitch it like a friend who just watched it — mention the title naturally, '
        f'one vivid moment (not a plot summary), direct and personal.{runtime_note}{poor_match_note}\n'
        f'JSON: {{"tmdb_id": <id>, "description": "<pitch, max 300 chars>"}}'
    )

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Run LLM and TMDB enrichment in parallel; both fire immediately
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    def _call_llm() -> str:
        """Call the LLM with up to 3 retries for the transient 'prompt too long' server error.
        That error returns in <1s, so 3 retries costs at most ~1.5s total."""
        for attempt in range(3):
            try:
                response = client.chat(model=MODEL, messages=messages, options={"temperature": 0})
                return response.message.content
            except Exception as _e:
                err = str(_e).lower()
                if "prompt too long" in err and attempt < 2:
                    time.sleep(0.5)  # brief pause; error is immediate so total cost < 2s
                else:
                    return ""
        return ""

    llm_future = executor.submit(_call_llm)
    tmdb_future = executor.submit(_enrich_candidates, candidates)
    executor.shutdown(wait=False)

    content = ""
    try:
        content = llm_future.result(timeout=17)  # stays well within 20s requirement
    except Exception as _e:
        pass


    tmdb_details = {}
    try:
        tmdb_details = tmdb_future.result(timeout=1)
    except Exception:
        pass

    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    try:
        result = json.loads(json_match.group()) if json_match else {}
    except (json.JSONDecodeError, ValueError):
        result = {}
    result.pop("reasoning", None)

    valid_ids = set(int(x) for x in candidates['tmdb_id'].tolist())
    picked_id = int(result.get('tmdb_id') or -1)

    if picked_id not in valid_ids or picked_id in history_ids:
        # Fallback pick: random from full pool for surprises, top-ranked otherwise
        if is_surprise:
            picked_id = int(candidates.sample(1).iloc[0].tmdb_id)
        else:
            picked_id = int(candidates.iloc[0].tmdb_id)

    picked_row = candidates[candidates['tmdb_id'] == picked_id].iloc[0]
    picked_title = _clean(str(picked_row.title))

    description = str(result.get('description') or '').strip()
    if not description:
        if result:
            pass
        extra = tmdb_details.get(picked_id, {})
        tagline = _clean(extra.get("tagline", ""))
        description = _make_fallback(preferences, picked_row, tagline, is_surprise=is_surprise)
    elif is_surprise:
        # LLM gave a description but the user typed gibberish — prepend a surprise opener
        _SURPRISE_OPENERS = [
            "Not sure what you're in the mood for?",
            "Since you're not sure what to watch —",
            "Since you seem a bit lost on what to watch,",
            "Going with something unexpected tonight —",
        ]
        opener = random.choice(_SURPRISE_OPENERS)
        _ALREADY_HAS_OPENER = ("not sure", "since you", "going with", "surprise", "unexpected")
        if not any(description.lower().startswith(p) for p in _ALREADY_HAS_OPENER):
            description = f"{opener} {description}"
    if picked_title.lower() not in description.lower():
        description = f"{picked_title}: {description}"

    # If the user specified a runtime constraint, make sure the duration appears in the
    # description so they can confirm it fits — append "(X min)" as a safety net if the
    # LLM forgot to mention it.
    if has_runtime_constraint:
        rt_str = _runtime_str(picked_row)
        if rt_str and not re.search(r'\b\d+\s*min', description, re.IGNORECASE):
            description = f"{description.rstrip('. ')}. ({rt_str})"

    # Prepend a caveat when the pick misses the requested genre/nationality.
    # Only add it if the description doesn't already acknowledge the gap.
    if not is_surprise:
        caveat = _match_caveat(picked_row)
        _CAVEAT_SIGNALS = ("couldn't find", "couldn't find", "no exact", "closest", "not exactly", "best i could")
        if caveat and not any(s in description.lower() for s in _CAVEAT_SIGNALS):
            description = f"{caveat} {description}"

    return {
        "tmdb_id": picked_id,
        "description": description[:500],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    args = parser.parse_args()

    print("Movie recommender – type your preferences and press Enter.")
    preferences = (
        args.preferences.strip() if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip() if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\nThinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - start
    print(result)
    print(f"\nServed in {elapsed:.2f}s")
