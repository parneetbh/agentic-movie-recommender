"""
Fast retrieval regression tests — no API calls needed.

Checks that get_candidates() ranks the right movies highly for a wide range
of query types. Run this after any change to scoring logic.

Usage:
    python test_retrieval.py
"""

import sys
from llm import get_candidates, MOVIES_DF

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def top_ids(preferences, history_ids=None, n=5):
    return set(get_candidates(preferences, history_ids or [], n=n)['tmdb_id'].tolist())


def top_genres(preferences, history_ids=None, n=5):
    return " | ".join(
        get_candidates(preferences, history_ids or [], n=n)['genres'].tolist()
    )


def assert_in_top(label, preferences, must_contain_tmdb_id, n=5, history_ids=None):
    ids = top_ids(preferences, history_ids, n)
    ok = must_contain_tmdb_id in ids
    title = MOVIES_DF[MOVIES_DF['tmdb_id'] == must_contain_tmdb_id]['title'].iloc[0]
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         Expected '{title}' (id={must_contain_tmdb_id}) in top {n}")
        print(f"         Got: {top_genres(preferences, history_ids, n)}")
    return ok


def assert_not_in_top(label, preferences, must_exclude_tmdb_id, n=10, history_ids=None):
    ids = top_ids(preferences, history_ids, n)
    ok = must_exclude_tmdb_id not in ids
    title = MOVIES_DF[MOVIES_DF['tmdb_id'] == must_exclude_tmdb_id]['title'].iloc[0]
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         Expected '{title}' (id={must_exclude_tmdb_id}) NOT in top {n}")
    return ok


def assert_genres_in_top(label, preferences, required_genre, n=5, history_ids=None):
    candidates = get_candidates(preferences, history_ids or [], n=n)
    genre_hits = candidates['genres'].str.contains(required_genre, na=False).sum()
    ok = genre_hits >= (n // 2 + 1)  # majority should match
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         Only {genre_hits}/{n} top results contain genre '{required_genre}'")
        print(f"         Got: {top_genres(preferences, history_ids, n)}")
    return ok


def assert_excluded_not_in_top(label, preferences, excluded_genre, n=10, history_ids=None):
    candidates = get_candidates(preferences, history_ids or [], n=n)
    violations = candidates[candidates['genres'].str.contains(excluded_genre, na=False)]
    ok = len(violations) == 0
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if not ok:
        print(f"         Found {len(violations)} movies with excluded genre '{excluded_genre}' in top {n}:")
        print(f"         {violations['title'].tolist()}")
    return ok


def assert_history_excluded(label, preferences, history_ids, n=20):
    ids = top_ids(preferences, history_ids, n)
    violations = set(history_ids) & ids
    ok = len(violations) == 0
    status = PASS if ok else FAIL
    print(f"  [{status}] {label}")
    if not ok:
        titles = MOVIES_DF[MOVIES_DF['tmdb_id'].isin(violations)]['title'].tolist()
        print(f"         Watch history appeared in candidates: {titles}")
    return ok


results = []

# ── Country queries ────────────────────────────────────────────────────────────
print("\n=== Country Queries ===")
results.append(assert_in_top(
    "Iran -> 'It Was Just an Accident' in top 3",
    "a movie from Iran", 1456349, n=3
))
results.append(assert_in_top(
    "South Korea -> Parasite in top 3",
    "Korean movie", 496243, n=3
))
results.append(assert_in_top(
    "Japan -> Your Name in top 5",
    "Japanese animated movie", 372058, n=5
))

# ── Genre queries ──────────────────────────────────────────────────────────────
print("\n=== Genre Queries ===")
results.append(assert_genres_in_top(
    "Comedy request -> majority Comedy in top 5",
    "something funny and light", "Comedy", n=5
))
results.append(assert_genres_in_top(
    "Horror request -> majority Horror in top 5",
    "a really scary horror movie", "Horror", n=5
))
results.append(assert_genres_in_top(
    "Romance request -> majority Romance in top 5",
    "a romantic love story", "Romance", n=5
))
results.append(assert_genres_in_top(
    "Animation request -> majority Animation in top 5",
    "animated movie for the family", "Animation", n=5
))

# ── Negation queries ───────────────────────────────────────────────────────────
print("\n=== Negation Queries ===")
results.append(assert_excluded_not_in_top(
    "No thriller -> Thriller excluded from top 10",
    "funny movie no thriller", "Thriller", n=10
))
results.append(assert_excluded_not_in_top(
    "No horror -> Horror excluded from top 10",
    "I want something fun, no horror", "Horror", n=10
))
results.append(assert_excluded_not_in_top(
    "No romance -> Romance excluded from top 10",
    "action movie without romance", "Romance", n=10
))

# ── Light mood / dark genre penalty ───────────────────────────────────────────
print("\n=== Light Mood Queries ===")
results.append(assert_not_in_top(
    "Cooking/chill mood -> Parasite not in top 5",
    "something light while cooking", 496243, n=5
))
results.append(assert_genres_in_top(
    "Feel-good -> mostly Comedy/Family in top 5",
    "feel-good movie to watch with family", "Comedy", n=5
))

# ── Multi-genre queries ────────────────────────────────────────────────────────
print("\n=== Multi-Genre Queries ===")
results.append(assert_genres_in_top(
    "Action + comedy -> Action in top results",
    "action comedy movie", "Action", n=5
))

# ── Watch history exclusion ────────────────────────────────────────────────────
print("\n=== Watch History Exclusion ===")
results.append(assert_history_excluded(
    "Watched movies never appear in candidates",
    "great drama film",
    [496243, 155, 157336],  # Parasite, Dark Knight, Interstellar
    n=20
))

# ── Food/topic queries ─────────────────────────────────────────────────────────
print("\n=== Topic Queries ===")
results.append(assert_in_top(
    "Food movie -> The Menu in top 3",
    "food movie", 593643, n=3
))
results.append(assert_not_in_top(
    "Food movie no thriller -> The Menu excluded",
    "food movie no thriller", 593643, n=10
))

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(results)
total = len(results)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if passed < total:
    sys.exit(1)
