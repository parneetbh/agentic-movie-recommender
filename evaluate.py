"""
LLM-as-a-judge evaluator.

Usage:
    python evaluate.py

Runs get_recommendation() against a set of test prompts, then asks the same
LLM to score each result on relevance and persuasiveness (1-5 each).
"""

import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import ollama
from llm import get_recommendation, MODEL

JUDGE_PROMPT = """You are a movie recommendation judge.

A user said: "{preferences}"
The recommender suggested: "{title}" — "{description}"

Score this recommendation on two dimensions (integers 1-5):
1. relevance   — does this movie actually match what the user asked for?
2. persuasion  — does the description make you want to watch it?

Respond with ONLY valid JSON:
{{"relevance": <1-5>, "persuasion": <1-5>, "reasoning": "<one sentence>"}}"""

TEST_CASES = [
    {"preferences": "I want a thrilling sci-fi with mind-bending twists", "history": [], "history_ids": []},
    {"preferences": "Something funny and feel-good for a Friday night", "history": [], "history_ids": []},
    {"preferences": "A dark psychological thriller that keeps me guessing", "history": ["Gone Girl"], "history_ids": [128796]},
    {"preferences": "Epic fantasy adventure with great world-building", "history": [], "history_ids": []},
    {"preferences": "A heartwarming story about family and redemption", "history": ["The Dark Knight"], "history_ids": [155]},
]


def judge(preferences: str, title: str, description: str, client: ollama.Client) -> dict:
    prompt = JUDGE_PROMPT.format(
        preferences=preferences,
        title=title,
        description=description,
    )
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(response.message.content)


def main():
    if not os.environ.get("OLLAMA_API_KEY"):
        print("ERROR: OLLAMA_API_KEY is not set.")
        sys.exit(1)

    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )

    from llm import MOVIES_DF

    total_relevance = 0
    total_persuasion = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['preferences'][:60]}")
        result = get_recommendation(case["preferences"], case["history"], case["history_ids"])

        movie_row = MOVIES_DF[MOVIES_DF["tmdb_id"] == result["tmdb_id"]]
        title = movie_row.iloc[0]["title"] if not movie_row.empty else f"tmdb_id={result['tmdb_id']}"

        print(f"  Recommended : {title}")
        print(f"  Description : {result['description'][:100]}...")

        scores = judge(case["preferences"], title, result["description"], client)
        r, p = scores.get("relevance", 0), scores.get("persuasion", 0)
        total_relevance += r
        total_persuasion += p
        print(f"  Relevance   : {r}/5")
        print(f"  Persuasion  : {p}/5")
        print(f"  Reasoning   : {scores.get('reasoning', '')}")

    n = len(TEST_CASES)
    print(f"\n{'='*50}")
    print(f"Avg relevance  : {total_relevance/n:.1f}/5")
    print(f"Avg persuasion : {total_persuasion/n:.1f}/5")
    print(f"Overall        : {(total_relevance+total_persuasion)/(n*2):.1f}/5")


if __name__ == "__main__":
    main()
