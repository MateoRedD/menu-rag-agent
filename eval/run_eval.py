"""Retrieval evaluation: compares chunking strategies on a labeled question set.

A retrieved chunk counts as relevant if it contains any of the question's "expected" strings.
Metrics:
  hit@1 / hit@3 -> share of questions with a relevant chunk in the top 1 / top 3 (recall@k with one answer per question)
  MRR@5         -> mean of 1/rank of the first relevant chunk (0 if none in the top 5)

Usage: python -m eval.run_eval
"""
import json
from pathlib import Path

from rag import CHUNKERS, search

K = 5
questions = [json.loads(line) for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def first_relevant_rank(results, expected):
    for rank, r in enumerate(results, start=1):
        if any(e.lower() in r["content"].lower() for e in expected):
            return rank
    return None


for strategy in CHUNKERS:
    ranks, misses = [], []
    for item in questions:
        results = search(item["q"], strategy, K)
        rank = first_relevant_rank(results, item["expected"])
        ranks.append(rank)
        if rank != 1:
            top = results[0]["content"].splitlines()[0][:60]  # what the retriever ranked first instead
            misses.append(f'  [{item["lang"]}] {item["q"]} -> rank {rank} (top 1: "{top}")')

    n = len(ranks)
    hit1 = sum(r == 1 for r in ranks) / n
    hit3 = sum(r is not None and r <= 3 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n

    print(f"\n== strategy: {strategy} ==  hit@1 {hit1:.2f} | hit@3 {hit3:.2f} | MRR@5 {mrr:.2f}")
    print("Not at rank 1:" if misses else "All questions found at rank 1.")
    print("\n".join(misses))
