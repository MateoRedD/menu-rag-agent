"""Retrieval evaluation: compares chunking strategies on a labeled question set.

A retrieved chunk counts as relevant if it contains any of the question's "expected" strings.
Questions with an empty "expected" list are out of scope (nothing in the menu answers them).
Metrics (in-scope questions only):
  hit@1 / hit@3 -> share of questions with a relevant chunk in the top 1 / top 3 (recall@k with one answer per question)
  MRR@5         -> mean of 1/rank of the first relevant chunk (0 if none in the top 5)
Threshold check (rag.MIN_SIMILARITY):
  in-scope kept     -> the first relevant chunk is at or above the threshold
  out-of-scope cut  -> no chunk at all is at or above the threshold

Usage: python -m eval.run_eval
"""
import json
from pathlib import Path

from rag import CHUNKERS, MIN_SIMILARITY, search

K = 5
questions = [json.loads(line) for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
in_scope = [q for q in questions if q["expected"]]
out_of_scope = [q for q in questions if not q["expected"]]


def first_relevant(results, expected):
    """(rank, chunk) of the first relevant chunk, or (None, None)."""
    for rank, r in enumerate(results, start=1):
        if any(e.lower() in r["content"].lower() for e in expected):
            return rank, r
    return None, None


for strategy in CHUNKERS:
    ranks, misses, kept = [], [], 0
    for item in in_scope:
        results = search(item["q"], strategy, K)
        rank, chunk = first_relevant(results, item["expected"])
        ranks.append(rank)
        kept += chunk is not None and chunk["similarity"] >= MIN_SIMILARITY
        if rank != 1:
            top = results[0]["content"].splitlines()[0][:60]  # what the retriever ranked first instead
            misses.append(f'  [{item["lang"]}] {item["q"]} -> rank {rank} (top 1: "{top}")')

    cut, leaks = 0, []
    for item in out_of_scope:
        top = search(item["q"], strategy, 1)[0]
        if top["similarity"] < MIN_SIMILARITY:
            cut += 1
        else:
            leaks.append(f'  [{item["lang"]}] {item["q"]} -> {top["similarity"]:.2f} "{top["content"].splitlines()[0][:50]}"')

    n = len(ranks)
    hit1 = sum(r == 1 for r in ranks) / n
    hit3 = sum(r is not None and r <= 3 for r in ranks) / n
    mrr = sum(1 / r for r in ranks if r) / n

    print(f"\n== strategy: {strategy} ==  hit@1 {hit1:.2f} | hit@3 {hit3:.2f} | MRR@5 {mrr:.2f}")
    print("Not at rank 1:" if misses else "All questions found at rank 1.")
    print("\n".join(misses))
    print(f"Threshold {MIN_SIMILARITY}: in-scope kept {kept}/{n} | out-of-scope cut {cut}/{len(out_of_scope)}")
    print("Out-of-scope above threshold (the LLM must say we don't have it):")
    print("\n".join(leaks))
