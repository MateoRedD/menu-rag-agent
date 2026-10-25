"""Assign each message in handoff_dataset.jsonl to "dev" or "test", once, and save it in the file.

Stratified by (should_handoff, lang): each of the four groups is shuffled with a fixed seed and its
first 60% goes to dev. Committing the result (not just this script) keeps the split from ever changing,
so test results can't be improved by re-shuffling until the numbers look good.

Usage (run once): python -m eval.split_handoff_dataset
"""
import json
import random
from collections import defaultdict
from pathlib import Path

PATH = Path("eval/handoff_dataset.jsonl")
SEED = 42
DEV_SHARE = 0.6

rows = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
if any("split" in r for r in rows):
    raise SystemExit("The dataset is already split. Delete the 'split' fields first if you really mean to redo it.")

groups = defaultdict(list)
for r in rows:
    groups[(r["should_handoff"], r["lang"])].append(r)

rng = random.Random(SEED)
for key in sorted(groups):  # sorted so the order of groups, and therefore of random draws, is fixed
    group = sorted(groups[key], key=lambda r: r["id"])
    rng.shuffle(group)
    n_dev = round(len(group) * DEV_SHARE)
    for i, r in enumerate(group):
        r["split"] = "dev" if i < n_dev else "test"

PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
for key in sorted(groups):
    splits = [r["split"] for r in groups[key]]
    print(f"should_handoff={key[0]!s:5} lang={key[1]}: dev {splits.count('dev')}, test {splits.count('test')}")
