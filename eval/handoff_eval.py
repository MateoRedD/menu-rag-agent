"""Evaluate the handoff decision as a binary classifier.

Positive class = "should hand off to a human". Prediction = the agent called handoff_to_human.
Each message runs RUNS times with the real agent and real retrieval; database writes are faked, so
nothing is inserted into Supabase. The headline numbers use the majority vote of the runs, and the
per-run numbers show how much the LLM varies.

Usage: python -m eval.handoff_eval --split dev     (look at it and iterate as often as needed)
       python -m eval.handoff_eval --split test    (once, at the end; refuses to run twice)
Costs money (OpenAI calls), so it lives here and not in tests/: CI never runs it.
"""
import argparse
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

DATASET = Path("eval/handoff_dataset.jsonl")
RESULTS_DIR = Path("eval/results")
RUNS = 3


# ---------- Metrics (plain Python so the formulas are visible) ----------

def confusion(pairs):
    """pairs: (actual, predicted) booleans. Returns TP, FP, FN, TN."""
    tp = sum(a and p for a, p in pairs)
    fp = sum(not a and p for a, p in pairs)
    fn = sum(a and not p for a, p in pairs)
    tn = sum(not a and not p for a, p in pairs)
    return tp, fp, fn, tn


def ratio(num, den):
    """'0.90 (9/10)'. Counts always shown: with small samples one message moves the percentage a lot."""
    return f"{num / den:.2f} ({num}/{den})" if den else f"n/a (0/0)"


def scores(pairs):
    tp, fp, fn, tn = confusion(pairs)
    precision = tp / (tp + fp) if tp + fp else None  # of the handoffs the agent made, how many were right
    recall = tp / (tp + fn) if tp + fn else None      # of the handoffs that were needed, how many it made
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return SimpleNamespace(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1,
                           p_text=ratio(tp, tp + fp), r_text=ratio(tp, tp + fn),
                           f1_text=f"{f1:.2f}" if f1 is not None else "n/a")


def majority(votes):
    return sum(votes) * 2 > len(votes)


# ---------- Running the agent ----------

def run_agent(agent, text):
    """One fresh conversation. Returns (called handoff?, reason or None, final reply)."""
    result = agent.invoke({"messages": [{"role": "user", "content": text}]},
                          {"configurable": {"thread_id": str(uuid.uuid4())}})
    calls = [c for m in result["messages"] for c in getattr(m, "tool_calls", []) if c["name"] == "handoff_to_human"]
    reason = calls[0]["args"].get("reason") if calls else None
    return bool(calls), reason, result["messages"][-1].content


def fake_db():
    """Stands in for Supabase writes in tools.py (reads in rag.py stay real)."""
    def insert(row):
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"id": 0, **row}]))
    return SimpleNamespace(table=lambda name: SimpleNamespace(insert=insert))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args()

    out_path = RESULTS_DIR / f"handoff_{args.split}.json"
    if args.split == "test" and out_path.exists():
        raise SystemExit(f"{out_path} already exists: the test split is run once. Look at dev instead.")

    import tools  # imported here so the metric functions above can be unit-tested without the agent
    from agent import build_agent
    tools.db = fake_db

    rows = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == args.split]
    agent = build_agent()
    model = os.environ.get("CHAT_MODEL", "gpt-4.1-mini")
    print(f"== handoff eval: split={args.split}, messages={len(rows)}, runs={RUNS}, model={model} ==")

    for i, r in enumerate(rows, 1):
        r["runs"] = [dict(zip(("handoff", "reason", "reply"), run_agent(agent, r["text"]))) for _ in range(RUNS)]
        votes = [run["handoff"] for run in r["runs"]]
        r["predicted"] = majority(votes)
        reasons = [run["reason"] for run in r["runs"] if run["handoff"]]
        r["predicted_reason"] = Counter(reasons).most_common(1)[0][0] if reasons else None
        print(f"  [{i:2}/{len(rows)}] {r['id']} handoff votes {sum(votes)}/{RUNS}")

    # ----- Headline: majority vote -----
    s = scores([(r["should_handoff"], r["predicted"]) for r in rows])
    print(f"\nMajority vote (handoff in at least 2 of {RUNS} runs)")
    print("                     predicted handoff   predicted no handoff")
    print(f"  actual handoff     TP = {s.tp:<15} FN = {s.fn}")
    print(f"  actual no handoff  FP = {s.fp:<15} TN = {s.tn}")
    print(f"\n  {'':6} {'precision':18} {'recall':18} F1")
    for label, subset in [("all", rows), ("es", [r for r in rows if r["lang"] == "es"]),
                          ("en", [r for r in rows if r["lang"] == "en"])]:
        sub = scores([(r["should_handoff"], r["predicted"]) for r in subset])
        print(f"  {label:6} {sub.p_text:18} {sub.r_text:18} {sub.f1_text}")

    true_positives = [r for r in rows if r["should_handoff"] and r["predicted"]]
    correct = sum(r["predicted_reason"] == r["expected_reason"] for r in true_positives)
    print(f"\n  Reason correct in true positives: {ratio(correct, len(true_positives))}")
    for r in true_positives:
        if r["predicted_reason"] != r["expected_reason"]:
            print(f"    {r['id']} expected {r['expected_reason']}, got {r['predicted_reason']}: {r['text']}")

    # ----- Variability between runs -----
    print("\nPer run (how much the LLM varies)")
    for k in range(RUNS):
        sub = scores([(r["should_handoff"], r["runs"][k]["handoff"]) for r in rows])
        print(f"  run {k + 1}: precision {sub.p_text:14} recall {sub.r_text:14} F1 {sub.f1_text}")
    unstable = [r for r in rows if len({run["handoff"] for run in r["runs"]}) > 1]
    print(f"  Messages where the runs disagree: {ratio(len(unstable), len(rows))}")
    for r in unstable:
        print(f"    {r['id']} ({sum(run['handoff'] for run in r['runs'])}/{RUNS} handoff): {r['text']}")

    # ----- Errors, with text, to analyze -----
    for title, wrong in [("False negatives (needed a human, didn't get one)",
                          [r for r in rows if r["should_handoff"] and not r["predicted"]]),
                         ("False positives (handed off without need)",
                          [r for r in rows if not r["should_handoff"] and r["predicted"]])]:
        print(f"\n{title}: {len(wrong)}")
        for r in wrong:
            print(f"  {r['id']} [{r['lang']}] {r['text']}")
            print(f"      reply: {r['runs'][0]['reply'][:140]!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({
        "split": args.split, "model": model, "runs_per_message": RUNS,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tp": s.tp, "fp": s.fp, "fn": s.fn, "tn": s.tn,
        "precision": s.precision, "recall": s.recall, "f1": s.f1,
        "messages": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
