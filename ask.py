"""Quick manual check of what the retriever returns.

Usage: python ask.py "Do you deliver to Riomar?" [item|fixed]
"""
import sys

from rag import search

query = sys.argv[1]
strategy = sys.argv[2] if len(sys.argv) > 2 else "item"

for r in search(query, strategy, k=3):
    print(f"\n[{r['similarity']:.3f}] {r['content']}")
