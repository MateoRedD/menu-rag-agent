"""Chunk the menu with every strategy, embed the chunks and store them in Supabase.

Usage: python ingest.py
"""
from pathlib import Path

from rag import CHUNKERS, db, embed

text = Path("data/menu.md").read_text(encoding="utf-8")

for strategy, chunker in CHUNKERS.items():
    chunks = chunker(text)
    vectors = embed(chunks)  # one API call per strategy (batched)

    db().table("documents").delete().eq("strategy", strategy).execute()  # re-runs don't duplicate rows
    db().table("documents").insert([
        {"strategy": strategy, "content": c, "embedding": v, "metadata": {"source": "menu.md", "chunk": i}}
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]).execute()

    print(f"{strategy}: {len(chunks)} chunks stored")
