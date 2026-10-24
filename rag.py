"""Core RAG pieces: chunking, embeddings and vector search."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"


@lru_cache
def openai_client():
    return OpenAI()  # reads OPENAI_API_KEY from the environment


@lru_cache
def db():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


# ---------- Chunking ----------

def chunk_by_item(text: str) -> list[str]:
    """One chunk per '## ' block (a dish or a policy), prefixed with its '# ' section."""
    chunks, section, current = [], "", []

    def flush():
        if current:
            chunks.append("\n".join(current).strip())
            current.clear()

    for line in text.splitlines():
        if line.startswith("# "):
            flush()
            section = line[2:].strip()
        elif line.startswith("## "):
            flush()
            current.append(f"{section} > {line[3:].strip()}")
        elif current:
            current.append(line)
    flush()
    return chunks


def chunk_fixed(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Baseline: fixed-size character windows with overlap, ignoring the document structure."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


CHUNKERS = {"item": chunk_by_item, "fixed": chunk_fixed}


# ---------- Embeddings and search ----------

def embed(texts: list[str]) -> list[list[float]]:
    resp = openai_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def search(query: str, strategy: str = "item", k: int = 3) -> list[dict]:
    query_embedding = embed([query])[0]
    resp = db().rpc(
        "match_documents",
        {"query_embedding": query_embedding, "match_count": k, "filter_strategy": strategy},
    ).execute()
    return resp.data
