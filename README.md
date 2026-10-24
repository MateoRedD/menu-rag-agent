# Menu RAG Agent

A retrieval-augmented generation (RAG) pipeline that lets a restaurant's customer-service agent answer questions from an approved source (the menu and house policies), in Spanish or English.


## How it works

1. **Chunking** – `data/menu.md` is split with two strategies so they can be compared:
   - `item`: one chunk per dish or policy, prefixed with its section (structure-aware).
   - `fixed`: 300-character windows with 50 characters of overlap (baseline).
2. **Embeddings** – each chunk is embedded with OpenAI `text-embedding-3-small` (1536 dims).
3. **Vector store** – chunks and vectors live in Supabase Postgres with `pgvector`; a SQL function ranks them by cosine similarity.
4. **Evaluation** – 18 labeled questions (half in English against a Spanish menu) measure hit@1, hit@3 and MRR@5 for each strategy.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill in your keys
```

Run `sql/schema.sql` once in the Supabase SQL Editor, then:

```bash
python ingest.py                         # chunk, embed and store
python ask.py "Do you deliver to Riomar?"  # inspect what the retriever returns
python -m eval.run_eval                  # compare chunking strategies
```

## Results

| Strategy |   hit@1   |   hit@3   |   MRR@5   |
|----------|  -------  |  -------  |  -------  | 
| item     |    0.98   |   1.00    |   0.97    |
| fixed    |    0.78   |   1.00    |    0.88   |

**What I learned:** _(fill in after running the evaluation)_

## Roadmap

- [x] Phase 1 – chunking, embeddings, vector search, retrieval evaluation
- [ ] Phase 2 – tool-calling agent (LangChain) with retrieval, order and human-handoff tools
- [ ] Phase 3 – FastAPI endpoint with API-key auth and rate limiting, Docker, deploy to Azure
