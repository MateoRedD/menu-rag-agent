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

18 questions (9 Spanish, 9 English), `item` = 21 chunks, `fixed` = 13 chunks:

| Strategy | hit@1 | hit@3 | MRR@5 |
|----------|-------|-------|-------|
| item     | 0.94  | 1.00  | 0.97  |
| fixed    | 0.78  | 1.00  | 0.88  |

## What I learned

- **Structure-aware chunking wins at rank 1** (hit@1 0.94 vs 0.78). All four `fixed` misses
  were chunks cut mid-sentence: for "group of 8" the top result started with "de 6 personas o más…"
  and the actual answer ("24 horas de anticipación") had fallen into the previous window.
- **The section prefix helps, but adds noise.** The only `item` miss, "¿Dónde queda el restaurante?",
  ranked *Opening hours* first because every policy chunk starts with "Políticas del restaurante".
- **The eval is too easy.** hit@3 is 1.00 for both strategies because the top 3 is already 14% of a
  21-chunk corpus, so it can't tell the strategies apart. Substring matching is also lenient: a
  half-cut chunk still counts as a hit.
- **Top-k always returns k results**, even irrelevant ones ("Do you deliver to Riomar?" brings
  carimañolas at 0.27 similarity in 3rd place). The agent needs a similarity threshold before
  passing context to the LLM.
- **Cross-lingual retrieval works, with lower scores**: English questions against the Spanish menu
  score ~0.42 vs ~0.57 for Spanish, but the ranking is still correct.

## Roadmap

- [x] Phase 1 – chunking, embeddings, vector search, retrieval evaluation
- [ ] Phase 2 – tool-calling agent (LangChain) with retrieval, order and human-handoff tools
- [ ] Phase 3 – FastAPI endpoint with API-key auth and rate limiting, Docker, deploy to Azure
