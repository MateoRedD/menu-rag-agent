# Menu RAG Agent

A customer-service agent for a (fictional) restaurant in Barranquilla, Colombia. It answers questions in Spanish or English using only an approved source (`data/menu.md`: dishes, prices, hours, delivery, payments, reservations, allergens and location), takes orders with totals computed in code, and hands conversations off to a human when it should.

## Example

A real conversation from `python chat.py` (tool calls are printed by the CLI):

```text
You: ¿Tienen sushi?
  [tool] search_menu({'query': 'sushi'})
Agent: No, el restaurante no ofrece sushi en su menú. ¿Quieres que te cuente sobre otras opciones que tienen?

You: Do you deliver to Riomar? How much is it?
  [tool] search_menu({'query': 'domicilios a Riomar'})
Agent: Yes, we deliver to Riomar. The delivery fee is 5,000 COP, but it's free for orders over 80,000 COP. Would you like to place an order?

You: I want 2 arepas de huevo and one limonada de coco, delivery to Calle 80 # 52-10, Riomar. My name is Ana
  [tool] create_order({'customer_name': 'Ana', 'items': [{'name': 'Arepa de huevo', 'quantity': 2}, {'name': 'Limonada de coco', 'quantity': 1}], 'order_type': 'delivery', 'confirmed': False, 'address': 'Calle 80 # 52-10, Riomar'})
Agent: Your order is 2 Arepa de huevo and 1 Limonada de coco for delivery to Calle 80 # 52-10, Riomar. The delivery fee is 5,000 COP, and the total is 28,000 COP. Do you confirm the order?
```

## How it works

**Retrieval (Phase 1)**

1. **Chunking** – `data/menu.md` is split with two strategies so they can be compared:
   - `item`: one chunk per dish or policy, prefixed with its section (structure-aware). Used by the agent.
   - `fixed`: 300-character windows with 50 characters of overlap (baseline).
2. **Embeddings** – each chunk is embedded with OpenAI `text-embedding-3-small` (1536 dims).
3. **Vector store** – chunks and vectors live in Supabase Postgres with `pgvector`; a SQL function ranks them by cosine similarity.

**Agent (Phase 2)** – built with LangChain `create_agent` (LangChain 1.4, LangGraph 1.2) and an OpenAI chat model set by `CHAT_MODEL` (default `gpt-4.1-mini`, temperature 0).

| Tool | What it does |
|------|--------------|
| `search_menu(query)` | Top 4 `item` chunks above a similarity floor of 0.20, or an explicit `NO_RESULTS`. |
| `create_order(...)` | Validates dishes against the menu, quantities, delivery zone and weekend-only dishes, and computes the total in Python (prices are parsed from `menu.md`). With `confirmed=false` it only returns a quote; with `confirmed=true` it saves to the `orders` table. |
| `handoff_to_human(reason, summary)` | Saves a record in the `handoffs` table with one of five reasons: `complaint`, `allergy`, `group_booking`, `unknown_answer`, `other`. |

**Guardrails**

- The system prompt tells the model to answer only from `search_menu` results, to say "we don't have it" instead of guessing, to never calculate prices, to confirm an order before saving it, and when to hand off.
- Prices and totals are computed in code. Invalid orders come back as an error message the model can act on.
- The reply language is detected in code from the latest user message and stated explicitly in the prompt (see *What I learned*).
- The database rejects bad rows on its own (`CHECK` constraints) and has Row Level Security on with no policies, so only the server-side key can read or write.
- The prompt is rebuilt on every model call with the current date and time in Colombia, so the agent knows today's day.

**Memory** – conversation history per `thread_id` with LangGraph's `InMemorySaver` (lost when the process stops).

## Project structure

```text
data/menu.md          approved source (Spanish)
sql/schema.sql        documents table + match_documents() similarity search
sql/phase2.sql        orders and handoffs tables
rag.py                chunking, embeddings, vector search with similarity floor
ingest.py             chunk, embed and store the menu
menu.py               price parsing and order validation/totals
tools.py              the three agent tools
agent.py              system prompt, language detection, agent setup
chat.py               interactive CLI
ask.py                inspect what the retriever returns
eval/                 retrieval evaluation (26 labeled questions)
tests/                unit tests (offline) and integration tests (real API)
```

## Setup

Tested on Python 3.14. Python 3.15 doesn't work yet: `zstandard`, a LangChain dependency, has no prebuilt wheel for it.

```bash
python -m venv .venv            # Windows: py -3.14 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # Windows: copy .env.example .env, then fill in your keys
```

Run `sql/schema.sql` and then `sql/phase2.sql` once in the Supabase SQL Editor, then:

```bash
python ingest.py                           # chunk, embed and store
python chat.py                             # talk to the agent
python ask.py "Do you deliver to Riomar?"  # inspect what the retriever returns
python -m eval.run_eval                    # retrieval evaluation
pytest                                     # unit tests: offline, free
pytest -m integration                      # integration tests: real OpenAI calls, a few US cents
```

## Results

**Retrieval** – 18 answerable questions (9 Spanish, 9 English), `item` = 21 chunks, `fixed` = 13 chunks:

| Strategy | hit@1 | hit@3 | MRR@5 |
|----------|-------|-------|-------|
| item     | 0.94  | 1.00  | 0.97  |
| fixed    | 0.78  | 1.00  | 0.88  |

Similarity floor of 0.20 on `item`: keeps the relevant chunk for 18/18 answerable questions and filters out 3 of 8 out-of-scope questions (weather, pets, pizza). The other 5 are left to the model.

**Tests**

- 40 unit tests (no API): price parsing, totals and the free-delivery boundary, order validation, the similarity floor, the tools (with Supabase faked) and language detection.
- 15 integration tests (real model and retrieval, database writes faked): Spanish and English answers, a follow-up question using memory, out-of-scope questions (sushi, pizza, kids' menu), quote vs. confirmed order, rejected orders (delivery outside the zones, sancocho on a Tuesday), handoffs for a complaint, a severe allergy, a group booking and a request for a person, and a prompt-injection attempt.

The integration tests check behavior (which tool was called, with which arguments, the reply language, the total), not exact wording, because model output varies slightly between runs.

## What I learned

**Retrieval**

- **Structure-aware chunking wins at rank 1** (hit@1 0.94 vs 0.78). All four `fixed` misses
  were chunks cut mid-sentence: for "group of 8" the top result started with "de 6 personas o más…"
  and the actual answer ("24 horas de anticipación") had fallen into the previous window.
- **The section prefix helps, but adds noise.** The only `item` miss, "¿Dónde queda el restaurante?",
  ranked *Opening hours* first because every policy chunk starts with "Políticas del restaurante".
- **The eval is too easy.** hit@3 is 1.00 for both strategies because the top 3 is already 14% of a
  21-chunk corpus, so it can't tell the strategies apart. Substring matching is also lenient: a
  half-cut chunk still counts as a hit.
- **Cross-lingual retrieval works, with lower scores**: English questions against the Spanish menu
  score ~0.42 vs ~0.57 for Spanish, but the ranking is still correct. Rewriting the query in Spanish
  raised the weakest English question from 0.26 to 0.48, so the search tool asks the model for Spanish queries.

**Agent**

- **A similarity threshold can't tell in-scope from out-of-scope.** Answerable questions scored as
  low as 0.26 and unanswerable ones as high as 0.43 ("¿Tienen menú infantil?"), so any single cutoff
  either drops real answers or lets junk through. I kept a low floor (0.20) for clearly off-topic
  queries and left "we don't have that" to the prompt, which the integration tests check.
- **"Reply in the customer's language" made the model reply in Spanish to English questions.**
  I ruled out the query wording, three prompt rewrites and LangChain itself (same result with the raw
  OpenAI SDK). In a minimal SDK test the ambiguous rule was the trigger (without it the model answered in
  English), but in the full agent the Spanish context still won even with no rule. Detecting the language
  in code and writing "Reply in English" or "Reply in Spanish" fixed it in every test.
- **Arithmetic and business rules belong in code.** The model never sees a price it could miscalculate:
  totals, the free-delivery boundary (exactly 80,000 COP still pays) and weekend-only dishes are
  validated in Python and unit-tested.

## Limitations

- `handoff_to_human` only stores a record; it doesn't notify anyone yet.
- Conversation memory is in RAM, so it's lost when the process restarts.
- The integration tests detect "the agent declined" by looking for words like "no" or "sorry", which is coarse.
- Orders don't collect a phone number, aren't checked against opening hours, and the age limit for beer is only a prompt rule.
- Language detection is a word-list heuristic for Spanish and English only.

## Roadmap

- [x] Phase 1 – chunking, embeddings, vector search, retrieval evaluation
- [x] Phase 2 – tool-calling agent (LangChain) with retrieval, order and human-handoff tools, guardrails, unit and integration tests
- [ ] Phase 3 – FastAPI endpoint with API-key auth and rate limiting, Docker, deploy to Azure, unit tests in GitHub Actions
- [ ] Persistent memory (`PostgresSaver`), handoff notifications, LLM-as-judge for the integration tests
