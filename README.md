# Menu RAG Agent

![Unit tests](https://github.com/MateoRedD/menu-rag-agent/actions/workflows/tests.yml/badge.svg)

A customer-service agent for a (fictional) restaurant in Barranquilla, Colombia. It answers questions in Spanish or English using only an approved source (`data/menu.md`: dishes, prices, hours, delivery, payments, reservations, allergens and location), takes orders with totals computed in code, and hands conversations off to a human when it should. It runs as an authenticated, rate-limited HTTP API in a Docker container on Azure Container Apps.

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

**API (Phase 3)** – FastAPI app in `api.py`:

| Endpoint | Auth | What it does |
|----------|------|--------------|
| `GET /health` | none | Liveness check for the platform. |
| `POST /v1/chat` | `X-API-Key` header | Sends one message; pass the returned `conversation_id` to continue the conversation. |

```bash
curl -X POST https://<your-app>/v1/chat \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"message": "¿Cuánto cuesta la mojarra frita?"}'
```

```json
{"reply": "La mojarra frita cuesta 38.000 COP. ¿Quieres saber algo más o te gustaría hacer un pedido?",
 "conversation_id": "396c8640-f2d6-48f8-a8e6-362cc666655b",
 "tools_used": ["search_menu"]}
```

- **Auth:** the key is compared in constant time (as bytes, so a non-ASCII key gets a 401 instead of a 500). If `API_KEY` is not configured, every request is rejected.
- **Rate limit:** 10 requests per 60-second sliding window per API key; the 11th gets `429` with a `Retry-After` header. It's keyed by API key because behind Azure's proxy every request arrives from the same internal IP.
- **Input limit:** messages of 1–1,000 characters (`422` otherwise), which caps what one request can spend.
- **Errors:** if the model call fails, the client gets a generic `502`; the details go only to the logs.
- **Logs:** one line per chat with `conversation_id`, tools used and latency. Message text is never logged because it contains names and addresses.

**Deployment** – the image built from the `Dockerfile` (Python 3.14 slim, runs as a non-root user, no secrets inside) is pushed to Azure Container Registry and runs on Azure Container Apps:

- Secrets (`OPENAI_API_KEY`, `SUPABASE_SERVICE_KEY`, `API_KEY`) are Container Apps secrets passed as environment variables, never baked into the image. `.dockerignore` keeps `.env` out of the build context.
- The image is pulled with a user-assigned managed identity, so there are no registry passwords.
- Scale from 0 to 1 replica: no cost while idle, and a single replica because memory and the rate limit live in process memory.
- Each release is a tagged image (`v1`, `v2`), so rolling back is one `az containerapp update --image`.
- Logs go to Log Analytics; see *Monitoring*.

**CI** – GitHub Actions runs the unit tests on every push and pull request with Python 3.14 (`.github/workflows/tests.yml`). Integration tests stay manual because they need API keys and cost money.

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
api.py                HTTP API: auth, rate limit, logging
ask.py                inspect what the retriever returns
eval/                 retrieval evaluation (26 labeled questions)
tests/                unit tests (offline) and integration tests (real API)
Dockerfile            container image for the API
.github/workflows/    CI: unit tests on every push
requirements.txt      runtime dependencies (what goes into the image)
requirements-dev.txt  runtime + pytest
```

## Setup

Tested on Python 3.14. Python 3.15 doesn't work yet: `zstandard`, a LangChain dependency, has no prebuilt wheel for it.

```bash
python -m venv .venv            # Windows: py -3.14 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env            # Windows: copy .env.example .env, then fill in your keys
```

Run `sql/schema.sql` and then `sql/phase2.sql` once in the Supabase SQL Editor, then:

```bash
python ingest.py                           # chunk, embed and store
python chat.py                             # talk to the agent
uvicorn api:app --reload                   # run the HTTP API on http://localhost:8000
python ask.py "Do you deliver to Riomar?"  # inspect what the retriever returns
python -m eval.run_eval                    # retrieval evaluation
pytest                                     # unit tests: offline, free
pytest -m integration                      # integration tests: real OpenAI calls, a few US cents
```

**Docker**

```bash
docker build -t menu-rag-agent .
docker run --rm -p 8000:8000 --env-file .env menu-rag-agent
```

**Azure Container Apps** (PowerShell; the variables `$RG`, `$LOCATION`, `$ACR`, `$ENV_NAME`, `$APP`, `$IDENTITY` and `$IMAGE` hold your names). Azure for Students subscriptions only allow some regions; check yours with `az policy assignment list`.

```powershell
az group create --name $RG --location $LOCATION
az acr create --resource-group $RG --name $ACR --sku Basic
az acr login --name $ACR; docker tag menu-rag-agent $IMAGE; docker push $IMAGE
az identity create --name $IDENTITY --resource-group $RG
$IDENTITY_ID = az identity show --name $IDENTITY --resource-group $RG --query id -o tsv
az containerapp env create --name $ENV_NAME --resource-group $RG --location $LOCATION
# Read .env into $cfg so key values never appear in the shell history
$cfg = @{}; Get-Content .env | ForEach-Object { if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$') { $cfg[$Matches[1]] = $Matches[2].Trim() } }
az containerapp create --name $APP --resource-group $RG --environment $ENV_NAME --image $IMAGE `
  --user-assigned $IDENTITY_ID --registry-identity $IDENTITY_ID --registry-server "$ACR.azurecr.io" `
  --ingress external --target-port 8000 --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 `
  --secrets "openai-api-key=$($cfg.OPENAI_API_KEY)" "supabase-service-key=$($cfg.SUPABASE_SERVICE_KEY)" "api-key=$($cfg.API_KEY)" `
  --env-vars "OPENAI_API_KEY=secretref:openai-api-key" "SUPABASE_SERVICE_KEY=secretref:supabase-service-key" `
             "API_KEY=secretref:api-key" "SUPABASE_URL=$($cfg.SUPABASE_URL)" "CHAT_MODEL=$($cfg.CHAT_MODEL)"
# New release: build, tag and push a new image, then
az containerapp update --name $APP --resource-group $RG --image "$ACR.azurecr.io/menu-rag-agent:v2"
```

## Monitoring

Container logs go to Log Analytics (portal: the container app → *Monitoring → Logs*; live output in *Monitoring → Log stream*). Example query, chat latency per hour:

```kusto
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "menu-rag-agent" and Log_s has "latency_ms="
| extend ms = toint(extract(@"latency_ms=(\d+)", 1, Log_s))
| summarize chats = count(), median_ms = percentile(ms, 50), max_ms = max(ms) by bin(TimeGenerated, 1h)
```

## Results

**Retrieval** – 18 answerable questions (9 Spanish, 9 English), `item` = 21 chunks, `fixed` = 13 chunks:

| Strategy | hit@1 | hit@3 | MRR@5 |
|----------|-------|-------|-------|
| item     | 0.94  | 1.00  | 0.97  |
| fixed    | 0.78  | 1.00  | 0.88  |

Similarity floor of 0.20 on `item`: keeps the relevant chunk for 18/18 answerable questions and filters out 3 of 8 out-of-scope questions (weather, pets, pizza). The other 5 are left to the model.

**In production (Azure, first day)** – 12 chat requests: median latency 1.6 s, slowest 5.4 s, measured inside the app (cold starts after scaling to zero add more). Unauthenticated requests got `401`, and the 11th request within a minute got `429`.

**Tests**

- 52 unit tests (no API), run in CI on every push: price parsing, totals and the free-delivery boundary, order validation, the similarity floor, the tools (with Supabase faked), language detection, and the HTTP API (auth, input limits, error handling, rate limit with a fake clock, all with the agent faked).
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

**Deployment**

- **Behind a cloud proxy the client IP is useless for rate limiting.** In the Azure logs every request
  came from `192.168.0.1`, the platform's proxy, so a per-IP limit would have been one shared bucket for
  all clients. The limit is keyed by API key instead.
- **`secrets.compare_digest` raises on non-ASCII strings.** A key containing "ñ" would have produced a
  500 instead of a 401; comparing bytes fixes it, and a unit test covers it.
- **The app must fail fast without its secrets.** It crashes at startup without `OPENAI_API_KEY`, so the
  secrets are passed when the container app is created rather than added afterwards.
- **The cloud has its own constraints.** The student subscription only allows five regions, and
  `az containerapp logs show` fails on this environment type (a CLI bug), so live logs come from the portal.

## Limitations

- `handoff_to_human` only stores a record; it doesn't notify anyone yet.
- Conversation memory and the rate-limit counters are in RAM: they reset when the app scales to zero or restarts, and they require a single replica.
- With one API key the rate limit is effectively global; per-client limits need one key per client.
- After scaling to zero, the first request waits for the container to start.
- The integration tests detect "the agent declined" by looking for words like "no" or "sorry", which is coarse.
- Orders don't collect a phone number, aren't checked against opening hours, and the age limit for beer is only a prompt rule.
- Language detection is a word-list heuristic for Spanish and English only.

## Roadmap

- [x] Phase 1 – chunking, embeddings, vector search, retrieval evaluation
- [x] Phase 2 – tool-calling agent (LangChain) with retrieval, order and human-handoff tools, guardrails, unit and integration tests
- [x] Phase 3 – FastAPI endpoint with API-key auth and rate limiting, Docker, deploy to Azure Container Apps, unit tests in GitHub Actions
- [ ] Persistent memory (`PostgresSaver`) and a shared rate-limit store, so the app can run more than one replica
- [ ] Handoff notifications and LLM-as-judge for the integration tests
