"""HTTP API for the agent.

POST /v1/chat  (requires the X-API-Key header)
GET  /health   (no auth: used by the cloud platform to check the app is alive)

Run locally: uvicorn api:app --reload
"""
import logging
import os
import secrets
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from agent import build_agent

# Logs go to stdout, which Docker and Azure collect. Message text is never logged: it has names and addresses.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
for noisy in ("httpx", "httpx2"):  # one line per OpenAI/Supabase call is noise (and log storage is billed)
    logging.getLogger(noisy).setLevel(logging.WARNING)

app = FastAPI(title="Menu RAG Agent")
agent = build_agent()  # one agent per process; conversations live in its in-memory checkpointer

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    expected = os.environ.get("API_KEY")
    # Fail closed: if API_KEY is not configured, nobody gets in.
    # Compare bytes: compare_digest raises on non-ASCII str, which would turn a bad key into a 500.
    if not expected or not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)  # caps the tokens a single request can spend
    conversation_id: str | None = Field(default=None, max_length=100)


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    tools_used: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


# Plain "def", not "async def": agent.invoke blocks, so FastAPI runs it in a worker thread.
@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(body: ChatRequest):
    conversation_id = body.conversation_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": conversation_id}}
    previous = len(agent.get_state(config).values.get("messages", []))

    start = time.perf_counter()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": body.message}]}, config)
    except Exception:
        log.exception("agent failed conversation_id=%s", conversation_id)
        raise HTTPException(status_code=502, detail="The assistant is unavailable right now. Please try again.")

    new_messages = result["messages"][previous:]
    tools_used = [call["name"] for m in new_messages for call in getattr(m, "tool_calls", [])]
    log.info("chat conversation_id=%s tools=%s latency_ms=%d",
             conversation_id, tools_used, (time.perf_counter() - start) * 1000)
    return ChatResponse(reply=result["messages"][-1].content, conversation_id=conversation_id, tools_used=tools_used)
