"""Unit tests for the HTTP API. The agent is faked, so no OpenAI or Supabase calls are made."""
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

KEY = "test-api-key"


class FakeAgent:
    """Mimics the two methods api.py uses. Every turn calls search_menu once and replies 'reply N'."""

    def __init__(self, fail=False):
        self.threads, self.fail = {}, fail

    def get_state(self, config):
        return SimpleNamespace(values={"messages": self.threads.get(config["configurable"]["thread_id"], [])})

    def invoke(self, payload, config):
        if self.fail:
            raise RuntimeError("OpenAI is down")
        messages = self.threads.setdefault(config["configurable"]["thread_id"], [])
        messages += [
            HumanMessage(payload["messages"][0]["content"]),
            AIMessage("", tool_calls=[{"name": "search_menu", "args": {"query": "x"}, "id": "1"}]),
            AIMessage(f"reply {len(messages) // 3 + 1}"),
        ]
        return {"messages": list(messages)}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def api(monkeypatch):
    # api.py builds the real agent on import, which needs an OpenAI key. CI has none, so use a dummy one,
    # but never replace a real key (the integration tests may run in the same process).
    if "OPENAI_API_KEY" not in os.environ:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    import api
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.setattr(api, "agent", FakeAgent())
    clock = FakeClock()
    monkeypatch.setattr(api, "limiter", api.RateLimiter(limit=3, window=60, clock=clock))
    return SimpleNamespace(module=api, client=TestClient(api.app), clock=clock)


def post(api, json, key=KEY):
    headers = {"X-API-Key": key} if key is not None else {}
    return api.client.post("/v1/chat", json=json, headers=headers)


# ---------- Health and auth ----------

def test_health_needs_no_key(api):
    assert api.client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("key", [None, "wrong-key", "ñandú".encode()])  # bytes: a client sending non-ASCII
def test_bad_or_missing_key_is_rejected(api, key):
    assert post(api, {"message": "hola"}, key=key).status_code == 401


def test_fails_closed_when_api_key_is_not_configured(api, monkeypatch):
    monkeypatch.delenv("API_KEY")
    assert post(api, {"message": "hola"}).status_code == 401


# ---------- Chat ----------

def test_chat_returns_reply_new_conversation_and_tools(api):
    body = post(api, {"message": "¿Tienen sushi?"}).json()
    assert body["reply"] == "reply 1"
    assert len(body["conversation_id"]) == 36  # a fresh UUID
    assert body["tools_used"] == ["search_menu"]


def test_conversation_id_continues_and_tools_are_only_from_this_turn(api):
    first = post(api, {"message": "hola"}).json()
    second = post(api, {"message": "¿y el domicilio?", "conversation_id": first["conversation_id"]}).json()
    assert second["reply"] == "reply 2"
    assert second["tools_used"] == ["search_menu"]  # not the two calls in the history


@pytest.mark.parametrize("message", ["", "a" * 1001])
def test_message_length_is_validated(api, message):
    assert post(api, {"message": message}).status_code == 422


def test_agent_failure_returns_502_without_details(api, monkeypatch):
    monkeypatch.setattr(api.module, "agent", FakeAgent(fail=True))
    response = post(api, {"message": "hola"})
    assert response.status_code == 502
    assert "OpenAI" not in response.text


# ---------- Rate limit (3 per 60 s in these tests) ----------

def test_rate_limit_blocks_then_recovers(api):
    assert [post(api, {"message": "hola"}).status_code for _ in range(3)] == [200, 200, 200]

    blocked = post(api, {"message": "hola"})
    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"] == "60"

    api.clock.now = 30
    assert post(api, {"message": "hola"}).headers["Retry-After"] == "30"

    api.clock.now = 61  # the first three hits are now outside the window
    assert post(api, {"message": "hola"}).status_code == 200


def test_rejected_keys_do_not_use_up_the_limit(api):
    for _ in range(5):
        post(api, {"message": "hola"}, key="wrong-key")
    assert post(api, {"message": "hola"}).status_code == 200
