"""Terminal client for the HTTP API: like chat.py, but every message goes through POST /v1/chat
(API key, rate limit, the deployed app). Shows the tools the agent used and the response time.

Usage: python client.py                          (API_URL and API_KEY from .env; default http://localhost:8000)
       python client.py --url https://<your-app>
Type "exit" or press Ctrl+C to quit.
"""
import argparse
import os

import httpx
from dotenv import load_dotenv

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument("--url", help="API base URL (overrides API_URL)")
args = parser.parse_args()

url = (args.url or os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")
conversation_id = None  # set by the server on the first reply, then sent back to keep the conversation

print(f"Connected to {url}. Type 'exit' to quit.\n")
with httpx.Client(base_url=url, headers={"X-API-Key": os.environ["API_KEY"]}, timeout=60) as client:
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("exit", "salir"):
            break
        if not text:
            continue

        response = client.post("/v1/chat", json={"message": text, "conversation_id": conversation_id})
        if response.status_code != 200:
            wait = response.headers.get("Retry-After")
            print(f"  [HTTP {response.status_code}] {response.json().get('detail')}"
                  + (f" (retry in {wait} s)" if wait else "") + "\n")
            continue

        body = response.json()
        conversation_id = body["conversation_id"]
        tools = ", ".join(body["tools_used"]) or "none"
        print(f"  [tools: {tools} | {response.elapsed.total_seconds():.1f} s]")
        print(f"Agent: {body['reply']}\n")
