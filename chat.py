"""Interactive CLI to talk to the agent. Shows each tool call so you can see what the agent is doing.

Usage: python chat.py   (type "exit" or press Ctrl+C to quit)
"""
import uuid

from agent import build_agent

agent = build_agent()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # one conversation per run

print("Restaurant assistant. Type 'exit' to quit.\n")
seen = 0
while True:
    try:
        text = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if text.lower() in ("exit", "salir"):
        break
    if not text:
        continue

    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    new_messages, seen = result["messages"][seen:], len(result["messages"])
    for m in new_messages:
        for call in getattr(m, "tool_calls", []):
            print(f"  [tool] {call['name']}({call['args']})")
    print(f"Agent: {result['messages'][-1].content}\n")
