"""REPL: python main.py  (needs ANTHROPIC_API_KEY; set ANTHROPIC_MODEL to use a different model)."""
import os
import sys
import uuid

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from graph import build_graph

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set. Export it first; the tests and evals run without it.")

graph = build_graph(ChatAnthropic(model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"), max_tokens=1024))
config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # one thread per REPL session

while True:
    try:
        q = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    try:
        out = graph.invoke({"messages": [HumanMessage(q)]}, config)
    except anthropic.APIError as e:
        # Rate limits, overloads, network drops: report and keep the session (and its thread) alive.
        print(f"error> {type(e).__name__}: {e}")
        continue
    print("bot>", out["messages"][-1].text)
