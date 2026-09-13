"""REPL: python main.py  (needs ANTHROPIC_API_KEY)."""
import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from graph import build_graph

graph = build_graph(ChatAnthropic(model="claude-opus-5", max_tokens=1024))
config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # one thread per REPL session

while True:
    try:
        q = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    out = graph.invoke({"messages": [HumanMessage(q)]}, config)
    print("bot>", out["messages"][-1].text)
