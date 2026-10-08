import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from retrieval import DOCS, retrieve
from tools import get_application_status

SYSTEM = """You are the Keystone support assistant. Answer using the documentation below.
If the user asks about a specific application, call get_application_status with the exact id.
Never invent application ids or statuses. If the docs do not cover the question, say so.

<docs>
{context}
</docs>"""

# API keys, AWS keys, SSN-shaped, and 13-16 digit card-shaped numbers.
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|\b\d{3}-\d{2}-\d{4}\b|\b(?:\d[ -]?){13,16}\b")
APP_ID = re.compile(r"APP-\d+")
FALLBACK = "I can't provide that answer safely. Please contact support@keystone.example with your application id."


class State(MessagesState):
    context: str
    retries: int


def retrieve_node(state: State) -> dict:
    query = state["messages"][-1].content
    names = retrieve(query)
    context = "\n\n".join(f"## {n}\n{DOCS[n]}" for n in names)
    return {"context": context, "retries": 0}  # retries resets every user turn


def check(state: State) -> str | None:
    """Return a correction message if the last answer violates a rule, else None."""
    text = state["messages"][-1].text
    if SECRET.search(text):
        return "Your answer contained something that looks like a secret or personal identifier. Rewrite it without that."
    # An id is "known" only if the user typed it or a tool returned it this thread.
    known = set()
    for m in state["messages"]:
        if isinstance(m, (HumanMessage, ToolMessage)):
            known.update(APP_ID.findall(m.text))
    if set(APP_ID.findall(text)) - known:
        # Correction deliberately does not repeat the id, or it would become "known" on the retry.
        return "Your answer cited an application id that does not exist in this conversation. Rewrite it without inventing ids."
    return None


def guardrail_node(state: State) -> dict:
    problem = check(state)
    if problem is None:
        return {}
    if state["retries"] == 0:
        return {"messages": [HumanMessage(problem)], "retries": 1}
    # Second failure: drop the first bad answer, the correction, and everything the
    # retry produced (tool calls included) so the violating text never stays in the
    # thread, then emit a fixed fallback. The correction is the last HumanMessage.
    msgs = state["messages"]
    correction = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    drop = [RemoveMessage(id=m.id) for m in msgs[correction - 1 :]]
    return {"messages": [*drop, AIMessage(FALLBACK)]}


def after_guardrail(state: State) -> str:
    # A trailing HumanMessage means guardrail just appended a correction.
    return "answer" if isinstance(state["messages"][-1], HumanMessage) else END


def build_graph(model):
    llm = model.bind_tools([get_application_status])

    def answer_node(state: State) -> dict:
        system = SystemMessage(SYSTEM.format(context=state["context"]))
        return {"messages": [llm.invoke([system, *state["messages"]])]}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)
    g.add_node("tools", ToolNode([get_application_status]))
    g.add_node("guardrail", guardrail_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges("answer", tools_condition, {"tools": "tools", END: "guardrail"})
    g.add_edge("tools", "answer")
    g.add_conditional_edges("guardrail", after_guardrail, {"answer": "answer", END: END})
    return g.compile(checkpointer=MemorySaver())
