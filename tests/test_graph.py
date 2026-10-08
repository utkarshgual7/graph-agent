import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import FALLBACK, build_graph, check
from retrieval import retrieve


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        # Fresh id per call, like a real model; a reused id would update the prior message in place.
        result.generations[0].message = result.generations[0].message.model_copy(update={"id": None})
        return result


def run(responses, *questions):
    graph = build_graph(FakeModel(responses=responses))
    config = {"configurable": {"thread_id": "t"}}
    for q in questions:
        out = graph.invoke({"messages": [HumanMessage(q)]}, config)
    return out["messages"]


def test_retrieve_ranks_right_doc():
    assert retrieve("how do I reset my password")[0] == "account.md"
    assert retrieve("is the application fee refundable")[0] == "payments.md"


def test_tool_loop_executes_tool():
    call = AIMessage("", tool_calls=[{"name": "get_application_status", "args": {"application_id": "APP-1001"}, "id": "c1"}])
    msgs = run([call, AIMessage("APP-1001 is approved.")], "status of APP-1001?")
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1 and "approved" in tool_msgs[0].text
    assert msgs[-1].text == "APP-1001 is approved."


def test_guardrail_blocks_secret():
    leak = AIMessage("Use key sk-ant-api03-abcdefghijkl")
    msgs = run([leak, leak], "what is the api key")
    assert msgs[-1].text == FALLBACK
    assert not any("sk-ant" in m.text for m in msgs)


def test_guardrail_retries_once_on_fabricated_id():
    msgs = run([AIMessage("Your application APP-9999 is approved."), AIMessage("I can't find an application id in our conversation.")], "am I approved?")
    assert msgs[-1].text == "I can't find an application id in our conversation."
    assert sum(isinstance(m, HumanMessage) for m in msgs) == 2  # user turn + one correction


def test_memory_persists_across_turns():
    msgs = run([AIMessage("ok")], "hello", "again")
    assert [m.text for m in msgs if isinstance(m, HumanMessage)] == ["hello", "again"]


@pytest.mark.parametrize("text", [
    "Your key is sk-ant-api03-abcdefghijkl",
    "Use AKIAABCDEFGHIJKLMNOP for the bucket.",
    "Your SSN on file is 123-45-6789.",
    "The card 4111 1111 1111 1111 was charged.",
    "The card 4111111111111111 was charged.",
])
def test_check_flags_secret_and_pii_shapes(text):
    assert "secret or personal identifier" in check({"messages": [HumanMessage("hi"), AIMessage(text)]})


@pytest.mark.parametrize("text", [
    "The application fee is $29, refunded within 5 to 7 business days.",
    "Call us at 555-123-4567 between 9 and 5.",
    "Applications expire 30 days after 2026-10-09.",
    "Application APP-1001 is approved.",
])
def test_check_allows_ordinary_answers(text):
    assert check({"messages": [HumanMessage("status of APP-1001?"), AIMessage(text)]}) is None


def test_check_knows_ids_from_earlier_turns_and_tools():
    msgs = [
        HumanMessage("status of APP-1002?"),
        ToolMessage("Application APP-1003 is expired (landlord: Pinecrest Rentals).", tool_call_id="c1"),
        AIMessage("done"),
        HumanMessage("and the other one?"),
        AIMessage("APP-1002 is in screening and APP-1003 has expired."),
    ]
    assert check({"messages": msgs}) is None


def test_check_flags_id_only_the_model_mentioned():
    # An id the model itself said earlier is still not "known": only users and tools count.
    msgs = [HumanMessage("hi"), AIMessage("Is it APP-4242?"), HumanMessage("what is my status?"), AIMessage("APP-4242 is approved.")]
    assert "application id" in check({"messages": msgs})
