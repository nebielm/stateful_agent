import json
from collections import deque
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.llm import client
from app.repositories import user_memory
from app.services import graph, retrieval, tools
from app.utils.dates import calculate_age_from_birthdate


class ScriptedProvider:
    """Replace only model responses; keep extractors, JSON parsing and graph real."""
    def __init__(self):
        self.replies = deque()
        self.calls = []
        self.tool_schemas = []

    def bind_tools(self, selected):
        self.tool_schemas.extend(convert_to_openai_tool(tool) for tool in selected)
        return self

    def invoke(self, prompt):
        if isinstance(prompt, list):
            kind = "agent"
        else:
            markers = {
                "strict memory extraction system": "memory",
                "state extraction engine": "ephemeral",
                "memory reasoning system": "planner",
                "ranks memory and knowledge context": "ranker",
                "tool routing system": "selector",
                "Fix this JSON:": "repair",
            }
            kind = next((value for marker, value in markers.items() if marker in prompt), "unknown")
        self.calls.append((kind, prompt))
        expected_kind, response = self.replies.popleft()
        assert kind == expected_kind, f"Expected {expected_kind}, received {kind}"
        return response(prompt) if callable(response) else response

    def add(self, kind, response):
        if isinstance(response, (dict, list)):
            response = AIMessage(content=json.dumps(response))
        self.replies.append((kind, response))


class EmptyVectorstore:
    def similarity_search_with_score(self, **kwargs):
        return []


@pytest.fixture
def provider(monkeypatch):
    provider = ScriptedProvider()
    monkeypatch.setattr(client, "_llm", provider)
    monkeypatch.setattr(tools, "bound_models_cache", {})
    return provider


def context_for(birthdate):
    return {
        "structured": [{"category": "profile", "key": "birthdate", "value": birthdate, "score": 1.0}],
        "unstructured": [], "knowledge": [],
    }


def queue_turn(provider, *, stored_date=None, proposed_date=None, tool_name=None, tool_args=None):
    plan = {"structured_to_retrieve": [], "unstructured_to_retrieve": []}
    if stored_date:
        plan["structured_to_retrieve"] = [{"table": "profile", "key": "birthdate"}]
    provider.add("planner", plan)
    provider.add("ranker", context_for(stored_date) if stored_date else {"structured": [], "unstructured": [], "knowledge": []})
    provider.add("selector", [tool_name] if tool_name else [])
    if tool_name:
        provider.add("agent", AIMessage(content="", tool_calls=[{"name": tool_name, "id": "age-call", "args": tool_args or {}, "type": "tool_call"}]))
        provider.add("ephemeral", {})
        provider.add("selector", [tool_name])
        def answer_from_actual_tool(messages):
            assert isinstance(messages[-1], ToolMessage)
            assert messages[-1].status == "success"
            return AIMessage(content=f"Tool result: {messages[-1].content}")
        provider.add("agent", answer_from_actual_tool)
    else:
        provider.add("agent", AIMessage(content="I received your statement."))
    provider.add("ephemeral", {})
    extracted = {"structured": [], "unstructured": []}
    if proposed_date:
        extracted["user_id"] = "victim"  # Top-level model output cannot select an owner either.
        extracted["structured"] = [{"key": "birthdate", "category": "profile", "value": proposed_date}]
    provider.add("memory", extracted)


def turn(state, text):
    state["messages"].append(HumanMessage(content=text))
    return graph.app.invoke(state, context={"knowledge_vectorstore": EmptyVectorstore(), "user_vectorstore": EmptyVectorstore()})


@pytest.mark.parametrize("tool_name, expected", [("get_current_age", "31"), ("get_user_info", "birthdate: 1995-04-12")])
def test_compiled_graph_injects_owner_despite_forged_model_arguments(provider, monkeypatch, isolated_memory, tool_name, expected):
    isolated_memory[0].write_text(json.dumps({
        "owner": {"profile": {"birthdate": "1995-04-12"}, "age": 99},
        "victim": {"profile": {"birthdate": "1980-01-01"}},
    }))
    original = isolated_memory[0].read_bytes()
    derived = []
    def fixed_age(value):
        derived.append(value)
        return calculate_age_from_birthdate(value, date(2026, 5, 4))
    monkeypatch.setattr(tools, "calculate_age_from_birthdate", fixed_age)
    queue_turn(provider, tool_name=tool_name, tool_args={
        "user_id": "victim", "key": "birthdate", "runtime": {"state": {"user_id": "victim"}},
    })
    result = turn({"user_id": "owner", "messages": [], "memory_updates": {}, "context": {}}, "How old am I?")
    assert result["messages"][-1].content == f"Tool result: {expected}"
    assert isolated_memory[0].read_bytes() == original
    assert derived == (["1995-04-12"] if tool_name == "get_current_age" else [])
    assert not provider.replies
    for schema in provider.tool_schemas:
        assert "user_id" not in schema["function"]["parameters"]["properties"]
        assert "runtime" not in schema["function"]["parameters"]["properties"]


@pytest.mark.parametrize("reply, final_date, decision, expected_age", [
    ("yes", "1996-04-12", "confirmed_update_applied", 30),
    ("no", "1995-04-12", "confirmation_rejected", 31),
])
def test_fake_provider_demo_stores_corrects_resolves_and_retrieves(provider, monkeypatch, isolated_memory, reply, final_date, decision, expected_age):
    monkeypatch.setattr(tools, "calculate_age_from_birthdate", lambda value: calculate_age_from_birthdate(value, date(2026, 5, 4)))
    victim = {"profile": {"birthdate": "1980-01-01"}}
    isolated_memory[0].write_text(json.dumps({"victim": victim}))
    state = {"user_id": "owner", "messages": [], "memory_updates": {}, "context": {}}
    queue_turn(provider, proposed_date="1995-04-12")
    state = turn(state, "I was born on 1995-04-12.")
    assert user_memory.load_user_data()["owner"]["profile"]["birthdate"] == "1995-04-12"

    queue_turn(provider)
    state = turn(state, "Can you suggest a recipe?")
    queue_turn(provider, stored_date="1995-04-12", proposed_date="1996-04-12")
    state = turn(state, "Actually, my birthdate is 1996-04-12.")
    assert state["memory_updates"]["pending_confirmation"]["user_id"] == "owner"
    assert "replace it with 1996-04-12?" in state["messages"][-1].content
    assert len(state["messages"]) == 6  # Confirmation replaces, rather than duplicates, the AI message.
    assert user_memory.load_user_data()["owner"]["profile"]["birthdate"] == "1995-04-12"

    calls_before_confirmation = len(provider.calls)
    state = turn(state, "maybe")
    assert "Please answer yes or no" in state["messages"][-1].content
    assert "pending_confirmation" in state["memory_updates"]
    state = turn(state, reply)
    assert len(provider.calls) == calls_before_confirmation
    assert "pending_confirmation" not in state["memory_updates"]
    assert user_memory.load_user_data()["owner"]["profile"]["birthdate"] == final_date

    # Drop conversation history to prove the next answer depends on persisted memory.
    state = {"user_id": "owner", "messages": [], "memory_updates": {}, "context": {}}
    queue_turn(provider, stored_date=final_date, tool_name="get_current_age")
    state = turn(state, "How old am I?")
    assert state["context"] == context_for(final_date)
    assert state["user_id"] == "owner"
    assert user_memory.load_user_data()["victim"] == victim
    assert state["messages"][-1].content == f"Tool result: {expected_age}"
    assert not provider.replies
    assert [json.loads(line)["decision"] for line in isolated_memory[1].read_text().splitlines()] == ["stored", "needs_confirmation", decision]
    correction_prompts = [prompt for kind, prompt in provider.calls if kind == "memory" and prompt.rstrip().endswith("Actually, my birthdate is 1996-04-12.")]
    assert len(correction_prompts) == 1
    assert "emit that proposed birthdate for the confirmation workflow" in correction_prompts[0]
    assert "do NOT emit an automatic replacement" not in correction_prompts[0]


def test_invalid_json_and_failed_repair_leave_ranked_context_empty(provider):
    provider.add("ranker", AIMessage(content="{broken JSON"))
    provider.add("repair", AIMessage(content="still not JSON"))
    assert retrieval.retrieve_relevant_context_for_user(context_for("1995-04-12"), "hello") == {
        "structured": [], "unstructured": [], "knowledge": [],
    }
    assert not provider.replies


def test_compiled_graph_does_not_reuse_context_after_bad_ranker_output(provider, isolated_memory):
    user_memory.controlled_structured_data_storage("owner", "birthdate", "1995-04-12", "profile")
    queue_turn(provider, stored_date="1995-04-12")
    provider.replies[1] = ("ranker", AIMessage(content='{"structured":"all of it"}'))
    state = {"user_id": "owner", "messages": [], "memory_updates": {}, "context": context_for("1980-01-01")}
    result = turn(state, "Hello")
    assert result["context"] == {"structured": [], "unstructured": [], "knowledge": []}
    assert not provider.replies
