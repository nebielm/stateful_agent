import json
from datetime import date
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from app.repositories import memory_decision_log
from app.repositories import user_memory
from app.services import graph, tools
from app.utils.dates import calculate_age_from_birthdate


def test_birthdate_is_saved_when_missing(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))
    monkeypatch.setattr(
        graph,
        "extract_memory_updates",
        lambda text: {
            "structured": [{"key": "birthdate", "value": "1995-04-12", "category": "profile"}],
            "unstructured": [],
        },
    )

    state = {
        "user_id": "user-1",
        "messages": [
            HumanMessage(content="I was born on 1995-04-12."),
            AIMessage(content="Thanks, I will remember that."),
        ],
    }
    runtime = SimpleNamespace(context={"user_vectorstore": object()})

    graph.memory_updater_node(state, runtime)

    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}
    assert state["memory_updates"]["structured_results"] == [
        {
            "decision": "stored",
            "field": "birthdate",
            "category": "profile",
            "existing_value": None,
            "proposed_value": "1995-04-12",
            "reason": "value stored",
        }
    ]
    assert "pending_confirmation" not in state["memory_updates"]
    assert json.loads(log_file.read_text().strip())["field"] == "birthdate"


def test_birthdate_is_not_overwritten_when_already_present(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    result = user_memory.controlled_structured_data_storage(
        user_id="user-1",
        key="birthdate",
        value="1998-08-30",
        category="profile",
    )

    assert result == {
        "decision": "needs_confirmation",
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1998-08-30",
        "reason": "immutable field conflict",
    }
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}
    assert json.loads(log_file.read_text().strip())["decision"] == "needs_confirmation"


def test_same_immutable_value_is_no_op(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    result = user_memory.controlled_structured_data_storage(
        user_id="user-1",
        key="birthdate",
        value="1995-04-12",
        category="profile",
    )

    assert result == {
        "decision": "no_change",
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1995-04-12",
        "reason": "same immutable value",
    }
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}
    assert json.loads(log_file.read_text().strip())["decision"] == "no_change"


def test_unrelated_message_does_not_overwrite_birthdate(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_memory_updates",
        lambda text: {"structured": [], "unstructured": []},
    )

    state = {
        "user_id": "user-1",
        "messages": [
            HumanMessage(content="Can you suggest a recipe?"),
            AIMessage(content="You could try a tomato pasta."),
        ],
    }
    runtime = SimpleNamespace(context={"user_vectorstore": object()})

    graph.memory_updater_node(state, runtime)

    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}


def test_another_person_birthdate_does_not_overwrite_user_birthdate(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))
    monkeypatch.setattr(
        graph,
        "extract_memory_updates",
        lambda text: {
            "structured": [{"key": "birthdate", "value": "2001-02-03", "category": "profile"}],
            "unstructured": [],
        },
    )

    state = {
        "user_id": "user-1",
        "messages": [
            HumanMessage(content="My sister was born on 2001-02-03."),
            AIMessage(content="That is your sister's birthdate."),
        ],
    }
    runtime = SimpleNamespace(context={"user_vectorstore": object()})

    graph.memory_updater_node(state, runtime)

    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}
    assert state["memory_updates"]["structured_results"][0]["decision"] == "ignored"
    assert "pending_confirmation" not in state["memory_updates"]
    assert "replace it" not in state["messages"][-1].content
    assert json.loads(log_file.read_text().strip())["source"] == "memory_item_validation"


def test_conflict_generates_confirmation_message_with_existing_and_proposed_values(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_memory_updates",
        lambda text: {
            "structured": [{"key": "birthdate", "value": "1996-04-12", "category": "profile"}],
            "unstructured": [],
        },
    )

    state = {
        "user_id": "user-1",
        "messages": [
            HumanMessage(content="My birthdate is actually 1996-04-12."),
            AIMessage(content="Thanks for the correction."),
        ],
        "memory_updates": {},
    }
    runtime = SimpleNamespace(context={"user_vectorstore": object()})

    graph.memory_updater_node(state, runtime)

    assert state["messages"][-1].content == (
        "Thanks for the correction.\n\n"
        "I currently have 1995-04-12 saved as your birthdate. "
        "Do you want me to replace it with 1996-04-12?"
    )
    assert state["memory_updates"]["pending_confirmation"] == {
        "user_id": "user-1",
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1996-04-12",
        "reason": "immutable field conflict",
    }
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}


def test_non_conflicting_results_do_not_generate_confirmation_message(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_memory_updates",
        lambda text: {
            "structured": [{"key": "birthdate", "value": "1995-04-12", "category": "profile"}],
            "unstructured": [],
        },
    )

    state = {
        "user_id": "user-1",
        "messages": [
            HumanMessage(content="I was born on 1995-04-12."),
            AIMessage(content="Thanks, I will remember that."),
        ],
        "memory_updates": {},
    }
    runtime = SimpleNamespace(context={"user_vectorstore": object()})

    graph.memory_updater_node(state, runtime)

    assert state["messages"][-1].content == "Thanks, I will remember that."
    assert "pending_confirmation" not in state["memory_updates"]


def test_confirmed_pending_correction_is_applied_on_next_turn(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="yes")],
        "memory_updates": {
            "pending_confirmation": {
                "user_id": "user-1",
                "field": "birthdate",
                "category": "profile",
                "existing_value": "1995-04-12",
                "proposed_value": "1996-04-12",
                "reason": "immutable field conflict",
            }
        },
        "context": {},
    }

    result = graph.confirmation_resolution_node(state)

    assert result["memory_updates"]["confirmation_resolution"]["status"] == "confirmed"
    assert "pending_confirmation" not in result["memory_updates"]
    assert result["messages"][-1].content == "Got it - I updated your birthdate to 1996-04-12."
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1996-04-12"}}}


def test_rejected_pending_correction_is_discarded_on_next_turn(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="no")],
        "memory_updates": {
            "pending_confirmation": {
                "user_id": "user-1",
                "field": "birthdate",
                "category": "profile",
                "existing_value": "1995-04-12",
                "proposed_value": "1996-04-12",
                "reason": "immutable field conflict",
            }
        },
        "context": {},
    }

    result = graph.confirmation_resolution_node(state)

    assert result["memory_updates"]["confirmation_resolution"]["status"] == "rejected"
    assert "pending_confirmation" not in result["memory_updates"]
    assert result["messages"][-1].content == "Okay, I kept your birthdate as 1995-04-12."
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}


def test_unclear_pending_correction_is_kept_on_next_turn(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="maybe")],
        "memory_updates": {
            "pending_confirmation": {
                "user_id": "user-1",
                "field": "birthdate",
                "category": "profile",
                "existing_value": "1995-04-12",
                "proposed_value": "1996-04-12",
                "reason": "immutable field conflict",
            }
        },
        "context": {},
    }

    result = graph.confirmation_resolution_node(state)

    assert result["memory_updates"]["confirmation_resolution"]["status"] == "unclear"
    assert result["memory_updates"]["pending_confirmation"] == {
        "user_id": "user-1",
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1996-04-12",
        "reason": "immutable field conflict",
    }
    assert result["messages"][-1].content == "Please answer yes or no so I know whether to update your birthdate."


def test_birthdate_is_retrieved_for_age_question(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {
            "structured_to_retrieve": [{"category": "profile", "key": "birthdate"}],
            "unstructured_to_retrieve": [],
        },
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="How old am I?")],
        "context": {},
    }
    runtime = SimpleNamespace(
        context={
            "knowledge_vectorstore": object(),
            "user_vectorstore": object(),
        }
    )

    result = graph.context_retrieval_node(state, runtime)

    assert result["context"]["structured"] == [
        {
            "key": "birthdate",
            "value": "1995-04-12",
            "category": "profile",
            "score": 1.0,
        }
    ]


def test_age_is_derived_from_birthdate_with_fixed_current_date():
    assert calculate_age_from_birthdate("1995-04-12", current_date=date(2026, 5, 4)) == 31


def test_conversation_remembers_birthdate_after_unrelated_turns(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_memory_updates",
        lambda text: {"structured": [], "unstructured": []},
    )
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {
            "structured_to_retrieve": [{"category": "profile", "key": "birthdate"}],
            "unstructured_to_retrieve": [],
        },
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)
    monkeypatch.setattr(tools, "load_user_data", lambda: json.loads(data_file.read_text()))
    monkeypatch.setattr(
        tools,
        "calculate_age_from_birthdate",
        lambda birthdate_value: calculate_age_from_birthdate(birthdate_value, current_date=date(2026, 5, 4)),
    )

    unrelated_state = {
        "user_id": "user-1",
        "messages": [
            HumanMessage(content="Can you suggest a recipe?"),
            AIMessage(content="You could try lentil soup."),
        ],
    }
    memory_runtime = SimpleNamespace(context={"user_vectorstore": object()})
    graph.memory_updater_node(unrelated_state, memory_runtime)

    age_state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="How old am I?")],
        "context": {},
    }
    retrieval_runtime = SimpleNamespace(
        context={
            "knowledge_vectorstore": object(),
            "user_vectorstore": object(),
        }
    )

    retrieved_state = graph.context_retrieval_node(age_state, retrieval_runtime)
    derived_age = tools.get_current_age.func(SimpleNamespace(state={"user_id": "user-1"}))

    assert retrieved_state["context"]["structured"][0]["value"] == "1995-04-12"
    assert derived_age == 31
