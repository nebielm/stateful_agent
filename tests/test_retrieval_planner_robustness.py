import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.repositories import user_memory
from app.services import graph


def make_runtime():
    return SimpleNamespace(
        context={
            "knowledge_vectorstore": object(),
            "user_vectorstore": object(),
        }
    )


def test_valid_planner_output_retrieves_requested_memory(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12"},
                    "preferences": {"favorite_food": "pasta"},
                }
            }
        )
    )
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

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == [
        {
            "key": "birthdate",
            "value": "1995-04-12",
            "category": "profile",
            "score": 1.0,
        }
    ]


def test_missing_structured_to_retrieve_does_not_dump_all_memory(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12", "city": "Berlin"},
                    "preferences": {"favorite_food": "pasta"},
                }
            }
        )
    )
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {"unstructured_to_retrieve": []},
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="How old am I?")],
        "context": {},
    }

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == []
    assert result["context"]["unstructured"] == []


def test_missing_unstructured_to_retrieve_does_not_crash_and_keeps_structured_results(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {"structured_to_retrieve": [{"category": "profile", "key": "birthdate"}]},
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="How old am I?")],
        "context": {},
    }

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == [
        {
            "key": "birthdate",
            "value": "1995-04-12",
            "category": "profile",
            "score": 1.0,
        }
    ]
    assert result["context"]["unstructured"] == []


def test_unknown_structured_fields_are_ignored_safely(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {
            "structured_to_retrieve": [{"category": "profile", "key": "unknown_field"}],
            "unstructured_to_retrieve": [],
        },
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="Tell me about my stored info")],
        "context": {},
    }

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == []


def test_empty_retrieval_plan_produces_empty_user_memory_context(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12", "city": "Berlin"},
                    "preferences": {"favorite_food": "pasta"},
                    "dynamic": {"weight": 92},
                }
            }
        )
    )
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {
            "structured_to_retrieve": [],
            "unstructured_to_retrieve": [],
        },
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="Hello")],
        "context": {},
    }

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == []
    assert result["context"]["unstructured"] == []


def test_over_broad_retrieval_plan_ignores_invalid_fields_and_keeps_valid_ones(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "dynamic": {"weight": 92, "current_goal": "lose weight"},
                    "profile": {"birthdate": "1995-04-12"},
                }
            }
        )
    )
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {
            "structured_to_retrieve": [
                {"category": "dynamic", "key": "weight"},
                {"category": "dynamic", "key": "unknown_key"},
                {"category": "unknown_category", "key": "birthdate"},
            ],
            "unstructured_to_retrieve": [],
        },
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="How is my weight trending?")],
        "context": {},
    }

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == [
        {
            "key": "weight",
            "value": 92,
            "category": "dynamic",
            "score": 1.0,
        }
    ]


def test_malformed_planner_response_falls_back_safely_without_crashing(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12"},
                    "preferences": {"favorite_food": "pasta"},
                }
            }
        )
    )
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(
        graph,
        "extract_retrieval_plan",
        lambda text: {
            "structured_to_retrieve": [None, "bad", {"category": "profile", "key": "birthdate"}],
            "unstructured_to_retrieve": "not-a-list",
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

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == [
        {
            "key": "birthdate",
            "value": "1995-04-12",
            "category": "profile",
            "score": 1.0,
        }
    ]
    assert result["context"]["unstructured"] == []


def test_non_dict_planner_response_does_not_dump_all_memory(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12", "city": "Berlin"},
                    "preferences": {"favorite_food": "pasta"},
                }
            }
        )
    )
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(graph, "extract_retrieval_plan", lambda text: None)
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="hello")],
        "context": {},
    }

    result = graph.context_retrieval_node(state, make_runtime())

    assert result["context"]["structured"] == []
    assert result["context"]["unstructured"] == []
