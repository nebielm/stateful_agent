import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from app.repositories import user_memory
from app.services import graph


def test_age_question_retrieves_birthdate_without_unrelated_food_memory(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12", "city": "Berlin"},
                    "preferences": {"favorite_food": "pasta"},
                    "dynamic": {"current_goal": "lose weight"},
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
    assert result["context"]["unstructured"] == []
    assert result["context"]["knowledge"] == []


def test_food_question_retrieves_preferences_goal_and_dislikes_without_birthdate(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12"},
                    "preferences": {"favorite_food": "pasta"},
                    "dynamic": {"current_goal": "lose weight"},
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
                {"category": "preferences", "key": "favorite_food"},
                {"category": "dynamic", "key": "current_goal"},
            ],
            "unstructured_to_retrieve": [{"type": "dislike"}],
        },
    )
    monkeypatch.setattr(
        graph,
        "retrieve_unstructured_memory",
        lambda **kwargs: [
            {"text": "user dislikes pork", "type": "dislike", "score": 0.95},
        ],
    )
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="What should I cook today?")],
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
            "key": "favorite_food",
            "value": "pasta",
            "category": "preferences",
            "score": 1.0,
        },
        {
            "key": "current_goal",
            "value": "lose weight",
            "category": "dynamic",
            "score": 1.0,
        },
    ]
    assert result["context"]["unstructured"] == [
        {"text": "user dislikes pork", "type": "dislike", "score": 0.95}
    ]
    assert all(item["key"] != "birthdate" for item in result["context"]["structured"])


def test_fitness_question_retrieves_persisted_weight_and_current_goal(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "dynamic": {"weight": 92, "current_goal": "lose weight"},
                    "profile": {"city": "Berlin"},
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
                {"category": "dynamic", "key": "current_goal"},
            ],
            "unstructured_to_retrieve": [],
        },
    )
    monkeypatch.setattr(graph, "retrieve_unstructured_memory", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_knowledge_docs", lambda **kwargs: [])
    monkeypatch.setattr(graph, "retrieve_relevant_context_for_user", lambda all_context, message, k=5: all_context)

    state = {
        "user_id": "user-1",
        "messages": [HumanMessage(content="How am I doing with my fitness goal?")],
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
            "key": "weight",
            "value": 92,
            "category": "dynamic",
            "score": 1.0,
        },
        {
            "key": "current_goal",
            "value": "lose weight",
            "category": "dynamic",
            "score": 1.0,
        },
    ]
    assert all(item["key"] != "city" for item in result["context"]["structured"])


def test_specific_retrieval_plan_prevents_dumping_all_memory_into_context(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    data_file.write_text(
        json.dumps(
            {
                "user-1": {
                    "profile": {"birthdate": "1995-04-12", "city": "Berlin", "job": "Engineer"},
                    "preferences": {"favorite_food": "pasta", "diet": "vegetarian"},
                    "dynamic": {"weight": 92, "current_goal": "lose weight"},
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
        "messages": [HumanMessage(content="Remind me of my age.")],
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
