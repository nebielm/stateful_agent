from app.services import retrieval


def sample_context():
    return {
        "structured": [
            {"key": "birthdate", "value": "1995-04-12", "category": "profile", "score": 1.0},
            {"key": "favorite_food", "value": "pasta", "category": "preferences", "score": 1.0},
            {"key": "current_goal", "value": "lose weight", "category": "dynamic", "score": 1.0},
        ],
        "unstructured": [
            {"text": "user dislikes pork", "type": "dislike", "score": 0.95},
            {"text": "user likes simple meals", "type": "preference", "score": 0.75},
        ],
        "knowledge": [
            {"text": "protein helps satiety", "category": "nutrition", "tags": ["protein"], "score": 0.88},
            {"text": "walking supports cardio health", "category": "fitness", "tags": ["walking"], "score": 0.8},
        ],
    }


def test_valid_ranker_output_works_normally(monkeypatch):
    all_context = sample_context()
    monkeypatch.setattr(
        retrieval,
        "call_llm_json",
        lambda prompt, default: {
            "structured": [all_context["structured"][0]],
            "unstructured": [all_context["unstructured"][0]],
            "knowledge": [all_context["knowledge"][0]],
        },
    )

    result = retrieval.retrieve_relevant_context_for_user(all_context, "How old am I?", k=2)

    assert result == {
        "structured": [all_context["structured"][0]],
        "unstructured": [all_context["unstructured"][0]],
        "knowledge": [all_context["knowledge"][0]],
    }


def test_empty_ranker_output_falls_back_safely(monkeypatch):
    all_context = sample_context()
    monkeypatch.setattr(retrieval, "call_llm_json", lambda prompt, default: {})

    result = retrieval.retrieve_relevant_context_for_user(all_context, "How old am I?", k=2)

    assert result == {
        "structured": [],
        "unstructured": [],
        "knowledge": [],
    }


def test_malformed_ranker_output_falls_back_safely(monkeypatch):
    all_context = sample_context()
    monkeypatch.setattr(
        retrieval,
        "call_llm_json",
        lambda prompt, default: {
            "structured": "not-a-list",
            "unstructured": [all_context["unstructured"][0]],
            "knowledge": None,
        },
    )

    result = retrieval.retrieve_relevant_context_for_user(all_context, "What should I cook?", k=2)

    assert result == {
        "structured": [],
        "unstructured": [all_context["unstructured"][0]],
        "knowledge": [],
    }


def test_non_dict_ranker_output_falls_back_safely(monkeypatch):
    all_context = sample_context()
    monkeypatch.setattr(retrieval, "call_llm_json", lambda prompt, default: ["bad-output"])

    result = retrieval.retrieve_relevant_context_for_user(all_context, "What should I cook?", k=2)

    assert result == {
        "structured": [],
        "unstructured": [],
        "knowledge": [],
    }


def test_ranker_output_cannot_add_items_not_retrieved_upstream(monkeypatch):
    all_context = sample_context()
    invented_item = {"key": "bank_balance", "value": "1000000", "category": "finance", "score": 1.0}
    monkeypatch.setattr(
        retrieval,
        "call_llm_json",
        lambda prompt, default: {
            "structured": [invented_item, all_context["structured"][0]],
            "unstructured": [],
            "knowledge": [],
        },
    )

    result = retrieval.retrieve_relevant_context_for_user(all_context, "How old am I?", k=2)

    assert result["structured"] == [all_context["structured"][0]]


def test_ranker_output_deduplicates_context_items(monkeypatch):
    all_context = sample_context()
    monkeypatch.setattr(
        retrieval,
        "call_llm_json",
        lambda prompt, default: {
            "structured": [
                all_context["structured"][0],
                all_context["structured"][0],
                all_context["structured"][1],
            ],
            "unstructured": [],
            "knowledge": [],
        },
    )

    result = retrieval.retrieve_relevant_context_for_user(all_context, "How old am I?", k=3)

    assert result["structured"] == [
        all_context["structured"][0],
        all_context["structured"][1],
    ]


def test_ranker_output_is_limited_to_k_items(monkeypatch):
    all_context = sample_context()
    monkeypatch.setattr(
        retrieval,
        "call_llm_json",
        lambda prompt, default: {
            "structured": all_context["structured"],
            "unstructured": all_context["unstructured"],
            "knowledge": all_context["knowledge"],
        },
    )

    result = retrieval.retrieve_relevant_context_for_user(all_context, "Tell me everything", k=1)

    assert result == {
        "structured": [all_context["structured"][0]],
        "unstructured": [all_context["unstructured"][0]],
        "knowledge": [all_context["knowledge"][0]],
    }


def test_ranker_exception_returns_no_context(monkeypatch):
    def failed_call(*args, **kwargs):
        raise RuntimeError("fake provider failed")
    monkeypatch.setattr(retrieval, "call_llm_json", failed_call)
    assert retrieval.retrieve_relevant_context_for_user(sample_context(), "hello") == {
        "structured": [], "unstructured": [], "knowledge": [],
    }
