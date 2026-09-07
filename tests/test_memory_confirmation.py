import json

from langchain_core.messages import AIMessage

from app.repositories import memory_decision_log, user_memory
from app.services.memory_confirmation import (
    apply_confirmation_prompt_to_state,
    build_confirmation_message,
    build_pending_memory_confirmation,
    classify_confirmation_reply,
    resolve_pending_confirmation,
)


def test_build_pending_memory_confirmation_returns_none_for_non_conflicts():
    assert build_pending_memory_confirmation({"decision": "stored"}, user_id="user-1") is None
    assert build_pending_memory_confirmation({"decision": "no_change"}, user_id="user-1") is None
    assert build_pending_memory_confirmation({"decision": "ignored"}, user_id="user-1") is None


def test_build_pending_memory_confirmation_returns_pending_structure_for_conflict():
    pending = build_pending_memory_confirmation(
        {
            "decision": "needs_confirmation",
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        },
        user_id="user-1",
    )

    assert pending == {
        "user_id": "user-1",
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1996-04-12",
        "reason": "immutable field conflict",
    }


def test_build_confirmation_message_includes_field_existing_and_proposed_values():
    message = build_confirmation_message(
        {
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        }
    )

    assert message == (
        "I currently have 1995-04-12 saved as your birthdate. "
        "Do you want me to replace it with 1996-04-12?"
    )


def test_apply_confirmation_prompt_to_state_updates_last_ai_message():
    state = {
        "messages": [AIMessage(content="Thanks for the correction.")],
    }
    pending_confirmation = {
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1996-04-12",
        "reason": "immutable field conflict",
    }

    apply_confirmation_prompt_to_state(state, pending_confirmation)

    assert state["messages"][-1].content == (
        "Thanks for the correction.\n\n"
        "I currently have 1995-04-12 saved as your birthdate. "
        "Do you want me to replace it with 1996-04-12?"
    )


def test_classify_confirmation_reply_detects_clear_yes_replies():
    assert classify_confirmation_reply("yes") == "confirm"
    assert classify_confirmation_reply("ja") == "confirm"
    assert classify_confirmation_reply("correct") == "confirm"


def test_classify_confirmation_reply_detects_clear_no_replies():
    assert classify_confirmation_reply("no") == "reject"
    assert classify_confirmation_reply("nein") == "reject"
    assert classify_confirmation_reply("keep old") == "reject"


def test_classify_confirmation_reply_keeps_unclear_replies_unclear():
    assert classify_confirmation_reply("maybe") == "unclear"
    assert classify_confirmation_reply("I changed my mind") == "unclear"


def test_confirmed_pending_correction_updates_immutable_value(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    resolution = resolve_pending_confirmation(
        user_id="user-1",
        reply_text="yes",
        pending_confirmation={
            "user_id": "user-1",
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        },
    )

    assert resolution["status"] == "confirmed"
    assert resolution["message"] == "Got it - I updated your birthdate to 1996-04-12."
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1996-04-12"}}}
    assert json.loads(log_file.read_text().strip())["decision"] == "confirmed_update_applied"


def test_rejected_pending_correction_preserves_old_value(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    resolution = resolve_pending_confirmation(
        user_id="user-1",
        reply_text="no",
        pending_confirmation={
            "user_id": "user-1",
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        },
    )

    assert resolution["status"] == "rejected"
    assert resolution["message"] == "Okay, I kept your birthdate as 1995-04-12."
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}
    assert json.loads(log_file.read_text().strip())["decision"] == "confirmation_rejected"


def test_unclear_reply_preserves_pending_confirmation():
    resolution = resolve_pending_confirmation(
        user_id="user-1",
        reply_text="maybe",
        pending_confirmation={
            "user_id": "user-1",
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        },
    )

    assert resolution["status"] == "unclear"
    assert resolution["message"] == "Please answer yes or no so I know whether to update your birthdate."


def test_confirmation_resolver_only_updates_exact_pending_field_category(tmp_path, monkeypatch):
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    data_file.write_text(json.dumps({"user-1": {"profile": {"birthdate": "1995-04-12"}}}))
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))

    resolution = resolve_pending_confirmation(
        user_id="user-1",
        reply_text="yes",
        pending_confirmation={
            "user_id": "user-1",
            "field": "birthdate",
            "category": "preferences",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        },
    )

    assert resolution["status"] == "invalid"
    assert resolution["message"] == "That pending correction is invalid. No memory was changed."
    stored = json.loads(data_file.read_text())
    assert stored == {"user-1": {"profile": {"birthdate": "1995-04-12"}}}
    assert resolution["result"]["decision"] == "ignored"
