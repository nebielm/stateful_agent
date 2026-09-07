import json

from app.repositories.memory_decision_log import append_memory_decision_log


def test_stored_result_creates_jsonl_log_entry(tmp_path):
    log_path = tmp_path / "logs" / "memory_decision_log.jsonl"

    entry = append_memory_decision_log(
        user_id="user-1",
        result={
            "decision": "stored",
            "field": "city",
            "category": "profile",
            "existing_value": None,
            "proposed_value": "Berlin",
            "reason": "value stored",
        },
        timestamp="2026-05-04T12:00:00",
        log_path=str(log_path),
    )

    assert entry == {
        "timestamp": "2026-05-04T12:00:00",
        "user_id": "user-1",
        "category": "profile",
        "field": "city",
        "has_proposed_value": True,
        "has_existing_value": False,
        "decision": "stored",
        "reason": "value stored",
        "source": "structured_memory_storage",
    }
    assert json.loads(log_path.read_text().strip()) == entry


def test_ignored_result_creates_jsonl_log_entry(tmp_path):
    log_path = tmp_path / "memory_decision_log.jsonl"

    append_memory_decision_log(
        user_id="user-1",
        result={
            "decision": "ignored",
            "field": "city",
            "category": "preferences",
            "existing_value": None,
            "proposed_value": "Berlin",
            "reason": "invalid schema key",
        },
        timestamp="2026-05-04T12:00:00",
        log_path=str(log_path),
    )

    assert json.loads(log_path.read_text().strip())["decision"] == "ignored"


def test_needs_confirmation_log_records_decision_without_duplicating_values(tmp_path):
    log_path = tmp_path / "memory_decision_log.jsonl"

    append_memory_decision_log(
        user_id="user-1",
        result={
            "decision": "needs_confirmation",
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        },
        timestamp="2026-05-04T12:00:00",
        log_path=str(log_path),
    )

    entry = json.loads(log_path.read_text().strip())
    assert entry["has_existing_value"] is True
    assert entry["has_proposed_value"] is True
    assert "1995-04-12" not in log_path.read_text()
    assert "1996-04-12" not in log_path.read_text()
    assert entry["decision"] == "needs_confirmation"


def test_logger_creates_parent_directory_if_missing(tmp_path):
    log_path = tmp_path / "nested" / "logs" / "memory_decision_log.jsonl"

    append_memory_decision_log(
        user_id="user-1",
        result={
            "decision": "stored",
            "field": "city",
            "category": "profile",
            "existing_value": None,
            "proposed_value": "Berlin",
            "reason": "value stored",
        },
        timestamp="2026-05-04T12:00:00",
        log_path=str(log_path),
    )

    assert log_path.exists()
