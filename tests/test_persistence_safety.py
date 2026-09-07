import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.repositories import memory_decision_log, user_memory
from app.services import memory_confirmation
from app.services import graph


def pending():
    return {
        "user_id": "owner", "field": "birthdate", "category": "profile",
        "existing_value": "1995-04-12", "proposed_value": "1996-04-12",
    }


@pytest.mark.parametrize("contents", ['{"owner":', '[]', '{"owner": []}', '{"owner": {"profile": []}}', '\xff'])
def test_corrupt_store_is_not_empty_and_cannot_be_overwritten(isolated_memory, contents, caplog):
    path, log_path = isolated_memory
    original = contents.encode("latin1")
    path.write_bytes(original)
    with pytest.raises(user_memory.MemoryStoreError, match="preserved"):
        user_memory.load_user_data()
    result = user_memory.controlled_structured_data_storage("owner", "city", "Berlin", "profile")
    assert result["decision"] == "failed"
    assert path.read_bytes() == original
    assert "preserved" in caplog.text
    assert json.loads(log_path.read_text())["decision"] == "failed"


def test_corrupt_store_cannot_be_overwritten_by_confirmation(isolated_memory):
    path, _ = isolated_memory
    path.write_text('{"owner":')
    result = memory_confirmation.resolve_pending_confirmation("owner", "yes", pending())
    assert result["status"] == "failed"
    assert "unchanged" in result["message"]
    assert path.read_text() == '{"owner":'


@pytest.mark.parametrize("correction", [False, True])
def test_writes_replace_a_complete_same_directory_temporary_file(isolated_memory, monkeypatch, correction):
    path, _ = isolated_memory
    original = {"owner": {"profile": {"birthdate": "1995-04-12"}}}
    path.write_text(json.dumps(original))
    replace, fsync = user_memory.os.replace, user_memory.os.fsync
    events = []

    def record_fsync(fd):
        events.append("flush")
        fsync(fd)

    def replace_complete(source, target):
        assert events == ["flush"]
        assert Path(source).parent == path.parent
        assert Path(target) == path
        assert json.loads(path.read_text()) == original
        contents = json.loads(Path(source).read_text())
        assert contents["owner"]["profile"]["birthdate"] == ("1996-04-12" if correction else "1995-04-12")
        events.append("replace")
        replace(source, target)

    monkeypatch.setattr(user_memory.os, "fsync", record_fsync)
    monkeypatch.setattr(user_memory.os, "replace", replace_complete)
    if correction:
        result = memory_confirmation.resolve_pending_confirmation("owner", "yes", pending())
        assert result["status"] == "confirmed"
    else:
        assert user_memory.controlled_structured_data_storage("owner", "city", "Berlin", "profile")["decision"] == "stored"
    assert events == ["flush", "replace"]
    assert not list(path.parent.glob(f".{path.name}.*"))


@pytest.mark.parametrize("failure_point", ["fsync", "replace"])
@pytest.mark.parametrize("correction", [False, True])
def test_failed_write_preserves_original_and_cleans_temp(isolated_memory, monkeypatch, failure_point, correction):
    path, _ = isolated_memory
    original = json.dumps({"owner": {"profile": {"birthdate": "1995-04-12"}}})
    path.write_text(original)

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(user_memory.os, failure_point, fail)
    if correction:
        result = memory_confirmation.resolve_pending_confirmation("owner", "yes", pending())
        assert result["status"] == "failed"
    else:
        assert user_memory.controlled_structured_data_storage("owner", "city", "Paris", "profile")["decision"] == "failed"
    assert path.read_text() == original
    assert not list(path.parent.glob(f".{path.name}.*"))


@pytest.mark.parametrize("action,decision", [
    ("store", "stored"), ("yes", "confirmed_update_applied"), ("no", "confirmation_rejected"),
])
def test_audit_failure_does_not_change_memory_outcome(isolated_memory, monkeypatch, caplog, action, decision):
    path, _ = isolated_memory
    path.write_text(json.dumps({"owner": {"profile": {"birthdate": "1995-04-12"}}}))
    blocked_log = path.parent / "blocked"
    blocked_log.mkdir()
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(blocked_log))
    if action == "store":
        result = user_memory.controlled_structured_data_storage("owner", "city", "Berlin", "profile")
        assert user_memory.load_user_data()["owner"]["profile"]["city"] == "Berlin"
    else:
        resolution = memory_confirmation.resolve_pending_confirmation("owner", action, pending())
        assert resolution["status"] == ("confirmed" if action == "yes" else "rejected")
        result = resolution["result"]
    assert result["decision"] == decision
    assert user_memory.load_user_data()["owner"]["profile"]["birthdate"] == ("1996-04-12" if action == "yes" else "1995-04-12")
    assert "Decision log append failed" in caplog.text
    assert f"memory decision remains {decision}" in caplog.text


def test_memory_values_are_not_duplicated_in_normal_logs(isolated_memory, caplog):
    user_memory.controlled_structured_data_storage("owner", "birthdate", "1995-04-12", "profile")
    user_memory.controlled_structured_data_storage("owner", "birthdate", "1996-04-12", "profile")
    combined = caplog.text + isolated_memory[1].read_text()
    assert "1995-04-12" not in combined
    assert "1996-04-12" not in combined
    assert "needs_confirmation" in combined


def test_graph_reports_mid_session_storage_failure_to_user(isolated_memory, monkeypatch):
    path, _ = isolated_memory
    path.write_text('{"owner":')
    monkeypatch.setattr(graph, "extract_memory_updates", lambda **kwargs: {
        "structured": [{"key": "city", "category": "profile", "value": "Berlin"}],
        "unstructured": [],
    })
    state = {"user_id": "owner", "messages": [HumanMessage(content="I live in Berlin."), AIMessage(content="Noted.", id="reply")]}
    result = graph.memory_updater_node(state, SimpleNamespace(context={"user_vectorstore": object()}))
    assert result["memory_updates"]["structured_results"][0]["decision"] == "failed"
    assert "updates were not saved" in result["messages"][-1].content
    assert result["messages"][-1].id == "reply"
    assert len(result["messages"]) == 2
    assert path.read_text() == '{"owner":'
