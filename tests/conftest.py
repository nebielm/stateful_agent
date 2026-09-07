import socket
from datetime import datetime

import pytest

from app.llm import client
from app.repositories import memory_decision_log, user_memory
from app.utils import dates


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Keep every test away from personal data, credentials, and live services."""
    data_file = tmp_path / "user_info.json"
    log_file = tmp_path / "memory_decision_log.jsonl"
    monkeypatch.setattr(user_memory, "USER_INFO_PATH", str(data_file))
    monkeypatch.setattr(memory_decision_log, "MEMORY_DECISION_LOG_PATH", str(log_file))
    monkeypatch.setattr(client.settings, "OPENROUTER_API_KEY", None)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 5, 4, 12, tzinfo=tz)

    monkeypatch.setattr(dates, "datetime", FixedDateTime)

    def no_network(*args, **kwargs):
        raise AssertionError("Tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    return data_file, log_file
