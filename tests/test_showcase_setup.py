import importlib
import json
import logging
import runpy
from pathlib import Path
from types import SimpleNamespace

import dotenv
import pytest
from langchain_core.messages import AIMessage

from app.core import settings
from app.core.logging import JsonFormatter
from app.llm import client
from app.repositories.user_memory import MemoryStoreError


def test_settings_require_openrouter_key_and_load_project_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(dotenv, "load_dotenv", lambda path: calls.append(path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "not-an-openrouter-key")
    monkeypatch.chdir(tmp_path)
    loaded = runpy.run_path(settings.__file__)
    assert loaded["OPENROUTER_API_KEY"] == ""
    assert loaded["DATA_DIR"] == settings.PROJECT_ROOT / "data"
    assert calls == [settings.PROJECT_ROOT / ".env"]
    monkeypatch.setenv("OPENROUTER_API_KEY", " placeholder-router-key ")
    assert runpy.run_path(settings.__file__)["OPENROUTER_API_KEY"] == "placeholder-router-key"


def test_missing_provider_key_fails_before_embedding_or_vector_initialization(monkeypatch):
    chat = importlib.import_module("app.services.chat")
    monkeypatch.setattr(client, "_llm", client._UNINITIALIZED)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(chat, "get_user_id", lambda: "owner")
    monkeypatch.setattr(chat, "runtime_context", lambda: pytest.fail("Must not initialize vectors without provider configuration"))
    with pytest.raises(ValueError, match="Missing OPENROUTER_API_KEY"):
        chat.chat()


def test_corrupt_memory_fails_before_any_runtime_initialization(monkeypatch, isolated_memory):
    chat = importlib.import_module("app.services.chat")
    isolated_memory[0].write_text('{"owner":')
    monkeypatch.setattr(chat, "get_user_id", lambda: "owner")
    monkeypatch.setattr(chat, "get_llm", lambda: pytest.fail("Must not initialize provider with corrupt memory"))
    with pytest.raises(MemoryStoreError, match="preserved"):
        chat.chat()


def test_offline_cli_keeps_answer_visible_without_logging_conversation(monkeypatch, capsys, caplog):
    chat = importlib.import_module("app.services.chat")
    inputs = iter(["personal user text", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(chat, "get_user_id", lambda: "owner")
    monkeypatch.setattr(chat, "get_llm", lambda: object())
    monkeypatch.setattr(chat, "runtime_context", lambda: {})

    def invoke(input, context):
        input["messages"].append(AIMessage(content="personal answer text"))
        return input

    monkeypatch.setattr(chat, "app", SimpleNamespace(invoke=invoke))
    runpy.run_path(str(settings.PROJECT_ROOT / "main.py"), run_name="__main__")
    assert "AI: personal answer text" in capsys.readouterr().out
    assert "personal user text" not in caplog.text
    assert "personal answer text" not in caplog.text
    assert "Chat turn completed" in caplog.text


def test_json_log_formatter_escapes_quotes_and_newlines():
    message = 'quoted "message"\nnext line'
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, (), None)
    rendered = JsonFormatter().format(record)
    assert len(rendered.splitlines()) == 1
    assert json.loads(rendered)["message"] == message


def test_readme_and_demo_local_links_exist():
    import re

    for name in ("README.md", "demo_script.md"):
        source = (settings.PROJECT_ROOT / name).read_text()
        assert "/Users/" not in source
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source):
            if "://" in target or target.startswith("#"):
                continue
            assert not Path(target).is_absolute()
            assert (settings.PROJECT_ROOT / target.split("#")[0]).is_file(), target
