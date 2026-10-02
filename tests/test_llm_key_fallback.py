"""Credential failover exercises the real provider HTTP request path."""
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bis_assistant import rag_config, rag_llm


def config():
    return {"provider": "openai-compatible", "model": "main", "api_key": "first",
            "api_keys": ["first", "second", "third"],
            "base_url": "https://api.groq.com/openai/v1", "retries": 1}


def test_config_preserves_primary_and_deduplicates_backups(monkeypatch):
    monkeypatch.setenv("BIS_LLM_API_KEY", "first")
    monkeypatch.setenv("BIS_LLM_FALLBACK_API_KEYS", " second,first,,third,second ")
    cfg = rag_config.load_llm_config()
    assert cfg["api_key"] == "first"
    assert cfg["api_keys"] == ["first", "second", "third"]


@pytest.mark.parametrize("status", [401, 403, 429, 503])
@pytest.mark.parametrize("key_count", [3, 7])
def test_failed_credentials_reach_last_key(monkeypatch, status, key_count):
    seen = []
    keys = [f"key-{index}" for index in range(key_count)]
    cfg = config() | {"api_key": keys[0], "api_keys": keys}
    monkeypatch.setattr(rag_llm.time, "sleep", lambda _: None)

    def post(url, payload, headers, timeout):
        seen.append(headers["Authorization"])
        if headers["Authorization"] != f"Bearer {keys[-1]}":
            raise urllib.error.HTTPError(url, status, "failure", {}, None)
        return {"choices": [{"message": {"content": "recovered"}}]}

    monkeypatch.setattr(rag_llm, "_post_json", post)
    assert rag_llm.chat_complete([], cfg) == "recovered"
    expected = [f"Bearer {key}" for key in keys]
    if status == 503:
        expected = [f"Bearer {key}" for key in keys[:-1] for _ in range(2)] + [
            f"Bearer {keys[-1]}"]
    assert seen == expected
    assert rag_llm.last_failure() == ""


@pytest.mark.parametrize("status,count", [(401, 3), (400, 1)])
def test_exhaustion_and_invalid_requests_are_bounded(monkeypatch, status, count):
    seen = []

    def post(url, payload, headers, timeout):
        seen.append(headers["Authorization"])
        raise urllib.error.HTTPError(url, status, "failure", {}, None)

    monkeypatch.setattr(rag_llm, "_post_json", post)
    assert rag_llm.chat_complete([], config()) is None
    assert len(seen) == count
    assert rag_llm.last_failure() == f"http_{status}"


def test_model_fallback_remains_available_after_key_failover(monkeypatch):
    seen = []
    cfg = config() | {"fallback_model": "backup", "retries": 0}

    def post(url, payload, headers, timeout):
        seen.append((payload["model"], headers["Authorization"]))
        if payload["model"] == "main":
            raise urllib.error.HTTPError(url, 429, "limited", {}, None)
        return {"choices": [{"message": {"content": "backup answer"}}]}

    monkeypatch.setattr(rag_llm, "_post_json", post)
    assert rag_llm.chat_complete([], cfg) == "backup answer"
    assert seen == [("main", f"Bearer {key}") for key in ("first", "second", "third")] + [
        ("backup", "Bearer first")]


def test_groq_voice_uses_backup_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    seen = []

    def transcribe(data, mime, key):
        seen.append(key)
        if key != "third":
            raise urllib.error.HTTPError("https://api.groq.com", 401, "expired", {}, None)
        return "transcript"

    monkeypatch.setattr(rag_llm, "_transcribe_groq", transcribe)
    assert rag_llm.transcribe_audio(b"audio", cfg=config()) == "transcript"
    assert seen == ["first", "second", "third"]
