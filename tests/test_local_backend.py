"""Pointing the LLM client at a local server (vLLM / Ollama)."""

from agentsynth.utils import LLMClient


def test_api_base_is_threaded_and_keyed_for_local_servers():
    client = LLMClient(model="openai/my-served-model", api_base="http://localhost:8000/v1")
    assert client.api_base == "http://localhost:8000/v1"
    assert client.model == "openai/my-served-model"
    # an OpenAI-compatible server (vLLM) needs a non-empty key; we default one
    assert client.api_key == "local"


def test_no_api_base_leaves_the_key_alone():
    client = LLMClient(model="gpt-4o-mini")
    assert client.api_base is None
    assert client.api_key is None


def test_env_vars_drive_a_local_backend(monkeypatch):
    monkeypatch.setenv("AGENTSYNTH_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("AGENTSYNTH_MODEL", "ollama/llama3")
    client = LLMClient()
    assert client.api_base == "http://localhost:11434"
    assert client.model == "ollama/llama3"
    assert client.api_key == "local"


def test_explicit_args_win_over_env(monkeypatch):
    monkeypatch.setenv("AGENTSYNTH_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("AGENTSYNTH_MODEL", "ollama/llama3")
    client = LLMClient(model="openai/other", api_base="http://localhost:9000/v1", api_key="sk-x")
    assert client.model == "openai/other"
    assert client.api_base == "http://localhost:9000/v1"
    assert client.api_key == "sk-x"
