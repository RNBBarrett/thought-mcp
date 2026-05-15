"""Tests for v0.4 local-LLM setup helpers and the SessionStart context hook."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from thought.cli import app
from thought.hooks.setup import (
    KNOWN_OLLAMA_EMBED_MODELS,
    ping_lmstudio,
    ping_ollama,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------- ping_ollama

def test_ping_ollama_reachable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [
            {"name": "nomic-embed-text:latest"},
            {"name": "llama3.1:8b"},
        ]})
    r = ping_ollama(client=_client(handler))
    assert r.reachable
    assert "nomic-embed-text:latest" in r.models
    assert r.suggested_model is not None
    assert "nomic-embed-text" in r.suggested_model


def test_ping_ollama_unreachable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("daemon down")
    r = ping_ollama(client=_client(handler))
    assert not r.reachable
    assert "ollama serve" in r.error


def test_ping_ollama_no_embed_model() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    r = ping_ollama(client=_client(handler))
    assert r.reachable
    assert r.suggested_model is None


# ---------------------------------------------------------------- ping_lmstudio

def test_ping_lmstudio_reachable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": [{"id": "nomic-embed-text-v1.5"}, {"id": "openai/gpt-oss-20b"}],
        })
    r = ping_lmstudio(client=_client(handler))
    assert r.reachable
    assert "nomic-embed-text-v1.5" in r.models


def test_ping_lmstudio_unreachable() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("server down")
    r = ping_lmstudio(client=_client(handler))
    assert not r.reachable
    assert "LM Studio" in r.error


# ---------------------------------------------------------------- CLI

@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_ollama_setup_unreachable_exits_1(workspace, monkeypatch) -> None:
    # Make all httpx requests fail with ConnectError.
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")
    # The CLI command builds its own client; we patch httpx.Client at module level.
    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client_cls(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", fake_client)
    runner = CliRunner()
    r = runner.invoke(app, ["ollama-setup"])
    assert r.exit_code == 1


def test_lmstudio_setup_unreachable_exits_1(workspace, monkeypatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")
    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client_cls(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", fake_client)
    runner = CliRunner()
    r = runner.invoke(app, ["lmstudio-setup"])
    assert r.exit_code == 1


def test_ollama_setup_writes_config(workspace, monkeypatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "models": [{"name": "nomic-embed-text:latest"}],
        })
    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client_cls(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", fake_client)
    runner = CliRunner()
    r = runner.invoke(app, ["ollama-setup", "--write"])
    assert r.exit_code == 0, r.stdout
    cfg = (workspace / "thought.toml").read_text(encoding="utf-8")
    assert 'choice = "ollama"' in cfg
    assert "nomic-embed-text" in cfg


def test_lmstudio_setup_writes_config(workspace, monkeypatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": [{"id": "nomic-embed-text-v1.5"}],
        })
    real_client_cls = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs.pop("timeout", None)
        return real_client_cls(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", fake_client)
    runner = CliRunner()
    r = runner.invoke(app, ["lmstudio-setup", "--write"])
    assert r.exit_code == 0, r.stdout
    cfg = (workspace / "thought.toml").read_text(encoding="utf-8")
    assert 'choice = "lmstudio"' in cfg


# ---------------------------------------------------------------- known-embed-models constant

def test_known_ollama_embed_models_non_empty() -> None:
    assert len(KNOWN_OLLAMA_EMBED_MODELS) > 0
    assert "nomic-embed-text" in KNOWN_OLLAMA_EMBED_MODELS


# ---------------------------------------------------------------- reembed CLI

def test_reembed_cli(workspace) -> None:
    runner = CliRunner()
    r = runner.invoke(
        app, ["init", "--db-path", str(workspace / ".thought" / "r.db"),
              "--embedder", "deterministic", "--quick", "--no-claude-md"],
    )
    assert r.exit_code == 0
    runner.invoke(app, ["ingest", "Alice owns Acme.", "--scope", "shared"])
    r = runner.invoke(app, ["reembed", "--to", "deterministic", "--dim", "256"])
    assert r.exit_code == 0, r.stdout
    assert "reembedded" in r.stdout.lower() or "deterministic" in r.stdout.lower()


# ---------------------------------------------------------------- hook install --context

def test_hook_install_context(workspace) -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["hook", "install", "--context"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(
        (workspace / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert "SessionStart" in data["hooks"]


def test_hook_install_both_plus_context(workspace) -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["hook", "install", "--both", "--context"])
    assert r.exit_code == 0, r.stdout
    data = json.loads(
        (workspace / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    assert "UserPromptSubmit" in data["hooks"]
    assert "Stop" in data["hooks"]
    assert "SessionStart" in data["hooks"]
