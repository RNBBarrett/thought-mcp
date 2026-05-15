"""v0.5 tests — multi-language extractors + agents + scan + working_context + adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from thought.adapters.claude_sdk import ThoughtMemoryProvider
from thought.cli import app
from thought.ingest.code.ast_extractor import detect_language, extract
from thought.memory import Memory

# ---------------------------------------------------------------- language detection

@pytest.mark.parametrize("path,expected", [
    ("main.py", "python"),
    ("Cat.java", "java"),
    ("server.go", "go"),
    ("lib.rs", "rust"),
    ("class.php", "php"),
    ("app.ts", "typescript"),
    ("app.js", "javascript"),
    ("app.tsx", "typescript"),
    ("README.md", None),
])
def test_detect_language(path, expected) -> None:
    assert detect_language(path) == expected


# ---------------------------------------------------------------- per-language extractors

def test_go_extractor_emits_module_function_struct_method() -> None:
    src = """package main
import "fmt"
type Cat struct { Name string }
func (c *Cat) Meow() string { return c.Name }
func main() { fmt.Println("hi") }
"""
    ents, edges = extract(src, language="go", file_path="main.go")
    names = {(e.name, e.type_) for e in ents}
    assert ("main", "module") in names
    assert ("Cat", "class") in names
    assert ("Cat.Meow", "method") in names
    assert ("main", "function") in names
    rels = {(e.source_name, e.relation_type, e.target_name) for e in edges}
    assert ("main", "IMPORTS", "fmt") in rels
    assert ("Cat", "DEFINES", "Cat.Meow") in rels


def test_rust_extractor() -> None:
    src = """use std::io::Read;
pub struct Cat { name: String }
impl Cat {
    pub fn meow(&self) -> String { self.name.clone() }
}
fn main() {}
"""
    ents, edges = extract(src, language="rust", file_path="main.rs")
    names = {(e.name, e.type_) for e in ents}
    assert ("Cat", "class") in names
    assert ("Cat.meow", "method") in names
    assert ("main", "function") in names
    rels = {(e.source_name, e.relation_type, e.target_name) for e in edges}
    assert any(r == "INHERITS_FROM" or r == "DEFINES" for _, r, _ in rels)


def test_java_extractor_with_inheritance() -> None:
    src = """package com.acme;
import java.util.List;
public class Cat extends Animal {
    public String meow() { return "meow"; }
    public Cat(String n) {}
}
"""
    ents, edges = extract(src, language="java", file_path="Cat.java")
    types_by_name = {e.name: e.type_ for e in ents}
    assert types_by_name.get("com.acme") == "module"
    assert types_by_name.get("Cat") == "class"
    assert types_by_name.get("Cat.meow") == "method"
    rels = {(e.source_name, e.relation_type, e.target_name) for e in edges}
    assert ("Cat", "INHERITS_FROM", "Animal") in rels
    assert ("Cat", "DEFINES", "Cat.meow") in rels


def test_php_extractor() -> None:
    src = """<?php
namespace App;
use Symfony\\Component\\HttpFoundation\\Request;
class Cat extends Animal {
    public function meow(): string { return "meow"; }
}
function hi() { return "hi"; }
"""
    ents, edges = extract(src, language="php", file_path="Cat.php")
    names = {(e.name, e.type_) for e in ents}
    assert ("Cat", "class") in names
    assert ("Cat.meow", "method") in names
    assert ("hi", "function") in names
    rels = {(e.source_name, e.relation_type, e.target_name) for e in edges}
    assert ("Cat", "INHERITS_FROM", "Animal") in rels


def test_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        extract("foo bar", language="cobol", file_path="x.cob")


# ---------------------------------------------------------------- agents + scan

@pytest.fixture()
def mem(tmp_path):
    m = Memory.open(
        db_path=str(tmp_path / "v05.db"),
        embedder_choice="deterministic", embedder_dim=64,
    )
    yield m
    m.close()


def test_register_agent_idempotent(mem) -> None:
    a = mem.register_agent("vuln-scanner", description="CVE detector")
    assert a["name"] == "vuln-scanner"
    again = mem.register_agent("vuln-scanner")
    assert again["id"] == a["id"]
    assert {a["name"] for a in mem.list_agents()} == {"vuln-scanner"}


def test_scan_full_then_incremental(mem, tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("def alpha(): return 1\n")
    (repo / "b.go").write_text("package main\nfunc Beta() string { return \"\" }\n")
    # First scan: full
    r1 = mem.scan(repo, agent="bot")
    assert r1["files_scanned"] == 2
    assert r1["entities_added"] >= 2  # at least 2 modules / functions
    # Second scan (no changes via git): re-ingests but dedups idempotently
    r2 = mem.scan(repo, agent="bot")
    # No new entities the second time (canonical-name dedup absorbs replays).
    assert r2["entities_added"] == 0


def test_scan_log_persists(mem, tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("def x(): pass\n")
    mem.scan(repo, agent="bot", note="first run")
    log = mem.scan_log(agent="bot")
    assert log
    assert log[0]["note"] == "first run"


def test_scan_missing_path_raises(mem, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        mem.scan(tmp_path / "nope")


# ---------------------------------------------------------------- working_context

def test_working_context_returns_payload(mem, tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "auth.py").write_text(
        "def authenticate(token):\n    return token\n"
        "def helper():\n    return authenticate('x')\n"
    )
    mem.scan(repo, agent="agent-x")
    wc = mem.working_context("authenticate")
    assert wc["anchor"] is not None
    assert wc["anchor"]["name"] == "authenticate"
    assert "neighbours" in wc
    assert "recent_contradictions" in wc


def test_working_context_token_budget_trims(mem, tmp_path) -> None:
    # Seed many neighbours, then ask for a tiny budget.
    repo = tmp_path / "r"
    repo.mkdir()
    body = "\n".join([f"def f_{i}(): return {i}" for i in range(50)])
    (repo / "many.py").write_text(body)
    mem.scan(repo)
    wc = mem.working_context("many", budget_tokens=200)
    # Payload JSON length stays under (very roughly) the budget × 4 chars/token.
    assert len(json.dumps(wc)) < 200 * 4 * 2  # generous bound; trim is heuristic


def test_working_context_unknown_target(mem) -> None:
    wc = mem.working_context("does_not_exist")
    assert wc["anchor"] is None
    assert wc["neighbours"] == []


# ---------------------------------------------------------------- CLI

@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_scan_cli(workspace) -> None:
    runner = CliRunner()
    runner.invoke(app, [
        "init", "--db-path", ".thought/v05.db",
        "--embedder", "deterministic", "--quick", "--no-claude-md",
    ])
    (workspace / "lib.py").write_text("def hi(): return 1\n")
    r = runner.invoke(app, ["scan", str(workspace), "--as-agent", "test"])
    assert r.exit_code == 0, r.stdout


def test_agent_register_and_list_cli(workspace) -> None:
    runner = CliRunner()
    runner.invoke(app, [
        "init", "--db-path", ".thought/a.db",
        "--embedder", "deterministic", "--quick", "--no-claude-md",
    ])
    r = runner.invoke(app, [
        "agent", "register", "vuln-scanner",
        "--description", "CVE pattern matcher",
        "--cap", "scan-code", "--cap", "find-similar",
    ])
    assert r.exit_code == 0, r.stdout
    r = runner.invoke(app, ["agent", "list", "--json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert any(a["name"] == "vuln-scanner" for a in data["agents"])


def test_codebase_map_cli(workspace) -> None:
    runner = CliRunner()
    runner.invoke(app, [
        "init", "--db-path", ".thought/cm.db",
        "--embedder", "deterministic", "--quick", "--no-claude-md",
    ])
    (workspace / "src.py").write_text(
        "def main(): return helper()\ndef helper(): return 1\n"
    )
    runner.invoke(app, ["scan", str(workspace)])
    r = runner.invoke(app, ["codebase-map", "--budget-tokens", "500"])
    assert r.exit_code == 0, r.stdout


# ---------------------------------------------------------------- adapter

def test_claude_sdk_adapter_round_trip(tmp_path) -> None:
    db = str(tmp_path / "adapter.db")
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "lib.py").write_text("def authenticate(t): return t\n")
    with ThoughtMemoryProvider(
        db_path=db, agent="vuln-scanner",
        embedder_choice="deterministic", embedder_dim=64,
    ) as provider:
        scan_summary = provider.scan(str(repo))
        assert scan_summary["files_scanned"] == 1
        ctx = provider.context_for("authenticate")
        assert ctx["anchor"] is not None
        rendered = provider.render_context("authenticate")
        assert "authenticate" in rendered
        r = provider.record("Alice owns Acme Corp.")
        assert r["source_id"]
