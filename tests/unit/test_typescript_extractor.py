"""TypeScript AST extractor tests — mirror the Python tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from thought.ingest.code.ast_extractor import extract

FIXTURE = Path(__file__).parent.parent / "fixtures" / "code" / "typescript" / "auth.ts"


@pytest.fixture(scope="module")
def parsed():
    return extract(
        FIXTURE.read_text(encoding="utf-8"),
        language="typescript", file_path="auth.ts",
    )


def _names_by_type(entities, type_):
    return {e.name for e in entities if e.type_ == type_}


def test_module_entity_present(parsed):
    entities, _ = parsed
    assert any(e.type_ == "module" and e.name == "auth" for e in entities)


def test_classes_extracted(parsed):
    entities, _ = parsed
    classes = _names_by_type(entities, "class")
    assert "AuthBackend" in classes
    assert "AuthError" in classes
    assert "JWTAuth" in classes


def test_top_level_function_extracted(parsed):
    entities, _ = parsed
    functions = _names_by_type(entities, "function")
    # `authenticateUser` is exported, `_decodeToken` is module-level.
    assert "authenticateUser" in functions or "_decodeToken" in functions


def test_methods_qualified(parsed):
    entities, _ = parsed
    methods = _names_by_type(entities, "method")
    assert "JWTAuth.verify" in methods
    assert "AuthError.withContext" in methods


def test_inheritance_edge(parsed):
    _, edges = parsed
    inh = [(e.source_name, e.target_name) for e in edges if e.relation_type == "INHERITS_FROM"]
    # JWTAuth extends AuthBackend; AuthError extends Error
    assert ("JWTAuth", "AuthBackend") in inh
    assert any(s == "AuthError" and t == "Error" for s, t in inh)


def test_imports_extracted(parsed):
    _, edges = parsed
    imports = {e.target_name for e in edges if e.relation_type == "IMPORTS"}
    # The fixture imports 'jsonwebtoken' and './errors'.
    assert "jsonwebtoken" in imports
    assert "./errors" in imports


def test_javascript_alias_works():
    """Passing ``language='javascript'`` should route to the same extractor."""
    entities, _ = extract(
        "function foo() { return 42; }",
        language="javascript", file_path="snippet.js",
    )
    fn_names = {e.name for e in entities if e.type_ == "function"}
    assert "foo" in fn_names


def test_defines_edge_for_method(parsed):
    _, edges = parsed
    defines = [(e.source_name, e.target_name) for e in edges if e.relation_type == "DEFINES"]
    assert ("JWTAuth", "JWTAuth.verify") in defines
