"""AST extractor tests — Python first.

The extractor takes ``(source_code, language, file_path)`` and returns a list
of ``CodeEntity`` dataclasses plus a list of ``CodeEdge`` drafts (imports +
inheritance only at this phase; call-graph edges come from a separate
extractor in Phase 2 so we can iterate them independently).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from thought.ingest.code.ast_extractor import extract
from thought.ingest.code.types import CodeEdge, CodeEntity

FIXTURE = Path(__file__).parent.parent / "fixtures" / "code" / "python" / "auth.py"


@pytest.fixture(scope="module")
def parsed() -> tuple[list[CodeEntity], list[CodeEdge]]:
    source = FIXTURE.read_text(encoding="utf-8")
    return extract(source, language="python", file_path="auth.py")


def _names_by_type(entities, type_):
    return {e.name for e in entities if e.type_ == type_}


def test_module_entity_present(parsed):
    entities, _ = parsed
    assert any(e.type_ == "module" and e.name == "auth" for e in entities), (
        f"expected module 'auth'; got {[(e.type_, e.name) for e in entities]}"
    )


def test_top_level_functions_extracted(parsed):
    entities, _ = parsed
    functions = _names_by_type(entities, "function")
    assert "authenticate_user" in functions
    assert "_decode_token" in functions


def test_classes_extracted(parsed):
    entities, _ = parsed
    classes = _names_by_type(entities, "class")
    assert "AuthBackend" in classes
    assert "AuthError" in classes
    assert "JWTAuth" in classes


def test_methods_extracted_as_method_type(parsed):
    entities, _ = parsed
    methods = _names_by_type(entities, "method")
    # We expect class-qualified names so `__init__` doesn't collide across classes.
    assert "AuthError.__init__" in methods
    assert "AuthError.with_context" in methods
    assert "JWTAuth.__init__" in methods
    assert "JWTAuth.verify" in methods


def test_entities_carry_language_and_file_path(parsed):
    entities, _ = parsed
    for e in entities:
        assert e.language == "python"
        assert e.file_path == "auth.py"


def test_entities_carry_line_numbers(parsed):
    entities, _ = parsed
    fn = next(e for e in entities if e.type_ == "function" and e.name == "authenticate_user")
    assert fn.line_start > 0
    assert fn.line_end > fn.line_start


def test_function_signature_captured(parsed):
    entities, _ = parsed
    fn = next(e for e in entities if e.type_ == "function" and e.name == "authenticate_user")
    # We don't pin the exact format, just that the captured signature contains the params.
    assert "token" in fn.signature
    assert "secret" in fn.signature


def test_docstring_captured(parsed):
    entities, _ = parsed
    fn = next(e for e in entities if e.type_ == "function" and e.name == "authenticate_user")
    assert "JWT" in (fn.docstring or "")


def test_imports_extracted(parsed):
    _, edges = parsed
    import_edges = [e for e in edges if e.relation_type == "IMPORTS"]
    targets = {e.target_name for e in import_edges}
    assert "jwt" in targets
    assert "datetime" in targets
    # relative import — we keep the leading dot in the target name for resolution
    assert any(t.endswith("errors") for t in targets), f"missing relative import; got {targets}"


def test_inheritance_edge_extracted(parsed):
    _, edges = parsed
    inh = [e for e in edges if e.relation_type == "INHERITS_FROM"]
    # JWTAuth INHERITS_FROM AuthBackend
    assert any(
        e.source_name == "JWTAuth" and e.target_name == "AuthBackend" for e in inh
    ), f"expected JWTAuth INHERITS_FROM AuthBackend; got {[(e.source_name, e.target_name) for e in inh]}"


def test_methods_link_to_their_class_via_defines_edge(parsed):
    _, edges = parsed
    defines = [e for e in edges if e.relation_type == "DEFINES"]
    # JWTAuth DEFINES JWTAuth.verify
    assert any(
        e.source_name == "JWTAuth" and e.target_name == "JWTAuth.verify"
        for e in defines
    )


def test_unknown_language_raises(parsed):
    with pytest.raises(ValueError, match="unsupported language"):
        extract("doesnt matter", language="brainfuck", file_path="x.bf")
