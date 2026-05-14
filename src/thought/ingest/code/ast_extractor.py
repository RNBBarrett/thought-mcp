"""Language-agnostic AST extractor entry point.

Dispatches to per-language extractors that share a single output shape
(``list[CodeEntity], list[CodeEdge]``). Each language plugin owns the
tree-sitter grammar handle and the node-walk logic — adding a language is
one file under ``src/thought/ingest/code/``.

The extractor is *structural* only: it produces entity drafts and the
edges discoverable from a single file (IMPORTS / INHERITS_FROM / DEFINES).
Cross-file edges (CALLS targeting another file's function) come from the
call-graph pass in Phase 2, after all files have been ingested as entities.
"""
from __future__ import annotations

from collections.abc import Callable

from .types import CodeEdge, CodeEntity

# Lazy: don't import grammars at module load — they pull in C extensions.
_REGISTRY: dict[str, Callable[[str, str], tuple[list[CodeEntity], list[CodeEdge]]]] = {}


def _python_extractor():
    from . import python_extractor
    return python_extractor.extract


def _typescript_extractor():  # pragma: no cover — wired in Phase 5
    from . import typescript_extractor
    return typescript_extractor.extract


_LOADERS = {
    "python": _python_extractor,
    "typescript": _typescript_extractor,
    "javascript": _typescript_extractor,  # same grammar package, different mode
}


def extract(
    source: str, *, language: str, file_path: str,
) -> tuple[list[CodeEntity], list[CodeEdge]]:
    """Parse ``source`` with the language-specific extractor.

    Args:
        source: the source code as a string.
        language: ``"python"`` / ``"typescript"`` / ``"javascript"``.
        file_path: the file's path relative to the ingest root. Stored on
            every emitted CodeEntity for traceability.

    Raises:
        ValueError: when ``language`` isn't supported.
    """
    if language not in _REGISTRY:
        if language not in _LOADERS:
            raise ValueError(
                f"unsupported language: {language!r} "
                f"(known: {sorted(_LOADERS)})"
            )
        _REGISTRY[language] = _LOADERS[language]()
    return _REGISTRY[language](source, file_path)


def detect_language(file_path: str) -> str | None:
    """Best-effort language detection from a file extension."""
    p = file_path.lower()
    if p.endswith(".py"):
        return "python"
    if p.endswith((".ts", ".tsx")):
        return "typescript"
    if p.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript"
    return None
