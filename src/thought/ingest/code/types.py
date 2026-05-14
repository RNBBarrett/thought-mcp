"""Code-ingest DTOs.

Lightweight dataclasses produced by the AST extractor, before they get
written through the storage backend. Kept separate from ``thought.models``
because these are *drafts* — the storage layer assigns ULIDs, timestamps,
and source_refs on insert.

Naming convention for ``CodeEntity.name``:

- module:   the module's dotted name relative to the ingest root (``auth``,
  ``mypkg.utils``).
- function: bare name (``authenticate_user``).
- class:    bare name (``JWTAuth``).
- method:   ``ClassName.method_name`` — qualified so methods don't collide
  across classes in the same scope.
- file:     POSIX-style relative path (``src/auth/middleware.py``).
- commit:   ``<short-sha>``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CodeEntityType = Literal["module", "function", "class", "method", "file", "package", "commit"]


@dataclass(frozen=True)
class CodeEntity:
    name: str
    type_: CodeEntityType
    language: str
    file_path: str
    line_start: int = 0
    line_end: int = 0
    signature: str = ""
    docstring: str | None = None
    visibility: Literal["public", "private"] = "public"
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CodeEdge:
    source_name: str
    target_name: str
    relation_type: str  # IMPORTS | INHERITS_FROM | OVERRIDES | DEFINES | CALLS
    line_number: int = 0
    # When True the target couldn't be resolved to an in-package entity
    # (e.g. a stdlib import or a dynamic attribute call). Used to set
    # confidence_class="inferred" on the eventual edge row.
    unresolved: bool = False
    attrs: dict = field(default_factory=dict)
