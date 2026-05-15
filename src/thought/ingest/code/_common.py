"""Helpers shared across the per-language tree-sitter extractors.

Each language extractor follows the same shape — module entity + function +
class/struct + method + IMPORTS + INHERITS_FROM + DEFINES edges. The
boilerplate (text extraction, visibility, signature shaping) lives here so
the per-language files focus on the AST node names that matter for them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node


def text_of(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def visibility_of(name: str) -> str:
    """Heuristic — leading underscore (or lowercase-first in Go) = private."""
    if not name:
        return "public"
    if name.startswith("_") and not name.startswith("__"):
        return "private"
    return "public"


def go_visibility(name: str) -> str:
    """Go convention: capital first letter = exported."""
    if not name:
        return "public"
    return "public" if name[0].isupper() else "private"


def module_from_path(file_path: str, ext: str) -> str:
    """Derive ``a.b.c`` from ``a/b/c.<ext>``."""
    p = file_path.replace("\\", "/")
    if p.endswith(ext):
        p = p[:-len(ext)]
    parts = [seg for seg in p.split("/") if seg and seg not in {"src", "internal"}]
    return ".".join(parts) if parts else "<root>"


def first_child_named(node: Node, kind: str) -> Node | None:
    for c in node.named_children:
        if c.type == kind:
            return c
    return None


def all_named_descendants(node: Node, kinds: set[str]):
    """Walk every descendant; yield ones whose ``.type`` is in ``kinds``."""
    stack = list(node.named_children)
    while stack:
        n = stack.pop()
        if n.type in kinds:
            yield n
        stack.extend(n.named_children)
