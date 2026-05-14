"""TypeScript / JavaScript AST extractor via tree-sitter.

Mirrors the Python extractor — same output shape (``list[CodeEntity],
list[CodeEdge]``). Tree-sitter-typescript ships two grammars (``typescript``
+ ``tsx``); we use the plain TypeScript grammar for ``.ts`` files and TSX
for ``.tsx``. The JavaScript grammar is a subset that the TypeScript
grammar handles too.

Node types used:
    program                       (root)
    function_declaration          (top-level ``function foo()``)
    arrow_function                (``const foo = () => ...``)
    class_declaration             (``class Foo``)
    method_definition             (inside class body)
    import_statement              (``import * from ...``)
    extends_clause / class_heritage
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CodeEdge, CodeEntity

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

_TS_PARSER = None
_TSX_PARSER = None


def _get_parser(use_tsx: bool):
    global _TS_PARSER, _TSX_PARSER
    import tree_sitter_typescript
    from tree_sitter import Language, Parser
    if use_tsx:
        if _TSX_PARSER is None:
            _TSX_PARSER = Parser(Language(tree_sitter_typescript.language_tsx()))
        return _TSX_PARSER
    if _TS_PARSER is None:
        _TS_PARSER = Parser(Language(tree_sitter_typescript.language_typescript()))
    return _TS_PARSER


def _text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _module_name_from_path(file_path: str) -> str:
    p = file_path.replace("\\", "/")
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
        if p.endswith(ext):
            p = p[: -len(ext)]
            break
    parts = [seg for seg in p.split("/") if seg]
    if parts and parts[-1] == "index":
        parts.pop()
    return ".".join(parts) if parts else "<root>"


def _docstring_of(body_node: Node | None, source_bytes: bytes) -> str | None:
    """JSDoc lookup — TS doesn't have first-class docstrings; we read the
    leading ``/** ... */`` comment if one precedes the declaration.
    """
    if body_node is None:
        return None
    prev = body_node.prev_sibling
    if prev is not None and prev.type == "comment":
        text = _text(prev, source_bytes).strip()
        if text.startswith("/**"):
            return text.removeprefix("/**").removesuffix("*/").strip()
    return None


def _visibility(name: str) -> str:
    return "private" if name.startswith("_") and not name.startswith("__") else "public"


def _signature_of(func: Node, source_bytes: bytes) -> str:
    params = func.child_by_field_name("parameters")
    ret = func.child_by_field_name("return_type")
    sig = _text(params, source_bytes) if params else "()"
    if ret is not None:
        sig += " " + _text(ret, source_bytes)
    return sig


def _emit_class(
    cls: Node,
    *,
    file_path: str,
    source_bytes: bytes,
    out_entities: list[CodeEntity],
    out_edges: list[CodeEdge],
) -> None:
    name_node = cls.child_by_field_name("name")
    if name_node is None:
        return
    class_name = _text(name_node, source_bytes)
    body = cls.child_by_field_name("body")

    out_entities.append(CodeEntity(
        name=class_name,
        type_="class",
        language="typescript",
        file_path=file_path,
        line_start=cls.start_point[0] + 1,
        line_end=cls.end_point[0] + 1,
        signature=f"class {class_name}",
        docstring=_docstring_of(cls, source_bytes),
        visibility=_visibility(class_name),
    ))

    # Heritage — ``class Foo extends Bar implements Baz``
    heritage = cls.child_by_field_name("heritage") or cls.child_by_field_name("class_heritage")
    if heritage is None:
        # tree-sitter-typescript exposes it via named children.
        for c in cls.named_children:
            if c.type in ("class_heritage", "extends_clause"):
                heritage = c
                break
    if heritage is not None:
        for hc in heritage.named_children:
            if hc.type == "extends_clause":
                for v in hc.named_children:
                    parent = _text(v, source_bytes)
                    if parent:
                        out_edges.append(CodeEdge(
                            source_name=class_name, target_name=parent,
                            relation_type="INHERITS_FROM",
                            line_number=cls.start_point[0] + 1,
                        ))
            else:
                # Sometimes the extends target is a direct child.
                parent = _text(hc, source_bytes)
                if parent and parent not in {"extends", "implements"}:
                    out_edges.append(CodeEdge(
                        source_name=class_name, target_name=parent,
                        relation_type="INHERITS_FROM",
                        line_number=cls.start_point[0] + 1,
                    ))

    if body is not None:
        for m in body.named_children:
            if m.type == "method_definition":
                m_name_node = m.child_by_field_name("name")
                if m_name_node is None:
                    continue
                method_short = _text(m_name_node, source_bytes)
                method_qualified = f"{class_name}.{method_short}"
                out_entities.append(CodeEntity(
                    name=method_qualified,
                    type_="method",
                    language="typescript",
                    file_path=file_path,
                    line_start=m.start_point[0] + 1,
                    line_end=m.end_point[0] + 1,
                    signature=_signature_of(m, source_bytes),
                    docstring=_docstring_of(m, source_bytes),
                    visibility=_visibility(method_short),
                    attrs={"class": class_name},
                ))
                out_edges.append(CodeEdge(
                    source_name=class_name,
                    target_name=method_qualified,
                    relation_type="DEFINES",
                    line_number=m.start_point[0] + 1,
                ))


def _emit_import(stmt: Node, *, module_name: str, source_bytes: bytes, out_edges: list[CodeEdge]) -> None:
    """Extract the source string from an ``import`` statement.

    ``import * from 'foo'``      → IMPORTS('module', 'foo')
    ``import { X } from './bar'`` → IMPORTS('module', './bar')
    """
    src = stmt.child_by_field_name("source")
    if src is None:
        # Find the first string child as a fallback.
        for c in stmt.named_children:
            if c.type == "string":
                src = c
                break
    if src is None:
        return
    target = _text(src, source_bytes).strip("'\"")
    if target:
        out_edges.append(CodeEdge(
            source_name=module_name, target_name=target,
            relation_type="IMPORTS",
            line_number=stmt.start_point[0] + 1,
        ))


def extract(source: str, file_path: str) -> tuple[list[CodeEntity], list[CodeEdge]]:
    use_tsx = file_path.endswith((".tsx", ".jsx"))
    parser = _get_parser(use_tsx=use_tsx)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    module_name = _module_name_from_path(file_path)
    entities: list[CodeEntity] = []
    edges: list[CodeEdge] = []

    entities.append(CodeEntity(
        name=module_name,
        type_="module",
        language="typescript",
        file_path=file_path,
        line_start=1,
        line_end=root.end_point[0] + 1,
        signature=f"module {module_name}",
        docstring=None,
        visibility="public",
    ))

    def _walk(node: Node, scope_class: str | None = None) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_declaration":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    fn_name = _text(name_node, source_bytes)
                    entities.append(CodeEntity(
                        name=fn_name,
                        type_="function",
                        language="typescript",
                        file_path=file_path,
                        line_start=child.start_point[0] + 1,
                        line_end=child.end_point[0] + 1,
                        signature=_signature_of(child, source_bytes),
                        docstring=_docstring_of(child, source_bytes),
                        visibility=_visibility(fn_name),
                    ))
            elif t == "class_declaration":
                _emit_class(
                    child, file_path=file_path, source_bytes=source_bytes,
                    out_entities=entities, out_edges=edges,
                )
            elif t == "import_statement":
                _emit_import(child, module_name=module_name, source_bytes=source_bytes, out_edges=edges)
            elif t == "export_statement":
                # Recurse — ``export class X`` / ``export function Y`` wrap the declaration.
                _walk(child, scope_class=scope_class)

    _walk(root)
    return entities, edges
