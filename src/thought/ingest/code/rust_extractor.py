"""Rust AST extractor via tree-sitter."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import module_from_path, text_of
from .types import CodeEdge, CodeEntity

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        import tree_sitter_rust
        from tree_sitter import Language, Parser
        _PARSER = Parser(Language(tree_sitter_rust.language()))
    return _PARSER


def _rust_visibility(node: Node, source_bytes: bytes):  # -> Literal["public","private"]
    """Rust uses ``pub`` keyword. Default is private."""
    for c in node.children:
        if c.type == "visibility_modifier":
            return "public"
    return "private"


def _signature_of(func_node: Node, source_bytes: bytes) -> str:
    params = func_node.child_by_field_name("parameters")
    ret = func_node.child_by_field_name("return_type")
    sig = text_of(params, source_bytes) if params else "()"
    if ret is not None:
        sig += " -> " + text_of(ret, source_bytes)
    return sig


def _walk_impl(
    impl: Node, *, file_path: str, source_bytes: bytes,
    out_entities: list[CodeEntity], out_edges: list[CodeEdge],
) -> None:
    """An ``impl Foo`` or ``impl Trait for Foo`` block defines methods on Foo."""
    type_node = impl.child_by_field_name("type")
    trait_node = impl.child_by_field_name("trait")
    if type_node is None:
        return
    type_name = text_of(type_node, source_bytes).strip()
    body = impl.child_by_field_name("body")
    if trait_node is not None:
        trait_name = text_of(trait_node, source_bytes).strip()
        out_edges.append(CodeEdge(
            source_name=type_name, target_name=trait_name,
            relation_type="INHERITS_FROM",
            line_number=impl.start_point[0] + 1,
        ))
    if body is None:
        return
    for child in body.named_children:
        if child.type != "function_item":
            continue
        name_node = child.child_by_field_name("name")
        if name_node is None:
            continue
        m_short = text_of(name_node, source_bytes)
        qualified = f"{type_name}.{m_short}"
        out_entities.append(CodeEntity(
            name=qualified, type_="method", language="rust",
            file_path=file_path,
            line_start=child.start_point[0] + 1,
            line_end=child.end_point[0] + 1,
            signature=_signature_of(child, source_bytes),
            visibility=_rust_visibility(child, source_bytes),
            attrs={"impl_type": type_name},
        ))
        out_edges.append(CodeEdge(
            source_name=type_name, target_name=qualified,
            relation_type="DEFINES",
            line_number=child.start_point[0] + 1,
        ))


def extract(source: str, file_path: str) -> tuple[list[CodeEntity], list[CodeEdge]]:
    parser = _get_parser()
    source_bytes = source.encode("utf-8")
    root = parser.parse(source_bytes).root_node

    module_name = module_from_path(file_path, ".rs")
    entities: list[CodeEntity] = []
    edges: list[CodeEdge] = []

    entities.append(CodeEntity(
        name=module_name, type_="module", language="rust",
        file_path=file_path, line_start=1, line_end=root.end_point[0] + 1,
        signature=f"mod {module_name}", visibility="public",
    ))

    for child in root.named_children:
        t = child.type
        if t == "function_item":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            fn_name = text_of(name_node, source_bytes)
            entities.append(CodeEntity(
                name=fn_name, type_="function", language="rust",
                file_path=file_path,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                signature=_signature_of(child, source_bytes),
                visibility=_rust_visibility(child, source_bytes),
            ))
        elif t in ("struct_item", "enum_item", "trait_item", "union_item"):
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            name = text_of(name_node, source_bytes)
            entities.append(CodeEntity(
                name=name, type_="class", language="rust",
                file_path=file_path,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                signature=f"{t.replace('_item', '')} {name}",
                visibility=_rust_visibility(child, source_bytes),
                attrs={"rust_kind": t.replace("_item", "")},
            ))
        elif t == "impl_item":
            _walk_impl(
                child, file_path=file_path, source_bytes=source_bytes,
                out_entities=entities, out_edges=edges,
            )
        elif t == "use_declaration":
            arg = child.child_by_field_name("argument")
            if arg is not None:
                target = text_of(arg, source_bytes).strip().rstrip(";")
                edges.append(CodeEdge(
                    source_name=module_name, target_name=target,
                    relation_type="IMPORTS",
                    line_number=child.start_point[0] + 1,
                ))

    return entities, edges
