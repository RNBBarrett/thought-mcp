"""Go AST extractor via tree-sitter.

Emits: package module, top-level function entities, struct entities,
method entities (receiver-qualified as ``StructName.method``), IMPORTS edges,
DEFINES edges from struct to methods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import (
    all_named_descendants,
    go_visibility,
    module_from_path,
    text_of,
)
from .types import CodeEdge, CodeEntity

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        import tree_sitter_go
        from tree_sitter import Language, Parser
        _PARSER = Parser(Language(tree_sitter_go.language()))
    return _PARSER


def _signature_of(func_node: Node, source_bytes: bytes) -> str:
    params = func_node.child_by_field_name("parameters")
    ret = func_node.child_by_field_name("result")
    sig = text_of(params, source_bytes) if params else "()"
    if ret is not None:
        sig += " " + text_of(ret, source_bytes)
    return sig


def extract(source: str, file_path: str) -> tuple[list[CodeEntity], list[CodeEdge]]:
    parser = _get_parser()
    source_bytes = source.encode("utf-8")
    root = parser.parse(source_bytes).root_node

    module_name = module_from_path(file_path, ".go")
    entities: list[CodeEntity] = []
    edges: list[CodeEdge] = []

    entities.append(CodeEntity(
        name=module_name, type_="module", language="go",
        file_path=file_path, line_start=1, line_end=root.end_point[0] + 1,
        signature=f"package {module_name}", visibility="public",
    ))

    # Iterate top-level children.
    for child in root.named_children:
        t = child.type
        if t == "import_declaration":
            for spec in all_named_descendants(child, {"import_spec", "interpreted_string_literal"}):
                if spec.type == "interpreted_string_literal":
                    target = text_of(spec, source_bytes).strip('"')
                    edges.append(CodeEdge(
                        source_name=module_name, target_name=target,
                        relation_type="IMPORTS",
                        line_number=child.start_point[0] + 1,
                    ))
        elif t == "function_declaration":
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            fn_name = text_of(name_node, source_bytes)
            entities.append(CodeEntity(
                name=fn_name, type_="function", language="go",
                file_path=file_path,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                signature=_signature_of(child, source_bytes),
                visibility=go_visibility(fn_name),
            ))
        elif t == "method_declaration":
            name_node = child.child_by_field_name("name")
            receiver_node = child.child_by_field_name("receiver")
            if name_node is None:
                continue
            method_short = text_of(name_node, source_bytes)
            # Receiver text looks like ``(c *Cat)`` — extract the type name.
            receiver_type = ""
            if receiver_node is not None:
                rtxt = text_of(receiver_node, source_bytes).strip("()")
                # Strip leading variable name + spaces + leading * for pointer receivers.
                parts = rtxt.split()
                if parts:
                    receiver_type = parts[-1].lstrip("*")
            qualified = f"{receiver_type}.{method_short}" if receiver_type else method_short
            entities.append(CodeEntity(
                name=qualified, type_="method", language="go",
                file_path=file_path,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                signature=_signature_of(child, source_bytes),
                visibility=go_visibility(method_short),
                attrs={"receiver": receiver_type} if receiver_type else {},
            ))
            if receiver_type:
                edges.append(CodeEdge(
                    source_name=receiver_type, target_name=qualified,
                    relation_type="DEFINES",
                    line_number=child.start_point[0] + 1,
                ))
        elif t == "type_declaration":
            for spec in child.named_children:
                if spec.type != "type_spec":
                    continue
                name_node = spec.child_by_field_name("name")
                type_node = spec.child_by_field_name("type")
                if name_node is None or type_node is None:
                    continue
                name = text_of(name_node, source_bytes)
                is_interface = type_node.type == "interface_type"
                entities.append(CodeEntity(
                    name=name, type_="class", language="go",
                    file_path=file_path,
                    line_start=spec.start_point[0] + 1,
                    line_end=spec.end_point[0] + 1,
                    signature=f"type {name} {type_node.type}",
                    visibility=go_visibility(name),
                    attrs={"go_kind": "interface" if is_interface else "struct"},
                ))

    return entities, edges
