"""PHP AST extractor via tree-sitter."""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import module_from_path, text_of, visibility_of
from .types import CodeEdge, CodeEntity

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        import tree_sitter_php
        from tree_sitter import Language, Parser
        # tree_sitter_php exports two: language_php() and language_php_only().
        # The "_only" variant is for files with PHP-only content (no <?php tags).
        try:
            lang = tree_sitter_php.language_php()
        except AttributeError:  # pragma: no cover — older packaging
            lang = tree_sitter_php.language()
        _PARSER = Parser(Language(lang))
    return _PARSER


def _php_method_visibility(node: Node, source_bytes: bytes):
    """Look for visibility_modifier child."""
    for c in node.children:
        if c.type == "visibility_modifier":
            t = text_of(c, source_bytes).strip()
            return "private" if t in {"private", "protected"} else "public"
    return "public"


def _signature_of(method: Node, source_bytes: bytes) -> str:
    params = method.child_by_field_name("parameters")
    return_type = method.child_by_field_name("return_type")
    sig = text_of(params, source_bytes) if params else "()"
    if return_type is not None:
        sig += ": " + text_of(return_type, source_bytes)
    return sig


def _walk_class(
    cls: Node, *, file_path: str, source_bytes: bytes,
    out_entities: list[CodeEntity], out_edges: list[CodeEdge],
) -> None:
    # Find the name child (PHP exposes ``name`` as a named child, not a field).
    name_node = None
    body_node = None
    for c in cls.named_children:
        if c.type == "name" and name_node is None:
            name_node = c
        elif c.type == "declaration_list":
            body_node = c
    if name_node is None:
        return
    class_name = text_of(name_node, source_bytes)
    is_interface = cls.type == "interface_declaration"
    is_trait = cls.type == "trait_declaration"
    out_entities.append(CodeEntity(
        name=class_name, type_="class", language="php",
        file_path=file_path,
        line_start=cls.start_point[0] + 1,
        line_end=cls.end_point[0] + 1,
        signature=("interface " if is_interface else "trait " if is_trait else "class ") + class_name,
        visibility=visibility_of(class_name),
        attrs={"php_kind": "interface" if is_interface else "trait" if is_trait else "class"},
    ))
    # base_clause (extends) + class_interface_clause (implements) — both are
    # named children, not fields.
    for c in cls.named_children:
        if c.type in ("base_clause", "class_interface_clause"):
            for inner in c.named_children:
                base = text_of(inner, source_bytes).strip()
                if base:
                    out_edges.append(CodeEdge(
                        source_name=class_name, target_name=base,
                        relation_type="INHERITS_FROM",
                        line_number=cls.start_point[0] + 1,
                    ))
    if body_node is None:
        return
    for child in body_node.named_children:
        if child.type != "method_declaration":
            continue
        m_name_node = child.child_by_field_name("name")
        if m_name_node is None:
            continue
        m_short = text_of(m_name_node, source_bytes)
        qualified = f"{class_name}.{m_short}"
        out_entities.append(CodeEntity(
            name=qualified, type_="method", language="php",
            file_path=file_path,
            line_start=child.start_point[0] + 1,
            line_end=child.end_point[0] + 1,
            signature=_signature_of(child, source_bytes),
            visibility=_php_method_visibility(child, source_bytes),
            attrs={"class": class_name},
        ))
        out_edges.append(CodeEdge(
            source_name=class_name, target_name=qualified,
            relation_type="DEFINES",
            line_number=child.start_point[0] + 1,
        ))


def extract(source: str, file_path: str) -> tuple[list[CodeEntity], list[CodeEdge]]:
    parser = _get_parser()
    source_bytes = source.encode("utf-8")
    root = parser.parse(source_bytes).root_node

    module_name = module_from_path(file_path, ".php")
    entities: list[CodeEntity] = []
    edges: list[CodeEdge] = []

    entities.append(CodeEntity(
        name=module_name, type_="module", language="php",
        file_path=file_path, line_start=1, line_end=root.end_point[0] + 1,
        signature=f"namespace {module_name}", visibility="public",
    ))

    # PHP files start with <?php; the meaningful children are inside a
    # ``program``'s named children directly. Use a recursive scan because
    # function_definition / class_declaration can appear nested under
    # namespace_definition blocks.
    def _scan(node: Node) -> None:
        for child in node.named_children:
            t = child.type
            if t == "function_definition":
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    continue
                fn_name = text_of(name_node, source_bytes)
                entities.append(CodeEntity(
                    name=fn_name, type_="function", language="php",
                    file_path=file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=_signature_of(child, source_bytes),
                    visibility=visibility_of(fn_name),
                ))
            elif t in ("class_declaration", "interface_declaration", "trait_declaration"):
                _walk_class(
                    child, file_path=file_path, source_bytes=source_bytes,
                    out_entities=entities, out_edges=edges,
                )
            elif t == "namespace_use_declaration":
                # ``use Foo\Bar;`` — emit an IMPORTS edge.
                txt = text_of(child, source_bytes).strip().rstrip(";")
                target = txt[len("use "):].strip() if txt.startswith("use ") else txt
                edges.append(CodeEdge(
                    source_name=module_name, target_name=target,
                    relation_type="IMPORTS",
                    line_number=child.start_point[0] + 1,
                ))
            elif t == "namespace_definition":
                _scan(child)

    _scan(root)
    return entities, edges
