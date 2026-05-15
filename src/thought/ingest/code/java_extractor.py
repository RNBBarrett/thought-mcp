"""Java AST extractor via tree-sitter."""
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
        import tree_sitter_java
        from tree_sitter import Language, Parser
        _PARSER = Parser(Language(tree_sitter_java.language()))
    return _PARSER


def _java_visibility(node: Node, source_bytes: bytes):
    """``public`` / ``protected`` / `package-private` (default) / ``private`` modifiers."""
    mods = node.child_by_field_name("modifiers")
    if mods is None:
        return "public"
    text = text_of(mods, source_bytes)
    if "private" in text:
        return "private"
    return "public"


def _signature_of(method: Node, source_bytes: bytes) -> str:
    params = method.child_by_field_name("parameters")
    ret = method.child_by_field_name("type")
    sig = text_of(params, source_bytes) if params else "()"
    if ret is not None:
        sig = text_of(ret, source_bytes) + " " + sig
    return sig


def _walk_class(
    cls: Node, *, file_path: str, source_bytes: bytes,
    out_entities: list[CodeEntity], out_edges: list[CodeEdge],
) -> None:
    name_node = cls.child_by_field_name("name")
    if name_node is None:
        return
    class_name = text_of(name_node, source_bytes)
    is_interface = cls.type == "interface_declaration"
    body = cls.child_by_field_name("body")
    out_entities.append(CodeEntity(
        name=class_name, type_="class", language="java",
        file_path=file_path,
        line_start=cls.start_point[0] + 1,
        line_end=cls.end_point[0] + 1,
        signature=("interface " if is_interface else "class ") + class_name,
        visibility=_java_visibility(cls, source_bytes),
        attrs={"java_kind": "interface" if is_interface else "class"},
    ))
    # extends / implements → INHERITS_FROM edges
    for clause_name in ("superclass", "interfaces", "extends_interfaces"):
        node = cls.child_by_field_name(clause_name)
        if node is None:
            continue
        for desc in node.named_children:
            base = text_of(desc, source_bytes).strip()
            if base:
                out_edges.append(CodeEdge(
                    source_name=class_name, target_name=base,
                    relation_type="INHERITS_FROM",
                    line_number=cls.start_point[0] + 1,
                ))
    if body is None:
        return
    for child in body.named_children:
        if child.type not in ("method_declaration", "constructor_declaration"):
            continue
        m_name_node = child.child_by_field_name("name")
        if m_name_node is None:
            continue
        m_short = text_of(m_name_node, source_bytes)
        qualified = f"{class_name}.{m_short}"
        out_entities.append(CodeEntity(
            name=qualified, type_="method", language="java",
            file_path=file_path,
            line_start=child.start_point[0] + 1,
            line_end=child.end_point[0] + 1,
            signature=_signature_of(child, source_bytes),
            visibility=_java_visibility(child, source_bytes),
            attrs={"class": class_name,
                   "is_constructor": child.type == "constructor_declaration"},
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

    module_name = module_from_path(file_path, ".java")
    entities: list[CodeEntity] = []
    edges: list[CodeEdge] = []

    # Package declaration overrides our path-derived name if present.
    for child in root.named_children:
        if child.type == "package_declaration":
            pkg = child.named_children[0] if child.named_children else None
            if pkg is not None:
                module_name = text_of(pkg, source_bytes)
            break

    entities.append(CodeEntity(
        name=module_name, type_="module", language="java",
        file_path=file_path, line_start=1, line_end=root.end_point[0] + 1,
        signature=f"package {module_name}", visibility="public",
    ))

    for child in root.named_children:
        t = child.type
        if t == "import_declaration":
            # Take the literal text minus "import " and trailing ";".
            txt = text_of(child, source_bytes).strip().rstrip(";")
            target = txt[len("import "):].strip() if txt.startswith("import ") else txt
            edges.append(CodeEdge(
                source_name=module_name, target_name=target,
                relation_type="IMPORTS",
                line_number=child.start_point[0] + 1,
            ))
        elif t in ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration"):
            _walk_class(
                child, file_path=file_path, source_bytes=source_bytes,
                out_entities=entities, out_edges=edges,
            )

    return entities, edges
