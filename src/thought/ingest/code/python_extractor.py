"""Python AST extractor via tree-sitter.

Walks the parsed tree-sitter tree once and emits:
- one ``module`` entity per file
- one ``function`` entity per top-level ``def`` / ``async def``
- one ``class`` entity per ``class`` statement
- one ``method`` entity per ``def`` inside a class body, named ``ClassName.method``
- IMPORTS edges from the module to each imported name
- INHERITS_FROM edges from each class to each base class
- DEFINES edges from each class to its methods (so a class → method traversal
  is one hop in the graph layer)

Tree-sitter node types used (per tree-sitter-python grammar):
    module
    function_definition          (top-level or inside class_body)
    class_definition
    expression_statement → string (docstring detection)
    import_statement, import_from_statement
    parameters, identifier, dotted_name
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .types import CodeEdge, CodeEntity

if TYPE_CHECKING:  # pragma: no cover
    from tree_sitter import Node

_PARSER = None


def _get_parser():
    """Lazily build the tree-sitter Python parser. Cached process-wide."""
    global _PARSER
    if _PARSER is None:
        import tree_sitter_python
        from tree_sitter import Language, Parser
        _PARSER = Parser(Language(tree_sitter_python.language()))
    return _PARSER


def _text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _module_name_from_path(file_path: str) -> str:
    """Derive ``a.b.c`` from ``a/b/c.py``. Trim trailing ``__init__``."""
    p = file_path.replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    parts = [seg for seg in p.split("/") if seg]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else "<root>"


def _docstring_of(body_node: Node | None, source_bytes: bytes) -> str | None:
    """Return the docstring text if the first statement in ``body_node`` is a string."""
    if body_node is None:
        return None
    for child in body_node.named_children:
        if child.type == "expression_statement":
            string_node = child.named_children[0] if child.named_children else None
            if string_node and string_node.type == "string":
                text = _text(string_node, source_bytes)
                # Strip triple/quote delimiters + the string-prefix flags.
                for prefix in ('"""', "'''", '"', "'"):
                    if text.startswith(prefix) and text.endswith(prefix):
                        return text[len(prefix) : -len(prefix)].strip()
            return None
        # First non-string statement → no docstring.
        return None
    return None


def _visibility(name: str) -> str:
    return "private" if name.startswith("_") and not name.startswith("__") else "public"


def _signature_of(func_node: Node, source_bytes: bytes) -> str:
    """Return the function's parameter list as source text, e.g. ``(token: str) -> dict``."""
    params = func_node.child_by_field_name("parameters")
    ret = func_node.child_by_field_name("return_type")
    sig = _text(params, source_bytes) if params else "()"
    if ret is not None:
        sig += " -> " + _text(ret, source_bytes)
    return sig


def _walk_class(
    cls: Node,
    *,
    module_name: str,
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
        language="python",
        file_path=file_path,
        line_start=cls.start_point[0] + 1,
        line_end=cls.end_point[0] + 1,
        signature=f"class {class_name}",
        docstring=_docstring_of(body, source_bytes),
        visibility=_visibility(class_name),
    ))

    # INHERITS_FROM — read the superclass argument list.
    superclasses_node = cls.child_by_field_name("superclasses")
    if superclasses_node is not None:
        for sc in superclasses_node.named_children:
            base_name = _text(sc, source_bytes)
            if base_name:
                out_edges.append(CodeEdge(
                    source_name=class_name,
                    target_name=base_name,
                    relation_type="INHERITS_FROM",
                    line_number=cls.start_point[0] + 1,
                ))

    # Methods inside the class body.
    if body is not None:
        for child in body.named_children:
            if child.type in ("function_definition", "async_function_definition"):
                m_name_node = child.child_by_field_name("name")
                if m_name_node is None:
                    continue
                method_short = _text(m_name_node, source_bytes)
                method_qualified = f"{class_name}.{method_short}"
                out_entities.append(CodeEntity(
                    name=method_qualified,
                    type_="method",
                    language="python",
                    file_path=file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=_signature_of(child, source_bytes),
                    docstring=_docstring_of(child.child_by_field_name("body"), source_bytes),
                    visibility=_visibility(method_short),
                    attrs={"class": class_name},
                ))
                out_edges.append(CodeEdge(
                    source_name=class_name,
                    target_name=method_qualified,
                    relation_type="DEFINES",
                    line_number=child.start_point[0] + 1,
                ))


def _walk_module(
    root: Node,
    *,
    module_name: str,
    file_path: str,
    source_bytes: bytes,
    out_entities: list[CodeEntity],
    out_edges: list[CodeEdge],
) -> None:
    for child in root.named_children:
        t = child.type
        if t in ("function_definition", "async_function_definition"):
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            fn_name = _text(name_node, source_bytes)
            out_entities.append(CodeEntity(
                name=fn_name,
                type_="function",
                language="python",
                file_path=file_path,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                signature=_signature_of(child, source_bytes),
                docstring=_docstring_of(child.child_by_field_name("body"), source_bytes),
                visibility=_visibility(fn_name),
            ))
        elif t == "class_definition":
            _walk_class(
                child, module_name=module_name, file_path=file_path,
                source_bytes=source_bytes,
                out_entities=out_entities, out_edges=out_edges,
            )
        elif t == "import_statement":
            # import a, b.c — multiple targets possible.
            for dn in child.named_children:
                if dn.type == "dotted_name":
                    out_edges.append(CodeEdge(
                        source_name=module_name,
                        target_name=_text(dn, source_bytes),
                        relation_type="IMPORTS",
                        line_number=child.start_point[0] + 1,
                    ))
                elif dn.type == "aliased_import":
                    inner = dn.child_by_field_name("name")
                    if inner is not None:
                        out_edges.append(CodeEdge(
                            source_name=module_name,
                            target_name=_text(inner, source_bytes),
                            relation_type="IMPORTS",
                            line_number=child.start_point[0] + 1,
                        ))
        elif t == "import_from_statement":
            mod_node = child.child_by_field_name("module_name")
            target: str | None = None
            if mod_node is not None:
                target = _text(mod_node, source_bytes)
            else:
                # Relative import: ``from .errors import X`` produces a
                # ``relative_import`` child instead of a ``module_name`` field.
                # Use the literal text (e.g. ".errors", "..pkg.mod") so the
                # leading dot count survives — it carries package-level
                # information the resolver later uses.
                for c in child.named_children:
                    if c.type == "relative_import":
                        target = _text(c, source_bytes)
                        break
                if target is None:
                    target = "."  # bare ``from . import x``
            out_edges.append(CodeEdge(
                source_name=module_name,
                target_name=target,
                relation_type="IMPORTS",
                line_number=child.start_point[0] + 1,
                attrs={"from_import": True},
            ))


def extract(source: str, file_path: str) -> tuple[list[CodeEntity], list[CodeEdge]]:
    parser = _get_parser()
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    module_name = _module_name_from_path(file_path)
    entities: list[CodeEntity] = []
    edges: list[CodeEdge] = []

    # Emit the module entity first.
    entities.append(CodeEntity(
        name=module_name,
        type_="module",
        language="python",
        file_path=file_path,
        line_start=1,
        line_end=root.end_point[0] + 1,
        signature=f"module {module_name}",
        docstring=_docstring_of(root, source_bytes),
        visibility="public",
    ))

    _walk_module(
        root, module_name=module_name, file_path=file_path,
        source_bytes=source_bytes,
        out_entities=entities, out_edges=edges,
    )
    return entities, edges
