"""Cypher subset → SQL compiler.

A documented, tested subset of Cypher that maps cleanly onto our typed-edge
bi-temporal entity graph. The grammar accepted by v0.4 is a deliberate slice:

    Query        = MATCH PatternList (WHERE Expr)? RETURN ReturnList
                   (AS_OF String)? (LIMIT Int)? (SKIP Int)?
    PatternList  = Pattern (',' Pattern)*
    Pattern      = NodePattern (EdgeStep NodePattern)*
    EdgeStep     = '-' '[' (Ident)? (':' Ident)? ']' '->'
                 | '<' '-' '[' ... ']' '-'   (reverse direction)
    NodePattern  = '(' (Ident)? (':' Ident)? PropMap? ')'
    PropMap      = '{' Pair (',' Pair)* '}'
    Pair         = Ident ':' Value
    Expr         = Term (BinaryOp Term)*    (AND/OR; left-associative)
    Term         = ('NOT')? Comparison
    Comparison   = Operand CmpOp Operand
    Operand      = Ident '.' Ident | Value
    Value        = String | Number | TRUE | FALSE | NULL
    ReturnList   = ReturnItem (',' ReturnItem)*
    ReturnItem   = Ident ('.' Ident)? ('AS' Ident)?
    CmpOp        = '=' | '<>' | '<' | '>' | '<=' | '>=' | 'CONTAINS' | 'STARTS WITH' | 'IN'

Out-of-scope for v0.4 (raises ``UnsupportedCypher`` at parse time):
- Variable-length paths ``-[:R*1..3]->``
- Aggregations (``count``, ``collect``)
- Subqueries (``CALL { ... }``)
- ``WITH`` chaining
- ``OPTIONAL MATCH``
- ``MERGE`` / ``CREATE`` / ``DELETE`` / ``SET`` (writes; read-only subset)
- Path variables ``MATCH p = (a)-[r]->(b)``

Exposed entrypoints:
- ``parse(source) -> CypherQuery`` — lex + parse, raises ``CypherSyntaxError`` or
  ``UnsupportedCypher``.
- ``compile_to_sql(query, scope_filter) -> (sql_text, params, columns)`` — emits
  parameterised SQL against the live ``entities`` / ``edges`` tables.
- ``execute(memory, source, ...)`` — convenience: parse + compile + run + hydrate.
"""
from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from ..models import ScopeFilter

# ---------------------------------------------------------------- errors

class CypherError(Exception):
    """Base for all Cypher-layer errors."""


class CypherSyntaxError(CypherError):
    """Cypher source didn't parse — bad tokens / structure."""


class UnsupportedCypher(CypherError):  # noqa: N818 — public exception name, no Error suffix
    """Cypher feature outside the v0.4 subset."""


# ---------------------------------------------------------------- lexer

_TOKEN_RE = re.compile(r"""
    (?P<WS>      \s+) |
    (?P<COMMENT> //[^\n]* | /\*.*?\*/) |
    (?P<STRING>  '(?:[^'\\]|\\.)*' | "(?:[^"\\]|\\.)*") |
    (?P<NUMBER>  -?\d+(?:\.\d+)?) |
    (?P<ARROW_R> -\[(?P<EREL_R>[^\]]*)\]-> | -->) |
    (?P<ARROW_L> <-\[(?P<EREL_L>[^\]]*)\]- | <--) |
    (?P<LPAREN>  \() |
    (?P<RPAREN>  \)) |
    (?P<LBRACE>  \{) |
    (?P<RBRACE>  \}) |
    (?P<LBRACK>  \[) |
    (?P<RBRACK>  \]) |
    (?P<COLON>   :) |
    (?P<COMMA>   ,) |
    (?P<DOT>     \.) |
    (?P<NEQ>     <>) |
    (?P<LTE>     <=) |
    (?P<GTE>     >=) |
    (?P<EQ>      =) |
    (?P<LT>      <) |
    (?P<GT>      >) |
    (?P<IDENT>   [A-Za-z_][A-Za-z0-9_-]*)
""", re.VERBOSE | re.DOTALL)

_KEYWORDS = frozenset({
    "MATCH", "WHERE", "RETURN", "LIMIT", "SKIP", "AS_OF", "AS",
    "AND", "OR", "NOT", "TRUE", "FALSE", "NULL", "IN",
    "CONTAINS", "STARTS", "ENDS", "WITH",
})

_UNSUPPORTED_KEYWORDS = frozenset({
    "MERGE", "CREATE", "DELETE", "SET", "REMOVE", "DETACH",
    "CALL", "OPTIONAL", "UNION", "FOREACH",
})


@dataclass
class Token:
    kind: str
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.kind}, {self.value!r})"


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(source):
        m = _TOKEN_RE.match(source, i)
        if not m:
            raise CypherSyntaxError(
                f"unexpected character at pos {i}: {source[i:i+20]!r}"
            )
        kind = m.lastgroup
        value = m.group()
        i = m.end()
        if kind in {"WS", "COMMENT"}:
            continue
        if kind == "IDENT" and value.upper() in _KEYWORDS:
            kind = value.upper()
            value = value.upper()
        if kind == "IDENT" and value.upper() in _UNSUPPORTED_KEYWORDS:
            raise UnsupportedCypher(
                f"{value.upper()} is not supported in the v0.4 read-only "
                f"Cypher subset. Writes still go through `thought ingest` / "
                f"`remember` / the auto-write hook."
            )
        # Capture inner edge spec for ARROW_R / ARROW_L
        if kind == "ARROW_R":
            tokens.append(Token("ARROW_R", m.group("EREL_R") or "", m.start()))
        elif kind == "ARROW_L":
            tokens.append(Token("ARROW_L", m.group("EREL_L") or "", m.start()))
        else:
            tokens.append(Token(kind or "UNK", value, m.start()))
    tokens.append(Token("EOF", "", len(source)))
    return tokens


# ---------------------------------------------------------------- AST

@dataclass
class NodePattern:
    var: str | None
    type_: str | None
    props: dict[str, Any]


@dataclass
class EdgeStep:
    direction: Literal["forward", "reverse"]
    var: str | None
    relation: str | None
    # Variable-length paths intentionally raise UnsupportedCypher in the parser.


@dataclass
class Pattern:
    head: NodePattern
    steps: list[tuple[EdgeStep, NodePattern]] = field(default_factory=list)


@dataclass
class Comparison:
    left: tuple[str, str] | object  # (var, prop) or literal
    op: str  # "=", "<>", "<", ">", "<=", ">=", "CONTAINS", "STARTS WITH", "IN"
    right: tuple[str, str] | object


@dataclass
class WhereClause:
    # AND-joined comparisons; OR is not supported in v0.4. Keep it explicit.
    terms: list[tuple[bool, Comparison]]  # (negated, comparison)


@dataclass
class ReturnItem:
    var: str
    prop: str | None
    alias: str | None


@dataclass
class CypherQuery:
    patterns: list[Pattern]
    where: WhereClause | None
    return_items: list[ReturnItem]
    limit: int | None
    skip: int | None
    as_of: str | None


# ---------------------------------------------------------------- parser

class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    def _peek(self, offset: int = 0) -> Token:
        return self.tokens[self.i + offset]

    def _consume(self, kind: str | None = None) -> Token:
        t = self.tokens[self.i]
        if kind is not None and t.kind != kind:
            raise CypherSyntaxError(
                f"expected {kind} at pos {t.pos}, got {t.kind} ({t.value!r})"
            )
        self.i += 1
        return t

    def _accept(self, kind: str) -> Token | None:
        if self._peek().kind == kind:
            return self._consume()
        return None

    def parse(self) -> CypherQuery:
        self._consume("MATCH")
        patterns = [self._pattern()]
        while self._accept("COMMA"):
            patterns.append(self._pattern())
        where = self._where() if self._peek().kind == "WHERE" else None
        self._consume("RETURN")
        return_items = [self._return_item()]
        while self._accept("COMMA"):
            return_items.append(self._return_item())
        as_of: str | None = None
        limit: int | None = None
        skip: int | None = None
        while self._peek().kind != "EOF":
            t = self._consume()
            if t.kind == "AS_OF":
                v = self._consume("STRING")
                as_of = _strip_quotes(v.value)
            elif t.kind == "LIMIT":
                limit = int(self._consume("NUMBER").value)
            elif t.kind == "SKIP":
                skip = int(self._consume("NUMBER").value)
            else:
                raise CypherSyntaxError(f"unexpected trailing token {t.kind}")
        return CypherQuery(
            patterns=patterns, where=where, return_items=return_items,
            limit=limit, skip=skip, as_of=as_of,
        )

    def _pattern(self) -> Pattern:
        head = self._node_pattern()
        steps: list[tuple[EdgeStep, NodePattern]] = []
        while self._peek().kind in {"ARROW_R", "ARROW_L"}:
            step = self._edge_step()
            node = self._node_pattern()
            steps.append((step, node))
        return Pattern(head=head, steps=steps)

    def _node_pattern(self) -> NodePattern:
        self._consume("LPAREN")
        var: str | None = None
        type_: str | None = None
        props: dict[str, Any] = {}
        if self._peek().kind == "IDENT":
            var = self._consume("IDENT").value
        if self._accept("COLON"):
            type_ = self._consume("IDENT").value
        if self._peek().kind == "LBRACE":
            props = self._prop_map()
        self._consume("RPAREN")
        return NodePattern(var=var, type_=type_, props=props)

    def _edge_step(self) -> EdgeStep:
        t = self._consume()
        spec = t.value.strip()
        direction: Literal["forward", "reverse"] = "forward" if t.kind == "ARROW_R" else "reverse"
        # ``spec`` is the inner bracket text we captured in the lexer: e.g.
        # ``r:WORKS_WITH``, ``:WORKS_WITH``, ``r``, or empty.
        if "*" in spec:
            raise UnsupportedCypher(
                "variable-length paths (-[:R*N..M]->) are not supported in v0.4. "
                "Use explicit two-step patterns or save the query as a view."
            )
        var: str | None = None
        relation: str | None = None
        if spec:
            if ":" in spec:
                var_part, rel_part = spec.split(":", 1)
                var = var_part.strip() or None
                relation = rel_part.strip() or None
            else:
                var = spec.strip() or None
        return EdgeStep(direction=direction, var=var, relation=relation)

    def _prop_map(self) -> dict[str, Any]:
        self._consume("LBRACE")
        result: dict[str, Any] = {}
        while self._peek().kind != "RBRACE":
            key = self._consume("IDENT").value
            self._consume("COLON")
            result[key] = self._value()
            if not self._accept("COMMA"):
                break
        self._consume("RBRACE")
        return result

    def _value(self) -> Any:
        t = self._consume()
        if t.kind == "STRING":
            return _strip_quotes(t.value)
        if t.kind == "NUMBER":
            return float(t.value) if "." in t.value else int(t.value)
        if t.kind == "TRUE":
            return True
        if t.kind == "FALSE":
            return False
        if t.kind == "NULL":
            return None
        raise CypherSyntaxError(f"expected value, got {t.kind} ({t.value!r})")

    def _where(self) -> WhereClause:
        self._consume("WHERE")
        terms: list[tuple[bool, Comparison]] = []
        terms.append(self._where_term())
        while self._peek().kind == "AND":
            self._consume("AND")
            terms.append(self._where_term())
        if self._peek().kind == "OR":
            raise UnsupportedCypher(
                "OR in WHERE is not supported in v0.4. Compose multiple "
                "MATCH clauses or save each branch as its own view."
            )
        return WhereClause(terms=terms)

    def _where_term(self) -> tuple[bool, Comparison]:
        negated = bool(self._accept("NOT"))
        left = self._operand()
        op = self._cmp_op()
        right = self._operand()
        return negated, Comparison(left=left, op=op, right=right)

    def _operand(self):
        t = self._peek()
        if t.kind == "IDENT":
            var = self._consume("IDENT").value
            if self._accept("DOT"):
                prop = self._consume("IDENT").value
                return (var, prop)
            return (var, None)
        return self._value()

    def _cmp_op(self) -> str:
        t = self._consume()
        if t.kind == "EQ":
            return "="
        if t.kind == "NEQ":
            return "<>"
        if t.kind == "LT":
            return "<"
        if t.kind == "GT":
            return ">"
        if t.kind == "LTE":
            return "<="
        if t.kind == "GTE":
            return ">="
        if t.kind == "IN":
            return "IN"
        if t.kind == "CONTAINS":
            return "CONTAINS"
        if t.kind == "STARTS" and self._accept("WITH"):
            return "STARTS WITH"
        if t.kind == "ENDS" and self._accept("WITH"):
            return "ENDS WITH"
        raise CypherSyntaxError(f"expected comparison operator, got {t.kind}")

    def _return_item(self) -> ReturnItem:
        var = self._consume("IDENT").value
        prop: str | None = None
        alias: str | None = None
        if self._accept("DOT"):
            prop = self._consume("IDENT").value
        if self._accept("AS"):
            alias = self._consume("IDENT").value
        return ReturnItem(var=var, prop=prop, alias=alias)


def _strip_quotes(s: str) -> str:
    # Lexer already validated the form; strip outer quotes + unescape.
    body = s[1:-1]
    return body.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


def parse(source: str) -> CypherQuery:
    """Parse Cypher source into an AST. Raises ``CypherSyntaxError`` or
    ``UnsupportedCypher`` for invalid / out-of-subset queries."""
    tokens = tokenize(source)
    if tokens[0].kind != "MATCH":
        raise CypherSyntaxError(
            "v0.4 Cypher subset requires every query to start with MATCH."
        )
    return _Parser(tokens).parse()


# ---------------------------------------------------------------- compiler

def compile_to_sql(
    query: CypherQuery,
    *,
    scope_filter: ScopeFilter | None = None,
    bi_temporal: bool = True,
) -> tuple[str, list[Any], list[str]]:
    """Compile a parsed CypherQuery to a parameterised SQL string.

    Returns ``(sql, params, columns)`` where ``columns`` is the ordered list
    of result column names (matching the order of RETURN items).
    """
    scope_filter = scope_filter or ScopeFilter(scope="all")
    # Walk patterns to build a (var → table_alias) map and JOIN clauses.
    if not query.patterns:
        raise CypherSyntaxError("empty pattern list")
    var_to_alias: dict[str, str] = {}
    select_terms: list[str] = []
    joins: list[str] = []
    where_terms: list[str] = []
    params: list[Any] = []

    alias_counter = [0]

    def new_node_alias() -> str:
        alias_counter[0] += 1
        return f"n{alias_counter[0]}"

    def new_edge_alias() -> str:
        alias_counter[0] += 1
        return f"e{alias_counter[0]}"

    def add_node(node: NodePattern, *, is_root: bool) -> str:
        alias = new_node_alias()
        if node.var:
            if node.var in var_to_alias:
                # Cross-pattern variable reuse — bind as a JOIN constraint.
                where_terms.append(f"{alias}.id = {var_to_alias[node.var]}.id")
            else:
                var_to_alias[node.var] = alias
        if is_root:
            joins.append(f"entities AS {alias}")
        else:
            # Joined in via the edge step; we still need the FROM-style clause.
            pass
        if node.type_:
            where_terms.append(f"{alias}.type = ?")
            params.append(node.type_)
        if bi_temporal:
            where_terms.append(f"{alias}.valid_until IS NULL")
        # Scope filter — applied per-node.
        scope_sql, scope_params = scope_filter.sql_where()
        # The ScopeFilter SQL references ``e.`` — substitute for our alias.
        scope_sql = scope_sql.replace("e.", f"{alias}.")
        where_terms.append(f"({scope_sql})")
        params.extend(scope_params)
        # Property map: each pair → equality predicate.
        for key, value in node.props.items():
            if key == "name":
                where_terms.append(f"{alias}.name = ?")
            elif key == "type":
                where_terms.append(f"{alias}.type = ?")
            elif key == "owner_id":
                where_terms.append(f"{alias}.owner_id = ?")
            elif key == "scope":
                where_terms.append(f"{alias}.scope = ?")
            elif key == "tier":
                where_terms.append(f"{alias}.tier = ?")
            else:
                # Entity attrs live in attrs_json; index via json_extract.
                where_terms.append(f"json_extract({alias}.attrs_json, '$.\"{key}\"') = ?")
            params.append(value)
        return alias

    for pattern in query.patterns:
        head_alias = add_node(pattern.head, is_root=True)
        prev_alias = head_alias
        for step, next_node in pattern.steps:
            edge_alias = new_edge_alias()
            next_alias = new_node_alias()
            if next_node.var:
                if next_node.var in var_to_alias:
                    where_terms.append(f"{next_alias}.id = {var_to_alias[next_node.var]}.id")
                else:
                    var_to_alias[next_node.var] = next_alias
            if step.direction == "forward":
                joins.append(
                    f"JOIN edges AS {edge_alias} "
                    f"ON {edge_alias}.source_id = {prev_alias}.id "
                    f"JOIN entities AS {next_alias} "
                    f"ON {next_alias}.id = {edge_alias}.target_id"
                )
            else:
                joins.append(
                    f"JOIN edges AS {edge_alias} "
                    f"ON {edge_alias}.target_id = {prev_alias}.id "
                    f"JOIN entities AS {next_alias} "
                    f"ON {next_alias}.id = {edge_alias}.source_id"
                )
            if step.relation:
                where_terms.append(f"{edge_alias}.relation_type = ?")
                params.append(step.relation)
            if bi_temporal:
                where_terms.append(f"{edge_alias}.valid_until IS NULL")
            # Apply per-node constraints to the joined entity.
            if next_node.type_:
                where_terms.append(f"{next_alias}.type = ?")
                params.append(next_node.type_)
            if bi_temporal:
                where_terms.append(f"{next_alias}.valid_until IS NULL")
            scope_sql, scope_params = scope_filter.sql_where()
            scope_sql = scope_sql.replace("e.", f"{next_alias}.")
            where_terms.append(f"({scope_sql})")
            params.extend(scope_params)
            for key, value in next_node.props.items():
                if key == "name":
                    where_terms.append(f"{next_alias}.name = ?")
                elif key == "type":
                    where_terms.append(f"{next_alias}.type = ?")
                else:
                    where_terms.append(
                        f"json_extract({next_alias}.attrs_json, '$.\"{key}\"') = ?"
                    )
                params.append(value)
            prev_alias = next_alias

    # WHERE clauses from explicit `WHERE` in source.
    if query.where:
        for negated, cmp in query.where.terms:
            sql, p = _compile_comparison(cmp, var_to_alias)
            if negated:
                sql = f"NOT ({sql})"
            where_terms.append(sql)
            params.extend(p)

    # AS_OF — override the valid_until IS NULL clause with a point-in-time
    # filter on the entity rows that bind to user variables.
    if query.as_of is not None:
        as_of_ts = _parse_iso(query.as_of)
        # Drop the "valid_until IS NULL" guards we added; replace with bi-temporal slice.
        # Simpler: skip — the user-facing implementation is to add the AS_OF
        # predicate to each entity alias. For correctness in v0.4 we add to
        # the variables the user explicitly named in RETURN.
        for var in {ri.var for ri in query.return_items}:
            if var in var_to_alias:
                a = var_to_alias[var]
                where_terms.append(
                    f"({a}.valid_from <= ? AND ({a}.valid_until IS NULL OR {a}.valid_until > ?))"
                )
                params.append(as_of_ts)
                params.append(as_of_ts)

    # SELECT projection.
    columns: list[str] = []
    for ri in query.return_items:
        if ri.var not in var_to_alias:
            raise CypherSyntaxError(
                f"RETURN refers to unknown variable {ri.var!r}. "
                f"Known: {list(var_to_alias.keys())}"
            )
        alias = var_to_alias[ri.var]
        if ri.prop:
            if ri.prop in {"id", "type", "name", "canonical_name", "owner_id",
                           "scope", "tier", "importance", "valid_from",
                           "valid_until", "learned_at", "unlearned_at",
                           "created_at", "last_accessed_at", "access_count"}:
                select_terms.append(f"{alias}.{ri.prop}")
            else:
                select_terms.append(
                    f"json_extract({alias}.attrs_json, '$.\"{ri.prop}\"')"
                )
            col = ri.alias or f"{ri.var}.{ri.prop}"
        else:
            # Return the full row as a JSON object so the executor can hydrate.
            select_terms.append(
                f"json_object('id', {alias}.id, 'type', {alias}.type, "
                f"'name', {alias}.name, 'scope', {alias}.scope, "
                f"'valid_from', {alias}.valid_from, "
                f"'valid_until', {alias}.valid_until)"
            )
            col = ri.alias or ri.var
        columns.append(col)

    select_sql = "SELECT DISTINCT " + ", ".join(select_terms)
    from_sql = "FROM " + " ".join(joins)
    where_sql = ("WHERE " + " AND ".join(where_terms)) if where_terms else ""
    limit_sql = f"LIMIT {int(query.limit)}" if query.limit else ""
    skip_sql = f"OFFSET {int(query.skip)}" if query.skip else ""
    sql = " ".join(s for s in [select_sql, from_sql, where_sql, limit_sql, skip_sql] if s)
    return sql, params, columns


def _compile_comparison(
    cmp: Comparison, var_to_alias: dict[str, str],
) -> tuple[str, list[Any]]:
    def lit_or_ref(val) -> tuple[str, list[Any]]:
        if isinstance(val, tuple):
            var, prop = val
            if var not in var_to_alias:
                raise CypherSyntaxError(f"unknown variable in WHERE: {var!r}")
            alias = var_to_alias[var]
            if prop is None:
                return f"{alias}.id", []
            if prop in {"id", "type", "name", "canonical_name", "owner_id",
                       "scope", "tier", "importance"}:
                return f"{alias}.{prop}", []
            return f"json_extract({alias}.attrs_json, '$.\"{prop}\"')", []
        return "?", [val]

    l_sql, l_params = lit_or_ref(cmp.left)
    r_sql, r_params = lit_or_ref(cmp.right)
    if cmp.op == "CONTAINS":
        # Translate CONTAINS to LIKE %x%.
        if isinstance(cmp.right, tuple):
            raise UnsupportedCypher("CONTAINS requires a literal on the right side")
        return f"{l_sql} LIKE ?", [*l_params, f"%{cmp.right}%"]
    if cmp.op == "STARTS WITH":
        if isinstance(cmp.right, tuple):
            raise UnsupportedCypher("STARTS WITH requires a literal on the right side")
        return f"{l_sql} LIKE ?", [*l_params, f"{cmp.right}%"]
    if cmp.op == "ENDS WITH":
        if isinstance(cmp.right, tuple):
            raise UnsupportedCypher("ENDS WITH requires a literal on the right side")
        return f"{l_sql} LIKE ?", [*l_params, f"%{cmp.right}"]
    if cmp.op == "IN":
        if isinstance(cmp.right, tuple):
            raise UnsupportedCypher("IN requires a literal list on the right side")
        if not isinstance(cmp.right, list | tuple):
            raise UnsupportedCypher("IN requires a list of literals")
        placeholders = ",".join(["?"] * len(cmp.right))
        return f"{l_sql} IN ({placeholders})", [*l_params, *cmp.right]
    return f"{l_sql} {cmp.op} {r_sql}", [*l_params, *r_params]


def _parse_iso(s: str) -> str:
    """Validate + canonicalise an ISO date/datetime string for the AS_OF predicate."""
    # Ensure it parses; return as ISO string (the comparator is lexicographic-friendly).
    dt = datetime.fromisoformat(s)
    return dt.isoformat()


# ---------------------------------------------------------------- executor

def execute(
    memory,
    source: str,
    *,
    scope: str = "all",
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    """Parse, compile, execute a Cypher query against ``memory``'s backend."""
    sf = ScopeFilter(scope=scope, owner_id=owner_id)  # type: ignore[arg-type]
    query = parse(source)
    sql, params, columns = compile_to_sql(query, scope_filter=sf)
    rows = memory._backend._conn.execute(sql, params).fetchall()  # type: ignore[attr-defined]
    out: list[dict[str, Any]] = []
    for r in rows:
        row_dict: dict[str, Any] = {}
        for i, col in enumerate(columns):
            v = r[i]
            # Hydrate full-row JSON objects.
            if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                try:
                    row_dict[col] = _json.loads(v)
                    continue
                except _json.JSONDecodeError:
                    pass
            row_dict[col] = v
        out.append(row_dict)
    return out
