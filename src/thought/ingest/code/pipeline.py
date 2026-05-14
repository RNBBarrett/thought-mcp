"""CodeIngestPipeline — wraps the v0.1 storage backend for code-typed writes.

Differences from the v0.1 ``IngestPipeline``:

- Entities have ``type ∈ {module, function, class, method}`` and carry
  ``code_file`` / ``code_language`` / ``code_commit_sha`` columns.
- Edges are emitted from the AST extractor (IMPORTS / INHERITS_FROM /
  DEFINES) rather than being mined from natural-language triples.
- Idempotent on ``(canonical_name, type, code_file, code_commit_sha)`` —
  re-ingesting the same file at the same commit is a no-op. A different
  commit produces a new entity *version* with the new SHA stamped.

This is the Phase 1 surface — call-graph edges (CALLS) come from a
separate ``call_graph.py`` pass in Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ...embeddings.base import Embedder, vector_to_bytes
from ...models import ScopeName
from ...storage.sqlite.backend import SQLiteBackend
from .ast_extractor import detect_language, extract
from .types import CodeEdge, CodeEntity


@dataclass(frozen=True)
class CodeIngestResult:
    source_id: str
    entity_ids: list[str]
    edge_ids: list[str]
    embeddings_created: int
    unresolved_edges: int


class CodeIngestPipeline:
    """Ingest a parsed file into the THOUGHT graph."""

    def __init__(
        self,
        *,
        backend: SQLiteBackend,
        embedder: Embedder,
        scope: ScopeName = "shared",
        owner_id: str | None = None,
    ) -> None:
        self._backend = backend
        self._embedder = embedder
        self._default_scope = scope
        self._default_owner = owner_id

    def ingest_code_file(
        self,
        path: Path,
        *,
        commit_sha: str | None,
        now: datetime,
        language: str | None = None,
        scope: ScopeName | None = None,
        owner_id: str | None = None,
        repo_root: Path | None = None,
    ) -> CodeIngestResult:
        """Parse a file with tree-sitter and write its entities + structural edges.

        Args:
            path: filesystem path to the source file.
            commit_sha: the commit at which this file is being observed. ``None``
                means "current snapshot" — facts get no commit pin.
            now: write timestamp (caller-supplied so backfills behave correctly).
            language: explicit override. ``None`` → auto-detect by extension.
            repo_root: if provided, ``code_file`` is recorded relative to this
                directory. Defaults to the file's parent (single-file ingest).
        """
        source = path.read_text(encoding="utf-8")
        lang = language or detect_language(str(path))
        if lang is None:
            raise ValueError(
                f"could not detect language for {path!r}; pass language= explicitly"
            )
        root = repo_root or path.parent
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.name

        entities, edges = extract(source, language=lang, file_path=rel)
        scope_to_use = scope or self._default_scope
        owner_to_use = owner_id or self._default_owner

        # Hash file content → synthetic source row so every edge can point at it.
        source_id = self._backend.upsert_source(
            content=f"{rel}@{commit_sha or 'HEAD'}\n\n{source}",
            mime_type=f"text/x-{lang}",
        )

        # Open a single transaction for the whole file's writes.
        self._backend.begin()
        try:
            name_to_id, embeddings_created = self._write_entities(
                entities, scope_to_use, owner_to_use, source_id,
                commit_sha, lang, now,
            )
            edge_ids, unresolved = self._write_edges(
                edges, name_to_id, source_id, now,
                scope=scope_to_use, owner_id=owner_to_use,
            )
            self._backend.commit()
        except Exception:
            self._backend.rollback()
            raise

        return CodeIngestResult(
            source_id=source_id,
            entity_ids=list(name_to_id.values()),
            edge_ids=edge_ids,
            embeddings_created=embeddings_created,
            unresolved_edges=unresolved,
        )

    # ------------------------------------------------------------------ helpers

    def _write_entities(
        self,
        entities: list[CodeEntity],
        scope: ScopeName,
        owner_id: str | None,
        source_id: str,
        commit_sha: str | None,
        language: str,
        now: datetime,
    ) -> tuple[dict[str, str], int]:
        """Upsert each entity, embed its signature + docstring, return name→id map."""
        name_to_id: dict[str, str] = {}
        n_embedded = 0
        for ent in entities:
            eid = self._backend.upsert_entity(
                type_=ent.type_,
                name=ent.name,
                scope=scope,
                owner_id=owner_id,
                valid_from=now,
                learned_at=now,
                source_ref=source_id,
                tier="hot",
                attrs={
                    "signature": ent.signature,
                    "visibility": ent.visibility,
                    "line_start": ent.line_start,
                    "line_end": ent.line_end,
                    **ent.attrs,
                },
                code_file=ent.file_path,
                code_language=language,
                code_commit_sha=commit_sha,
            )
            name_to_id[ent.name] = eid

            # Embed signature + docstring so VIBE recall can find the function
            # by intent rather than exact name. Skipped if both are empty.
            embed_text_parts = [ent.name, ent.signature]
            if ent.docstring:
                embed_text_parts.append(ent.docstring)
            embed_text = "\n".join(p for p in embed_text_parts if p)
            if embed_text.strip():
                vec = self._embedder.embed(embed_text)
                self._backend.store_embedding(
                    entity_id=eid,
                    model_name=self._embedder.model_name,
                    model_version=self._embedder.model_version,
                    dim=self._embedder.dim,
                    vector=vector_to_bytes(vec),
                )
                n_embedded += 1
        return name_to_id, n_embedded

    def _write_edges(
        self,
        edges: list[CodeEdge],
        name_to_id: dict[str, str],
        source_id: str,
        now: datetime,
        scope: ScopeName = "shared",
        owner_id: str | None = None,
    ) -> tuple[list[str], int]:
        """Insert the structural edges produced by the extractor.

        IMPORTS / INHERITS_FROM targets that aren't in ``name_to_id`` are
        materialised as stub ``module`` / ``class`` entities tagged
        ``confidence_class='inferred'`` — the agent gets a real graph
        traversal for "what does this file import" without us having to
        parse the imported package. Phase 2's cross-package resolver will
        upgrade these stubs when their real definition is later ingested.
        """
        edge_ids: list[str] = []
        unresolved = 0
        for edge in edges:
            src_id = name_to_id.get(edge.source_name)
            tgt_id = name_to_id.get(edge.target_name)

            if src_id is None:
                # We don't auto-create source-side stubs — source is always a
                # thing we parsed. Skip and count as unresolved.
                unresolved += 1
                continue

            if tgt_id is None:
                # Target not yet seen — create a stub entity so the edge has
                # somewhere to point. This is the right call for IMPORTS
                # (external packages) and for INHERITS_FROM where the parent
                # class is defined in another file.
                stub_type = "module" if edge.relation_type == "IMPORTS" else "class"
                tgt_id = self._backend.upsert_entity(
                    type_=stub_type,
                    name=edge.target_name,
                    scope=scope,
                    owner_id=owner_id,
                    valid_from=now,
                    learned_at=now,
                    source_ref=source_id,
                    tier="hot",
                    attrs={"stub": True, "reason": "unresolved import / inheritance"},
                )
                name_to_id[edge.target_name] = tgt_id
                unresolved += 1

            edge_ids.append(self._backend.upsert_edge(
                source_id=src_id,
                target_id=tgt_id,
                relation_type=edge.relation_type,
                source_ref=source_id,
                confidence_score=0.5 if edge.unresolved else 1.0,
                confidence_class=(
                    "inferred" if edge.unresolved else "source_grounded"
                ),
                valid_from=now,
                learned_at=now,
                attrs={"line": edge.line_number, **edge.attrs},
            ))
        return edge_ids, unresolved
