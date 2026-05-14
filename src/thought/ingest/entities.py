"""Entity / relation extraction.

The MVP uses a deterministic baseline:
- Proper-noun token spotting (Capitalized words and ALLCAPS) → entity
  candidates, typed as ``CONCEPT`` by default.
- A small set of verb / preposition patterns produces typed edges:
   - ``X owns Y``      → OWNS
   - ``X prefers Y``   → PREFERS
   - ``X works at Y``  → WORKS_AT
   - ``X is part of Y``→ PART_OF
   - ``X depends on Y``→ DEPENDS_ON
- Any other ``X <verb> Y`` produces a generic ``RELATED_TO`` edge.

When an LLM is configured (``llm`` parameter non-None on the pipeline) the
baseline output is replaced by the LLM's structured response. The LLM is
deferred to v0.2 implementation; the baseline keeps full functionality offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_VERB_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bowns\b", re.I), "OWNS"),
    (re.compile(r"\bprefers?\b", re.I), "PREFERS"),
    (re.compile(r"\bworks (at|for)\b", re.I), "WORKS_AT"),
    (re.compile(r"\bis (a )?part of\b", re.I), "PART_OF"),
    (re.compile(r"\bdepends on\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\bcauses\b", re.I), "CAUSES"),
    (re.compile(r"\breports to\b", re.I), "REPORTS_TO"),
)


@dataclass(frozen=True)
class EntityDraft:
    name: str
    type_: str = "CONCEPT"


@dataclass(frozen=True)
class TripleDraft:
    subject: EntityDraft
    predicate: str
    object: EntityDraft  # noqa: A003 — matches the SPO term


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _proper_nouns(sentence: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(sentence):
        # Skip sentence-initial 'The', 'A', 'An', 'This'
        if tok.lower() in {"the", "a", "an", "this", "that", "these", "those"}:
            continue
        # Treat any capitalized OR all-caps token of length >=2 as a candidate.
        if (tok[0].isupper() and len(tok) >= 2) or (tok.isupper() and len(tok) >= 2):
            out.append(tok)
    return out


def extract(text: str) -> tuple[list[EntityDraft], list[TripleDraft]]:
    entities: dict[str, EntityDraft] = {}
    triples: list[TripleDraft] = []

    for sentence in _split_sentences(text):
        nouns = _proper_nouns(sentence)
        for n in nouns:
            entities.setdefault(n.lower(), EntityDraft(name=n))

        if len(nouns) < 2:
            continue
        predicate: str | None = None
        for pat, rel in _VERB_RULES:
            if pat.search(sentence):
                predicate = rel
                break
        if predicate is None:
            predicate = "RELATED_TO"

        # Pair the first noun with each subsequent noun in the sentence.
        subj = entities[nouns[0].lower()]
        for obj_name in nouns[1:]:
            obj = entities[obj_name.lower()]
            triples.append(TripleDraft(subject=subj, predicate=predicate, object=obj))

    return list(entities.values()), triples


def triple_fingerprint(t: TripleDraft) -> str:
    """Stable fingerprint for Jaccard-style dedup.

    Uses canonical (lower-cased) names + predicate.
    """
    return f"{t.subject.name.lower()}|{t.predicate}|{t.object.name.lower()}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
