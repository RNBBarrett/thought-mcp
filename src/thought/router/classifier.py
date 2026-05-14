"""Rule-based query classifier.

Reads ``rules.yaml`` and tags every incoming query with one of four classes:
VIBE, FACT, CHANGE, HYBRID. The classifier is deterministic and zero-latency
(<1ms in the benchmark) — it does not require an LLM call. Users can extend or
replace the rule set by passing their own ``rules`` mapping.

If a query trips signals across multiple classes above threshold, the
classifier returns HYBRID and includes the per-class signal counts so the
dispatcher can fan out properly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..models import QueryClass

_RULES_PATH = Path(__file__).parent / "rules.yaml"


@dataclass
class _Compiled:
    vibe: list[re.Pattern[str]]
    fact: list[re.Pattern[str]]
    change: list[re.Pattern[str]]
    code: list[re.Pattern[str]]


class RuleBasedClassifier:
    def __init__(self, rules: dict[str, list[str]]) -> None:
        self._rules = _Compiled(
            vibe=[re.compile(p, re.I) for p in rules.get("vibe", [])],
            fact=[re.compile(p, re.I) for p in rules.get("fact", [])],
            change=[re.compile(p, re.I) for p in rules.get("change", [])],
            code=[re.compile(p, re.I) for p in rules.get("code", [])],
        )

    @classmethod
    def with_defaults(cls) -> RuleBasedClassifier:
        rules = yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8"))
        return cls(rules)

    def classify(self, query: str) -> tuple[QueryClass, dict[str, int]]:
        counts = {
            "vibe": sum(1 for p in self._rules.vibe if p.search(query)),
            "fact": sum(1 for p in self._rules.fact if p.search(query)),
            "change": sum(1 for p in self._rules.change if p.search(query)),
            "code": sum(1 for p in self._rules.code if p.search(query)),
        }
        nonzero = [(name, c) for name, c in counts.items() if c > 0]
        if not nonzero:
            # Default: treat undecorated queries as VIBE (semantic search).
            return QueryClass.VIBE, counts
        # Code + change combine into HYBRID (e.g. "what calls X since v1.0").
        # Code on its own is its own class — distinct from FACT because the
        # dispatch uses CodeLayer's call-graph helpers, not generic PageRank
        # seeded by NER hits.
        if counts["code"] > 0 and counts["change"] > 0:
            return QueryClass.HYBRID, counts
        if len(nonzero) >= 2 and counts["code"] == 0:
            return QueryClass.HYBRID, counts
        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        return {
            "vibe": QueryClass.VIBE,
            "fact": QueryClass.FACT,
            "change": QueryClass.CHANGE,
            "code": QueryClass.CODE,
        }[winner], counts
