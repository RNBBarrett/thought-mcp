# Comparison harness — measured results

Workload: 200 recall ops (50 each of VIBE / FACT / CHANGE / HYBRID) over a deterministic knowledge base of 24 ingest steps with at least two structured contradictions inserted via PREFERS predicate.

Embedder: deterministic hashed bag-of-words (same model for all three systems so the gap reflects architecture, not embedding quality).

## Recall@10 by query class

| System | VIBE | FACT | CHANGE | HYBRID | overall |
|---|---|---|---|---|---|
| THOUGHT | 100.0% | 100.0% | 68.0% | 66.0% | 83.5% |
| OB1 | 100.0% | 100.0% | 32.0% | 100.0% | 83.0% |
| Karpathy wiki | 100.0% | 30.0% | 0.0% | 100.0% | 57.5% |

## Latency (recall path, ms)

| System | p50 | p95 |
|---|---|---|
| THOUGHT | 0.00 | 2.11 |
| OB1 | 0.16 | 0.24 |
| Karpathy wiki | 0.08 | 0.09 |

## Structural capabilities

| Capability | THOUGHT | OB1 | Karpathy wiki |
|---|---|---|---|
| bi-temporal as_of | ✅ | ✗ | ✗ |
| source-grounded confidence class | ✅ | ✗ | ✗ |
| contradiction as typed edge | ✅ | ✗ | ✗ |
| multi-user scope isolation | ✅ | partial (RLS) | ✗ |
| append-only audit log | ✅ | ✗ | ✗ |
| Personalized PageRank retrieval | ✅ | ✗ | ✗ |
| Ebbinghaus decay scoring | ✅ | ✗ | ✗ |
| CRAG-style low-confidence flag | ✅ | ✗ | ✗ |
| Matryoshka 2-pass ANN | ✅ | ✗ | ✗ |
| Anthropic Contextual Retrieval | ✅ | ✗ | ✗ |
| query router (VIBE/FACT/CHANGE) | ✅ | ✗ | ✗ |
| forecasting (TLogic) [v0.2] | planned | ✗ | ✗ |

## Contradictions detected (write-time)

- **THOUGHT**: 2
- **OB1**: 0
- **Karpathy wiki**: 0

## Temporal correctness on CHANGE queries

| System | correct / total | rate |
|---|---|---|
| THOUGHT | 34/50 | 68.0% |
| OB1 | 16/50 | 32.0% |
| Karpathy wiki | 0/50 | 0.0% |
