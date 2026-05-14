# Ablation study — marginal contribution of Tier A techniques

Each row turns OFF one frontier technique to measure its marginal contribution to THOUGHT's accuracy on the 200-op comparison workload. Higher overall, FACT, CHANGE, HYBRID = better.

| Variant | Overall | FACT | CHANGE | HYBRID | Contradictions detected |
|---|---|---|---|---|---|
| Full v0.1 (all Tier A) | 83.5% | 100.0% | 68.0% | 66.0% | 2 |
| − HippoRAG bidirectional PPR | 66.0% | 30.0% | 68.0% | 66.0% | 2 |
| − Bi-temporal edge retirement | 75.0% | 100.0% | 34.0% | 66.0% | 2 |
| − Query router (force VIBE) | 65.5% | 30.0% | 32.0% | 100.0% | 2 |
