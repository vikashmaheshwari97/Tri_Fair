# Tri-Fair × GNN-RAG extension plan

This extension is intentionally separate from the current BBQ / Bias in Bios /
Civil Comments 5M runs. Those runs should remain unchanged for comparability.

## Why a separate extension?

The current Tri-Fair datasets are classification/fairness benchmarks. They do
not include an explicit knowledge graph, entity linking, GNN retrieval, candidate
answer nodes, or reasoning paths. GNN-RAG is a KGQA pipeline, so a fair
comparison requires KGQA-style datasets such as WebQSP, CWQ, or MetaQA, plus
group labels or counterfactual group annotations.

## Recommended first experiment

Use precomputed GNN-RAG retrieval outputs and treat the graph context as part of
the prompt input. Tri-Fair optimizes prompt instruction, graph-to-text
verbalization wording, path ordering, maximum number of paths, and optional
fairness instructions.

Do not train or modify the GNN during the first experiment. Keep GNN retrieval
fixed and optimize only prompt/verbalization choices. This gives a clean
Tri-Fair extension without exploding compute cost.

## Baselines

1. No-RAG LLM.
2. Text/long-context RAG.
3. GNN-RAG with original fixed prompt.
4. GNN-RAG + MO-CAPO-style quality/cost optimization.
5. GNN-RAG + Tri-Fair quality/cost/fairness optimization.
6. Optional: GNN-RAG + retrieval augmentation if available.

## Objectives

Quality: KGQA Hit / F1 / H@1.

Cost: input tokens, output tokens, number of paths/triples, optional GNN latency.

Unfairness: start with end-to-end answer unfairness; add retrieval unfairness if
candidate-answer metadata is available.

Recommended first unfairness:
`U_E2E = lambda * U_retrieval + (1 - lambda) * U_answer`.

Keep entity-linking, subgraph, path, and verbalization fairness as diagnostics
until the data supports reliable objective-level measurement.
