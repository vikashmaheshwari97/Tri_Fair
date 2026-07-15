# Tri-Fair-GR Version B-full

This patch implements an expensive but practical Version B of Tri-Fair-GR:

- fixed pretrained GNN-RAG retrieval outputs;
- optimized graph-context and prompt policy;
- many candidate policies evaluated with vLLM;
- Pareto selection over answer quality, graph-context cost proxy, and end-to-end unfairness.

This is stronger than prompt-only Version A because the policy changes not only
instruction wording, but also path count, path ordering, graph-to-text
verbalization, answer format, and fairness wording.

It is still not Version C: it does not retrain the GNN. Version C would add GNN
architecture/threshold/training parameters and is a separate, much larger study.

## Typical workflow

Generate a policy grid:

```bash
python -m scripts.graphrag_generate_policy_grid \
  --size compact \
  --out analysis/output/trifair_gr_vb/webqsp/policies.jsonl
```

Materialize policy prompts:

```bash
python -m scripts.graphrag_materialize_policy_prompts \
  --base-predictions ../GNN-RAG/llm/results/KGQA-GNN-RAG-RA/rearev-sbert/RoG-webqsp/RoG/test/results_gen_rule_path_RoG-webqsp_RoG_test_predictions_3_False_jsonl/False/predictions.jsonl \
  --policies analysis/output/trifair_gr_vb/webqsp/policies.jsonl \
  --out-dir analysis/output/trifair_gr_vb/webqsp/prompts \
  --limit 0
```

Run one policy locally on a GPU node:

```bash
python -m scripts.graphrag_run_policy_vllm \
  --prompts analysis/output/trifair_gr_vb/webqsp/prompts/<POLICY>/prompts.jsonl \
  --out analysis/output/trifair_gr_vb/webqsp/predictions/<POLICY>/predictions.jsonl \
  --model-path ../models/Qwen3-30B \
  --max-output-tokens 32 \
  --batch-size 32 \
  --trust-remote-code
```

Run many policies as a Slurm array by setting `PROMPTS_ROOT` and `OUTPUT_ROOT`.

Evaluate policies:

```bash
python -m scripts.graphrag_evaluate_policy_grid \
  --predictions-root analysis/output/trifair_gr_vb/webqsp/predictions \
  --metadata data/graphrag/webqsp_west_proxy_metadata.csv \
  --group-col protected_group \
  --out-dir analysis/output/trifair_gr_vb/webqsp/eval
```

Select Pareto policies:

```bash
python -m scripts.graphrag_select_pareto_policies \
  --summary analysis/output/trifair_gr_vb/webqsp/eval/policy_eval_summary.csv \
  --out analysis/output/trifair_gr_vb/webqsp/eval/policy_pareto.csv
```
