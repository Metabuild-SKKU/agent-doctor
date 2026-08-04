# Eval Ablation Report

This helper is for the person who runs the expensive KorQuAD or large-corpus
experiment. It turns existing logs into a small Markdown/CSV report without
rerunning probes, retrieval, or LLM judges.

## Why

Issue #67 discusses moving Optimize from label-first selection to
action-centered selection. To make that discussion concrete, we need evidence
for which config actions were actually tried, kept, or rolled back.

This report summarizes:

- score changes across Eval visits
- dominant weighted labels
- retrieval bottleneck signals such as `recall@k=0.00` and `missed_gold_ranks`
- Optimize prescriptions and their canonical action keys
- KEEP/ROLLBACK evidence when the log contains it

## Usage

Run the expensive corpus once, then pass the saved log file to the tool.

```bash
python tools/eval_ablation_report.py output/logs/web_run_YYYYMMDD_HHMMSS.log
```

To compare several logs:

```bash
python tools/eval_ablation_report.py output/logs/run_a.log output/logs/run_b.log --output-dir output/eval_ablation
```

The command writes:

- `output/eval_ablation/ablation_report.md`
- `output/eval_ablation/ablation_summary.csv`

## How To Read It

Use `Eval Timeline` to see whether the run improved or regressed.

Use `Retrieval Bottleneck Signals` to check whether `retrieval_low_rank`,
`recall@k=0.00`, or `missed_gold_ranks` remains the main bottleneck.

Use `Action-Centered Summary` for Issue #67. It groups prescriptions by their
actual config action, such as:

- `top_k:increase`
- `rerank_candidates:increase`
- `use_reranker:enable`
- `context_compression:enable`

This makes it easier to discuss which action deserves the next experiment
instead of deciding from label names alone.

## Notes

Generated reports are experiment artifacts and should not be committed unless
the team explicitly wants a frozen benchmark note.
