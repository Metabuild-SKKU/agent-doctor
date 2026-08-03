"""Build an Eval/Optimize ablation report from existing logs.

The script is intentionally log-only: it never reruns probes, retrieval, or LLM
judges. It helps the person who runs the expensive KorQuAD experiment turn a
single large log into tables for action-centered Optimize discussions.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_OUTPUT_DIR = Path("output/eval_ablation")
KNOWN_LABELS = {
    "bad_gold_answer",
    "chunking_context_mismatch",
    "chunking_underchunking",
    "context_noise_interference",
    "generation_hallucination",
    "generation_misinterpretation",
    "generation_partial_answer",
    "retrieval_low_rank",
    "retrieval_missing_gold",
    "retrieval_rerank_candidate_miss",
    "retrieval_reranker_demotion",
    "retrieval_semantic_mismatch",
}
PRESCRIPTION_ACTIONS = {
    "context_compression": "context_compression:enable",
    "checklist_review_step": "answer_checklist_review:enable",
    "completeness_prompt": "generation.completeness_mode:enable",
    "decrease_chunk_size": "chunk_size:decrease",
    "decrease_top_k": "top_k:decrease",
    "dynamic_top_k": "top_k:increase",
    "enable_adaptive_retrieval": "adaptive_retrieval:enable",
    "enable_bridge_entity_verifier": "bridge_entity_verifier:enable",
    "enable_calculation_check": "calculation_check:enable",
    "enable_hybrid": "use_hybrid:enable",
    "enable_mmr": "mmr:enable",
    "enable_noise_filter": "noise_filter:enable",
    "enable_query_decomposition": "query_rewrite:decompose",
    "enable_reranker": "use_reranker:enable",
    "expand_bridge_entity_query": "bridge_entity_expansion:enable",
    "expand_query": "query_rewrite:expand",
    "force_hop_evidence_binding": "answer_format:cot_chained",
    "increase_chunk_overlap": "chunk_overlap:increase",
    "increase_chunk_size": "chunk_size:increase",
    "increase_top_k": "top_k:increase",
    "llm_verification_pass": "verifier_on:enable",
    "lower_temperature": "generation.temperature:decrease",
    "relax_reranker_threshold": "reranker_threshold:decrease",
    "reorder_context_edges": "context_ordering:most_relevant_edges",
    "require_citation": "generation.require_citation:enable",
    "require_numeric_citation": "numeric_citation_required:enable",
    "restate_question": "generation.restate_question:enable",
    "shrink_chunk_size": "chunk_size:decrease",
    "strengthen_abstention": "generation.abstention_strict:enable",
    "strict_conflict_prompt": "conflict_resolution_prompt:prefer_high_confidence_evidence",
    "swap_embedding_model": "embedding_model:change",
    "swap_reranker_model": "reranker_model:change",
    "switch_to_markdown_recursive": "chunk_strategy:markdown_recursive",
    "switch_to_recursive_sentence": "chunk_strategy:recursive_sentence",
    "tighten_reranker_threshold": "reranker_threshold:increase",
    "upgrade_generation_model": "generation.model:upgrade",
    "widen_rerank_candidates": "rerank_candidates:increase",
}

SCORE_LINE_RE = re.compile(
    r"(?P<composite>\d+(?:\.\d+)?)/100.*?"
    r"\((?:[^0-9-]*(?P<quality>-?\d+(?:\.\d+)?))\s*/\s*"
    r"(?:[^0-9-]*(?P<reliability>-?\d+(?:\.\d+)?))\).*?"
    r"overall\s*(?P<overall>-?\d+(?:\.\d+)?).*?"
    r"pass\s*(?P<passed>True|False|true|false|[^\s]+)"
)
CONFIG_RE = re.compile(r"config:\s*(?P<config>.+)")
ACTION_RE = re.compile(r"id=(?P<id>[^,\s]+),\s*label=(?P<label>[^,\s]+)")
SELECTED_LABEL_RE = re.compile(r"(?:label|선택한 라벨):\s*(?P<label>[^,\s]+)")
VERDICT_RE = re.compile(
    r"(?P<decision>KEEP|ROLLBACK).*?"
    r"prescription=(?P<prescription>[^,\s]+).*?"
    r"before=(?P<before>-?\d+(?:\.\d+)?),\s*after=(?P<after>-?\d+(?:\.\d+)?)"
)
OPTIMIZE_EVAL_RE = re.compile(
    r"\[Optimize\].*?(?:Eval 결과|Eval result):.*?"
    r"overall=(?P<overall>-?\d+(?:\.\d+)?),\s*"
    r"composite=(?P<composite>-?\d+(?:\.\d+)?),\s*"
    r"pass=(?P<passed>True|False|true|false)"
)
FINAL_GATE_RE = re.compile(r"gate_pass\s*=\s*(?P<passed>True|False|true|false)")


@dataclass
class EvalSnapshot:
    index: int
    composite: float | None = None
    quality: float | None = None
    reliability: float | None = None
    overall: float | None = None
    passed: bool | None = None
    probes: int | None = None
    failures: int | None = None
    label_distribution: dict[str, float] = field(default_factory=dict)
    type_distribution: dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizeAction:
    eval_after: int
    label: str = "-"
    prescription: str = "-"
    before_config: str = ""
    after_config: str = ""
    verdict: str = ""
    verdict_before: float | None = None
    verdict_after: float | None = None

    @property
    def canonical_action(self) -> str:
        if self.prescription in PRESCRIPTION_ACTIONS:
            return PRESCRIPTION_ACTIONS[self.prescription]
        before_items = _parse_config_assignments(self.before_config)
        after_items = _parse_config_assignments(self.after_config)
        after_by_key = dict(after_items)
        for key, before_value in before_items:
            if key in after_by_key:
                direction = _direction(before_value, after_by_key[key])
                return f"{key}:{direction}"
        if after_items:
            return f"{after_items[0][0]}:set"
        return self.prescription


@dataclass
class ParsedLog:
    source: str
    evals: list[EvalSnapshot] = field(default_factory=list)
    actions: list[OptimizeAction] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


def parse_log_text(text: str, source: str = "<memory>") -> ParsedLog:
    parsed = ParsedLog(source=source)
    current_eval: EvalSnapshot | None = None
    current_action: OptimizeAction | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        _update_global_counts(parsed.counts, line)

        optimize_eval = OPTIMIZE_EVAL_RE.search(line)
        if optimize_eval and parsed.evals:
            _update_eval_gate(
                parsed.evals,
                composite=_to_float(optimize_eval.group("composite")),
                passed=_parse_pass(optimize_eval.group("passed")),
            )
            continue

        final_gate = FINAL_GATE_RE.search(line)
        if final_gate and parsed.evals:
            parsed.evals[-1].passed = _parse_pass(final_gate.group("passed"))
            continue

        score = SCORE_LINE_RE.search(line)
        if score:
            overall = _to_float(score.group("overall"))
            current_eval = EvalSnapshot(
                index=len(parsed.evals) + 1,
                composite=_to_float(score.group("composite")),
                quality=_to_float(score.group("quality")),
                reliability=_to_float(score.group("reliability")),
                overall=overall,
                passed=_parse_pass(score.group("passed")),
            )
            parsed.evals.append(current_eval)
            continue

        distribution = _distribution_from_line(line)
        if distribution and current_eval is not None:
            if _looks_like_label_distribution(distribution):
                current_eval.label_distribution = distribution
            else:
                current_eval.type_distribution = distribution
            continue

        action = ACTION_RE.search(line)
        if action:
            current_action = OptimizeAction(
                eval_after=len(parsed.evals),
                label=action.group("label"),
                prescription=action.group("id"),
            )
            parsed.actions.append(current_action)
            continue

        verdict = VERDICT_RE.search(line)
        if verdict:
            current_action = _attach_verdict(
                parsed,
                prescription=verdict.group("prescription"),
                decision=verdict.group("decision"),
                before=_to_float(verdict.group("before")),
                after=_to_float(verdict.group("after")),
            )
            continue

        if current_action is None:
            continue

        selected_label = SELECTED_LABEL_RE.search(line)
        if selected_label and current_action.label == "-":
            current_action.label = selected_label.group("label").strip()
            continue

        config = CONFIG_RE.search(line)
        if config:
            value = config.group("config").strip()
            if not current_action.before_config:
                current_action.before_config = value
            elif not current_action.after_config:
                current_action.after_config = value

    return parsed


def build_markdown(logs: Iterable[ParsedLog]) -> str:
    logs = list(logs)
    lines = [
        "# Eval Ablation Report",
        "",
        "This report was built from existing logs only. It is meant to help choose",
        "the next Optimize action without rerunning a large KorQuAD experiment.",
        "",
        "## Sources",
    ]
    for log in logs:
        lines.append(f"- `{log.source}`")

    lines.extend(["", "## Eval Timeline", ""])
    lines.append("| log | eval | score | quality | reliability | overall | gate pass | top labels |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for log in logs:
        for snapshot in log.evals:
            lines.append(
                f"| {Path(log.source).name} | {snapshot.index} | {_fmt(snapshot.composite)} | "
                f"{_fmt(snapshot.quality)} | {_fmt(snapshot.reliability)} | "
                f"{_fmt(snapshot.overall)} | {_fmt_bool(snapshot.passed)} | "
                f"{_top_distribution(snapshot.label_distribution)} |"
            )

    lines.extend(["", "## Score Changes", ""])
    lines.extend(_score_change_lines(logs))

    lines.extend(["", "## Retrieval Bottleneck Signals", ""])
    lines.append(
        "| log | retrieval_low_rank | recall@k=0.00 | missed_gold_ranks | reranker=applied | search_fallback=true |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for log in logs:
        lines.append(
            f"| {Path(log.source).name} | {log.counts['retrieval_low_rank']} | "
            f"{log.counts['recall_zero']} | {log.counts['missed_gold_ranks']} | "
            f"{log.counts['reranker_applied']} | {log.counts['search_fallback_true']} |"
        )

    lines.extend(["", "## Action-Centered Summary", ""])
    lines.append(
        "| action | tries | keep | rollback | supporting labels | best observed delta |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | ---: |")
    action_rows = _action_summary(logs)
    if action_rows:
        for row in action_rows:
            lines.append(
                f"| {row['action']} | {row['tries']} | {row['keep']} | {row['rollback']} | "
                f"{row['labels']} | {row['best_delta']} |"
            )
    else:
        lines.append("| - | 0 | 0 | 0 | No Optimize action lines found | - |")

    lines.extend(["", "## Optimize Timeline", ""])
    lines.append("| log | after eval | label | prescription | canonical action | before | after | verdict |")
    lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
    action_count = 0
    for log in logs:
        for action in log.actions:
            action_count += 1
            verdict = _verdict_text(action)
            lines.append(
                f"| {Path(log.source).name} | {action.eval_after} | {action.label} | "
                f"{action.prescription} | {action.canonical_action} | "
                f"{action.before_config or '-'} | {action.after_config or '-'} | {verdict or '-'} |"
            )
    if action_count == 0:
        lines.append("| - | - | - | - | - | - | - | No Optimize action lines found |")

    lines.extend(["", "## Suggested Discussion Points", ""])
    lines.extend(_suggestions(logs))
    lines.append("")
    return "\n".join(lines)


def write_csv(logs: Iterable[ParsedLog], output: Path) -> None:
    logs = list(logs)
    output.parent.mkdir(parents=True, exist_ok=True)
    label_keys = sorted({
        key
        for log in logs
        for snapshot in log.evals
        for key in snapshot.label_distribution
    })
    fields = [
        "log_file",
        "eval_index",
        "score",
        "quality",
        "reliability",
        "overall",
        "gate_pass",
        *[f"{key}_weight" for key in label_keys],
        "top_labels",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for log in logs:
            for snapshot in log.evals:
                labels = snapshot.label_distribution
                row = {
                    "log_file": log.source,
                    "eval_index": snapshot.index,
                    "score": _fmt(snapshot.composite),
                    "quality": _fmt(snapshot.quality),
                    "reliability": _fmt(snapshot.reliability),
                    "overall": _fmt(snapshot.overall),
                    "gate_pass": _fmt_bool(snapshot.passed),
                    "top_labels": _top_distribution(labels, limit=5),
                }
                for key in label_keys:
                    row[f"{key}_weight"] = labels.get(key, 0)
                writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logs = []
    for path in args.logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        logs.append(parse_log_text(text, source=str(path)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = args.output_dir / args.markdown_name
    csv_path = args.output_dir / args.csv_name
    markdown_path.write_text(build_markdown(logs), encoding="utf-8")
    write_csv(logs, csv_path)

    print(f"wrote_markdown={markdown_path.resolve()}")
    print(f"wrote_csv={csv_path.resolve()}")
    print(f"evals={sum(len(log.evals) for log in logs)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Log files to summarize")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--markdown-name", default="ablation_report.md")
    parser.add_argument("--csv-name", default="ablation_summary.csv")
    return parser.parse_args(argv)


def _score_change_lines(logs: list[ParsedLog]) -> list[str]:
    lines = []
    for log in logs:
        evals = [item for item in log.evals if item.composite is not None]
        for before, after in zip(evals, evals[1:]):
            delta = after.composite - before.composite
            direction = "up" if delta > 0 else "down" if delta < 0 else "same"
            lines.append(
                f"- `{Path(log.source).name}` eval {before.index}->{after.index}: "
                f"{before.composite:.0f} -> {after.composite:.0f} ({direction} {delta:+.0f})"
            )
    return lines or ["- Only one Eval snapshot was found, so no score transition was available."]


def _action_summary(logs: list[ParsedLog]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, object]] = defaultdict(
        lambda: {"tries": 0, "keep": 0, "rollback": 0, "labels": Counter(), "deltas": []}
    )
    for log in logs:
        for action in log.actions:
            key = action.canonical_action
            row = grouped[key]
            row["tries"] = int(row["tries"]) + 1
            if action.label and action.label != "-":
                row["labels"][action.label] += 1  # type: ignore[index]
            if action.verdict == "KEEP":
                row["keep"] = int(row["keep"]) + 1
            elif action.verdict == "ROLLBACK":
                row["rollback"] = int(row["rollback"]) + 1
            if action.verdict_before is not None and action.verdict_after is not None:
                row["deltas"].append(action.verdict_after - action.verdict_before)  # type: ignore[index]

    rows = []
    for action, row in grouped.items():
        deltas = row["deltas"]  # type: ignore[assignment]
        rows.append(
            {
                "action": action,
                "tries": str(row["tries"]),
                "keep": str(row["keep"]),
                "rollback": str(row["rollback"]),
                "labels": _top_distribution(dict(row["labels"]), limit=4),  # type: ignore[arg-type]
                "best_delta": _fmt(max(deltas)) if deltas else "-",
            }
        )
    rows.sort(key=lambda item: (-int(item["tries"]), item["action"]))
    return rows


def _update_eval_gate(
    evals: list[EvalSnapshot],
    composite: float | None,
    passed: bool | None,
) -> None:
    if passed is None:
        return
    evals[-1].passed = passed


def _attach_verdict(
    parsed: ParsedLog,
    prescription: str,
    decision: str,
    before: float | None,
    after: float | None,
) -> OptimizeAction:
    for action in reversed(parsed.actions):
        if action.prescription == prescription and not action.verdict:
            action.verdict = decision
            action.verdict_before = before
            action.verdict_after = after
            return action

    action = OptimizeAction(
        eval_after=len(parsed.evals),
        prescription=prescription,
        verdict=decision,
        verdict_before=before,
        verdict_after=after,
    )
    parsed.actions.append(action)
    return action


def _distribution_from_line(line: str) -> dict[str, float]:
    if "{" not in line or "}" not in line:
        return {}
    raw = line[line.find("{") : line.rfind("}") + 1]
    try:
        data = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    result = {}
    for key, value in data.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _looks_like_label_distribution(data: dict[str, float]) -> bool:
    return any(key in KNOWN_LABELS for key in data)


def _parse_config_assignment(raw: str) -> tuple[str, str] | None:
    items = _parse_config_assignments(raw)
    return items[0] if items else None


def _parse_config_assignments(raw: str) -> list[tuple[str, str]]:
    if not raw or raw == "-":
        return []
    if "=" not in raw:
        return []

    assignments: list[tuple[str, str]] = []
    for part in _split_config_assignments(raw):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = _clean_config_token(key)
        value = _clean_config_token(value)
        if key:
            assignments.append((key, value))
    return assignments


def _split_config_assignments(raw: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(raw):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in "([{":
            depth += 1
            continue
        if char in ")]}" and depth > 0:
            depth -= 1
            continue
        if char == "," and depth == 0:
            parts.append(raw[start:index].strip())
            start = index + 1
    parts.append(raw[start:].strip())
    return [part for part in parts if part]


def _clean_config_token(value: str) -> str:
    return value.strip().strip("{}").strip().strip("'\"")


def _direction(before: str, after: str) -> str:
    before_num = _to_float(before)
    after_num = _to_float(after)
    if before_num is not None and after_num is not None:
        if after_num > before_num:
            return "increase"
        if after_num < before_num:
            return "decrease"
        return "same"
    if before.lower() in {"false", "none", "0"} and after.lower() in {"true", "1"}:
        return "enable"
    if before.lower() in {"true", "1"} and after.lower() in {"false", "0"}:
        return "disable"
    return "change"


def _update_global_counts(counter: Counter[str], line: str) -> None:
    if _is_distribution_line(line):
        return
    patterns = {
        "retrieval_low_rank": "retrieval_low_rank",
        "context_noise_interference": "context_noise_interference",
        "bad_gold_answer": "bad_gold_answer",
        "generation_misinterpretation": "generation_misinterpretation",
        "missed_gold_ranks": "missed_gold_ranks",
        "recall_zero": "recall@k=0.00",
        "recall_half": "recall@k=0.50",
        "recall_one": "recall@k=1.00",
        "reranker_applied": "reranker=applied",
        "search_fallback_true": "search_fallback=true",
    }
    for key, needle in patterns.items():
        if needle in line:
            counter[key] += line.count(needle)


def _is_distribution_line(line: str) -> bool:
    return any(
        marker in line
        for marker in (
            "label distribution",
            "type distribution",
            "라벨분포",
            "타입분포",
        )
    )


def _suggestions(logs: list[ParsedLog]) -> list[str]:
    total = Counter()
    for log in logs:
        total.update(log.counts)

    suggestions = []
    if total["retrieval_low_rank"] or total["missed_gold_ranks"]:
        suggestions.append(
            "- If retrieval_low_rank dominates, compare action rows for top_k, rerank_candidates, and reranker.enabled before changing defaults."
        )
    if total["context_noise_interference"]:
        suggestions.append(
            "- If context_noise_interference remains high, compare context_compression/MMR with recall@k changes to avoid hiding gold context."
        )
    if total["bad_gold_answer"]:
        suggestions.append(
            "- If bad_gold_answer is high, separate probe/gold calibration from retrieval tuning before blaming top_k or reranker."
        )
    suggestions.append(
        "- For Issue #67, use the Action-Centered Summary as evidence for which canonical actions are repeatedly supported or rolled back."
    )
    return suggestions


def _top_distribution(data: dict[str, float], limit: int = 3) -> str:
    if not data:
        return "-"
    items = sorted(data.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{key}={value:g}" for key, value in items)


def _verdict_text(action: OptimizeAction) -> str:
    if not action.verdict:
        return ""
    if action.verdict_before is None or action.verdict_after is None:
        return action.verdict
    return f"{action.verdict} {action.verdict_before:.2f}->{action.verdict_after:.2f}"


def _to_float(value: str | None) -> float | None:
    if value in {None, "-"}:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "true" if value else "false"


def _parse_pass(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "pass", "ok"} or value == "\u2713":
        return True
    if lowered in {"false", "fail"} or value == "\u2717":
        return False
    return None


if __name__ == "__main__":
    main()
