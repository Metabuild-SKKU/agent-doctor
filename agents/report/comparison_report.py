"""Build a before/after performance report for the RAG pipeline.

The report compares the first optimization baseline with the final Eval result.
If no optimization was attempted, it still writes a final Eval summary so every
pipeline run can leave one report artifact.
"""
from __future__ import annotations

import dataclasses
import html as html_lib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from core.schema import DiagnosticReport
from core.state import AgentDoctorState

DEFAULT_OUTPUT_DIR = Path("output") / "reports"


def save_pipeline_report(
    state: AgentDoctorState,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Write markdown/json reports and return their paths."""
    payload = build_comparison_payload(state)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"rag_pipeline_report_{stamp}"
    markdown_path = out_dir / f"{base}.md"
    json_path = out_dir / f"{base}.json"
    html_path = out_dir / f"{base}.html"

    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "markdown": str(markdown_path),
        "json": str(json_path),
        "html": str(html_path),
        "comparison_available": payload["comparison"]["available"],
    }


def build_comparison_payload(state: AgentDoctorState) -> dict[str, Any]:
    """Create a JSON-ready before/after report payload from pipeline state."""
    final_report = state.report
    trial = _baseline_trial(state.optimization_history)

    before = _before_snapshot(trial)
    after = _after_snapshot(state, trial)
    metric_rows = _metric_rows(before.get("metrics", {}), after.get("metrics", {}))
    config_rows = _config_rows(before.get("config", {}), after.get("config", {}))

    return {
        "report_type": "rag_pipeline_before_after",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": {
            "status": state.status,
            "iteration": state.iteration,
            "max_iterations": state.max_iterations,
            "documents": len(state.documents),
            "chunks": len(state.chunks),
            "probes": len(state.probes),
        },
        "comparison": {
            "available": trial is not None,
            "basis": (
                "first optimization trial vs final eval"
                if trial is not None
                else "final eval only; optimization was not attempted"
            ),
            "before": before,
            "after": after,
            "metric_rows": metric_rows,
            "config_rows": config_rows,
        },
        "optimization": {
            "trials": [_trial_summary(item) for item in state.optimization_history],
            "latest_report": _safe_dataclass(state.optimization_report),
        },
        "final_eval": _report_snapshot(final_report),
        "interpretation": _interpret(metric_rows, trial is not None),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the comparison payload as a human-readable markdown report."""
    pipeline = payload["pipeline"]
    comparison = payload["comparison"]
    final_eval = payload["final_eval"]
    lines = [
        "# RAG Pipeline Before/After Report",
        "",
        "## 1. 실행 요약",
        f"- 생성 시각: {payload['created_at']}",
        f"- 파이프라인 상태: {pipeline['status']}",
        f"- 반복: {pipeline['iteration']} / {pipeline['max_iterations']}",
        f"- 문서 수: {pipeline['documents']}",
        f"- 청크 수: {pipeline['chunks']}",
        f"- Probe 수: {pipeline['probes']}",
        f"- 비교 기준: {comparison['basis']}",
        "",
        "## 2. 성능 비교",
    ]

    metric_rows = comparison["metric_rows"]
    if metric_rows:
        lines.extend(_markdown_table(
            ["Metric", "Before", "After", "Change"],
            [
                [
                    row["metric"],
                    _fmt(row["before"]),
                    _fmt(row["after"]),
                    _fmt_delta(row["delta"]),
                ]
                for row in metric_rows
            ],
        ))
    else:
        lines.append("- 비교할 metric이 없습니다.")

    lines.extend(["", "## 3. 설정 비교"])
    config_rows = comparison["config_rows"]
    if config_rows:
        lines.extend(_markdown_table(
            ["Config", "Before", "After"],
            [
                [row["key"], _fmt(row["before"]), _fmt(row["after"])]
                for row in config_rows
            ],
        ))
    else:
        lines.append("- 변경된 설정이 없습니다.")

    lines.extend(["", "## 4. 최적화 시도 이력"])
    trials = payload["optimization"]["trials"]
    if trials:
        lines.extend(_markdown_table(
            ["Trial", "Status", "Prescription", "Before", "After", "Reason"],
            [
                [
                    trial["trial_id"],
                    trial["status"],
                    trial.get("selected_prescription_id") or "-",
                    _fmt(trial.get("before_score")),
                    _fmt(trial.get("after_score")),
                    trial.get("rollback_reason") or trial.get("reason") or "-",
                ]
                for trial in trials
            ],
        ))
    else:
        lines.append("- 최적화 시도 이력이 없습니다.")

    lines.extend([
        "",
        "## 5. 최종 Eval 요약",
        f"- report_id: {final_eval.get('report_id') or '-'}",
        f"- overall_score: {_fmt(final_eval.get('overall_score'))}",
        f"- pass_threshold: {final_eval.get('pass_threshold')}",
        f"- oracle_accuracy: {_fmt(final_eval.get('oracle_accuracy'))}",
        f"- findings: {final_eval.get('findings_count', 0)}",
        "",
        "## 6. 해석",
    ])
    lines.extend(f"- {item}" for item in payload["interpretation"])
    lines.append("")
    return "\n".join(lines)


def render_html(payload: dict[str, Any]) -> str:
    """Render the comparison payload as a standalone browser report."""
    pipeline = payload["pipeline"]
    comparison = payload["comparison"]
    final_eval = payload["final_eval"]
    metric_rows = comparison["metric_rows"]
    config_rows = comparison["config_rows"]
    trials = payload["optimization"]["trials"]

    metric_cards = _html_metric_cards(metric_rows)
    metric_table = _html_table(
        ["Metric", "Before", "After", "Change"],
        [
            [
                row["metric"],
                _fmt(row["before"]),
                _fmt(row["after"]),
                _fmt_delta(row["delta"]),
            ]
            for row in metric_rows
        ],
    )
    config_table = _html_table(
        ["Config", "Before", "After"],
        [[row["key"], _fmt(row["before"]), _fmt(row["after"])] for row in config_rows],
    )
    trial_table = _html_table(
        ["Trial", "Status", "Prescription", "Before", "After", "Reason"],
        [
            [
                trial["trial_id"],
                trial["status"],
                trial.get("selected_prescription_id") or "-",
                _fmt(trial.get("before_score")),
                _fmt(trial.get("after_score")),
                trial.get("rollback_reason") or trial.get("reason") or "-",
            ]
            for trial in trials
        ],
    )
    interpretation = "\n".join(
        f"<li>{_html_text(item)}</li>" for item in payload["interpretation"]
    )

    if not metric_rows:
        metric_table = '<p class="empty">No comparable metrics were recorded.</p>'
    if not config_rows:
        config_table = '<p class="empty">No changed configuration values were recorded.</p>'
    if not trials:
        trial_table = '<p class="empty">No optimization trials were recorded.</p>'

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Pipeline Before/After Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7fb;
      --surface: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d9dee8;
      --before: #64748b;
      --after: #2563eb;
      --positive: #15803d;
      --negative: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0 48px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 24px;
    }}
    h1, h2 {{ margin: 0; line-height: 1.2; }}
    h1 {{ font-size: 30px; }}
    h2 {{ font-size: 18px; margin-bottom: 14px; }}
    .subtle {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .card, section {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .card {{ padding: 14px 16px; }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 600;
    }}
    .delta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .delta.positive {{ color: var(--positive); }}
    .delta.negative {{ color: var(--negative); }}
    section {{
      padding: 18px;
      margin-top: 16px;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .metric-name {{ font-weight: 600; }}
    .bar-wrap {{
      min-width: 220px;
    }}
    .bar {{
      display: grid;
      gap: 5px;
    }}
    .bar-line {{
      display: grid;
      grid-template-columns: 52px 1fr 70px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .track {{
      height: 8px;
      border-radius: 999px;
      background: #eef2f7;
      overflow: hidden;
    }}
    .fill {{
      display: block;
      height: 100%;
      border-radius: inherit;
    }}
    .fill.before {{ background: var(--before); }}
    .fill.after {{ background: var(--after); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: #eef2ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 600;
    }}
    .empty {{ color: var(--muted); margin: 0; }}
    ul {{ margin: 0; padding-left: 20px; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1120px); padding-top: 20px; }}
      header {{ display: block; }}
      h1 {{ font-size: 24px; margin-bottom: 8px; }}
      section {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <p class="subtle">Generated at {_html_text(payload["created_at"])}</p>
        <h1>RAG Pipeline Before/After Report</h1>
      </div>
      <span class="pill">{_html_text(comparison["basis"])}</span>
    </header>

    <div class="grid">
      <div class="card">
        <div class="label">Pipeline status</div>
        <div class="value">{_html_text(pipeline["status"])}</div>
        <div class="delta">iteration {_html_value(pipeline["iteration"])} / {_html_value(pipeline["max_iterations"])}</div>
      </div>
      <div class="card">
        <div class="label">Documents / Chunks</div>
        <div class="value">{_html_value(pipeline["documents"])} / {_html_value(pipeline["chunks"])}</div>
        <div class="delta">probes {_html_value(pipeline["probes"])}</div>
      </div>
      <div class="card">
        <div class="label">Final overall score</div>
        <div class="value">{_html_value(final_eval.get("overall_score"))}</div>
        <div class="delta">pass threshold {_html_value(final_eval.get("pass_threshold"))}</div>
      </div>
    </div>

    {metric_cards}

    <section>
      <h2>Metric Comparison</h2>
      {metric_table}
    </section>

    <section>
      <h2>Configuration Changes</h2>
      {config_table}
    </section>

    <section>
      <h2>Optimization Trials</h2>
      {trial_table}
    </section>

    <section>
      <h2>Interpretation</h2>
      <ul>{interpretation}</ul>
    </section>
  </main>
</body>
</html>
"""


def _baseline_trial(history: list[Any]) -> Any | None:
    for item in history:
        if getattr(item, "before_config", None):
            return item
    return None


def _before_snapshot(trial: Any | None) -> dict[str, Any]:
    if trial is None:
        return {"config": {}, "metrics": {}}
    before_report = getattr(trial, "metadata", {}).get("before_report")
    metrics = _metrics_from_report(before_report)
    metrics.update(getattr(trial, "before_metrics", {}) or {})
    before_score = getattr(trial, "metadata", {}).get("before_score")
    if before_score is not None:
        metrics["overall_score"] = before_score
    return {
        "config": dict(getattr(trial, "before_config", {}) or {}),
        "metrics": metrics,
        "report": _report_snapshot(before_report),
    }


def _after_snapshot(state: AgentDoctorState, trial: Any | None) -> dict[str, Any]:
    metrics = _metrics_from_report(state.report)
    if not metrics and trial is not None and getattr(trial, "after_metrics", None):
        metrics.update(trial.after_metrics)
    after_score = getattr(trial, "metadata", {}).get("after_score") if trial is not None else None
    if "overall_score" not in metrics and after_score is not None:
        metrics["overall_score"] = after_score

    after_config = (
        state.index_config
        or (getattr(trial, "after_config", None) if trial is not None else {})
    )
    return {
        "config": dict(after_config or {}),
        "metrics": metrics,
        "report": _report_snapshot(state.report),
    }


def _metrics_from_report(report: DiagnosticReport | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = dict(report.ragas_scores or {})
    metrics["overall_score"] = report.overall_score
    metrics["oracle_accuracy"] = report.oracle_accuracy
    metrics["pass_threshold"] = report.pass_threshold
    return {key: value for key, value in metrics.items() if value is not None}


def _metric_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    keys = sorted(set(before) | set(after), key=_metric_sort_key)
    rows = []
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        rows.append({
            "metric": key,
            "before": before_value,
            "after": after_value,
            "delta": _delta(before_value, after_value),
        })
    return rows


def _metric_sort_key(name: str) -> tuple[int, str]:
    priority = {
        "overall_score": 0,
        "mean_recall_at_k": 1,
        "mean_f1": 2,
        "oracle_accuracy": 3,
        "pass_threshold": 4,
    }
    return priority.get(name, 100), name


def _config_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value == after_value:
            continue
        rows.append({"key": key, "before": before_value, "after": after_value})
    return rows


def _trial_summary(item: Any) -> dict[str, Any]:
    metadata = getattr(item, "metadata", {}) or {}
    return {
        "trial_id": _short(getattr(item, "trial_id", "")),
        "request_id": _short(getattr(item, "request_id", "")),
        "iteration": getattr(item, "iteration", None),
        "status": getattr(item, "status", ""),
        "failure_labels": list(getattr(item, "failure_labels", []) or []),
        "optimizer": getattr(item, "optimizer", ""),
        "selected_prescription_id": getattr(item, "selected_prescription_id", None),
        "before_score": metadata.get("before_score"),
        "after_score": metadata.get("after_score"),
        "target_metrics": list(getattr(item, "target_metrics", []) or []),
        "reason": getattr(item, "reason", ""),
        "rollback_reason": getattr(item, "rollback_reason", None),
        "pending": bool(metadata.get("pending")),
    }


def _report_snapshot(report: DiagnosticReport | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        "report_id": report.report_id,
        "iteration": report.iteration,
        "overall_score": report.overall_score,
        "pass_threshold": report.pass_threshold,
        "oracle_accuracy": report.oracle_accuracy,
        "ragas_scores": dict(report.ragas_scores or {}),
        "findings_count": len(report.findings or []),
        "findings_summary": dict(report.findings_summary or {}),
    }


def _safe_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


def _interpret(metric_rows: list[dict[str, Any]], has_comparison: bool) -> list[str]:
    if not has_comparison:
        return ["최적화 시도 이력이 없어 최종 Eval 결과만 요약했습니다."]

    messages = []
    by_name = {row["metric"]: row for row in metric_rows}
    score_delta = by_name.get("overall_score", {}).get("delta")
    if isinstance(score_delta, (int, float)):
        if score_delta > 0:
            messages.append(f"overall_score가 {_fmt_delta(score_delta)} 개선되었습니다.")
        elif score_delta < 0:
            messages.append(f"overall_score가 {_fmt_delta(score_delta)} 하락했습니다.")
        else:
            messages.append("overall_score는 변하지 않았습니다.")

    recall_delta = by_name.get("mean_recall_at_k", {}).get("delta")
    if isinstance(recall_delta, (int, float)) and recall_delta != 0:
        direction = "개선" if recall_delta > 0 else "하락"
        messages.append(f"검색 recall 지표가 {_fmt_delta(recall_delta)} {direction}되었습니다.")

    f1_delta = by_name.get("mean_f1", {}).get("delta")
    if isinstance(f1_delta, (int, float)) and f1_delta != 0:
        direction = "개선" if f1_delta > 0 else "하락"
        messages.append(f"답변 F1 지표가 {_fmt_delta(f1_delta)} {direction}되었습니다.")

    if not messages:
        messages.append("주요 지표 변화가 크지 않습니다. 질문별 상세 로그 확인이 필요합니다.")
    return messages


def _html_metric_cards(metric_rows: list[dict[str, Any]]) -> str:
    if not metric_rows:
        return ""

    preferred = {"overall_score", "mean_recall_at_k", "mean_f1", "oracle_accuracy"}
    selected = [row for row in metric_rows if row["metric"] in preferred][:4]
    if not selected:
        selected = metric_rows[:4]

    cards = []
    for row in selected:
        delta = row["delta"]
        delta_class = _html_delta_class(delta)
        bars = _html_metric_bars(row["before"], row["after"])
        cards.append(
            '<div class="card">'
            f'<div class="label">{_html_text(row["metric"])}</div>'
            f'<div class="value">{_html_value(row["after"])}</div>'
            f'<div class="delta {delta_class}">change {_html_text(_fmt_delta(delta))}</div>'
            f"{bars}"
            "</div>"
        )
    return '<div class="grid">' + "\n".join(cards) + "</div>"


def _html_metric_bars(before: Any, after: Any) -> str:
    if not _is_number(before) or not _is_number(after):
        return ""

    maximum = max(abs(before), abs(after), 1)
    before_width = min(100, max(0, abs(before) / maximum * 100))
    after_width = min(100, max(0, abs(after) / maximum * 100))
    return (
        '<div class="bar">'
        '<div class="bar-line">'
        '<span>Before</span>'
        '<span class="track">'
        f'<span class="fill before" style="width: {before_width:.1f}%"></span>'
        '</span>'
        f'<span>{_html_value(before)}</span>'
        '</div>'
        '<div class="bar-line">'
        '<span>After</span>'
        '<span class="track">'
        f'<span class="fill after" style="width: {after_width:.1f}%"></span>'
        '</span>'
        f'<span>{_html_value(after)}</span>'
        '</div>'
        '</div>'
    )


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_html = "".join(f"<th>{_html_text(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_html_text(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _html_delta_class(value: Any) -> str:
    if not _is_number(value):
        return ""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return ""


def _html_text(value: Any) -> str:
    return html_lib.escape(str(value))


def _html_value(value: Any) -> str:
    return html_lib.escape(_fmt(value))


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _delta(before: Any, after: Any) -> float | int | None:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_delta(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.4f}"


def _short(value: str) -> str:
    return value[:8] if value else ""
