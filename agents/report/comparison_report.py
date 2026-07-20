"""RAG 파이프라인 성능 비교 리포트를 만든다.

1. 첫 최적화 기준값과 최종 Eval 결과를 before/after로 비교한다.
2. 최적화 작업이 수행되지 않았으면 최종 Eval 요약만 작성한다.
3. 파이프라인 실행마다 Markdown/JSON/HTML 리포트 파일을 남긴다.
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


# 파이프라인 리포트를 파일로 저장하고 생성 경로를 반환한다.
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


# state에서 before/after 비교용 데이터를 만든다.
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


# payload를 Markdown 리포트 문자열로 바꾼다.
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


# payload를 브라우저용 HTML 리포트 문자열로 바꾼다.
def render_html(payload: dict[str, Any]) -> str:
    """Render the comparison payload as a standalone browser report."""
    pipeline = payload["pipeline"]
    comparison = payload["comparison"]
    final_eval = payload["final_eval"]
    metric_rows = comparison["metric_rows"]
    config_rows = comparison["config_rows"]
    trials = payload["optimization"]["trials"]
    metric_by_name = {row["metric"]: row for row in metric_rows}
    overall_row = metric_by_name.get("overall_score", {})
    overall_delta = overall_row.get("delta")
    overall_delta_class = _html_delta_class(overall_delta)
    summary_text = _report_summary_text(metric_rows, final_eval)

    quality_section = _html_quality_metric_section(metric_rows)
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
      --bg: #f4faf8;
      --surface: #ffffff;
      --surface-soft: #edf8f5;
      --surface-strong: #e8f1ff;
      --text: #17212b;
      --muted: #5f6f82;
      --line: rgba(31, 57, 88, 0.14);
      --line-strong: rgba(14, 148, 136, 0.38);
      --before: #8a98a8;
      --after: #0e9488;
      --after-strong: #0a6f68;
      --accent: #3656d4;
      --warm: #f28c52;
      --positive: #0c8a5d;
      --negative: #c44949;
      --shadow: 0 24px 70px rgba(31, 57, 88, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        linear-gradient(135deg, rgba(14, 148, 136, 0.10) 0%, transparent 32%),
        linear-gradient(180deg, #fbfefd 0%, var(--bg) 48%, #eef6fb 100%);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 26px 0 52px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 0 18px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }}
    .brand {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--text);
      font-weight: 700;
    }}
    .brand-icon {{
      display: inline-grid;
      width: 24px;
      height: 24px;
      place-items: center;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: var(--surface-soft);
      color: var(--after-strong);
      font-size: 13px;
    }}
    .nav-links {{
      display: inline-flex;
      gap: 26px;
      font-size: 13px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 44px;
      align-items: center;
      min-height: 360px;
      padding: 58px 0 42px;
    }}
    h1, h2 {{ margin: 0; line-height: 1.2; }}
    h1 {{
      max-width: 720px;
      font-size: clamp(42px, 5.2vw, 64px);
      letter-spacing: 0;
      line-height: 1.02;
    }}
    h1 span {{
      color: var(--accent);
    }}
    h2 {{ font-size: 18px; margin-bottom: 16px; }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .dot {{
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--after);
      box-shadow: 0 0 18px rgba(14, 148, 136, 0.48);
    }}
    .subtle {{
      max-width: 610px;
      margin: 20px 0 0;
      color: var(--muted);
      font-size: 17px;
    }}
    .hero-meta {{
      display: inline-flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px;
      margin-top: 26px;
    }}
    .hero-meta span, .tool-button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--text);
      font-size: 13px;
    }}
    .tool-button {{
      cursor: pointer;
      font-family: inherit;
    }}
    .tool-button:hover, .tool-button[aria-pressed="true"] {{
      border-color: var(--line-strong);
      background: var(--surface-soft);
      color: var(--text);
    }}
    .tool-button:focus-visible {{
      outline: 2px solid var(--after);
      outline-offset: 2px;
    }}
    .pulse-line {{
      width: min(420px, 100%);
      color: rgba(14, 148, 136, 0.46);
    }}
    .pulse-line svg {{
      display: block;
      width: 100%;
      height: 92px;
    }}
    .hero-score {{
      min-width: 270px;
      padding: 24px;
      background: linear-gradient(180deg, #ffffff, #f4fbf9);
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .hero-score .label {{ margin-bottom: 10px; }}
    .hero-score .value {{
      display: block;
      font-size: 48px;
      line-height: 1;
      letter-spacing: 0;
    }}
    .workspace {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
      gap: 14px;
      margin-bottom: 18px;
    }}
    .workspace-panel {{
      padding: 22px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .workspace-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 16px;
    }}
    .workspace-title {{
      display: flex;
      gap: 12px;
      align-items: baseline;
    }}
    .workspace-title h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .workspace-tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }}
    .tab-button {{
      border: 0;
      border-bottom: 2px solid transparent;
      padding: 0 4px 10px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
      font-weight: 700;
    }}
    .tab-button[aria-pressed="true"] {{
      border-bottom-color: var(--accent);
      color: var(--text);
    }}
    .dropzone {{
      display: grid;
      min-height: 178px;
      place-items: center;
      padding: 26px;
      border: 1px dashed rgba(14, 148, 136, 0.42);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(237, 248, 245, 0.72), rgba(255, 255, 255, 0.58));
      color: var(--text);
      cursor: pointer;
      text-align: center;
    }}
    .dropzone.is-dragover {{
      border-color: var(--accent);
      background: var(--surface-strong);
    }}
    .file-input {{
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }}
    .upload-icon {{
      display: inline-grid;
      width: 38px;
      height: 38px;
      margin-bottom: 12px;
      place-items: center;
      border-radius: 999px;
      background: var(--surface);
      color: var(--accent);
      box-shadow: 0 10px 24px rgba(31, 57, 88, 0.10);
      font-size: 22px;
      font-weight: 700;
    }}
    .upload-title {{
      display: block;
      font-size: 17px;
      font-weight: 800;
    }}
    .upload-help {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .file-list {{
      display: grid;
      gap: 8px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    .file-item {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
    }}
    .mode-grid {{
      display: grid;
      gap: 10px;
      margin: 12px 0 18px;
    }}
    .mode-button {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
      font: inherit;
      text-align: left;
    }}
    .mode-button[aria-pressed="true"] {{
      border-color: var(--line-strong);
      background: var(--surface-soft);
    }}
    .mode-button span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .start-button {{
      width: 100%;
      border: 0;
      border-radius: 8px;
      padding: 13px 16px;
      background: var(--accent);
      color: #ffffff;
      cursor: pointer;
      font: inherit;
      font-weight: 800;
      box-shadow: 0 12px 24px rgba(54, 86, 212, 0.22);
    }}
    .start-button:hover {{
      background: #2846bf;
    }}
    .upload-status {{
      min-height: 38px;
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(31, 57, 88, 0.08);
    }}
    .card {{ padding: 16px; }}
    .summary-card {{
      position: relative;
      overflow: hidden;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 600;
      letter-spacing: 0;
    }}
    .delta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .delta.positive {{ color: var(--positive); }}
    .delta.negative {{ color: var(--negative); }}
    .delta-pill {{
      display: inline-flex;
      align-items: center;
      width: fit-content;
      margin-top: 8px;
      border-radius: 999px;
      padding: 4px 9px;
      background: rgba(54, 86, 212, 0.10);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    .delta-pill.positive {{
      background: rgba(12, 138, 93, 0.10);
      color: var(--positive);
    }}
    .delta-pill.negative {{
      background: rgba(196, 73, 73, 0.10);
      color: var(--negative);
    }}
    .report-section {{
      margin-top: 18px;
      padding: 22px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 14px 34px rgba(31, 57, 88, 0.08);
      overflow-x: auto;
    }}
    .section-title {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .section-title h2 {{ margin-bottom: 0; }}
    .section-kicker {{
      color: var(--accent);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .quality-section {{
      margin-top: 18px;
      padding: 28px 30px;
      background: linear-gradient(180deg, #ffffff, #f8fcfb);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 46px rgba(31, 57, 88, 0.10);
    }}
    .quality-head {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
      margin-bottom: 26px;
    }}
    .quality-actions {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }}
    .quality-title {{
      display: flex;
      gap: 12px;
      align-items: baseline;
    }}
    .quality-title h2 {{
      margin: 0;
      font-size: 24px;
    }}
    .quality-count {{
      color: var(--muted);
      font-size: 13px;
    }}
    .quality-legend {{
      display: inline-flex;
      gap: 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .quality-legend span::before {{
      content: "";
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 7px;
      border-radius: 999px;
      background: var(--before);
    }}
    .quality-legend span:last-child::before {{
      background: var(--after);
    }}
    .quality-list {{
      display: grid;
      gap: 0;
    }}
    .quality-row {{
      display: grid;
      grid-template-columns: 240px 1fr 170px;
      gap: 24px;
      align-items: center;
      padding: 21px 0;
      border-top: 1px solid var(--line);
    }}
    .quality-row:last-child {{
      border-bottom: 1px solid var(--line);
    }}
    .quality-row[hidden] {{
      display: none;
    }}
    .quality-name {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      min-width: 0;
    }}
    .quality-label {{
      color: var(--text);
      font-size: 19px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .quality-code {{
      color: var(--muted);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
      font-weight: 700;
    }}
    .quality-track {{
      position: relative;
      height: 32px;
    }}
    .quality-line {{
      position: absolute;
      left: 0;
      right: 0;
      top: 50%;
      height: 4px;
      transform: translateY(-50%);
      border-radius: 999px;
      background: rgba(138, 152, 168, 0.24);
    }}
    .quality-range {{
      position: absolute;
      top: 50%;
      height: 4px;
      transform: translateY(-50%);
      border-radius: 999px;
      background: linear-gradient(90deg, var(--before), var(--after), var(--accent));
    }}
    .quality-dot {{
      position: absolute;
      top: 50%;
      width: 13px;
      height: 13px;
      transform: translate(-50%, -50%);
      border-radius: 999px;
      background: var(--before);
      box-shadow: 0 0 0 3px rgba(138, 152, 168, 0.20);
    }}
    .quality-dot.after {{
      background: var(--after);
      box-shadow: 0 0 0 4px rgba(14, 148, 136, 0.18);
    }}
    .quality-values {{
      display: flex;
      justify-content: flex-end;
      gap: 7px;
      align-items: baseline;
      color: var(--text);
      font-family: "SFMono-Regular", Consolas, monospace;
      font-weight: 700;
      white-space: nowrap;
    }}
    .quality-values .from {{
      color: var(--muted);
    }}
    .quality-values .delta-value {{
      color: var(--positive);
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 640px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 12px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    tbody tr:hover {{ background: rgba(14, 148, 136, 0.06); }}
    tr:last-child td {{ border-bottom: 0; }}
    .metric-card {{
      display: grid;
      gap: 12px;
    }}
    .metric-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }}
    .metric-name {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}
    .metric-value {{
      margin-top: 4px;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .bar {{
      display: grid;
      gap: 8px;
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
      height: 9px;
      border-radius: 999px;
      background: rgba(138, 152, 168, 0.18);
      overflow: hidden;
    }}
    .fill {{
      display: block;
      height: 100%;
      border-radius: inherit;
    }}
    .fill.before {{ background: var(--before); }}
    .fill.after {{ background: var(--after); }}
    .legend {{
      display: inline-flex;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 8px;
      height: 8px;
      margin-right: 5px;
      border-radius: 999px;
      background: var(--before);
    }}
    .legend span:last-child::before {{ background: var(--after); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 5px 10px;
      background: rgba(14, 148, 136, 0.10);
      color: var(--after-strong);
      font-size: 12px;
      font-weight: 600;
    }}
    .empty {{ color: var(--muted); margin: 0; }}
    .insights {{
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .insights li {{
      padding: 12px 14px;
      border-left: 4px solid var(--after);
      background: var(--surface-soft);
      border-radius: 8px;
    }}
    .footer {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-top: 38px;
      padding-top: 22px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1120px); padding-top: 20px; }}
      .topbar {{ align-items: flex-start; gap: 16px; }}
      .nav-links {{ display: none; }}
      .hero {{ display: block; min-height: 0; padding: 42px 0 26px; }}
      .hero-score {{ margin-top: 18px; min-width: 0; }}
      h1 {{ font-size: 24px; margin-bottom: 8px; }}
      .workspace {{ grid-template-columns: 1fr; }}
      .workspace-head {{ display: block; }}
      .workspace-panel {{ padding: 16px; }}
      .report-section {{ padding: 14px; }}
      .section-title {{ display: block; }}
      .quality-section {{ padding: 18px 14px; }}
      .quality-head {{ display: block; }}
      .quality-legend {{ margin-top: 12px; }}
      .quality-row {{
        grid-template-columns: 1fr;
        gap: 10px;
        padding: 18px 0;
      }}
      .quality-name {{ display: block; }}
      .quality-code {{ display: block; margin-top: 3px; }}
      .quality-values {{ justify-content: flex-start; }}
      .footer {{ display: block; }}
    }}
  </style>
</head>
<body>
  <main data-report data-summary="{_html_text(summary_text)}">
    <nav class="topbar">
      <div class="brand">
        <span class="brand-icon">~</span>
        <span>AgentDoctor</span>
      </div>
      <div class="nav-links">
        <span>RAG 리포트</span>
        <span>Before / After</span>
        <span>Eval 결과</span>
      </div>
    </nav>

    <header class="hero">
      <div>
        <p class="eyebrow"><span class="dot"></span> RAG 파이프라인 성능 비교</p>
        <h1>RAG 성능 개선을<br><span>전후 지표로 증명합니다</span></h1>
        <p class="subtle">최적화 전 기준값과 최종 Eval 결과를 비교해 검색, 답변, 설정 변화가 실제로 좋아졌는지 보여줍니다.</p>
        <div class="hero-meta">
          <span>Generated at {_html_text(payload["created_at"])}</span>
          <span>{_html_text(comparison["basis"])}</span>
          <button class="tool-button" type="button" data-copy-summary>요약 복사</button>
        </div>
      </div>
      <div>
        <div class="pulse-line" aria-hidden="true">
          <svg viewBox="0 0 420 92" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 58H92L106 58L118 22L130 72L146 58H198C203 51 214 51 220 58H306L320 58L334 21L346 72L364 58H420" stroke="currentColor" stroke-width="2"/>
          </svg>
        </div>
        <div class="hero-score">
          <div class="label">Final overall score</div>
          <strong class="value">{_html_value(final_eval.get("overall_score"))}</strong>
          <span class="delta-pill {overall_delta_class}">{_html_text(_fmt_delta(overall_delta))} vs before</span>
        </div>
      </div>
    </header>

    <section class="workspace" aria-labelledby="workspace-title">
      <div class="workspace-panel">
        <div class="workspace-head">
          <div class="workspace-title">
            <span class="section-kicker">01</span>
            <h2 id="workspace-title">문서 진단 시작</h2>
          </div>
          <span class="quality-count">PDF · MARKDOWN · TXT · DOCX · HTML</span>
        </div>
        <div class="workspace-tabs" aria-label="진단 입력 방식">
          <button class="tab-button" type="button" aria-pressed="true" data-workspace-tab="upload">문서 올리기</button>
          <button class="tab-button" type="button" aria-pressed="false" data-workspace-tab="existing">기존 RAG 연결</button>
        </div>
        <label class="dropzone" data-dropzone>
          <input class="file-input" type="file" multiple accept=".pdf,.md,.markdown,.txt,.docx,.html" data-file-input>
          <span>
            <span class="upload-icon">↑</span>
            <span class="upload-title">문서를 끌어다 놓거나 클릭해 업로드</span>
            <span class="upload-help">진단할 지식 문서를 넣으면 검색 파이프라인을 만들어 검사합니다.</span>
          </span>
        </label>
        <div class="file-list" data-file-list>선택된 문서가 없습니다.</div>
      </div>

      <aside class="workspace-panel">
        <div class="label">진단 깊이</div>
        <div class="mode-grid" aria-label="진단 깊이 선택">
          <button class="mode-button" type="button" aria-pressed="false" data-mode="quick">
            <strong>빠른 검진</strong><span>규칙 기반 · 약 30초</span>
          </button>
          <button class="mode-button" type="button" aria-pressed="true" data-mode="standard">
            <strong>표준 검진</strong><span>재검색 분석 · 약 2분</span>
          </button>
          <button class="mode-button" type="button" aria-pressed="false" data-mode="deep">
            <strong>정밀 검진</strong><span>LLM 검토 · 약 6분</span>
          </button>
        </div>
        <button class="start-button" type="button" data-start-button>진단 시작</button>
        <p class="upload-status" data-upload-status>문서를 선택하면 준비 상태가 표시됩니다.</p>
      </aside>
    </section>

    <div class="summary-grid">
      <div class="card summary-card">
        <div class="label">Pipeline status</div>
        <div class="value">{_html_text(pipeline["status"])}</div>
        <div class="delta">iteration {_html_value(pipeline["iteration"])} / {_html_value(pipeline["max_iterations"])}</div>
      </div>
      <div class="card summary-card">
        <div class="label">Documents / Chunks</div>
        <div class="value">{_html_value(pipeline["documents"])} / {_html_value(pipeline["chunks"])}</div>
        <div class="delta">probes {_html_value(pipeline["probes"])}</div>
      </div>
      <div class="card summary-card">
        <div class="label">Pass threshold</div>
        <div class="value">{_html_value(final_eval.get("pass_threshold"))}</div>
        <div class="delta">oracle accuracy {_html_value(final_eval.get("oracle_accuracy"))}</div>
      </div>
    </div>

    {quality_section}

    <section class="report-section">
      <div class="section-title">
        <h2>Metric Comparison</h2>
        <div class="legend"><span>Before</span><span>After</span></div>
      </div>
      {metric_table}
    </section>

    <section class="report-section">
      <div class="section-title">
        <h2>Configuration Changes</h2>
      </div>
      {config_table}
    </section>

    <section class="report-section">
      <div class="section-title">
        <h2>Optimization Trials</h2>
      </div>
      {trial_table}
    </section>

    <section class="report-section">
      <div class="section-title">
        <h2>Interpretation</h2>
      </div>
      <ul class="insights">{interpretation}</ul>
    </section>

    <footer class="footer">
      <span>AgentDoctor - RAG 성능 비교 리포트</span>
      <span>Ingest -> Index -> RAG -> Eval -> Optimize</span>
    </footer>
  </main>
  <script>
    (() => {{
      const report = document.querySelector("[data-report]");
      const rows = Array.from(document.querySelectorAll("[data-quality-row]"));
      const filterButtons = Array.from(document.querySelectorAll("[data-quality-filter]"));
      const sortButton = document.querySelector("[data-quality-sort]");
      const copyButton = document.querySelector("[data-copy-summary]");
      const list = document.querySelector("[data-quality-list]");
      const fileInput = document.querySelector("[data-file-input]");
      const fileList = document.querySelector("[data-file-list]");
      const dropzone = document.querySelector("[data-dropzone]");
      const status = document.querySelector("[data-upload-status]");
      const startButton = document.querySelector("[data-start-button]");
      const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
      const tabButtons = Array.from(document.querySelectorAll("[data-workspace-tab]"));
      let selectedFiles = [];

      function formatSize(bytes) {{
        if (!bytes) return "0 KB";
        const units = ["B", "KB", "MB", "GB"];
        let size = bytes;
        let unit = 0;
        while (size >= 1024 && unit < units.length - 1) {{
          size /= 1024;
          unit += 1;
        }}
        return `${{size.toFixed(unit === 0 ? 0 : 1)}} ${{units[unit]}}`;
      }}

      function renderFiles(files) {{
        selectedFiles = Array.from(files || []);
        if (!fileList || !status) return;
        if (!selectedFiles.length) {{
          fileList.textContent = "선택된 문서가 없습니다.";
          status.textContent = "문서를 선택하면 준비 상태가 표시됩니다.";
          return;
        }}
        fileList.innerHTML = "";
        selectedFiles.forEach((file) => {{
          const item = document.createElement("div");
          item.className = "file-item";
          const name = document.createElement("span");
          name.textContent = file.name;
          const size = document.createElement("span");
          size.textContent = formatSize(file.size);
          item.append(name, size);
          fileList.appendChild(item);
        }});
        const totalSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);
        status.textContent = `${{selectedFiles.length}}개 문서 준비됨 · 총 ${{formatSize(totalSize)}}`;
      }}

      function setFilter(filter) {{
        rows.forEach((row) => {{
          row.hidden = filter !== "all" && row.dataset.direction !== filter;
        }});
        filterButtons.forEach((button) => {{
          button.setAttribute("aria-pressed", String(button.dataset.qualityFilter === filter));
        }});
      }}

      filterButtons.forEach((button) => {{
        button.addEventListener("click", () => setFilter(button.dataset.qualityFilter));
      }});

      tabButtons.forEach((button) => {{
        button.addEventListener("click", () => {{
          tabButtons.forEach((item) => {{
            item.setAttribute("aria-pressed", String(item === button));
          }});
          if (status) {{
            status.textContent = button.dataset.workspaceTab === "existing"
              ? "기존 RAG 인덱스를 연결해 비교 진단할 수 있습니다."
              : selectedFiles.length
                ? `${{selectedFiles.length}}개 문서 준비됨`
                : "문서를 선택하면 준비 상태가 표시됩니다.";
          }}
        }});
      }});

      modeButtons.forEach((button) => {{
        button.addEventListener("click", () => {{
          modeButtons.forEach((item) => {{
            item.setAttribute("aria-pressed", String(item === button));
          }});
        }});
      }});

      if (fileInput) {{
        fileInput.addEventListener("change", () => renderFiles(fileInput.files));
      }}

      if (dropzone) {{
        dropzone.addEventListener("dragover", (event) => {{
          event.preventDefault();
          dropzone.classList.add("is-dragover");
        }});
        dropzone.addEventListener("dragleave", () => {{
          dropzone.classList.remove("is-dragover");
        }});
        dropzone.addEventListener("drop", (event) => {{
          event.preventDefault();
          dropzone.classList.remove("is-dragover");
          renderFiles(event.dataTransfer.files);
        }});
      }}

      if (startButton && status) {{
        startButton.addEventListener("click", () => {{
          const activeMode = modeButtons.find((button) => button.getAttribute("aria-pressed") === "true");
          const mode = activeMode ? activeMode.textContent.trim().replace(/\\s+/g, " ") : "표준 검진";
          status.textContent = selectedFiles.length
            ? `${{mode}} 준비 완료 · 아래 리포트에서 기존/최적화 결과를 확인하세요.`
            : "먼저 문서를 업로드하거나 기존 RAG 연결을 선택하세요.";
        }});
      }}

      if (sortButton && list) {{
        sortButton.addEventListener("click", () => {{
          const sorted = [...rows].sort((a, b) => Number(b.dataset.delta) - Number(a.dataset.delta));
          sorted.forEach((row) => list.appendChild(row));
          sortButton.textContent = "개선폭순 적용";
        }});
      }}

      if (copyButton && report) {{
        copyButton.addEventListener("click", async () => {{
          try {{
            await navigator.clipboard.writeText(report.dataset.summary || "");
            copyButton.textContent = "복사됨";
            setTimeout(() => {{ copyButton.textContent = "요약 복사"; }}, 1400);
          }} catch (error) {{
            copyButton.textContent = "복사 실패";
            setTimeout(() => {{ copyButton.textContent = "요약 복사"; }}, 1400);
          }}
        }});
      }}
    }})();
  </script>
</body>
</html>
"""


# 최적화 이력 중 기준이 되는 첫 trial을 찾는다.
def _baseline_trial(history: list[Any]) -> Any | None:
    for item in history:
        if getattr(item, "before_config", None):
            return item
    return None


# 최적화 전 설정과 metric 스냅샷을 만든다.
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


# 최종 평가 기준의 설정과 metric 스냅샷을 만든다.
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


# DiagnosticReport에서 비교할 metric만 꺼낸다.
def _metrics_from_report(report: DiagnosticReport | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = dict(report.ragas_scores or {})
    metrics["overall_score"] = report.overall_score
    metrics["oracle_accuracy"] = report.oracle_accuracy
    metrics["pass_threshold"] = report.pass_threshold
    return {key: value for key, value in metrics.items() if value is not None}


# before/after metric을 비교 행으로 묶는다.
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


# 중요한 metric이 먼저 보이도록 정렬 순서를 정한다.
def _metric_sort_key(name: str) -> tuple[int, str]:
    priority = {
        "overall_score": 0,
        "mean_recall_at_k": 1,
        "mean_f1": 2,
        "oracle_accuracy": 3,
        "pass_threshold": 4,
    }
    return priority.get(name, 100), name


# 변경된 설정값만 비교 행으로 묶는다.
def _config_rows(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value == after_value:
            continue
        rows.append({"key": key, "before": before_value, "after": after_value})
    return rows


# 최적화 trial 하나를 리포트용 요약 dict로 만든다.
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


# Eval 리포트를 저장 가능한 dict로 줄인다.
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


# dataclass 객체를 JSON에 넣을 수 있는 값으로 바꾼다.
def _safe_dataclass(value: Any) -> Any:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value


# metric 변화량을 사람이 읽을 해석 문장으로 만든다.
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


# 주요 품질 metric을 처방 전후 가로 비교 섹션으로 만든다.
def _html_quality_metric_section(metric_rows: list[dict[str, Any]]) -> str:
    rows = _quality_metric_rows(metric_rows)
    if not rows:
        return (
            '<section class="quality-section">'
            '<div class="quality-head">'
            '<div class="quality-title">'
            '<span class="section-kicker">02</span>'
            "<h2>품질 지표 · 처방 전후</h2>"
            "</div>"
            '<span class="quality-count">비교 가능한 지표 없음</span>'
            "</div>"
            '<p class="empty">숫자로 비교할 수 있는 before/after metric이 없습니다.</p>'
            "</section>"
        )

    items = "\n".join(_html_quality_metric_row(row) for row in rows)
    return (
        '<section class="quality-section">'
        '<div class="quality-head">'
        '<div class="quality-title">'
        '<span class="section-kicker">02</span>'
        "<h2>품질 지표 · 처방 전후</h2>"
        "</div>"
        '<div class="quality-actions">'
        '<button class="tool-button" type="button" data-quality-filter="all" aria-pressed="true">전체</button>'
        '<button class="tool-button" type="button" data-quality-filter="improved" aria-pressed="false">개선</button>'
        '<button class="tool-button" type="button" data-quality-filter="regressed" aria-pressed="false">하락</button>'
        '<button class="tool-button" type="button" data-quality-sort>개선폭순</button>'
        f'<span class="quality-count">RAGAS {len(rows)}개 지표</span>'
        "</div>"
        "</div>"
        '<div class="quality-legend"><span>처방 전</span><span>처방 후</span></div>'
        f'<div class="quality-list" data-quality-list>{items}</div>'
        "</section>"
    )


# 품질 비교 섹션에 보여줄 metric만 고른다.
def _quality_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {
        "faithfulness": "충실도",
        "context_recall": "정답 회수율",
        "context_precision": "검색 정확도",
        "response_relevancy": "답변 관련성",
        "mean_recall_at_k": "정답 회수율",
        "mean_f1": "답변 F1",
        "oracle_accuracy": "Oracle 정확도",
        "overall_score": "종합 점수",
    }
    preferred = [
        "faithfulness",
        "context_recall",
        "context_precision",
        "response_relevancy",
        "mean_recall_at_k",
        "mean_f1",
        "oracle_accuracy",
        "overall_score",
    ]
    by_name = {row["metric"]: row for row in metric_rows}
    selected = []
    for metric in preferred:
        row = by_name.get(metric)
        if row is None or not _is_number(row["before"]) or not _is_number(row["after"]):
            continue
        item = dict(row)
        item["label"] = labels.get(metric, metric)
        selected.append(item)
        if len(selected) == 4:
            break
    return selected


# 품질 metric 한 줄의 before/after 위치와 값을 만든다.
def _html_quality_metric_row(row: dict[str, Any]) -> str:
    before = row["before"]
    after = row["after"]
    left = _percent_position(min(before, after))
    right = _percent_position(max(before, after))
    before_pos = _percent_position(before)
    after_pos = _percent_position(after)
    delta = row["delta"]
    delta_text = _fmt_delta(delta)
    direction = _metric_direction(delta)
    return (
        f'<div class="quality-row" data-quality-row data-direction="{direction}" data-delta="{_delta_sort_value(delta):.6f}">'
        '<div class="quality-name">'
        f'<span class="quality-label">{_html_text(row["label"])}</span>'
        f'<span class="quality-code">{_html_text(row["metric"])}</span>'
        "</div>"
        '<div class="quality-track">'
        '<span class="quality-line"></span>'
        f'<span class="quality-range" style="left: {left:.2f}%; width: {right - left:.2f}%"></span>'
        f'<span class="quality-dot before" style="left: {before_pos:.2f}%"></span>'
        f'<span class="quality-dot after" style="left: {after_pos:.2f}%"></span>'
        "</div>"
        '<div class="quality-values">'
        f'<span class="from">{_html_value(before)}</span>'
        "<span>-></span>"
        f'<span>{_html_value(after)}</span>'
        f'<span class="delta-value">{_html_text(delta_text)}</span>'
        "</div>"
        "</div>"
    )


# HTML 상단에 보여줄 주요 metric 카드를 만든다.
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
            '<div class="card metric-card">'
            '<div class="metric-top">'
            "<div>"
            f'<div class="metric-name">{_html_text(row["metric"])}</div>'
            f'<div class="metric-value">{_html_value(row["after"])}</div>'
            "</div>"
            f'<span class="delta-pill {delta_class}">{_html_text(_fmt_delta(delta))}</span>'
            "</div>"
            f"{bars}"
            "</div>"
        )
    return '<div class="metric-grid">' + "\n".join(cards) + "</div>"


# 숫자 metric의 before/after 막대 그래프를 만든다.
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


# HTML 표 문자열을 만든다.
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


# delta 값에 따라 색상 class를 정한다.
def _html_delta_class(value: Any) -> str:
    if not _is_number(value):
        return ""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return ""


# metric 변화 방향을 필터용 값으로 바꾼다.
def _metric_direction(value: Any) -> str:
    if not _is_number(value):
        return "flat"
    if value > 0:
        return "improved"
    if value < 0:
        return "regressed"
    return "flat"


# 개선폭 정렬에 쓸 숫자값을 만든다.
def _delta_sort_value(value: Any) -> float:
    if not _is_number(value):
        return 0.0
    return float(value)


# 상단 요약 복사 버튼에 들어갈 문장을 만든다.
def _report_summary_text(metric_rows: list[dict[str, Any]], final_eval: dict[str, Any]) -> str:
    numeric_rows = [
        row for row in metric_rows
        if _is_number(row.get("delta")) and _is_number(row.get("after"))
    ]
    best = max(numeric_rows, key=lambda row: row["delta"], default=None)
    score = _fmt(final_eval.get("overall_score"))
    pass_threshold = final_eval.get("pass_threshold")
    if best is None:
        return f"RAG 최종 overall_score는 {score}이고 pass_threshold는 {pass_threshold}입니다."
    return (
        f"RAG 최종 overall_score는 {score}이고 pass_threshold는 {pass_threshold}입니다. "
        f"가장 크게 개선된 지표는 {best['metric']}({_fmt_delta(best['delta'])})입니다."
    )


# HTML에 넣을 일반 텍스트를 안전하게 escape한다.
def _html_text(value: Any) -> str:
    return html_lib.escape(str(value))


# HTML에 넣을 표시값을 포맷 후 escape한다.
def _html_value(value: Any) -> str:
    return html_lib.escape(_fmt(value))


# Markdown 표 문자열을 만든다.
def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


# 숫자 before/after 차이를 계산한다.
def _delta(before: Any, after: Any) -> float | int | None:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


# bool을 제외한 숫자인지 확인한다.
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# 0~1 점수를 가로 막대 위치 percent로 바꾼다.
def _percent_position(value: float | int) -> float:
    return min(100.0, max(0.0, float(value) * 100.0))


# 리포트 출력용 값 포맷을 통일한다.
def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


# delta 출력 형식을 통일한다.
def _fmt_delta(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.4f}"


# 긴 ID를 짧게 줄인다.
def _short(value: str) -> str:
    return value[:8] if value else ""
