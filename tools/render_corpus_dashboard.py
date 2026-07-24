"""Render a presentation-ready KorQuAD corpus dashboard."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


DEFAULT_SUMMARY = Path("output/korquad_corpus_visualization/corpus_summary.json")
DEFAULT_HTML = Path("output/korquad_corpus_visualization/corpus_visualization.html")

STOP_TERMS = {
    "the", "and", "for", "of", "to", "in", "on", "by", "with", "from", "as", "is", "are",
    "있다", "한다", "했다", "되었다", "확인함", "가기", "링크", "외부", "각주", "목차",
    "문서", "대한", "그는", "그의", "이후", "것을", "것은", "것이", "것이다", "것으로",
    "등을", "다른", "같은", "함께", "위해", "가장", "된다", "있었다", "부터", "하는",
    "따라", "의해", "다시", "통해", "자신의", "또는", "라고", "하였다", "현재", "영어",
    "많은", "에는", "으로", "에서", "에게", "모두의", "둘러보기로", "때문에", "com",
    "www", "http", "https", "보존된", "전거", "통제", "isbn", "lccn", "viaf", "ebay",
    "chapter", "displaystyle", "빈칸", "검색하러",
}


def main() -> None:
    args = _parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    dashboard = _dashboard_payload(data)
    args.output.write_text(_render_html(dashboard), encoding="utf-8")
    print(f"wrote_html={args.output.resolve()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_HTML)
    return parser.parse_args()


def _dashboard_payload(data: dict) -> dict:
    _validate_korquad_schema(data)
    summary = data["summary"]
    points = [
        {
            "x": point["x"],
            "y": point["y"],
            "cluster": point["cluster"],
            "tokens": point["tokens"],
            "chars": point["chars"],
            "qa": point["qa_count"],
            "title": point["title"],
            "doc": point["doc_id"],
            "chunk": point["chunk_id"],
            "preview": point["preview"],
            "url": point.get("url", ""),
        }
        for point in data["projection"]["points"]
    ]

    clusters = []
    for row in data["projection"]["clusters"]:
        terms = _clean_terms(row.get("top_terms", []), limit=5)
        docs = row.get("top_documents", [])[:3]
        fallback = [item["title"] for item in docs[:1]]
        label = " · ".join(terms[:2] or fallback or [f"군집 {row['cluster']}"])
        clusters.append(
            {
                "cluster": row["cluster"],
                "chunks": row["chunks"],
                "share": row["share"],
                "avg_tokens": row["avg_tokens"],
                "terms": terms,
                "label": label,
                "docs": docs,
            }
        )

    clean_top_terms = _clean_term_rows(data.get("top_terms", []), limit=18)
    chunk_count = max(summary["chunks"], 1)
    doc_count = max(summary["documents"], 1)
    short_ratio = summary["short_chunks"] / chunk_count
    long_ratio = summary["long_chunks"] / chunk_count
    duplicate_ratio = summary["duplicate_ratio"]
    length_ok = max(0.0, 1.0 - short_ratio - long_ratio)

    return {
        "summary": summary,
        "metrics": [
            {"label": "원본 문서", "value": _fmt(summary["documents"]), "note": "KorQuAD context"},
            {"label": "평가 QA", "value": _fmt(summary["qas"]), "note": f"문서당 {summary['qas'] / doc_count:.1f}개"},
            {"label": "검색 청크", "value": _fmt(summary["chunks"]), "note": f"문서당 {summary['avg_chunks_per_doc']:.1f}개"},
            {"label": "지도 표본", "value": _fmt(summary["sampled_points"]), "note": "균등 샘플링"},
            {"label": "중앙 길이", "value": f"{summary['median_tokens_per_chunk']:.0f}", "note": "토큰/청크"},
            {"label": "90% 길이", "value": f"{summary['p90_tokens_per_chunk']:.0f}", "note": "토큰/청크"},
        ],
        "signals": [
            {"label": "QA 연결 문서", "value": summary["contexts_with_qas"] / doc_count},
            {"label": "적정 길이 청크", "value": length_ok},
            {"label": "중복 낮음", "value": 1.0 - duplicate_ratio},
            {"label": "지도 표시 비율", "value": summary["sampled_points"] / chunk_count},
        ],
        "risks": [
            {
                "label": "짧은 청크",
                "value": summary["short_chunks"],
                "ratio": short_ratio,
                "tone": "warn" if short_ratio > 0.05 else "ok",
            },
            {
                "label": "긴 청크",
                "value": summary["long_chunks"],
                "ratio": long_ratio,
                "tone": "warn" if long_ratio > 0.02 else "ok",
            },
            {
                "label": "완전 중복",
                "value": summary["duplicate_chunks"],
                "ratio": duplicate_ratio,
                "tone": "warn" if duplicate_ratio > 0.01 else "ok",
            },
        ],
        "documents": data["documents"][:16],
        "token_bins": data["token_bins"],
        "chunk_doc_bins": data["chunk_doc_bins"],
        "qa_bins": data["qa_bins"],
        "terms": clean_top_terms,
        "points": points,
        "clusters": clusters,
        "explained": data["projection"]["explained_variance"],
    }


def _validate_korquad_schema(data: dict) -> None:
    required_summary = {
        "source_file",
        "version",
        "documents",
        "qas",
        "contexts_with_qas",
        "chunks",
        "sampled_points",
        "median_tokens_per_chunk",
        "p90_tokens_per_chunk",
        "short_chunks",
        "long_chunks",
        "duplicate_chunks",
        "duplicate_ratio",
    }
    required_top = {
        "summary",
        "documents",
        "token_bins",
        "chunk_doc_bins",
        "qa_bins",
        "top_terms",
        "projection",
    }
    _require_keys(data, required_top, "KorQuAD dashboard root")
    _require_keys(data["summary"], required_summary, "KorQuAD summary")
    projection = data["projection"]
    _require_keys(projection, {"points", "clusters", "explained_variance"}, "KorQuAD projection")

    required_point = {
        "x",
        "y",
        "cluster",
        "tokens",
        "chars",
        "qa_count",
        "title",
        "doc_id",
        "chunk_id",
        "preview",
    }
    for index, point in enumerate(projection["points"]):
        _require_keys(point, required_point, f"KorQuAD projection.points[{index}]")


def _require_keys(row: dict, keys: set[str], label: str) -> None:
    missing = sorted(key for key in keys if key not in row)
    if missing:
        raise ValueError(
            f"{label} is missing required KorQuAD dashboard fields: {', '.join(missing)}. "
            "Use tools/korquad_corpus_visualization.py or pass a KorQuAD-specific summary file."
        )


def _clean_term_rows(rows: list[dict], *, limit: int) -> list[dict]:
    cleaned = []
    for row in rows:
        term = _normalize_term(row["term"])
        if not _is_noise(term):
            cleaned.append({"term": term, "count": row["count"]})
    return cleaned[:limit]


def _clean_terms(terms: list[str], *, limit: int) -> list[str]:
    cleaned = []
    for term in terms:
        normalized = _normalize_term(term)
        if normalized and not _is_noise(normalized):
            cleaned.append(normalized)
    return cleaned[:limit]


def _normalize_term(term: str) -> str:
    return str(term).strip("._-+ ").lower()


def _is_noise(term: str) -> bool:
    if len(term) < 2 or term in STOP_TERMS:
        return True
    if re.fullmatch(r"\d+(?:년|월|일|개|명|회|차|위|대)?", term):
        return True
    if re.fullmatch(r"[a-z]{1,2}", term):
        return True
    return False


def _render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    summary = data["summary"]
    metrics = "\n".join(_metric_card(row) for row in data["metrics"])
    signals = "\n".join(_signal_row(row) for row in data["signals"])
    risks = "\n".join(_risk_pill(row) for row in data["risks"])
    source = html.escape(Path(summary["source_file"]).name)
    explained = " / ".join(f"{value * 100:.1f}%" for value in data["explained"])
    main_sentence = (
        f"{_fmt(summary['documents'])}개 문서를 {_fmt(summary['chunks'])}개 검색 단위로 나누고, "
        f"{_fmt(summary['sampled_points'])}개 청크를 의미 지도에 표시했습니다."
    )
    print_cards = "\n".join(
        [
            _print_item("문서", _fmt(summary["documents"])),
            _print_item("QA", _fmt(summary["qas"])),
            _print_item("청크", _fmt(summary["chunks"])),
            _print_item("중앙 토큰", f"{summary['median_tokens_per_chunk']:.0f}"),
            _print_item("짧은 청크", _fmt(summary["short_chunks"])),
            _print_item("중복 청크", _fmt(summary["duplicate_chunks"])),
        ]
    )

    return (
        HTML_TEMPLATE.replace("__PAYLOAD__", payload)
        .replace("__METRICS__", metrics)
        .replace("__SIGNALS__", signals)
        .replace("__RISKS__", risks)
        .replace("__SOURCE__", source)
        .replace("__VERSION__", html.escape(summary["version"]))
        .replace("__MAIN_SENTENCE__", html.escape(main_sentence))
        .replace("__EXPLAINED__", html.escape(explained))
        .replace("__PRINT_ITEMS__", print_cards)
    )


def _metric_card(row: dict) -> str:
    return (
        '<article class="metric-card">'
        f'<span>{html.escape(row["label"])}</span>'
        f'<strong>{html.escape(row["value"])}</strong>'
        f'<small>{html.escape(row["note"])}</small>'
        "</article>"
    )


def _signal_row(row: dict) -> str:
    percent = max(0, min(100, row["value"] * 100))
    return (
        '<div class="signal-row">'
        f'<div><b>{html.escape(row["label"])}</b><span>{percent:.1f}%</span></div>'
        '<div class="signal-track">'
        f'<i style="width:{percent:.2f}%"></i>'
        "</div>"
        "</div>"
    )


def _risk_pill(row: dict) -> str:
    return (
        f'<div class="risk-pill {row["tone"]}">'
        f'<span>{html.escape(row["label"])}</span>'
        f'<b>{_fmt(row["value"])}</b>'
        f'<small>{row["ratio"] * 100:.2f}%</small>'
        "</div>"
    )


def _print_item(label: str, value: str) -> str:
    return f'<div><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>'


def _fmt(value: int | float) -> str:
    return f"{int(round(value)):,}"


HTML_TEMPLATE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Doctor KorQuAD DB Atlas</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6fa;
      --surface: #ffffff;
      --surface-soft: #f8fafc;
      --ink: #111827;
      --muted: #667085;
      --line: #d8e0ea;
      --blue: #2864e8;
      --green: #148a62;
      --amber: #c97916;
      --rose: #c2416f;
      --violet: #6957d8;
      --cyan: #108498;
      --red: #c7352d;
      --shadow: 0 16px 36px rgba(17, 24, 39, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #edf2f8 0, #f4f6fa 360px), var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Pretendard, Arial, sans-serif;
      line-height: 1.5;
    }
    main {
      max-width: 1320px;
      margin: 0 auto;
      padding: 30px;
    }
    h1, h2, h3, p { margin: 0; }
    .masthead {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(340px, .8fr);
      gap: 24px;
      align-items: end;
      padding: 18px 0 22px;
    }
    .eyebrow {
      margin-bottom: 10px;
      color: var(--blue);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 830px;
      font-size: clamp(32px, 5vw, 60px);
      line-height: 1.04;
      font-weight: 760;
      letter-spacing: 0;
    }
    .lead {
      max-width: 780px;
      margin-top: 14px;
      color: #475467;
      font-size: 17px;
    }
    .source-stack {
      display: grid;
      gap: 10px;
      justify-items: end;
      color: var(--muted);
      font-size: 13px;
    }
    .source-stack span {
      max-width: 100%;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, .76);
      padding: 7px 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .metric-card, .panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .metric-card {
      min-height: 112px;
      padding: 15px;
    }
    .metric-card span {
      display: block;
      color: var(--muted);
      font-size: 13px;
    }
    .metric-card strong {
      display: block;
      margin-top: 6px;
      font-size: 30px;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }
    .metric-card small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .grid {
      display: grid;
      gap: 16px;
    }
    .intro-grid {
      grid-template-columns: minmax(0, .92fr) minmax(0, 1.08fr);
      margin-bottom: 16px;
    }
    .panel { padding: 18px; }
    .panel h2 {
      margin-bottom: 12px;
      font-size: 18px;
      font-weight: 760;
      letter-spacing: 0;
    }
    .panel-note {
      margin-bottom: 14px;
      color: var(--muted);
      font-size: 13px;
    }
    .signal-row {
      display: grid;
      gap: 8px;
      margin: 13px 0;
    }
    .signal-row div:first-child {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .signal-row b {
      color: var(--ink);
      font-weight: 650;
    }
    .signal-track {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e8edf5;
    }
    .signal-track i {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--green), var(--blue));
    }
    .risk-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .risk-pill {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-soft);
      padding: 12px;
    }
    .risk-pill.warn { border-color: #f7bf73; background: #fff8ed; }
    .risk-pill.ok { border-color: #bddfce; background: #f0faf5; }
    .risk-pill span, .risk-pill small {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .risk-pill b {
      display: block;
      margin: 4px 0 1px;
      font-size: 24px;
      font-weight: 760;
      font-variant-numeric: tabular-nums;
    }
    .db-panel {
      margin-bottom: 16px;
      padding: 0;
      overflow: hidden;
    }
    .db-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: start;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #fbfcff);
      padding: 20px;
    }
    .kicker {
      margin-bottom: 5px;
      color: var(--blue);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .db-head h2 {
      margin: 0 0 7px;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.12;
    }
    .map-stats {
      display: grid;
      grid-template-columns: repeat(3, max-content);
      gap: 8px;
    }
    .map-stats span {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      padding: 9px 11px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .map-stats b {
      display: block;
      color: var(--ink);
      font-size: 16px;
      font-variant-numeric: tabular-nums;
    }
    .map-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 0;
    }
    .map-stage {
      position: relative;
      min-height: 660px;
      background:
        linear-gradient(90deg, rgba(40,100,232,.05) 0, rgba(20,138,98,.04) 50%, rgba(194,65,111,.05) 100%),
        #fbfcff;
    }
    canvas {
      display: block;
      width: 100%;
      height: 660px;
    }
    .map-key {
      position: absolute;
      left: 18px;
      bottom: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      max-width: calc(100% - 36px);
      pointer-events: none;
    }
    .map-key span {
      border: 1px solid rgba(216, 224, 234, .88);
      border-radius: 999px;
      background: rgba(255, 255, 255, .86);
      padding: 6px 9px;
      color: #475467;
      font-size: 12px;
    }
    .tooltip {
      position: absolute;
      display: none;
      max-width: 280px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255,255,255,.97);
      box-shadow: 0 12px 26px rgba(17, 24, 39, .13);
      padding: 10px 11px;
      pointer-events: none;
      font-size: 12px;
      z-index: 2;
    }
    .tooltip b {
      display: block;
      margin-bottom: 3px;
      font-size: 13px;
    }
    .map-aside {
      border-left: 1px solid var(--line);
      background: #ffffff;
      padding: 16px;
    }
    .detail {
      margin-bottom: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 15px;
    }
    .detail h3 {
      margin-bottom: 8px;
      font-size: 19px;
      font-weight: 760;
    }
    .detail p {
      color: #344054;
      font-size: 14px;
    }
    .detail dl {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 7px 10px;
      margin: 16px 0 0;
      font-size: 13px;
    }
    .detail dt { color: var(--muted); }
    .detail dd {
      margin: 0;
      overflow-wrap: anywhere;
    }
    .cluster-list {
      display: grid;
      gap: 9px;
      max-height: 420px;
      overflow: auto;
      padding-right: 2px;
    }
    .cluster-card {
      cursor: pointer;
      border: 1px solid var(--line);
      border-left-width: 5px;
      border-radius: 8px;
      background: #fbfcff;
      padding: 11px;
      text-align: left;
      font: inherit;
    }
    .cluster-card[aria-pressed="true"] {
      background: #f1f5ff;
      border-color: #aebff0;
      box-shadow: inset 0 0 0 1px rgba(40,100,232,.12);
    }
    .cluster-card b {
      display: block;
      color: var(--ink);
      font-size: 14px;
      font-weight: 760;
    }
    .cluster-card span {
      display: block;
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }
    .cluster-meter {
      height: 6px;
      margin-top: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: #e8edf5;
    }
    .cluster-meter i {
      display: block;
      height: 100%;
      border-radius: inherit;
    }
    .chart-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      margin-bottom: 16px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(110px, 1fr) minmax(160px, 1.6fr) 72px;
      gap: 10px;
      align-items: center;
      margin: 9px 0;
      font-size: 13px;
    }
    .bar-label {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .bar-track {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e9eef6;
    }
    .bar-track i {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: var(--blue);
    }
    .bar-value {
      color: var(--muted);
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .terms {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .term {
      display: inline-flex;
      gap: 7px;
      align-items: baseline;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--surface-soft);
      padding: 6px 9px;
      font-size: 13px;
    }
    .term small {
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 650;
    }
    td:nth-child(1), td:nth-child(2), td:nth-child(3) {
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }
    .print-sheet {
      display: grid;
      grid-template-columns: minmax(0, .8fr) minmax(0, 1.2fr);
      gap: 16px;
      margin-bottom: 8px;
    }
    .print-items {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .print-items div {
      border-top: 2px solid var(--line);
      padding-top: 8px;
    }
    .print-items span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .print-items b {
      display: block;
      margin-top: 3px;
      font-size: 22px;
      font-variant-numeric: tabular-nums;
    }
    code {
      border-radius: 5px;
      background: #eef2f8;
      padding: 2px 5px;
      font-family: Consolas, monospace;
      font-size: .94em;
    }
    @media (max-width: 1100px) {
      main { padding: 20px; }
      .masthead, .intro-grid, .map-layout, .print-sheet { grid-template-columns: 1fr; }
      .source-stack { justify-items: start; }
      .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .map-aside { border-left: 0; border-top: 1px solid var(--line); }
      .cluster-list { max-height: none; }
      .db-head { grid-template-columns: 1fr; }
      .map-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      main { padding: 14px; }
      .metrics, .chart-grid, .risk-row, .print-items, .map-stats { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: 1fr; gap: 5px; }
      .bar-value { text-align: left; }
      canvas { height: 500px; }
      .map-stage { min-height: 500px; }
      .map-key { position: static; padding: 0 12px 12px; }
    }
  </style>
</head>
<body>
<main>
  <section class="masthead">
    <div>
      <div class="eyebrow">Agent Doctor Corpus Map</div>
      <h1>KorQuAD 검색 DB의 구조를 한눈에 읽는 지도</h1>
      <p class="lead">__MAIN_SENTENCE__ 문서 규모, 청크 품질, 군집 구조를 함께 보면서 RAG 검색 DB가 어디에 치우쳐 있는지 확인합니다.</p>
    </div>
    <div class="source-stack">
      <span>데이터: __SOURCE__</span>
      <span>버전: __VERSION__</span>
      <span>투영 설명분산: __EXPLAINED__</span>
    </div>
  </section>

  <section class="metrics">__METRICS__</section>

  <section class="grid intro-grid">
    <div class="panel">
      <h2>빠른 판독</h2>
      <p class="panel-note">점수처럼 단정하기보다, DB가 RAG 검색에 들어가기 전 어떤 상태인지 확인하는 기준입니다.</p>
      __SIGNALS__
    </div>
    <div class="panel">
      <h2>주의해서 볼 부분</h2>
      <p class="panel-note">짧은 청크는 맥락 부족, 긴 청크는 검색 정밀도 저하, 중복은 검색 결과 반복으로 이어질 수 있습니다.</p>
      <div class="risk-row">__RISKS__</div>
    </div>
  </section>

  <section class="panel db-panel">
    <div class="db-head">
      <div>
        <div class="kicker">Vector DB Atlas</div>
        <h2>청크가 어떤 주제 덩어리로 모였는지</h2>
        <p class="panel-note">점 하나는 검색 DB에 들어갈 청크 하나입니다. 배경의 진한 영역은 청크가 많이 몰린 곳, 색은 자동 군집, 점 크기는 청크 길이를 뜻합니다.</p>
      </div>
      <div class="map-stats">
        <span><b id="pointCount">-</b>지도 점</span>
        <span><b id="clusterCount">-</b>군집</span>
        <span><b id="visibleCount">-</b>표시 중</span>
      </div>
    </div>
    <div class="map-layout">
      <div class="map-stage">
        <canvas id="mapCanvas" aria-label="KorQuAD 청크 산점도"></canvas>
        <div class="tooltip" id="tooltip"></div>
        <div class="map-key">
          <span>밀도 배경 = 청크가 몰린 영역</span>
          <span>색 = 군집</span>
          <span>큰 점 = 긴 청크</span>
        </div>
      </div>
      <aside class="map-aside">
        <div class="detail" id="detail">
          <h3>군집을 선택하세요</h3>
          <p>군집 카드를 누르면 해당 주제 묶음만 강조됩니다.</p>
        </div>
        <div class="cluster-list" id="clusterDeck"></div>
      </aside>
    </div>
  </section>

  <section class="grid chart-grid">
    <div class="panel">
      <h2>청크 길이 분포</h2>
      <div id="tokenBars"></div>
    </div>
    <div class="panel">
      <h2>문서별 청크 수 분포</h2>
      <div id="chunkDocBars"></div>
    </div>
    <div class="panel">
      <h2>문서별 QA 수</h2>
      <div id="qaBars"></div>
    </div>
    <div class="panel">
      <h2>자주 등장한 핵심어</h2>
      <div class="terms" id="termList"></div>
    </div>
  </section>

  <section class="panel">
    <h2>군집별 해석 표</h2>
    <table>
      <thead>
        <tr><th>군집</th><th>점</th><th>비율</th><th>대표 단어</th><th>대표 문서</th></tr>
      </thead>
      <tbody id="clusterTable"></tbody>
    </table>
  </section>

  <section class="panel print-sheet">
    <div>
      <h2>프린트용 요약</h2>
      <p class="panel-note">회의 자료에 넣을 때는 이 숫자 묶음과 DB 지도를 함께 쓰면 충분합니다.</p>
    </div>
    <div class="print-items">__PRINT_ITEMS__</div>
  </section>
</main>

<script type="application/json" id="dashboardData">__PAYLOAD__</script>
<script>
const dashboard = JSON.parse(document.getElementById("dashboardData").textContent);
const colors = ["#2864e8", "#148a62", "#c97916", "#c2416f", "#6957d8", "#108498", "#c7352d", "#667085", "#7a5c2e"];
const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const detail = document.getElementById("detail");
let activeCluster = 0;
let selectedPoint = dashboard.points[0] || null;
let positions = [];

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmt(value) {
  return Number(value || 0).toLocaleString("ko-KR");
}

function chartBox() {
  const rect = canvas.getBoundingClientRect();
  return {w: rect.width, h: rect.height, l: 58, r: 26, t: 34, b: 54};
}

function xy(point) {
  const box = chartBox();
  return {
    x: box.l + ((point.x + 1) / 2) * (box.w - box.l - box.r),
    y: box.h - box.b - ((point.y + 1) / 2) * (box.h - box.t - box.b)
  };
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(320, rect.width) * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawMap();
}

function visiblePoints() {
  return dashboard.points.filter((point) => !activeCluster || point.cluster === activeCluster);
}

function updateCounters() {
  document.getElementById("pointCount").textContent = fmt(dashboard.points.length);
  document.getElementById("clusterCount").textContent = fmt(dashboard.clusters.length);
  document.getElementById("visibleCount").textContent = fmt(visiblePoints().length);
}

function clusterStats(points = dashboard.points) {
  const grouped = new Map();
  points.forEach((point) => {
    if (!grouped.has(point.cluster)) grouped.set(point.cluster, []);
    grouped.get(point.cluster).push(xy(point));
  });
  return [...grouped.entries()].map(([cluster, pts]) => {
    const mx = pts.reduce((sum, p) => sum + p.x, 0) / pts.length;
    const my = pts.reduce((sum, p) => sum + p.y, 0) / pts.length;
    const sx = Math.sqrt(pts.reduce((sum, p) => sum + (p.x - mx) ** 2, 0) / pts.length);
    const sy = Math.sqrt(pts.reduce((sum, p) => sum + (p.y - my) ** 2, 0) / pts.length);
    return {cluster, x: mx, y: my, sx, sy, count: pts.length};
  });
}

function drawRoundedRect(x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawDensity(points) {
  const box = chartBox();
  const cols = 42;
  const rows = 28;
  const cellW = (box.w - box.l - box.r) / cols;
  const cellH = (box.h - box.t - box.b) / rows;
  const bins = Array.from({length: rows}, () => Array(cols).fill(0));
  points.forEach((point) => {
    const pos = xy(point);
    const cx = Math.max(0, Math.min(cols - 1, Math.floor((pos.x - box.l) / cellW)));
    const cy = Math.max(0, Math.min(rows - 1, Math.floor((pos.y - box.t) / cellH)));
    bins[cy][cx] += 1;
  });
  const max = Math.max(...bins.flat(), 1);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const value = bins[y][x];
      if (!value) continue;
      const alpha = Math.min(.34, .05 + (value / max) * .32);
      ctx.fillStyle = `rgba(40, 100, 232, ${alpha})`;
      ctx.fillRect(box.l + x * cellW, box.t + y * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function drawAxes() {
  const box = chartBox();
  ctx.strokeStyle = "#d8e0ea";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#667085";
  ctx.font = "12px Segoe UI, Arial";
  [-1, -0.5, 0, 0.5, 1].forEach((tick) => {
    const px = box.l + ((tick + 1) / 2) * (box.w - box.l - box.r);
    const py = box.h - box.b - ((tick + 1) / 2) * (box.h - box.t - box.b);
    ctx.globalAlpha = tick === 0 ? .95 : .42;
    ctx.beginPath();
    ctx.moveTo(px, box.t);
    ctx.lineTo(px, box.h - box.b);
    ctx.moveTo(box.l, py);
    ctx.lineTo(box.w - box.r, py);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillText(String(tick), px - 7, box.h - 24);
    ctx.fillText(String(tick), 20, py + 4);
  });
}

function drawClusterEnvelopes(stats) {
  stats.forEach((cluster) => {
    const color = colors[(cluster.cluster - 1) % colors.length];
    ctx.save();
    ctx.globalAlpha = activeCluster ? .28 : .16;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.ellipse(
      cluster.x,
      cluster.y,
      Math.max(38, cluster.sx * 1.36),
      Math.max(30, cluster.sy * 1.36),
      0,
      0,
      Math.PI * 2
    );
    ctx.fill();
    ctx.globalAlpha = .32;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  });
}

function drawPoints(points) {
  positions = dashboard.points.map((point) => ({point, ...xy(point)}));
  positions.forEach(({point, x, y}) => {
    const dim = activeCluster && point.cluster !== activeCluster;
    const r = Math.max(2.4, Math.min(7.6, 2.4 + point.tokens / 68));
    ctx.globalAlpha = dim ? .08 : .74;
    ctx.fillStyle = colors[(point.cluster - 1) % colors.length];
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    if (!dim) {
      ctx.globalAlpha = .38;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });
  ctx.globalAlpha = 1;
}

function drawClusterLabels(stats) {
  ctx.font = "700 12px Segoe UI, Arial";
  stats.forEach((stat) => {
    const cluster = dashboard.clusters.find((item) => item.cluster === stat.cluster);
    if (!cluster) return;
    const label = `${stat.cluster}. ${cluster.label}`;
    const color = colors[(stat.cluster - 1) % colors.length];
    const width = Math.min(190, ctx.measureText(label).width + 20);
    const x = Math.max(64, Math.min(stat.x - width / 2, chartBox().w - width - 30));
    const y = Math.max(42, Math.min(stat.y - 18, chartBox().h - 88));
    ctx.save();
    ctx.globalAlpha = activeCluster && activeCluster !== stat.cluster ? .18 : .96;
    ctx.fillStyle = "rgba(255,255,255,.92)";
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    drawRoundedRect(x, y, width, 26, 13);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(label.length > 24 ? label.slice(0, 23) + "…" : label, x + 10, y + 17);
    ctx.restore();
  });
}

function drawSelected() {
  if (!selectedPoint) return;
  const pos = xy(selectedPoint);
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(17,24,39,.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, 18, 0, Math.PI * 2);
  ctx.stroke();
}

function drawMap() {
  const box = chartBox();
  const shown = visiblePoints();
  ctx.clearRect(0, 0, box.w, box.h);
  ctx.fillStyle = "#fbfcff";
  ctx.fillRect(0, 0, box.w, box.h);
  drawDensity(shown);
  drawAxes();
  const stats = clusterStats(shown);
  drawClusterEnvelopes(stats);
  drawPoints(shown);
  drawClusterLabels(stats);
  drawSelected();
  ctx.fillStyle = "#667085";
  ctx.font = "12px Segoe UI, Arial";
  ctx.fillText("X축: 2D 투영 축 1", box.w / 2 - 54, box.h - 10);
  ctx.save();
  ctx.translate(14, box.h / 2 + 54);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Y축: 2D 투영 축 2", 0, 0);
  ctx.restore();
  updateCounters();
}

function nearestPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  let best = null;
  let bestDist = 144;
  positions.forEach((pos) => {
    if (activeCluster && pos.point.cluster !== activeCluster) return;
    const dist = (pos.x - mx) ** 2 + (pos.y - my) ** 2;
    if (dist < bestDist) {
      best = pos;
      bestDist = dist;
    }
  });
  return best;
}

function showTooltip(pos, event) {
  if (!pos) {
    tooltip.style.display = "none";
    return;
  }
  tooltip.innerHTML = `<b>${esc(pos.point.title)}</b><span>군집 ${pos.point.cluster} · ${fmt(pos.point.tokens)}토큰</span>`;
  tooltip.style.display = "block";
  const wrap = canvas.parentElement.getBoundingClientRect();
  tooltip.style.left = Math.min(event.clientX - wrap.left + 12, wrap.width - 292) + "px";
  tooltip.style.top = Math.max(12, event.clientY - wrap.top - 8) + "px";
}

function showDetail(point) {
  selectedPoint = point;
  const cluster = dashboard.clusters.find((item) => item.cluster === point.cluster);
  detail.innerHTML = `
    <h3>${esc(point.title)}</h3>
    <p>${esc(point.preview)}</p>
    <dl>
      <dt>군집</dt><dd>${point.cluster}${cluster ? ` · ${esc(cluster.label)}` : ""}</dd>
      <dt>청크</dt><dd>${esc(point.chunk)}</dd>
      <dt>문서</dt><dd>${esc(point.doc)}</dd>
      <dt>길이</dt><dd>${fmt(point.tokens)}토큰 · ${fmt(point.chars)}자</dd>
      <dt>QA</dt><dd>${fmt(point.qa)}개</dd>
      <dt>원문</dt><dd>${point.url ? `<a href="${esc(point.url)}" target="_blank" rel="noreferrer">Wikipedia</a>` : "-"}</dd>
    </dl>
  `;
  drawMap();
}

function setCluster(cluster) {
  activeCluster = cluster;
  document.querySelectorAll(".cluster-card").forEach((button) => {
    button.setAttribute("aria-pressed", String(Number(button.dataset.cluster || 0) === cluster));
  });
  const candidate = dashboard.points.find((point) => cluster === 0 || point.cluster === cluster);
  if (candidate) {
    showDetail(candidate);
  } else {
    drawMap();
  }
}

function renderClusterDeck() {
  const root = document.getElementById("clusterDeck");
  const max = Math.max(...dashboard.clusters.map((cluster) => cluster.chunks), 1);
  const all = document.createElement("button");
  all.type = "button";
  all.className = "cluster-card";
  all.dataset.cluster = "0";
  all.setAttribute("aria-pressed", "true");
  all.style.borderLeftColor = "#111827";
  all.innerHTML = `<b>전체 보기</b><span>${fmt(dashboard.points.length)}개 점 · 모든 군집</span><div class="cluster-meter"><i style="width:100%;background:#111827"></i></div>`;
  all.addEventListener("click", () => setCluster(0));
  root.appendChild(all);
  dashboard.clusters.forEach((cluster) => {
    const color = colors[(cluster.cluster - 1) % colors.length];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cluster-card";
    button.dataset.cluster = String(cluster.cluster);
    button.setAttribute("aria-pressed", "false");
    button.style.borderLeftColor = color;
    button.innerHTML = `
      <b>${cluster.cluster}. ${esc(cluster.label)}</b>
      <span>${fmt(cluster.chunks)}개 점 · ${(cluster.share * 100).toFixed(1)}% · 평균 ${cluster.avg_tokens.toFixed(1)}토큰</span>
      <div class="cluster-meter"><i style="width:${(cluster.chunks / max * 100).toFixed(2)}%;background:${color}"></i></div>
    `;
    button.addEventListener("click", () => setCluster(cluster.cluster));
    root.appendChild(button);
  });
}

function renderBars(id, rows, color) {
  const root = document.getElementById(id);
  const max = Math.max(...rows.map((row) => row.count), 1);
  root.innerHTML = rows.map((row) => {
    const width = row.count / max * 100;
    return `<div class="bar-row">
      <div class="bar-label">${esc(row.label)}</div>
      <div class="bar-track"><i style="width:${width.toFixed(2)}%;background:${color}"></i></div>
      <div class="bar-value">${fmt(row.count)}</div>
    </div>`;
  }).join("");
}

function renderTerms() {
  document.getElementById("termList").innerHTML = dashboard.terms.map((row) =>
    `<span class="term"><b>${esc(row.term)}</b><small>${fmt(row.count)}</small></span>`
  ).join("");
}

function renderClusterTable() {
  document.getElementById("clusterTable").innerHTML = dashboard.clusters.map((cluster) => {
    const docs = cluster.docs.map((row) => row.title).join(", ");
    const terms = cluster.terms.join(", ");
    return `<tr>
      <td>${cluster.cluster}</td>
      <td>${fmt(cluster.chunks)}</td>
      <td>${(cluster.share * 100).toFixed(1)}%</td>
      <td>${esc(terms || cluster.label)}</td>
      <td>${esc(docs)}</td>
    </tr>`;
  }).join("");
}

canvas.addEventListener("mousemove", (event) => showTooltip(nearestPoint(event), event));
canvas.addEventListener("mouseleave", () => tooltip.style.display = "none");
canvas.addEventListener("click", (event) => {
  const pos = nearestPoint(event);
  if (pos) showDetail(pos.point);
});
window.addEventListener("resize", resizeCanvas);

renderClusterDeck();
renderBars("tokenBars", dashboard.token_bins, "#148a62");
renderBars("chunkDocBars", dashboard.chunk_doc_bins, "#2864e8");
renderBars("qaBars", dashboard.qa_bins, "#c2416f");
renderTerms();
renderClusterTable();
if (selectedPoint) showDetail(selectedPoint);
resizeCanvas();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
