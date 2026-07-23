"""Build report-ready corpus and vector DB visualization artifacts.

The visualizer is intentionally dependency-light. It works directly from the
Chunk objects produced by Index, so the report layer can explain what is inside
the RAG corpus without reaching into Qdrant internals.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from core.schema import Chunk

DEFAULT_OUTPUT_DIR = "output/corpus_visualization"
DEFAULT_MAX_POINTS = 500

_STOPWORDS = {
    "그리고",
    "그러나",
    "대한",
    "위한",
    "있는",
    "한다",
    "에서",
    "으로",
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "was",
}


def build_corpus_visualization_artifacts(chunks: list[Chunk], config: dict) -> dict:
    """Write JSON and standalone HTML artifacts for report consumption."""
    output_dir = Path(config.get("corpus_visualization_output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_points = int(config.get("corpus_visualization_max_points", DEFAULT_MAX_POINTS))
    data = build_corpus_visualization_data(chunks, config, max_points=max_points)

    json_path = output_dir / "corpus_summary.json"
    html_path = output_dir / "corpus_visualization.html"
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(_render_html(data), encoding="utf-8")

    return {
        "html": str(html_path),
        "json": str(json_path),
        "summary": data["summary"],
        "projection_method": data["projection"]["method"],
        "cluster_count": len(data["projection"]["clusters"]),
        "point_count": len(data["projection"]["points"]),
    }


def build_corpus_visualization_data(
    chunks: list[Chunk],
    config: dict,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict:
    """Create the compact data model used by the report visualization."""
    documents = _document_summaries(chunks)
    tokens = [_chunk_tokens(chunk) for chunk in chunks]
    embedded_chunks = [chunk for chunk in chunks if _embedding(chunk)]
    vector_dim = len(_embedding(embedded_chunks[0])) if embedded_chunks else 0
    hashes = [chunk.hash or chunk.metadata.get("chunk_hash") for chunk in chunks]
    hashed_chunks = [value for value in hashes if value]
    unique_hashes = len(set(hashed_chunks))
    duplicate_ratio = (
        max(0, len(hashed_chunks) - unique_hashes) / len(hashed_chunks)
        if hashed_chunks
        else 0.0
    )

    summary = {
        "documents": len(documents),
        "chunks": len(chunks),
        "total_tokens": sum(tokens),
        "avg_tokens_per_chunk": _round(sum(tokens) / len(tokens)) if tokens else 0,
        "median_tokens_per_chunk": _round(median(tokens)) if tokens else 0,
        "avg_chunks_per_document": _round(len(chunks) / len(documents)) if documents else 0,
        "embedding_coverage": _round(len(embedded_chunks) / len(chunks)) if chunks else 0,
        "embedding_dimension": vector_dim,
        "duplicate_chunk_ratio": _round(duplicate_ratio),
        "chunk_strategy": config.get("chunk_strategy") or config.get("chunk_stage") or "",
        "embedding_model": config.get("embedding_model", ""),
    }

    return {
        "summary": summary,
        "documents": documents[:50],
        "token_bins": _token_bins(tokens),
        "top_terms": _top_terms(chunks),
        "projection": _embedding_projection(chunks, max_points=max_points),
    }


def _document_summaries(chunks: list[Chunk]) -> list[dict]:
    docs: dict[str, dict[str, Any]] = {}
    sections: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        metadata = chunk.metadata or {}
        doc_id = chunk.doc_id
        doc = docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": str(metadata.get("title") or doc_id),
                "source": str(metadata.get("source") or ""),
                "chunks": 0,
                "tokens": 0,
                "avg_tokens": 0,
                "sections": 0,
            },
        )
        doc["chunks"] += 1
        doc["tokens"] += _chunk_tokens(chunk)
        if chunk.section:
            sections[doc_id].add(chunk.section)

    for doc_id, doc in docs.items():
        doc["avg_tokens"] = _round(doc["tokens"] / doc["chunks"]) if doc["chunks"] else 0
        doc["sections"] = len(sections[doc_id])

    return sorted(docs.values(), key=lambda item: (-item["chunks"], item["title"]))


def _embedding_projection(chunks: list[Chunk], *, max_points: int) -> dict:
    sample = _sample_chunks([chunk for chunk in chunks if _embedding(chunk)], max_points)
    raw_points = []
    for chunk in sample:
        x, y = _project_vector(_embedding(chunk))
        raw_points.append((chunk, x, y))

    if not raw_points:
        return {"method": "none", "points": [], "clusters": []}

    xs = [point[1] for point in raw_points]
    ys = [point[2] for point in raw_points]
    normalized = [
        (chunk, _normalize_axis(x, min(xs), max(xs)), _normalize_axis(y, min(ys), max(ys)))
        for chunk, x, y in raw_points
    ]
    labels = _cluster_labels([(x, y) for _, x, y in normalized])

    points = []
    cluster_docs: dict[int, Counter] = defaultdict(Counter)
    cluster_tokens: dict[int, list[int]] = defaultdict(list)
    for (chunk, x, y), cluster in zip(normalized, labels):
        metadata = chunk.metadata or {}
        title = str(metadata.get("title") or chunk.doc_id)
        tokens = _chunk_tokens(chunk)
        cluster_docs[cluster][title] += 1
        cluster_tokens[cluster].append(tokens)
        points.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": title,
                "section": chunk.section or "",
                "tokens": tokens,
                "x": _round(x, 4),
                "y": _round(y, 4),
                "cluster": cluster + 1,
                "preview": _preview(chunk.text),
            }
        )

    clusters = []
    for cluster in sorted(set(labels)):
        token_values = cluster_tokens[cluster]
        top_doc, _ = cluster_docs[cluster].most_common(1)[0]
        clusters.append(
            {
                "cluster": cluster + 1,
                "chunks": len(token_values),
                "top_document": top_doc,
                "avg_tokens": _round(sum(token_values) / len(token_values)),
            }
        )

    return {
        "method": "deterministic_random_projection",
        "points": points,
        "clusters": clusters,
    }


def _cluster_labels(points: list[tuple[float, float]]) -> list[int]:
    if len(points) < 8:
        return [0 for _ in points]

    k = min(6, max(2, round(math.sqrt(len(points) / 3))))
    ordered = sorted(points)
    centers = [
        ordered[round(index * (len(ordered) - 1) / max(1, k - 1))]
        for index in range(k)
    ]
    labels = [0 for _ in points]

    for _ in range(12):
        changed = False
        for index, point in enumerate(points):
            label = min(
                range(k),
                key=lambda center: _distance(point, centers[center]),
            )
            if labels[index] != label:
                labels[index] = label
                changed = True
        if not changed:
            break

        grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for label, point in zip(labels, points):
            grouped[label].append(point)
        for label, values in grouped.items():
            centers[label] = (
                sum(point[0] for point in values) / len(values),
                sum(point[1] for point in values) / len(values),
            )

    return labels


def _sample_chunks(chunks: list[Chunk], max_points: int) -> list[Chunk]:
    if max_points <= 0 or len(chunks) <= max_points:
        return chunks
    if max_points == 1:
        return [chunks[0]]
    step = (len(chunks) - 1) / (max_points - 1)
    return [chunks[round(index * step)] for index in range(max_points)]


def _project_vector(vector: list[float]) -> tuple[float, float]:
    x = 0.0
    y = 0.0
    for index, value in enumerate(vector):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        x += numeric * math.sin((index + 1) * 12.9898)
        y += numeric * math.cos((index + 1) * 78.233)
    return x, y


def _token_bins(tokens: list[int]) -> list[dict]:
    if not tokens:
        return []
    low = min(tokens)
    high = max(tokens)
    if low == high:
        return [{"start": low, "end": high, "label": str(low), "count": len(tokens)}]

    bin_count = min(12, max(4, round(math.sqrt(len(tokens)))))
    width = max(1, math.ceil((high - low + 1) / bin_count))
    bins = []
    for start in range(low, high + 1, width):
        end = min(high, start + width - 1)
        bins.append(
            {
                "start": start,
                "end": end,
                "label": f"{start}-{end}",
                "count": sum(1 for value in tokens if start <= value <= end),
            }
        )
    return bins


def _top_terms(chunks: list[Chunk], limit: int = 12) -> list[dict]:
    counts: Counter[str] = Counter()
    for chunk in chunks:
        counts.update(
            token
            for token in _tokens(chunk.text)
            if token.lower() not in _STOPWORDS and len(token) > 1
        )
    return [
        {"term": term, "count": count}
        for term, count in counts.most_common(limit)
    ]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[가-힣]+|[A-Za-z][A-Za-z0-9_+.-]*|\d+", text.lower())


def _chunk_tokens(chunk: Chunk) -> int:
    if isinstance(chunk.token_count, int) and chunk.token_count > 0:
        return chunk.token_count
    return max(1, len(_tokens(chunk.text or "")))


def _embedding(chunk: Chunk) -> list[float]:
    return chunk.embedding or []


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def _normalize_axis(value: float, low: float, high: float) -> float:
    if math.isclose(low, high):
        return 0.0
    return ((value - low) / (high - low)) * 2 - 1


def _preview(text: str, limit: int = 180) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    summary = data["summary"]
    documents = data["documents"]
    token_bins = data["token_bins"]
    top_terms = data["top_terms"]
    clusters = data["projection"]["clusters"]

    document_rows = "\n".join(_document_bar(row, documents) for row in documents[:12])
    token_rows = "\n".join(_token_bar(row, token_bins) for row in token_bins)
    term_rows = "\n".join(
        f'<span class="term"><b>{_html(term["term"])}</b> {term["count"]}</span>'
        for term in top_terms
    )
    cluster_rows = "\n".join(
        (
            "<tr>"
            f"<td>{cluster['cluster']}</td>"
            f"<td>{cluster['chunks']}</td>"
            f"<td>{_html(cluster['top_document'])}</td>"
            f"<td>{cluster['avg_tokens']}</td>"
            "</tr>"
        )
        for cluster in clusters
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>검색 자료 현황</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #64748b;
      --line: #d8dee9;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --green: #059669;
      --amber: #d97706;
      --pink: #db2777;
      --purple: #7c3aed;
      --cyan: #0891b2;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Pretendard, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-end; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; font-weight: 500; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; font-weight: 500; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .badge {{ color: var(--accent); background: var(--accent-soft); padding: 6px 10px; border-radius: 999px; font-size: 13px; white-space: nowrap; }}
    .notice {{ background: #fff7ed; border: 1px solid #fed7aa; color: #7c2d12; border-radius: 10px; padding: 12px 14px; margin: 16px 0 22px; }}
    .grid {{ display: grid; gap: 16px; }}
    .stats {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .two {{ grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr); align-items: start; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 18px; box-shadow: var(--shadow); }}
    .stat-label {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
    .stat-value {{ font-size: 27px; font-weight: 500; }}
    .stat-note {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(120px, 1fr) minmax(160px, 2fr) 64px; gap: 10px; align-items: center; margin: 10px 0; }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }}
    .bar-track {{ height: 12px; border-radius: 999px; background: #eef2f7; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--accent), var(--cyan)); }}
    .bar-value {{ color: var(--muted); font-variant-numeric: tabular-nums; text-align: right; font-size: 13px; }}
    .hist .bar-fill {{ background: linear-gradient(90deg, var(--green), var(--amber)); }}
    .terms {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .term {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 9px; background: #f8fafc; color: var(--muted); font-size: 13px; }}
    .term b {{ color: var(--text); font-weight: 500; }}
    .scatter-wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 290px; gap: 16px; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .axis {{ stroke: var(--line); stroke-width: 1; }}
    .point {{ cursor: pointer; opacity: 0.86; stroke: #fff; stroke-width: 1.5; }}
    .point:hover, .point.active {{ opacity: 1; stroke: var(--text); stroke-width: 2; }}
    .detail {{ min-height: 260px; }}
    .detail h3 {{ margin: 0 0 8px; font-size: 16px; font-weight: 500; }}
    .detail dl {{ margin: 12px 0 0; display: grid; grid-template-columns: 82px 1fr; gap: 6px 10px; font-size: 13px; }}
    .detail dt {{ color: var(--muted); }}
    .detail dd {{ margin: 0; overflow-wrap: anywhere; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 500; }}
    td:nth-child(1), td:nth-child(2), td:nth-child(4) {{ font-variant-numeric: tabular-nums; }}
    details {{ margin-top: 16px; color: var(--muted); }}
    summary {{ cursor: pointer; color: var(--text); font-weight: 500; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 5px; }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      header, .scatter-wrap {{ grid-template-columns: 1fr; display: grid; align-items: start; }}
      .stats, .two {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 620px) {{
      .stats, .two {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>검색 자료 현황</h1>
      <p>인덱싱된 문서와 청크가 어떤 비율과 분포로 RAG 검색 DB에 들어갔는지 확인합니다.</p>
    </div>
    <span class="badge">{_html(str(summary.get("embedding_model") or "embedding"))}</span>
  </header>

  <div class="notice">
    이 화면은 점수 판단을 돕는 보조 자료입니다. 청크 수나 군집 모양만으로 답변 품질을 확정하지 말고, 실패한 검증 질문과 함께 확인해야 합니다.
  </div>

  <section class="grid stats">
    {_stat_card("문서", summary["documents"], "인덱싱된 원본 문서 수")}
    {_stat_card("청크", summary["chunks"], "검색 단위로 쪼개진 텍스트 수")}
    {_stat_card("총 토큰", summary["total_tokens"], "청크 token_count 합계")}
    {_stat_card("임베딩 커버리지", f'{summary["embedding_coverage"] * 100:.0f}%', f'{summary["embedding_dimension"]}차원 벡터 기준')}
  </section>

  <section class="grid two" style="margin-top:16px;">
    <div class="card">
      <h2>문서별 청크 분포</h2>
      {document_rows or '<p>표시할 문서가 없습니다.</p>'}
    </div>
    <div class="card hist">
      <h2>청크 길이 분포</h2>
      {token_rows or '<p>표시할 토큰 정보가 없습니다.</p>'}
    </div>
  </section>

  <section class="card" style="margin-top:16px;">
    <h2>임베딩 분포와 군집</h2>
    <div class="scatter-wrap">
      <svg id="embedding-scatter" viewBox="0 0 760 420" role="img" aria-label="청크 임베딩 2차원 분포"></svg>
      <aside class="detail" id="point-detail">
        <h3>청크를 선택하세요</h3>
        <p>점 하나가 검색 DB에 저장된 청크 하나입니다. 가까운 점들은 임베딩 기준으로 비슷한 내용을 담고 있을 가능성이 큽니다.</p>
      </aside>
    </div>
  </section>

  <section class="grid two" style="margin-top:16px;">
    <div class="card">
      <h2>자주 등장한 용어</h2>
      <div class="terms">{term_rows or '<p>표시할 용어가 없습니다.</p>'}</div>
    </div>
    <div class="card">
      <h2>군집 요약</h2>
      <table>
        <thead><tr><th>군집</th><th>청크</th><th>대표 문서</th><th>평균 토큰</th></tr></thead>
        <tbody>{cluster_rows or '<tr><td colspan="4">임베딩 정보가 없습니다.</td></tr>'}</tbody>
      </table>
    </div>
  </section>

  <details>
    <summary>개발자용 산출물 설명</summary>
    <p>
      JSON 산출물은 <code>summary</code>, <code>documents</code>, <code>token_bins</code>,
      <code>top_terms</code>, <code>projection.points</code>를 포함합니다.
      현재 2D 좌표는 추가 의존성 없이 재현 가능한 deterministic projection으로 계산합니다.
    </p>
  </details>
</main>
<script type="application/json" id="corpus-data">{payload}</script>
<script>
  const corpusData = JSON.parse(document.getElementById("corpus-data").textContent);
  const points = corpusData.projection.points;
  const colors = ["#2563eb", "#059669", "#d97706", "#db2777", "#7c3aed", "#0891b2"];
  const svg = document.getElementById("embedding-scatter");
  const detail = document.getElementById("point-detail");

  function esc(value) {{
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }}

  function setDetail(point) {{
    detail.innerHTML = `
      <h3>${{esc(point.title)}}</h3>
      <p>${{esc(point.preview)}}</p>
      <dl>
        <dt>청크</dt><dd>${{esc(point.chunk_id)}}</dd>
        <dt>문서</dt><dd>${{esc(point.doc_id)}}</dd>
        <dt>섹션</dt><dd>${{esc(point.section || "-")}}</dd>
        <dt>군집</dt><dd>${{point.cluster}}</dd>
        <dt>토큰</dt><dd>${{point.tokens}}</dd>
      </dl>
    `;
  }}

  function drawScatter() {{
    const width = 760;
    const height = 420;
    const pad = 38;
    svg.innerHTML = "";

    const make = (name) => document.createElementNS("http://www.w3.org/2000/svg", name);
    const title = make("title");
    title.textContent = "청크 임베딩 2차원 분포";
    svg.appendChild(title);

    const xAxis = make("line");
    xAxis.setAttribute("class", "axis");
    xAxis.setAttribute("x1", pad);
    xAxis.setAttribute("x2", width - pad);
    xAxis.setAttribute("y1", height / 2);
    xAxis.setAttribute("y2", height / 2);
    svg.appendChild(xAxis);

    const yAxis = make("line");
    yAxis.setAttribute("class", "axis");
    yAxis.setAttribute("x1", width / 2);
    yAxis.setAttribute("x2", width / 2);
    yAxis.setAttribute("y1", pad);
    yAxis.setAttribute("y2", height - pad);
    svg.appendChild(yAxis);

    if (!points.length) {{
      const empty = make("text");
      empty.setAttribute("x", width / 2);
      empty.setAttribute("y", height / 2);
      empty.setAttribute("text-anchor", "middle");
      empty.setAttribute("fill", "#64748b");
      empty.textContent = "표시할 임베딩이 없습니다.";
      svg.appendChild(empty);
      return;
    }}

    points.forEach((point, index) => {{
      const cx = pad + ((point.x + 1) / 2) * (width - pad * 2);
      const cy = height - pad - ((point.y + 1) / 2) * (height - pad * 2);
      const circle = make("circle");
      circle.setAttribute("class", "point");
      circle.setAttribute("cx", cx.toFixed(1));
      circle.setAttribute("cy", cy.toFixed(1));
      circle.setAttribute("r", Math.max(4, Math.min(9, 4 + point.tokens / 180)).toFixed(1));
      circle.setAttribute("fill", colors[(point.cluster - 1) % colors.length]);
      circle.setAttribute("aria-label", `${{point.title}} ${{point.chunk_id}}`);
      circle.addEventListener("click", () => {{
        svg.querySelectorAll(".point.active").forEach((node) => node.classList.remove("active"));
        circle.classList.add("active");
        setDetail(point);
      }});
      svg.appendChild(circle);
      if (index === 0) {{
        circle.classList.add("active");
        setDetail(point);
      }}
    }});
  }}

  drawScatter();
</script>
</body>
</html>
"""


def _stat_card(label: str, value: Any, note: str) -> str:
    return (
        '<div class="card">'
        f'<div class="stat-label">{_html(label)}</div>'
        f'<div class="stat-value">{_html(str(value))}</div>'
        f'<div class="stat-note">{_html(note)}</div>'
        "</div>"
    )


def _document_bar(row: dict, rows: list[dict]) -> str:
    max_chunks = max((item["chunks"] for item in rows), default=1)
    width = (row["chunks"] / max_chunks) * 100 if max_chunks else 0
    return (
        '<div class="bar-row">'
        f'<div class="bar-label" title="{_html(row["title"])}">{_html(row["title"])}</div>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{width:.2f}%"></div>'
        "</div>"
        f'<div class="bar-value">{row["chunks"]}개</div>'
        "</div>"
    )


def _token_bar(row: dict, rows: list[dict]) -> str:
    max_count = max((item["count"] for item in rows), default=1)
    width = (row["count"] / max_count) * 100 if max_count else 0
    return (
        '<div class="bar-row">'
        f'<div class="bar-label">{_html(row["label"])}</div>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{width:.2f}%"></div>'
        "</div>"
        f'<div class="bar-value">{row["count"]}개</div>'
        "</div>"
    )


def _html(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
