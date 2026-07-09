"""Chunk에서 문서-청크-개념 관계를 만들고 시각화 파일로 저장한다."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from core.schema import Chunk

_STOPWORDS = {
    "그리고", "그러나", "대한", "위한", "있는", "한다", "에서", "으로",
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
}


def _keyword_entities(text: str, limit: int = 8) -> tuple[list[str], list[dict]]:
    tokens = re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9_+.-]{1,}", text)
    counts = Counter(token for token in tokens if token.lower() not in _STOPWORDS)
    entities = [token for token, _ in counts.most_common(limit)]
    relations = [
        {"source": left, "target": right, "type": "co_occurs"}
        for left, right in zip(entities, entities[1:])
    ]
    return entities, relations


def _llm_entities(text: str, model: str) -> tuple[list[str], list[dict]]:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "기술 문서에서 핵심 entity와 entity 간 relation을 추출한다. "
                    '반드시 {"entities":["..."],"relations":'
                    '[{"source":"...","target":"...","type":"..."}]} JSON으로 답한다.'
                ),
            },
            {"role": "user", "content": text[:8000]},
        ],
    )
    data = json.loads(response.choices[0].message.content or "{}")
    entities = [str(item).strip() for item in data.get("entities", []) if str(item).strip()]
    relations = [
        {
            "source": str(item.get("source", "")).strip(),
            "target": str(item.get("target", "")).strip(),
            "type": str(item.get("type", "related_to")).strip() or "related_to",
        }
        for item in data.get("relations", [])
        if item.get("source") and item.get("target")
    ]
    return entities[:12], relations[:20]


def _extract(chunk: Chunk, config: dict) -> tuple[list[str], list[dict], str]:
    mode = config.get("graph_extraction", "auto")
    if mode in {"auto", "llm"} and os.getenv("OPENAI_API_KEY"):
        try:
            entities, relations = _llm_entities(
                chunk.text,
                config.get("graph_llm_model", "gpt-4.1-mini"),
            )
            return entities, relations, "llm"
        except Exception as exc:
            print(f"[Index] LLM graph 추출 실패, keyword 방식 사용: {exc}")
    entities, relations = _keyword_entities(chunk.text)
    return entities, relations, "keyword"


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _safe_id(value: str) -> str:
    return "n" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _write_mermaid(graph: Any, path: Path) -> None:
    lines = ["```mermaid", "graph LR"]
    for node, data in graph.nodes(data=True):
        label = (
            str(data.get("label", node))
            .replace('"', "'")
            .replace("\n", " ")
        )
        lines.append(f'    {_safe_id(node)}["{label}"]')
    for source, target, data in graph.edges(data=True):
        relation = (
            str(data.get("relation", "related_to"))
            .replace('"', "'")
            .replace("|", "/")
            .replace("\n", " ")
        )
        lines.append(f"    {_safe_id(source)} -->|{relation}| {_safe_id(target)}")
    lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_graph_artifacts(chunks: list[Chunk], config: dict) -> dict:
    """NetworkX 그래프와 Mermaid/PyVis 산출물을 만든다."""
    import networkx as nx

    output_dir = Path(config.get("graph_output_dir", "output/index_graph"))
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    extraction_modes: set[str] = set()

    for chunk in chunks:
        document_node = f"doc:{chunk.doc_id}"
        chunk_node = f"chunk:{chunk.chunk_id}"
        graph.add_node(document_node, kind="document", label=chunk.metadata.get("title", chunk.doc_id))
        graph.add_node(chunk_node, kind="chunk", label=chunk.section or chunk.chunk_id)
        graph.add_edge(document_node, chunk_node, relation="contains")

        entities, relations, used_mode = _extract(chunk, config)
        extraction_modes.add(used_mode)
        for entity in entities:
            entity_node = f"entity:{entity.lower()}"
            graph.add_node(entity_node, kind="entity", label=entity)
            graph.add_edge(chunk_node, entity_node, relation="mentions")
        for relation in relations:
            source = f"entity:{relation['source'].lower()}"
            target = f"entity:{relation['target'].lower()}"
            graph.add_node(source, kind="entity", label=relation["source"])
            graph.add_node(target, kind="entity", label=relation["target"])
            graph.add_edge(source, target, relation=relation["type"])

    threshold = float(config.get("graph_similarity_threshold", 0.9))
    candidates = chunks[: int(config.get("graph_similarity_limit", 200))]
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            score = _cosine(left.embedding or [], right.embedding or [])
            if score >= threshold:
                graph.add_edge(
                    f"chunk:{left.chunk_id}",
                    f"chunk:{right.chunk_id}",
                    relation="similar_to",
                    score=float(score),
                )

    graphml_path = output_dir / "index_graph.graphml"
    mermaid_path = output_dir / "index_graph.md"
    nx.write_graphml(graph, graphml_path)
    _write_mermaid(graph, mermaid_path)

    pyvis_path: Path | None = None
    try:
        from pyvis.network import Network

        network = Network(
            height="750px",
            width="100%",
            directed=True,
            cdn_resources="remote",
        )
        network.from_nx(graph)
        pyvis_path = output_dir / "index_graph.html"
        network.write_html(str(pyvis_path), open_browser=False, notebook=False)
    except Exception as exc:
        print(f"[Index] PyVis 생성 생략: {exc}")

    return {
        "graphml": str(graphml_path),
        "mermaid": str(mermaid_path),
        "pyvis": str(pyvis_path) if pyvis_path else None,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "graph_extraction": sorted(extraction_modes),
    }
