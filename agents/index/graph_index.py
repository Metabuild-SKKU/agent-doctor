# chunk-document-entity 관계 그래프를 만드는 쪽. 검색보다는 구조 확인/시각화용이다.
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from core.llm_usage import log_usage
from core.schema import Chunk

_STOPWORDS = {
    "그리고", "그러나", "대한", "위한", "있는", "한다", "에서", "으로",
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
}


# OPENAI_API_KEY가 없어도 그래프 산출물이 나오도록 keyword 방식으로 대체한다.
def _keyword_entities(text: str, limit: int = 8) -> tuple[list[str], list[dict]]:
    tokens = re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9_+.-]{1,}", text)
    counts = Counter(token for token in tokens if token.lower() not in _STOPWORDS)
    entities = [token for token, _ in counts.most_common(limit)]
    relations = [
        {"source": left, "target": right, "type": "co_occurs"}
        for left, right in zip(entities, entities[1:])
    ]
    return entities, relations


# LLM을 쓸 수 있으면 entity/relation JSON만 받아온다.
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
    if response.usage:
        log_usage(model, response.usage.prompt_tokens, response.usage.completion_tokens, tag="Index")
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


# 설정과 API key 상태에 따라 LLM/keyword 추출을 고른다.
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


# ── 재생성 회피 캐시 ─────────────────────────────────────────────
# 그래프 산출물은 시각화/구조 확인용이라 튜닝 루프(Optimize→Index 재실행)마다
# 새로 만들 필요가 없다. 청크 집합·그래프 설정이 같으면 통째로 스킵하고(manifest),
# 일부 청크만 바뀐 재청킹에서는 바뀐 청크만 entity 추출을 다시 한다(entity cache).

def _extraction_ctx(config: dict) -> str:
    # entity 추출 결과에 영향을 주는 조건: 요청 모드, LLM 모델, 키 존재 여부.
    mode = config.get("graph_extraction", "auto")
    llm_on = 1 if (mode in {"auto", "llm"} and os.getenv("OPENAI_API_KEY")) else 0
    return f"{mode}:{config.get('graph_llm_model', 'gpt-4.1-mini')}:{llm_on}"


def _graph_signature(chunks: list[Chunk], config: dict) -> str:
    # 청크 텍스트(hash)·임베딩 모델(유사도 edge에 반영)·그래프 설정이 모두 같아야 재사용.
    chunk_keys = sorted(
        f"{c.chunk_id}:{c.hash}:{c.metadata.get('embedding_model', '')}" for c in chunks
    )
    payload = json.dumps(
        {
            "chunks": chunk_keys,
            "extraction": _extraction_ctx(config),
            "threshold": float(config.get("graph_similarity_threshold", 0.9)),
            "limit": int(config.get("graph_similarity_limit", 200)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_manifest_artifacts(path: Path, signature: str) -> dict | None:
    manifest = _load_json(path)
    if manifest.get("signature") != signature:
        return None
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    # 산출물 파일이 지워졌으면 캐시를 무효로 본다 (pyvis는 원래 None 허용).
    for key in ("graphml", "mermaid"):
        value = artifacts.get(key)
        if not value or not Path(value).exists():
            return None
    return artifacts


def _cached_extract(
    chunk: Chunk, config: dict, entity_cache: dict, ctx: str
) -> tuple[list[str], list[dict], str, bool]:
    """entity 추출(캐시 우선). 반환: (entities, relations, used_mode, cache_updated).

    캐시에는 실제 사용된 모드(llm 실패 → keyword 폴백 포함)가 그대로 남는다 —
    시각화 전용 산출물이라 폴백 결과 재사용을 허용한다."""
    key = f"{chunk.hash}:{ctx}"
    entry = entity_cache.get(key)
    if isinstance(entry, dict) and "entities" in entry and "relations" in entry:
        return entry["entities"], entry["relations"], entry.get("mode", "keyword"), False
    entities, relations, used_mode = _extract(chunk, config)
    entity_cache[key] = {"entities": entities, "relations": relations, "mode": used_mode}
    return entities, relations, used_mode, True


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _safe_id(value: str) -> str:
    return "n" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


# README나 Notion에 붙여 보기 쉽게 Mermaid 파일도 남긴다.
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


# 청크 쌍 유사도 계산: numpy가 있으면 행렬곱, 없으면 기존 이중 루프.
# (시각화 전용 edge라 float 미세차로 threshold 경계의 edge가 달라져도 평가와 무관.)
def _similar_pairs(candidates: list[Chunk], threshold: float) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    try:
        import numpy as np

        dims = {len(c.embedding) for c in candidates if c.embedding}
        if len(dims) == 1:
            indexed = [(i, c.embedding) for i, c in enumerate(candidates) if c.embedding]
            if len(indexed) >= 2:
                matrix = np.asarray([vec for _, vec in indexed], dtype=np.float64)
                norms = np.linalg.norm(matrix, axis=1)
                norms[norms == 0.0] = 1.0
                scores = (matrix / norms[:, None]) @ (matrix / norms[:, None]).T
                rows, cols = np.where(np.triu(scores, k=1) >= threshold)
                return [
                    (indexed[r][0], indexed[c][0], float(scores[r, c]))
                    for r, c in zip(rows.tolist(), cols.tolist())
                ]
            return []
        # 차원이 섞여 있으면(모델 혼재 등) 기존 루프가 0.0 처리까지 맡는다.
    except ImportError:
        pass
    for index, left in enumerate(candidates):
        for offset, right in enumerate(candidates[index + 1 :], start=index + 1):
            score = _cosine(left.embedding or [], right.embedding or [])
            if score >= threshold:
                pairs.append((index, offset, float(score)))
    return pairs


# NetworkX 그래프와 export 파일들을 한 번에 만든다.
def build_graph_artifacts(chunks: list[Chunk], config: dict) -> dict:
    import networkx as nx

    output_dir = Path(config.get("graph_output_dir", "output/index_graph"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 같은 청크 집합·설정이면 이전 산출물을 그대로 반환 (LLM 추출·유사도 계산 스킵)
    signature = _graph_signature(chunks, config)
    manifest_path = output_dir / "graph_manifest.json"
    cached = _load_manifest_artifacts(manifest_path, signature)
    if cached is not None:
        print("[Index] 그래프 산출물 재사용 (청크·설정 변화 없음)")
        return cached

    entity_cache_path = output_dir / "entity_cache.json"
    entity_cache = _load_json(entity_cache_path)
    extraction_ctx = _extraction_ctx(config)
    cache_dirty = False

    graph = nx.MultiDiGraph()
    extraction_modes: set[str] = set()

    for chunk in chunks:
        document_node = f"doc:{chunk.doc_id}"
        chunk_node = f"chunk:{chunk.chunk_id}"
        graph.add_node(document_node, kind="document", label=chunk.metadata.get("title", chunk.doc_id))
        graph.add_node(chunk_node, kind="chunk", label=chunk.section or chunk.chunk_id)
        graph.add_edge(document_node, chunk_node, relation="contains")

        entities, relations, used_mode, updated = _cached_extract(
            chunk, config, entity_cache, extraction_ctx
        )
        cache_dirty = cache_dirty or updated
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
    for left_index, right_index, score in _similar_pairs(candidates, threshold):
        graph.add_edge(
            f"chunk:{candidates[left_index].chunk_id}",
            f"chunk:{candidates[right_index].chunk_id}",
            relation="similar_to",
            score=score,
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

    artifacts = {
        "graphml": str(graphml_path),
        "mermaid": str(mermaid_path),
        "pyvis": str(pyvis_path) if pyvis_path else None,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "graph_extraction": sorted(extraction_modes),
    }

    # 캐시 저장은 best-effort — 실패해도 산출물 자체는 유효하다.
    try:
        if cache_dirty:
            entity_cache_path.write_text(
                json.dumps(entity_cache, ensure_ascii=False), encoding="utf-8"
            )
        manifest_path.write_text(
            json.dumps({"signature": signature, "artifacts": artifacts}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[Index] 그래프 캐시 저장 생략: {exc}")

    return artifacts
