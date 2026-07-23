"""
agents/eval/probe_gen.py
STEP1: Probe(진단용 질문) 생성

설계 문서 'STEP 1: Probe 생성' 구현.
Probe = RAG 파이프라인을 진단하기 위한 테스트 질문 집합.

Probe 소스별 신뢰도 (설계 §3):
    user_log(실사용 쿼리)  >  taxonomy(사람 작성)  >  llm_generated(자동 생성)

생성 우선순위:
    1) state.user_questions 가 있으면 → user_log Probe
    2) 없으면 → 지식그래프(knowledge_graph) 기반 RAGAS 스타일 4분면
       (단일/멀티홉 × 구체/추상) 생성. 그래프에 쓸 청크/엣지가 부족해
       결과가 비면 → Single-Hop Specific 단일홉 폴백(_from_chunks).

현재 구현: RAGAS 4분면(75%) + DataMorgana-lite(20%) + 무응답(5%, Held-out·False
Premise 반씩)을 _allocate_budget 비율대로 섞어 생성한다(질문 모두 LLM으로 생성 —
청크 내용을 그대로 베끼지 않도록 질문·정답을 LLM이 새로 구성. 실제 호출은
agents/eval/llm_provider.py 가 담당(OpenAI 기본, EVAL_LLM_PROVIDER=gemini 로
대체 가능) — 공용 RAG generator와 동일한 폴백 규칙: 키 없거나 호출 실패 시
휴리스틱 추출로 대체). testset_size(n) < 8 이면 5%/20%가 0~1개로 반올림돼 통계적
의미가 없으므로 _allocate_budget 이 전부 RAGAS 로 몰아준다(무응답/DataMorgana 없음).
gold_spans 가 채워진 probe(taxonomy 등 외부 소스)는 _resync_gold_chunk_ids 로
재청킹 후에도 gold_chunk_ids 를 다시 맞춘다.

[구현 포인트]  (다음 단계로 남겨둠)
    - eval_probes.json 영속화 + 문서 diff 기반 증분 생성(골든 테스트셋 재사용).
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable

from core.schema import Chunk, Document, Probe
from core.state import AgentDoctorState
from agents.eval import knowledge_graph, llm_provider
from agents.eval.knowledge_graph import KGNode
from agents.eval.types import (
    EVOL_DIRECTIONS,
    MULTIHOP_SUBTYPES,
    NO_ANSWER_MIX_RATIO,
    DATAMORGANA_MIX_RATIO,
    PERSONAS,
    QUERY_LENGTHS,
    QUERY_STYLES,
    RAGAS_MIX_RATIO,
    RAGAS_QUADRANT_WEIGHTS,
    PROBE_SOURCE_AUTO,
    PROBE_SOURCE_USER_LOG,
    PROBE_SOURCE_TAXONOMY,
    resolve_llm_concurrency,
    resolve_probe_source,
    taxonomy_qa_path,
)
from core.parallel import parallel_map

# 자동 생성 기본 개수 (설계: testset_size=5~10 으로 시작해 비용 확인 후 확대)
# EVAL_TESTSET_SIZE 환경변수로 오버라이드 가능 (임계값 캘리브레이션 시 30~50 권장)
DEFAULT_TESTSET_SIZE = 10


@dataclass
class _SynthesizedProbe:
    """질문 합성 결과와 원문 좌표 계산에 사용할 exact evidence 묶음."""

    question: str
    ground_truth: str
    evidence: list[dict[str, Any]]


def _testset_size() -> int:
    try:
        return max(1, int(os.getenv("EVAL_TESTSET_SIZE", str(DEFAULT_TESTSET_SIZE))))
    except (TypeError, ValueError):
        return DEFAULT_TESTSET_SIZE


def generate_probes(state: AgentDoctorState) -> list[Probe]:
    """
    state 를 보고 Probe 리스트를 생성한다.

    우선순위 (설계 §3, 소스 신뢰도 user_log > taxonomy > llm_generated):
        1) user_log 경로(uses_user_log) → user_log Probe (GT 없음)
        2) 아니면 → _allocate_budget 비율(RAGAS/DataMorgana/무응답)대로
           지식그래프 기반 4분면 + DataMorgana-lite + 무응답 Probe를 섞어 생성.
           RAGAS 몫이 그래프 부족으로 비면(단일 폴백만 있는 경우) → 단일홉
           폴백(_from_chunks)으로 전체를 대체(비중 섞기보다 최소 동작 보장 우선).

    user_log/auto 선택은 uses_user_log 이 EVAL_PROBE_SOURCE(auto|user_log) 스위치와
    state.user_questions 유무로 결정한다. 자동 생성 경로만 GT·gold 를 채운다.

    읽기: state.user_questions, state.chunks, state.documents
    """
    if resolve_probe_source() == PROBE_SOURCE_TAXONOMY:
        return _from_taxonomy(state)

    if uses_user_log(state):
        probes = _from_user_questions(state.user_questions)
        print(f"[Eval] STEP1: user_log Probe {len(probes)}개 생성")
        return _finalize_probes(probes, state)

    graph = knowledge_graph.build_graph(state.chunks, state.index_config)
    testset_size = _testset_size()
    budget = _allocate_budget(testset_size)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in state.chunks}
    documents_by_id = {document.doc_id: document for document in state.documents}

    ragas_probes = _generate_ragas_probes(
        graph,
        budget["ragas"],
        chunks_by_id,
        documents_by_id,
    )
    if not ragas_probes:
        probes = _from_chunks(
            state.chunks,
            testset_size,
            documents_by_id,
        )
        print(f"[Eval] STEP1: llm_generated(폴백) Probe {len(probes)}개 생성")
        return _finalize_probes(probes, state)

    datamorgana_probes = _generate_datamorgana_probes(
        graph,
        budget["datamorgana"],
        chunks_by_id,
        documents_by_id,
    )
    no_answer_probes = _generate_no_answer_probes(state.chunks, graph, budget["no_answer"])

    probes = ragas_probes + datamorgana_probes + no_answer_probes
    print(f"[Eval] STEP1: llm_generated Probe {len(probes)}개 생성 "
          f"(ragas={len(ragas_probes)}, datamorgana={len(datamorgana_probes)}, "
          f"no_answer={len(no_answer_probes)})")
    return _finalize_probes(probes, state)


# ── probe 소스 선택 (EVAL_PROBE_SOURCE 스위치) ────────────────────

def uses_user_log(state: AgentDoctorState) -> bool:
    """user_log 경로를 쓸지 결정.
    EVAL_PROBE_SOURCE(auto|user_log)가 지정되면 그걸 강제하고, 미지정이면 기존 자동
    판별(user_questions 유무)을 따른다. auto 는 질문이 있어도 무시하고 자동 생성으로 가며,
    user_log 는 강제하되 질문이 없으면 자동 생성으로 폴백한다(빈 probe 방지).

    agent.py STEP1 도 이 술어로 캐시 여부를 정한다 — state.user_questions 유무만 보면
    auto 일 때 실제로는 LLM 생성으로 가는데도 캐시를 건너뛴다."""
    source = resolve_probe_source()
    if source in (PROBE_SOURCE_AUTO, PROBE_SOURCE_TAXONOMY):
        return False    # 둘 다 자체 생성 경로(generate_probes 내부 분기)를 탄다
    if source == PROBE_SOURCE_USER_LOG:
        return bool(state.user_questions)
    return bool(state.user_questions)   # 미지정 → 기존 동작


# ── taxonomy: 외부 사람작성 QA 데이터셋 (KorQuAD 등) ──────────────

def _from_taxonomy(state: AgentDoctorState) -> list[Probe]:
    """taxonomy Probe(gold_spans 포함)를 로드하고, 현재 청크에 맞춰 gold_chunk_ids 를
    resync 한다(재청킹돼도 gold 유지).

    qa 는 EVAL_TAXONOMY_QA 에서, corpus(gold 좌표 조회용)는 state.source_url 에서
    가져온다 — Ingest 가 문서를 복원한 바로 그 파일이라 좌표계가 일치한다(설정 단일화).
    KORQUAD_MAX_DOCS / KORQUAD_QA_LIMIT 로 규모 제한(스모크). MAX_DOCS 는 Ingest 의
    corpus 로더와 같은 규칙이라 corpus/qa 가 같은 문서 집합을 본다."""
    from agents.eval.datasets.korquad import load_taxonomy_probes, DEFAULT_CORPUS
    from agents.eval.types import korquad_qa_limit, korquad_max_docs

    corpus_path = state.source_url or DEFAULT_CORPUS
    probes = load_taxonomy_probes(taxonomy_qa_path(), corpus_path,
                                  limit=korquad_qa_limit(),
                                  max_docs=korquad_max_docs())
    probes = _resync_gold_chunk_ids(probes, state.chunks, state.documents)
    matched = sum(1 for p in probes if p.gold_chunk_ids)
    print(f"[Eval] STEP1: taxonomy Probe {len(probes)}개 로드 "
          f"(gold 매칭 {matched}/{len(probes)})")
    return probes


# ── user_log: 실사용 질문 기반 ────────────────────────────────────

def _from_user_questions(questions: list[str]) -> list[Probe]:
    """
    사용자가 넘긴 질문을 Probe 로 변환.
    ground_truth·gold_chunk_ids 가 없으므로 recall@k·F1 은 계산 불가 →
    RAGAS 의 무정답 지표(Faithfulness, Response Relevancy)로만 평가된다.
    """
    probes = []
    for i, q in enumerate(questions):
        q = (q or "").strip()
        if not q:
            continue
        probes.append(Probe(
            probe_id=f"probe_user_{i:03d}",
            question=q,
            source="user_log",
            answer_exists=None,      # 미상 (Oracle 테스트로 추후 판정)
            ground_truth=None,
            gold_chunk_ids=[],
            qtype=None,
        ))
    return probes


# ── llm_generated: 청크 기반 (Single-Hop Specific) ────────────────

# 문장 종결부호 뒤의 공백 또는 줄바꿈만 경계로 본다. 토큰 내부의 점에는
# 공백이 없으므로 URL·이메일·버전·날짜·소수점을 중간에서 자르지 않는다.
_SENT_SPLIT = re.compile(r"(?<=[.!?。？！])[ \t]+|\r?\n+")
_HEURISTIC_EVIDENCE_MIN_CHARS = 20
_HEURISTIC_EVIDENCE_MAX_CHARS = 240


def _from_chunks(
    chunks: list[Chunk],
    size: int,
    documents_by_id: dict[str, Document] | None = None,
) -> list[Probe]:
    """
    청크마다 Single-Hop Specific Probe 를 만든다.
    각 청크에 대해 LLM 생성을 시도하고, 실패/미설정 시 휴리스틱 추출로 대체한다.
    """
    probes: list[Probe] = []
    # 너무 짧은 청크는 스킵, 앞에서부터 size 개 사용
    usable = [c for c in chunks if c.text and len(c.text.strip()) >= 20]
    targets = usable[:size]
    # LLM 합성만 병렬로 실행하고, 결과 조립과 gold span 계산은 입력 순서대로 처리한다.
    results = parallel_map(
        lambda chunk: _llm_generate_single_hop(chunk.text),
        targets,
        resolve_llm_concurrency(),
    )
    for i, (chunk, generated) in enumerate(zip(targets, results)):
        heuristic_quote: str | None = None
        if generated is None:
            question, ground_truth = _heuristic_single_hop(chunk.text)
            heuristic_quote = ground_truth
            gen_method = "heuristic_evidence"
        else:
            question, ground_truth = generated
            gen_method = "llm_single_hop"
        probe = Probe(
            probe_id=f"probe_gen_{i:03d}",
            question=question,
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=ground_truth,
            gold_chunk_ids=[chunk.chunk_id],
            qtype=None,                 # 단일홉
            metadata={"gen_method": gen_method},
        )
        document = (documents_by_id or {}).get(chunk.doc_id)
        spans = []
        exact_span_count = 0
        fallback_span_count = 0
        if document is not None:
            span = None
            if heuristic_quote:
                span = _locate_evidence_in_chunk(
                    heuristic_quote,
                    chunk,
                    document,
                    chunks,
                )
            if span is None:
                span = _chunk_fallback_span(chunk, document, chunks)
            if span is not None:
                spans.append(span)
                if span.get("_grounding_quality") == "exact":
                    exact_span_count = 1
                else:
                    fallback_span_count = 1
        _set_probe_gold_spans(
            probe,
            spans,
            expected_sources=1,
            located_sources=len(spans),
            exact_span_count=exact_span_count,
            fallback_span_count=fallback_span_count,
        )
        probes.append(probe)
    return probes


def _heuristic_single_hop(text: str) -> tuple[str, str]:
    """LLM 미사용/실패 시 원문의 대표 근거 구간으로 질문과 정답을 만든다."""
    evidence = _heuristic_evidence_of(text)
    topic = _topic_of(evidence or text)
    return f"{topic}에 대해 설명해줘.", evidence


def _heuristic_evidence_of(text: str) -> str:
    """
    원문에서 좌표를 정확히 계산할 수 있는 대표 문장 또는 짧은 연속 구간을 고른다.

    제목처럼 짧은 첫 줄보다 일정 길이 이상의 첫 문장을 우선한다. 문장 경계가
    없거나 한 문장이 지나치게 길면 앞부분을 제한하되, 가능한 경우 단어 중간을
    자르지 않는다. 반환값은 항상 입력 원문에 그대로 포함된 부분 문자열이다.
    """
    source = (text or "").strip()
    if not source:
        return ""

    candidates: list[str] = []
    for segment in _SENT_SPLIT.split(source):
        candidate = segment.strip().lstrip("#•-*> \t")
        if candidate:
            candidates.append(candidate)

    if not candidates:
        candidates = [source]

    evidence = next(
        (
            candidate
            for candidate in candidates
            if len(candidate) >= _HEURISTIC_EVIDENCE_MIN_CHARS
        ),
        max(candidates, key=len),
    )
    if len(evidence) <= _HEURISTIC_EVIDENCE_MAX_CHARS:
        return evidence

    bounded = evidence[:_HEURISTIC_EVIDENCE_MAX_CHARS].rstrip()
    word_break = max(bounded.rfind(" "), bounded.rfind("\t"))
    if word_break >= _HEURISTIC_EVIDENCE_MAX_CHARS // 2:
        bounded = bounded[:word_break].rstrip()
    return bounded


def _llm_generate_single_hop(chunk_text: str) -> tuple[str, str] | None:
    """
    LLM(OpenAI/Gemini/GitHub Models, EVAL_LLM_PROVIDER로 선택)으로 Single-Hop Specific (질문, 정답)
    쌍을 생성한다. 청크 문장을 그대로 베끼지 않도록 질문·정답 모두 새로 구성하게 지시한다.
    키 없거나 호출·파싱 실패 시 None(호출부가 휴리스틱으로 대체).
    """
    if not llm_provider.has_key():
        return None
    try:
        data = llm_provider.chat_json(
            system=("너는 RAG 파이프라인 평가용 테스트 질문을 설계하는 평가자다. "
                    "주어진 문서 조각(컨텍스트) 하나만으로 답할 수 있는, 실제 사용자가 물어볼 법한 "
                    "구체적인 사실 기반 질문(Single-Hop Specific) 하나와 그 정답을 만들어라. "
                    "질문과 정답 모두 컨텍스트 문장을 그대로 베끼지 말고 자기 말로 다시 구성하되, "
                    "정답은 컨텍스트에 있는 사실에서 벗어나면 안 된다. "
                    "반드시 {\"question\": str, \"ground_truth\": str} 형태의 JSON으로만 답하라."),
            user=f"[컨텍스트]\n{chunk_text}",
        )
        question = (data.get("question") or "").strip()
        ground_truth = (data.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            return None
        return question, ground_truth
    except Exception as e:
        print(f"[Eval] STEP1: Probe LLM 생성 실패({e}) → 휴리스틱 폴백")
        return None


def _topic_of(text: str) -> str:
    """청크 첫 문장/앞부분을 질문의 주제 문구로 사용."""
    text = text.strip()
    first = _SENT_SPLIT.split(text, maxsplit=1)[0]
    first = first.strip().strip("#•-*> ").strip()
    # 너무 길면 앞 40자만
    return first[:40] if len(first) > 40 else first


# ── gold span / 위치-인덱스 유틸 ──────────────────────────────────
#
# RAGAS 스타일 재구현(STEP1)에서 gold_char_span/gold_spans 로 정답 위치를
# 원문(Document.content) 절대 좌표로 저장해두면, Optimize→Index 재청킹 후에도
# (chunk_size/overlap이 달라져도) probe.gold_chunk_ids 를 다시 맞출 수 있다.
# generate_probes() 가 마지막 단계에서 _resync_gold_chunk_ids 를 호출한다.
# 답이 있는 자동 생성 Probe는 exact evidence를 우선한다. LLM 합성 실패 시에도
# 휴리스틱이 고른 짧은 원문 구간을 evidence로 쓰며, 좌표화까지 실패한 source만
# 청크 전체 좌표로 폴백한다.
#
# 전제: Chunk.text 는 Document.content 의 앞에서부터 순서대로 나온 부분 문자열이다
# (agents/index/agent.py::_chunk_text 가 고정 크기로 슬라이스하는 방식과, 향후
# 문장/의미 단위 청킹으로 바뀌어도 유지되는 유일한 불변식).

def _chunk_index_of(chunk: Chunk) -> int:
    """청크의 문서 내 순번. metadata['chunk_index'] 우선, 없으면 chunk_id 끝 숫자로 폴백."""
    idx = chunk.metadata.get("chunk_index") if chunk.metadata else None
    if isinstance(idx, int):
        return idx
    m = re.search(r"(\d+)$", chunk.chunk_id or "")
    return int(m.group(1)) if m else 0


def _locate_span(doc_content: str, needle: str, cursor: int) -> tuple[int, int] | None:
    """
    doc_content 에서 needle을 cursor 위치부터 찾아 (start, end) 를 반환.
    cursor 이후에 없으면(예: 검색용 cursor 추정이 어긋난 경우) 처음부터 한 번 더 찾아본다.
    둘 다 실패하면 None.
    """
    if not needle:
        return None
    idx = doc_content.find(needle, cursor)
    if idx == -1:
        idx = doc_content.find(needle)
        if idx == -1:
            return None
    return (idx, idx + len(needle))


def _build_doc_position_index(doc: Document, chunks: list[Chunk]) -> list[tuple[str, int, int]]:
    """
    doc에 속한 청크들을 chunk_index 순서로 doc.content 안에서 찾아
    [(chunk_id, start, end), ...] 로 반환한다(못 찾은 청크는 건너뜀).

    chunk_index 순서로 훑으면서 직전 청크의 start로 cursor를 옮기기 때문에
    (겹치는) 청크가 순서를 지켜 반복 등장해도 이전 위치를 앞지르지 않는다.
    """
    doc_chunks = sorted(
        (c for c in chunks if c.doc_id == doc.doc_id),
        key=_chunk_index_of,
    )
    index: list[tuple[str, int, int]] = []
    cursor = 0
    for c in doc_chunks:
        # Index 가 확정한 char_span(원문 좌표)을 우선 쓴다 — 본문 재검색은 동일 텍스트가
        # 반복될 때 여러 청크가 첫 위치로 몰려(cursor 방식) 뒤쪽 gold 가 비는 모호성이 있다.
        # 선언 span 이 실제 원문과 맞는지 검증 후 채택하고, 아니면 텍스트 검색으로 폴백.
        declared = _declared_chunk_span(c)
        span = None
        if declared is not None and _valid_document_span(
            doc,
            declared[0],
            declared[1],
            c.text,
        ):
            span = declared
        if span is None:
            span = _locate_span(doc.content, c.text, cursor)
        if span is None:
            continue
        start, end = span
        index.append((c.chunk_id, start, end))
        # overlap 청크도 허용하면서 동일 텍스트의 다음 등장을 찾을 수 있게 한 칸 전진한다.
        cursor = start + 1
    return index


def _declared_chunk_span(chunk: Chunk) -> tuple[int, int] | None:
    """Chunk 필드나 legacy metadata에서 선언된 원문 좌표를 읽는다."""

    raw = chunk.char_span
    if raw is None and chunk.metadata:
        raw = chunk.metadata.get("char_span")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    start, end = raw
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    return start, end


def _valid_document_span(
    document: Document,
    start: int,
    end: int,
    expected_text: str | None = None,
) -> bool:
    """좌표 범위와 선택적 원문 일치 조건을 검증한다."""

    if start < 0 or end <= start or end > len(document.content):
        return False
    return expected_text is None or document.content[start:end] == expected_text


def _chunk_fallback_span(
    chunk: Chunk,
    document: Document,
    chunks: Iterable[Chunk] | None = None,
) -> dict[str, Any] | None:
    """exact evidence가 없을 때 source chunk 전체의 원문 좌표를 구한다."""

    declared = _declared_chunk_span(chunk)
    if declared is not None:
        start, end = declared
        if _valid_document_span(document, start, end, chunk.text):
            return {
                "doc_id": chunk.doc_id,
                "start": start,
                "end": end,
                "_grounding_quality": "chunk_fallback",
            }

    if chunks is not None:
        position_index = _build_doc_position_index(document, list(chunks))
        for chunk_id, start, end in position_index:
            if chunk_id == chunk.chunk_id:
                return {
                    "doc_id": chunk.doc_id,
                    "start": start,
                    "end": end,
                    "_grounding_quality": "chunk_fallback",
                }

    located = _locate_span(document.content, chunk.text, declared[0] if declared else 0)
    if located is None:
        return None
    start, end = located
    if not _valid_document_span(document, start, end, chunk.text):
        return None
    return {
        "doc_id": chunk.doc_id,
        "start": start,
        "end": end,
        "_grounding_quality": "chunk_fallback",
    }


def _locate_evidence_in_chunk(
    quote: str,
    chunk: Chunk,
    document: Document,
    chunks: Iterable[Chunk] | None = None,
) -> dict[str, Any] | None:
    """선택된 source chunk 안에서 exact quote를 찾아 원문 절대좌표로 바꾼다."""

    if not quote:
        return None
    chunk_span = _chunk_fallback_span(chunk, document, chunks)
    if chunk_span is None:
        return None

    local_start = chunk.text.find(quote)
    if local_start >= 0:
        start = chunk_span["start"] + local_start
        end = start + len(quote)
        if _valid_document_span(document, start, end, quote):
            return {
                "doc_id": chunk.doc_id,
                "start": start,
                "end": end,
                "_grounding_quality": "exact",
            }

    start = document.content.find(
        quote,
        chunk_span["start"],
        chunk_span["end"],
    )
    if start == -1:
        return None
    end = start + len(quote)
    if not _valid_document_span(document, start, end, quote):
        return None
    return {
        "doc_id": chunk.doc_id,
        "start": start,
        "end": end,
        "_grounding_quality": "exact",
    }


def _parse_evidence(raw_evidence: Any) -> list[dict[str, Any]]:
    """LLM evidence를 안전한 source_index/quote 목록으로 정규화한다."""

    if not isinstance(raw_evidence, (list, tuple)):
        return []
    parsed: list[dict[str, Any]] = []
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            continue
        source_index = raw.get("source_index")
        quote = raw.get("quote")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
            or not isinstance(quote, str)
            or not quote.strip()
        ):
            continue
        parsed.append({"source_index": source_index, "quote": quote.strip()})
    return parsed


def _gold_spans_from_evidence(
    synthesized: _SynthesizedProbe,
    nodes: list[KGNode],
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
) -> tuple[list[dict[str, Any]], int, int, int]:
    """합성 evidence를 source별로 grounding하고 누락 source는 chunk로 폴백한다."""

    exact_by_source: dict[int, list[dict[str, Any]]] = {}
    for item in synthesized.evidence:
        source_index = item["source_index"]
        if source_index >= len(nodes):
            continue
        node = nodes[source_index]
        chunk = chunks_by_id.get(node.chunk_id)
        document = documents_by_id.get(node.doc_id)
        if chunk is None or document is None or chunk.doc_id != node.doc_id:
            continue
        span = _locate_evidence_in_chunk(
            item["quote"],
            chunk,
            document,
            list(chunks_by_id.values()),
        )
        if span is not None:
            exact_by_source.setdefault(source_index, []).append(span)

    spans: list[dict[str, Any]] = []
    exact_span_count = 0
    fallback_span_count = 0
    located_sources = 0
    for source_index, node in enumerate(nodes):
        exact_spans = exact_by_source.get(source_index, [])
        if exact_spans:
            spans.extend(exact_spans)
            exact_span_count += len(exact_spans)
            located_sources += 1
            continue
        chunk = chunks_by_id.get(node.chunk_id)
        document = documents_by_id.get(node.doc_id)
        if chunk is None or document is None or chunk.doc_id != node.doc_id:
            continue
        fallback = _chunk_fallback_span(
            chunk,
            document,
            list(chunks_by_id.values()),
        )
        if fallback is not None:
            spans.append(fallback)
            fallback_span_count += 1
            located_sources += 1

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for span in spans:
        key = (span["doc_id"], span["start"], span["end"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    # 상태 문자열과 실제 저장 span 목록이 어긋나지 않도록 dedup 이후 다시 센다.
    deduped_exact_count = sum(
        span.get("_grounding_quality") == "exact" for span in deduped
    )
    deduped_fallback_count = sum(
        span.get("_grounding_quality") == "chunk_fallback" for span in deduped
    )
    return (
        deduped,
        located_sources,
        deduped_exact_count,
        deduped_fallback_count,
    )


def _set_probe_gold_spans(
    probe: Probe,
    spans: list[dict[str, Any]],
    *,
    expected_sources: int,
    located_sources: int,
    exact_span_count: int,
    fallback_span_count: int,
) -> Probe:
    """Probe에 좌표와 품질 요약을 기록하고 단일 근거 호환 필드를 맞춘다."""

    span_qualities = [
        span.get("_grounding_quality", "unknown")
        for span in spans
    ]
    clean_spans = [
        {
            "doc_id": span["doc_id"],
            "start": span["start"],
            "end": span["end"],
        }
        for span in spans
    ]
    probe.gold_spans = clean_spans
    if len(clean_spans) == 1:
        span = clean_spans[0]
        probe.gold_doc_id = span["doc_id"]
        probe.gold_char_span = (span["start"], span["end"])
    else:
        probe.gold_doc_id = None
        probe.gold_char_span = None

    if not clean_spans:
        status = "failed"
    elif fallback_span_count == 0 and located_sources == expected_sources:
        status = "exact"
    elif exact_span_count == 0 and located_sources == expected_sources:
        status = "chunk_fallback"
    else:
        status = "partial"
    probe.metadata["span_grounding"] = {
        "status": status,
        "expected_sources": expected_sources,
        "located_sources": located_sources,
        "exact_span_count": exact_span_count,
        "fallback_span_count": fallback_span_count,
        "span_qualities": span_qualities,
    }
    return probe


def _finalize_probes(probes: list[Probe], state: AgentDoctorState) -> list[Probe]:
    """모든 생성 경로에서 현재 청킹 기준 gold chunk 캐시를 마지막에 동기화한다."""

    return _resync_gold_chunk_ids(probes, state.chunks, state.documents)


def _resync_gold_chunk_ids(
    probes: list[Probe], chunks: list[Chunk], documents: list[Document]
) -> list[Probe]:
    """
    probe.gold_spans(원문 절대 좌표, 재청킹해도 불변)를 기준으로 현재 chunks 와
    대응하는 최소 청크 id를 다시 찾아 probe.gold_chunk_ids 를 갱신한다(in-place + 반환).
    span 전체를 포함하는 청크가 있으면 문맥 낭비가 가장 작은 하나를 고르고,
    없으면 span을 빈틈없이 덮는 최소 분할 청크 조합을 고른다.
    gold_spans 가 없는 probe는 건드리지 않는다(기존 legacy 경로 그대로 유지).

    [설계 편차] 최초 계획엔 "state.chunks만 있으면 되고 state.documents는 필요 없다"고
    적혀 있었으나, 새 chunk의 (start, end)를 얻으려면 결국 원문(Document.content)에서
    다시 찾아야 하므로 documents 인자가 필요하다 — chunk_size/overlap로 좌표를 역산하면
    지금의 고정 크기 청킹에만 맞고 향후 청킹 전략 교체에 깨지므로 그 방식은 쓰지 않는다.
    """
    docs_by_id = {d.doc_id: d for d in documents}
    position_cache: dict[str, list[tuple[str, int, int]]] = {}

    def _position_index(doc_id: str) -> list[tuple[str, int, int]]:
        if doc_id not in position_cache:
            doc = docs_by_id.get(doc_id)
            position_cache[doc_id] = _build_doc_position_index(doc, chunks) if doc else []
        return position_cache[doc_id]

    def _minimal_gold_ids(
        positions: list[tuple[str, int, int]],
        span_start: int,
        span_end: int,
    ) -> list[str]:
        """span을 대표하는 단일 포함 청크 또는 최소 연속 커버를 반환한다."""

        containing = [
            item
            for item in positions
            if item[1] <= span_start and item[2] >= span_end
        ]
        if containing:
            best = min(
                containing,
                key=lambda item: (item[2] - item[1], item[1], item[0]),
            )
            return [best[0]]

        intersections = sorted(
            (
                chunk_id,
                max(span_start, chunk_start),
                min(span_end, chunk_end),
            )
            for chunk_id, chunk_start, chunk_end in positions
            if chunk_start < span_end and chunk_end > span_start
        )
        selected: list[str] = []
        cursor = span_start
        while cursor < span_end:
            eligible = [
                item
                for item in intersections
                if item[1] <= cursor < item[2]
            ]
            if not eligible:
                # 좌표 사이에 빈틈이 있으면 정보 손실을 피하려고 모든 교차 청크를 유지한다.
                return [item[0] for item in intersections]
            best = max(eligible, key=lambda item: (item[2], -item[1]))
            selected.append(best[0])
            cursor = best[2]
        return selected

    for probe in probes:
        if not probe.gold_spans:
            continue
        matched: list[str] = []
        for span in probe.gold_spans:
            doc_id = span.get("doc_id")
            s_start, s_end = span.get("start"), span.get("end")
            if (
                not isinstance(doc_id, str)
                or isinstance(s_start, bool)
                or isinstance(s_end, bool)
                or not isinstance(s_start, int)
                or not isinstance(s_end, int)
                or s_start < 0
                or s_end <= s_start
            ):
                continue
            matched.extend(
                _minimal_gold_ids(_position_index(doc_id), s_start, s_end)
            )
        # 매번 현재 청킹 결과로 교체한다. 매칭이 없을 때도 이전 청크 ID를 남기지 않는다.
        probe.gold_chunk_ids = list(dict.fromkeys(matched))  # 순서 유지 dedupe
    return probes


# ── RAGAS 스타일 시나리오 합성 ────────────────────────────────────
#
# knowledge_graph.build_graph()가 만든 그래프를 입력으로 받아 단일홉(구체/추상)
# + 멀티홉(bridge/comparison/aggregation) Probe를 생성한다.
# generate_probes()가 이를 호출하며, 결과가 비면(그래프에 쓸 청크/엣지 부족)
# 단일홉 폴백(_from_chunks)으로 대체한다. probe_store 영속화는 이후 단계.
#
# RAGAS의 "KG 구축 -> 시나리오 샘플링 -> 질문 합성" 3단계를 _synthesize_query()
# 하나의 LLM 호출로 압축했다 - testset_size당 호출 수를 3배로 늘리지 않기 위한
# 의도적 단순화(비용/지연 절충).

def _allocate_budget(n: int) -> dict[str, int]:
    """
    testset_size(n)을 RAGAS/DataMorgana/무응답 비율(75/20/5)로 나눈다.
    n<8이면 5%/20%가 0~1개로 반올림돼 통계적 의미가 없으므로 전부 RAGAS로
    몰아준다(합이 항상 n이 되도록 largest-remainder로 반올림).
    """
    if n <= 0:
        return {"ragas": 0, "datamorgana": 0, "no_answer": 0}
    if n < 8:
        return {"ragas": n, "datamorgana": 0, "no_answer": 0}
    raw = {
        "ragas": n * RAGAS_MIX_RATIO,
        "datamorgana": n * DATAMORGANA_MIX_RATIO,
        "no_answer": n * NO_ANSWER_MIX_RATIO,
    }
    return _largest_remainder_round(raw, n)


def _allocate_ragas_quadrants(n_ragas: int, has_multihop_edges: bool) -> dict[str, int]:
    """
    RAGAS 몫을 4분면(단일홉 구체/추상, 멀티홉 구체/추상)으로 largest-remainder
    배분한다. 그래프에 연결된 청크 쌍이 하나도 없으면(코퍼스가 작거나 서로
    무관) 멀티홉 몫을 단일홉 쪽으로 접어 무관한 청크를 엮은 멀티홉 질문을
    만들지 않는다.
    """
    if n_ragas <= 0:
        return {k: 0 for k in RAGAS_QUADRANT_WEIGHTS}
    weights = dict(RAGAS_QUADRANT_WEIGHTS)
    if not has_multihop_edges:
        weights["single_specific"] += weights["multi_specific"]
        weights["single_abstract"] += weights["multi_abstract"]
        weights["multi_specific"] = 0.0
        weights["multi_abstract"] = 0.0
    raw = {k: n_ragas * w for k, w in weights.items()}
    return _largest_remainder_round(raw, n_ragas)


def _largest_remainder_round(raw: dict[str, float], total: int) -> dict[str, int]:
    """비율 배분(raw, 합=total)을 정수로 반올림하되 합이 정확히 total이 되도록
    나머지가 큰 항목부터 1씩 더한다(Largest Remainder Method)."""
    floors = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:max(remainder, 0)]:
        floors[k] += 1
    return floors


def _round_robin(items: list[str], n: int) -> list[str]:
    if n <= 0 or not items:
        return []
    return [items[i % len(items)] for i in range(n)]


def _generate_ragas_probes(
    graph: knowledge_graph.KGraph,
    n: int,
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
) -> list[Probe]:
    """
    그래프에서 RAGAS 스타일 Probe n개를 만들고 source chunk/document를 이용해
    정답 evidence를 원문 절대좌표로 grounding한다.
    """
    usable = [node for node in graph.nodes.values() if node.text and len(node.text.strip()) >= 20]
    pairs = knowledge_graph.connected_pairs(graph, n=2)
    quadrants = _allocate_ragas_quadrants(n, has_multihop_edges=bool(pairs))

    remaining_pairs = list(pairs)

    def _next_pair() -> list[str] | None:
        if not remaining_pairs:
            return None
        return remaining_pairs.pop(random.randrange(len(remaining_pairs)))

    plan: list[tuple[str, str | None]] = (
        [("single_specific", None)] * quadrants["single_specific"]
        + [("single_abstract", None)] * quadrants["single_abstract"]
        + list(zip(
            ["multi_specific"] * quadrants["multi_specific"],
            _round_robin(MULTIHOP_SUBTYPES, quadrants["multi_specific"]),
        ))
        + list(zip(
            ["multi_abstract"] * quadrants["multi_abstract"],
            _round_robin(MULTIHOP_SUBTYPES, quadrants["multi_abstract"]),
        ))
    )

    # plan 단계(순차): 노드 선택·pair 소비·시나리오 샘플링 등 random 소비를 전부
    # 여기서 확정한다 — 병렬화가 RNG 소비 순서/공유 리스트(remaining_pairs)에
    # 영향을 주지 않도록 LLM 호출과 분리(동시성 1이든 4든 plan 은 동일).
    specs: list[dict] = []
    for quadrant, subtype in plan:
        if quadrant.startswith("single"):
            nodes = [random.choice(usable)] if usable else []
        else:
            pair_ids = _next_pair()
            nodes = [graph.nodes[cid] for cid in pair_ids if cid in graph.nodes] if pair_ids else []
        if len(nodes) < (1 if quadrant.startswith("single") else 2):
            continue
        specs.append(_plan_ragas_probe(nodes, quadrant, subtype))

    # 합성 단계(병렬): LLM 호출만. 실패는 태스크 안에서 None 으로 흡수(예외 무전파).
    results = parallel_map(_synthesize_ragas_query, specs, resolve_llm_concurrency())

    # 조립 단계(순차, plan 순서): probe_id 번호(성공분만 카운트) 규칙 보존
    probes: list[Probe] = []
    for spec, result in zip(specs, results):
        probe = _build_ragas_probe(
            spec,
            result,
            len(probes),
            chunks_by_id,
            documents_by_id,
        )
        if probe is not None:
            probes.append(probe)
    return probes


# ── DataMorgana-lite (예산 20%) ───────────────────────────────────
#
# 풀 DataMorgana(설정 가능한 페르소나/스타일 조합의 별도 파이프라인) 대신,
# 이미 있는 _llm_synthesize_query 시나리오 파라미터 중 "질문자가 덜 친절하게
# 묻는" 조합(비격식·긴 길이·breadth 확장)을 강제해 좀 더 거친 질문을 만드는
# 최소 버전으로 축소했다(설계 문서도 "시간 남으면 구체화" 항목으로 분류).

def _generate_datamorgana_probes(
    graph: knowledge_graph.KGraph,
    n: int,
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
) -> list[Probe]:
    """단일홉 노드에서 거친 스타일(conversational/long/breadth)로 질문을 합성한다."""
    if n <= 0:
        return []
    usable = [node for node in graph.nodes.values() if node.text and len(node.text.strip()) >= 20]
    # plan(순차): 노드·페르소나 샘플링 확정 → 합성(병렬) → 조립(순차, 번호 규칙 보존)
    specs: list[dict] = []
    for i in range(n):
        if not usable:
            break
        specs.append({"index": i, "node": random.choice(usable),
                      "persona": random.choice(PERSONAS)})
    results = parallel_map(
        lambda s: _llm_synthesize_query(
            [s["node"]], "single_specific", None,
            persona=s["persona"], style="conversational",
            length="long", evol_dir="breadth",
        ),
        specs, resolve_llm_concurrency())

    probes: list[Probe] = []
    for spec, result in zip(specs, results):
        node = spec["node"]
        if result is None:
            result = _heuristic_synthesize_query([node])
        if not result.question or not result.ground_truth:
            continue
        probe = Probe(
            probe_id=f"probe_datamorgana_{spec['index']:03d}",
            question=result.question,
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=result.ground_truth,
            gold_chunk_ids=[node.chunk_id],
            qtype=None,
            metadata={"gen_method": "datamorgana_lite", "style": "conversational"},
        )
        spans, located_sources, exact_count, fallback_count = _gold_spans_from_evidence(
            result,
            [node],
            chunks_by_id,
            documents_by_id,
        )
        probes.append(_set_probe_gold_spans(
            probe,
            spans,
            expected_sources=1,
            located_sources=located_sources,
            exact_span_count=exact_count,
            fallback_span_count=fallback_count,
        ))
    return probes


# ── 무응답 Probe (예산 5%, Held-out·False Premise 절반씩) ─────────
#
# 목적: diagnose.py 의 _no_diagnosis/is_abstention 게이팅이 "정답 없음을 올바르게
# 기권하는 경우"와 "무응답인데 답을 지어내는 생성 실패"를 구분해 진단할 수 있으려면,
# answer_exists=False 인 probe가 최소한 존재해야 한다. 두 방식 모두 정답(ground_truth)
# 없이 answer_exists=False 로 표시해 STEP2 가 "기권해야 정상"인 질문으로 다룬다.

def _generate_no_answer_probes(chunks: list[Chunk], graph: knowledge_graph.KGraph, n: int) -> list[Probe]:
    """Held-out(전반) / False Premise(후반)로 절반씩 나눠 생성. n=1이면 Held-out 하나만."""
    if n <= 0:
        return []
    n_held_out = (n + 1) // 2
    n_false_premise = n - n_held_out
    probes = _generate_held_out_probes(chunks, n_held_out)
    probes += _generate_false_premise_probes(graph, n_false_premise, start_index=len(probes))
    return probes


def _generate_held_out_probes(chunks: list[Chunk], n: int) -> list[Probe]:
    """
    코퍼스에 없는 정보를 묻는 질문. 실제 청크 하나를 주제로 삼되 gold_chunk_ids 를
    비워(코퍼스에서 답을 찾을 수 없는 것처럼) 검색이 반드시 실패하게 만든다
    (완전한 코퍼스 제외는 Index Agent 쪽 정보가 필요해 Eval 단독으로는 시뮬레이션만 가능).
    """
    usable = [c for c in chunks if c.text and len(c.text.strip()) >= 20]
    probes: list[Probe] = []
    for i in range(n):
        if not usable:
            break
        chunk = random.choice(usable)
        topic = _topic_of(chunk.text)
        probes.append(Probe(
            probe_id=f"probe_held_out_{i:03d}",
            question=f"{topic}과 관련해 아직 공개되지 않은 세부 내규는 무엇인가요?",
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=False,
            ground_truth=None,
            gold_chunk_ids=[],
            qtype=None,
            metadata={"gen_method": "no_answer_held_out"},
        ))
    return probes


def _generate_false_premise_probes(graph: knowledge_graph.KGraph, n: int, start_index: int = 0) -> list[Probe]:
    """
    질문 자체에 컨텍스트와 모순되는 잘못된 전제를 심는다(예: 실제로는 존재하지 않는
    정책이 있다고 전제하고 세부사항을 묻기) — LLM 이 그 전제를 그대로 받아 답을
    지어내면 생성 실패(hallucination 계열)로 잡혀야 한다.
    """
    if n <= 0:
        return []
    usable = [node for node in graph.nodes.values() if node.text and len(node.text.strip()) >= 20]
    # plan(순차): 노드 샘플링 확정 → 질문 생성(병렬, 태스크 내 휴리스틱 폴백) → 조립(순차)
    specs: list[dict] = []
    for i in range(n):
        if not usable:
            break
        specs.append({"index": i, "node": random.choice(usable)})
    results = parallel_map(lambda s: _false_premise_question(s["node"].text),
                           specs, resolve_llm_concurrency())

    probes: list[Probe] = []
    for spec, question in zip(specs, results):
        node = spec["node"]
        if question is None:
            continue
        probes.append(Probe(
            probe_id=f"probe_false_premise_{start_index + spec['index']:03d}",
            question=question,
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=False,
            ground_truth=None,
            gold_chunk_ids=[node.chunk_id],
            qtype=None,
            metadata={"gen_method": "no_answer_false_premise"},
        ))
    return probes


def _false_premise_question(chunk_text: str) -> str | None:
    """
    LLM(OpenAI/Gemini/GitHub Models, EVAL_LLM_PROVIDER로 선택)으로 잘못된 전제가 담긴 질문을
    만든다. 키 없거나 실패 시 휴리스틱(고정 템플릿)으로 대체 — LLM 없이도
    answer_exists=False probe가 최소 동작하도록 폴백을 항상 값 있게 유지한다
    (_llm_synthesize_query 와의 차이).
    """
    if llm_provider.has_key():
        try:
            data = llm_provider.chat_json(
                system=("너는 RAG 파이프라인 평가용 테스트 질문을 설계하는 평가자다. "
                        "주어진 컨텍스트와 모순되거나 컨텍스트에 없는 사실을 전제로 깔고, "
                        "그 전제가 사실인 것처럼 세부사항을 캐묻는 질문 하나를 한국어로 만들어라 "
                        "(예: 컨텍스트에 없는 제도가 있다고 가정하고 조건을 묻기). "
                        "반드시 {\"question\": str} 형태의 JSON으로만 답하라."),
                user=f"[컨텍스트]\n{chunk_text}",
            )
            question = (data.get("question") or "").strip()
            if question:
                return question
        except Exception as e:
            print(f"[Eval] STEP1: False Premise 질문 생성 실패({e}) → 휴리스틱 폴백")
    topic = _topic_of(chunk_text)
    return f"{topic}과 관련된 특별 예외 규정은 정확히 몇 조 몇 항에 명시되어 있나요?"


def _plan_ragas_probe(nodes: list[KGNode], quadrant: str, subtype: str | None) -> dict:
    """[plan 단계·순차] 노드(들)에 시나리오(페르소나/어투/길이/진화 방향)를 샘플링해
    합성 spec 을 확정한다. random 소비는 전부 여기서 끝난다."""
    return {
        "nodes": nodes,
        "quadrant": quadrant,
        "subtype": subtype,
        "persona": random.choice(PERSONAS),
        "style": random.choice(QUERY_STYLES),
        "length": random.choice(QUERY_LENGTHS),
        "evol_dir": random.choice(EVOL_DIRECTIONS),
    }


def _synthesize_ragas_query(spec: dict) -> _SynthesizedProbe | None:
    """[합성 단계·병렬 가능] spec 하나로 LLM 합성 1회. 실패 시 None (조립이 폴백)."""
    return _llm_synthesize_query(
        spec["nodes"], spec["quadrant"], spec["subtype"],
        spec["persona"], spec["style"], spec["length"], spec["evol_dir"],
    )


def _build_ragas_probe(
    spec: dict,
    result: _SynthesizedProbe | None,
    index: int,
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
) -> Probe | None:
    """[조립 단계·순차] 합성 결과(실패면 휴리스틱 폴백)로 Probe 하나를 만든다."""
    nodes, quadrant, subtype = spec["nodes"], spec["quadrant"], spec["subtype"]
    is_multi = quadrant.startswith("multi")

    if result is None:
        result = _heuristic_synthesize_query(nodes)
    if not result.question or not result.ground_truth:
        return None

    gen_method = f"ragas_{quadrant}_{subtype}" if is_multi else f"ragas_{quadrant}"
    probe = Probe(
        probe_id=f"probe_{quadrant}_{index:03d}",
        question=result.question,
        source="llm_generated",
        expected_difficulty="medium",
        answer_exists=True,
        ground_truth=result.ground_truth,
        gold_chunk_ids=[node.chunk_id for node in nodes],
        qtype=subtype if is_multi else None,
        metadata={
            "gen_method": gen_method,
            "persona": spec["persona"],
            "style": spec["style"],
            "length": spec["length"],
            "evol_direction": spec["evol_dir"],
        },
    )
    spans, located_sources, exact_count, fallback_count = _gold_spans_from_evidence(
        result,
        nodes,
        chunks_by_id,
        documents_by_id,
    )
    return _set_probe_gold_spans(
        probe,
        spans,
        expected_sources=len(nodes),
        located_sources=located_sources,
        exact_span_count=exact_count,
        fallback_span_count=fallback_count,
    )


def _make_ragas_probe(
    nodes: list[KGNode],
    quadrant: str,
    subtype: str | None,
    index: int,
    chunks_by_id: dict[str, Chunk],
    documents_by_id: dict[str, Document],
) -> Probe | None:
    """단일 Probe 생성용 호환 래퍼. 계획·합성·조립 단계를 순서대로 실행한다."""
    spec = _plan_ragas_probe(nodes, quadrant, subtype)
    result = _synthesize_ragas_query(spec)
    return _build_ragas_probe(
        spec,
        result,
        index,
        chunks_by_id,
        documents_by_id,
    )


def _heuristic_synthesize_query(nodes: list[KGNode]) -> _SynthesizedProbe:
    """LLM 미사용/실패 시 각 source의 대표 근거 구간으로 Probe를 합성한다."""
    evidence_quotes = [_heuristic_evidence_of(node.text) for node in nodes]
    if len(nodes) == 1:
        evidence = evidence_quotes[0]
        topic = _topic_of(evidence or nodes[0].text)
        return _SynthesizedProbe(
            question=f"{topic}에 대해 설명해줘.",
            ground_truth=evidence,
            evidence=[{"source_index": 0, "quote": evidence}],
        )
    topics = [
        _topic_of(evidence or node.text)
        for node, evidence in zip(nodes, evidence_quotes)
    ]
    question = " 그리고 ".join(topics) + "의 관계를 설명해줘."
    ground_truth = "\n".join(evidence_quotes)
    return _SynthesizedProbe(
        question=question,
        ground_truth=ground_truth,
        evidence=[
            {"source_index": index, "quote": quote}
            for index, quote in enumerate(evidence_quotes)
        ],
    )


def _format_sources_for_llm(nodes: list[KGNode]) -> str:
    """멀티홉 evidence가 출처를 가리킬 수 있도록 source 번호를 붙인다."""

    return "\n\n".join(
        f"[SOURCE {index}]\n{node.text}"
        for index, node in enumerate(nodes)
    )


def _llm_synthesize_query(
    nodes: list[KGNode],
    quadrant: str,
    subtype: str | None,
    persona: str,
    style: str,
    length: str,
    evol_dir: str,
) -> _SynthesizedProbe | None:
    """
    RAGAS 시나리오(quadrant/subtype/persona/style/length/evol_direction)를
    LLM(OpenAI/Gemini/GitHub Models, EVAL_LLM_PROVIDER로 선택) 호출 1번으로 합성한다. 기존
    _llm_generate_single_hop과 동일한 폴백 규칙: 키 없거나 호출·파싱 실패 시
    None(호출부가 휴리스틱으로 대체).
    """
    if not llm_provider.has_key():
        return None
    try:
        instruction = _quadrant_instruction(quadrant, subtype)
        data = llm_provider.chat_json(
            system=("너는 RAG 파이프라인 평가용 테스트 질문을 설계하는 평가자다. "
                    f"{instruction} "
                    f"질문자의 페르소나는 '{persona}', 어투는 '{style}', 길이는 '{length}' 이며 "
                    f"'{evol_dir}' 방향으로 질문의 난이도·범위를 조정해라. "
                     "질문과 정답은 컨텍스트 문장을 그대로 베끼지 말고 자기 말로 다시 구성하되, "
                     "정답은 컨텍스트에 있는 사실에서 벗어나면 안 된다. "
                     "evidence에는 정답에 실제 사용한 각 SOURCE의 짧고 연속된 원문을 "
                     "한 글자도 바꾸지 말고 복사하라. 단일홉은 최소 1개, 멀티홉은 사용한 "
                     "각 SOURCE마다 최소 1개를 반환하라. source_index는 표시된 SOURCE 번호다. "
                     "반드시 {\"question\": str, \"ground_truth\": str, "
                     "\"evidence\": [{\"source_index\": int, \"quote\": str}]} "
                     "형태의 JSON으로만 답하라."),
            user=f"[컨텍스트]\n{_format_sources_for_llm(nodes)}",
        )
        question = (data.get("question") or "").strip()
        ground_truth = (data.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            return None
        return _SynthesizedProbe(
            question=question,
            ground_truth=ground_truth,
            evidence=_parse_evidence(data.get("evidence")),
        )
    except Exception as e:
        print(f"[Eval] STEP1: RAGAS Probe 합성 실패({e}) -> 휴리스틱 폴백")
        return None


def _quadrant_instruction(quadrant: str, subtype: str | None) -> str:
    """4분면(+멀티홉 서브타입)별 LLM 지시문."""
    if quadrant == "single_specific":
        return "주어진 컨텍스트(청크 1개) 안의 구체적인 사실(숫자·기한·조건 등) 하나를 묻는 질문을 만들어라."
    if quadrant == "single_abstract":
        return "주어진 컨텍스트(청크 1개)의 취지·목적을 요약해서 답해야 하는, 특정 사실 하나만 콕 집지 않는 추상적인 질문을 만들어라."
    subtype_desc = {
        "bridge": "첫 번째 컨텍스트의 사실을 근거로 두 번째 컨텍스트의 내용을 연결해야 답할 수 있는 질문(다리형 멀티홉)을 만들어라.",
        "comparison": "두 컨텍스트의 내용을 서로 비교해야 답할 수 있는 질문을 만들어라.",
        "aggregation": "두 컨텍스트에 나뉘어 있는 정보를 종합(합산·목록화)해야 답할 수 있는 질문을 만들어라.",
    }
    base = subtype_desc.get(subtype or "bridge", subtype_desc["bridge"])
    if quadrant == "multi_abstract":
        base += " 다만 특정 숫자 하나보다는 두 컨텍스트를 아우르는 전체적인 맥락·취지를 종합하는 추상적인 질문으로 만들어라."
    return base
