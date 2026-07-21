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
대체 가능) — retrieval_temp.py 와 동일한 폴백 규칙: 키 없거나 호출 실패 시
휴리스틱 추출로 대체). testset_size(n) < 8 이면 5%/20%가 0~1개로 반올림돼 통계적
의미가 없으므로 _allocate_budget 이 전부 RAGAS 로 몰아준다(무응답/DataMorgana 없음).
gold_spans 가 채워진 probe(taxonomy 등 외부 소스)는 _resync_gold_chunk_ids 로
재청킹 후에도 gold_chunk_ids 를 다시 맞춘다.

[구현 포인트]  (다음 단계로 남겨둠)
    - eval_probes.json 영속화 + 문서 diff 기반 증분 생성(골든 테스트셋 재사용).
"""
from __future__ import annotations

import random
import re

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
    resolve_probe_source,
    taxonomy_qa_path,
    taxonomy_corpus_path,
)

# 자동 생성 기본 개수 (설계: testset_size=5~10 으로 시작해 비용 확인 후 확대)
DEFAULT_TESTSET_SIZE = 10


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
        return probes

    graph = knowledge_graph.build_graph(state.chunks)
    budget = _allocate_budget(DEFAULT_TESTSET_SIZE)

    ragas_probes = _generate_ragas_probes(graph, budget["ragas"])
    if not ragas_probes:
        probes = _from_chunks(state.chunks, DEFAULT_TESTSET_SIZE)
        print(f"[Eval] STEP1: llm_generated(폴백) Probe {len(probes)}개 생성")
        return probes

    datamorgana_probes = _generate_datamorgana_probes(graph, budget["datamorgana"])
    no_answer_probes = _generate_no_answer_probes(state.chunks, graph, budget["no_answer"])

    probes = ragas_probes + datamorgana_probes + no_answer_probes
    probes = _resync_gold_chunk_ids(probes, state.chunks, state.documents)
    print(f"[Eval] STEP1: llm_generated Probe {len(probes)}개 생성 "
          f"(ragas={len(ragas_probes)}, datamorgana={len(datamorgana_probes)}, "
          f"no_answer={len(no_answer_probes)})")
    return probes


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
    """EVAL_TAXONOMY_QA/CORPUS 에서 taxonomy Probe(gold_spans 포함)를 로드하고,
    현재 청크에 맞춰 gold_chunk_ids 를 resync 한다(재청킹돼도 gold 유지).

    KORQUAD_MAX_DOCS / KORQUAD_QA_LIMIT 로 규모 제한(스모크). MAX_DOCS 는 Ingest 의
    corpus 로더와 같은 규칙이라 corpus/qa 가 같은 문서 집합을 본다."""
    import os
    from agents.eval.datasets.korquad import load_taxonomy_probes

    def _pos_int(name):
        raw = os.getenv(name, "").strip()
        return int(raw) if raw.isdigit() and int(raw) > 0 else None  # 0/비정수/미설정 = 전체

    probes = load_taxonomy_probes(taxonomy_qa_path(), taxonomy_corpus_path(),
                                  limit=_pos_int("KORQUAD_QA_LIMIT"),
                                  max_docs=_pos_int("KORQUAD_MAX_DOCS"))
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

_SENT_SPLIT = re.compile(r"(?<=[.!?。？！\n])\s+")


def _from_chunks(chunks: list[Chunk], size: int) -> list[Probe]:
    """
    청크마다 Single-Hop Specific Probe 를 만든다.
    각 청크에 대해 LLM 생성을 시도하고, 실패/미설정 시 휴리스틱 추출로 대체한다.
    """
    probes: list[Probe] = []
    # 너무 짧은 청크는 스킵, 앞에서부터 size 개 사용
    usable = [c for c in chunks if c.text and len(c.text.strip()) >= 20]
    for i, chunk in enumerate(usable[:size]):
        question, ground_truth = _llm_generate_single_hop(chunk.text) or _heuristic_single_hop(chunk.text)
        probes.append(Probe(
            probe_id=f"probe_gen_{i:03d}",
            question=question,
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=ground_truth,
            gold_chunk_ids=[chunk.chunk_id],
            qtype=None,                 # 단일홉
        ))
    return probes


def _heuristic_single_hop(text: str) -> tuple[str, str]:
    """LLM 미사용/실패 시 폴백: 청크 앞부분을 질문 주제로, 청크 전문을 정답으로 그대로 사용."""
    topic = _topic_of(text)
    return f"{topic}에 대해 설명해줘.", text.strip()


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
# generate_probes() 가 마지막 단계에서 _resync_gold_chunk_ids 를 호출한다
# (gold_spans 가 없는 probe는 그대로 통과 — _generate_ragas_probes 가 만드는
# probe는 아직 gold_spans 를 채우지 않으므로 현재는 사실상 no-op).
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
        span = _locate_span(doc.content, c.text, cursor)
        if span is None:
            continue
        start, end = span
        index.append((c.chunk_id, start, end))
        cursor = start
    return index


def _resync_gold_chunk_ids(
    probes: list[Probe], chunks: list[Chunk], documents: list[Document]
) -> list[Probe]:
    """
    probe.gold_spans(원문 절대 좌표, 재청킹해도 불변)를 기준으로 현재 chunks 와
    구간이 겹치는 청크 id를 다시 찾아 probe.gold_chunk_ids 를 갱신한다(in-place + 반환).
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

    for probe in probes:
        if not probe.gold_spans:
            continue
        matched: list[str] = []
        for span in probe.gold_spans:
            doc_id = span.get("doc_id")
            s_start, s_end = span.get("start"), span.get("end")
            if doc_id is None or s_start is None or s_end is None:
                continue
            for chunk_id, c_start, c_end in _position_index(doc_id):
                if c_start < s_end and c_end > s_start:  # 구간 겹침
                    matched.append(chunk_id)
        if matched:
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


def _generate_ragas_probes(graph: knowledge_graph.KGraph, n: int) -> list[Probe]:
    """
    그래프 하나로 RAGAS 스타일 Probe n개를 생성한다(청크/문서 리스트를 별도로
    받지 않음 - KGNode가 chunk_id/doc_id/text를 이미 담고 있어 그래프만으로
    충분하다는 판단. 계획 문서의 _generate_ragas_probes(graph, docs, n) 시그
    니처에서 docs를 뺀 의도적 편차).
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

    probes: list[Probe] = []
    for quadrant, subtype in plan:
        if quadrant.startswith("single"):
            nodes = [random.choice(usable)] if usable else []
        else:
            pair_ids = _next_pair()
            nodes = [graph.nodes[cid] for cid in pair_ids if cid in graph.nodes] if pair_ids else []
        if len(nodes) < (1 if quadrant.startswith("single") else 2):
            continue
        probe = _make_ragas_probe(nodes, quadrant, subtype, len(probes))
        if probe is not None:
            probes.append(probe)
    return probes


# ── DataMorgana-lite (예산 20%) ───────────────────────────────────
#
# 풀 DataMorgana(설정 가능한 페르소나/스타일 조합의 별도 파이프라인) 대신,
# 이미 있는 _llm_synthesize_query 시나리오 파라미터 중 "질문자가 덜 친절하게
# 묻는" 조합(비격식·긴 길이·breadth 확장)을 강제해 좀 더 거친 질문을 만드는
# 최소 버전으로 축소했다(설계 문서도 "시간 남으면 구체화" 항목으로 분류).

def _generate_datamorgana_probes(graph: knowledge_graph.KGraph, n: int) -> list[Probe]:
    """단일홉 노드에서 거친 스타일(conversational/long/breadth)로 질문을 합성한다."""
    if n <= 0:
        return []
    usable = [node for node in graph.nodes.values() if node.text and len(node.text.strip()) >= 20]
    probes: list[Probe] = []
    for i in range(n):
        if not usable:
            break
        node = random.choice(usable)
        result = _llm_synthesize_query(
            node.text, "single_specific", None,
            persona=random.choice(PERSONAS), style="conversational",
            length="long", evol_dir="breadth",
        )
        if result is None:
            result = _heuristic_synthesize_query([node])
        question, ground_truth = result
        if not question or not ground_truth:
            continue
        probes.append(Probe(
            probe_id=f"probe_datamorgana_{i:03d}",
            question=question,
            source="llm_generated",
            expected_difficulty="medium",
            answer_exists=True,
            ground_truth=ground_truth,
            gold_chunk_ids=[node.chunk_id],
            qtype=None,
            metadata={"gen_method": "datamorgana_lite", "style": "conversational"},
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
    probes: list[Probe] = []
    for i in range(n):
        if not usable:
            break
        node = random.choice(usable)
        question = _false_premise_question(node.text)
        if question is None:
            continue
        probes.append(Probe(
            probe_id=f"probe_false_premise_{start_index + i:03d}",
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


def _make_ragas_probe(
    nodes: list[KGNode], quadrant: str, subtype: str | None, index: int
) -> Probe | None:
    """노드(들)로 시나리오를 샘플링하고 질문을 합성해 Probe 하나를 만든다."""
    persona = random.choice(PERSONAS)
    style = random.choice(QUERY_STYLES)
    length = random.choice(QUERY_LENGTHS)
    evol_dir = random.choice(EVOL_DIRECTIONS)
    is_multi = quadrant.startswith("multi")
    context = "\n\n".join(node.text for node in nodes)

    result = _llm_synthesize_query(context, quadrant, subtype, persona, style, length, evol_dir)
    if result is None:
        result = _heuristic_synthesize_query(nodes)
    question, ground_truth = result
    if not question or not ground_truth:
        return None

    gen_method = f"ragas_{quadrant}_{subtype}" if is_multi else f"ragas_{quadrant}"
    return Probe(
        probe_id=f"probe_{quadrant}_{index:03d}",
        question=question,
        source="llm_generated",
        expected_difficulty="medium",
        answer_exists=True,
        ground_truth=ground_truth,
        gold_chunk_ids=[node.chunk_id for node in nodes],
        qtype=subtype if is_multi else None,
        metadata={
            "gen_method": gen_method,
            "persona": persona,
            "style": style,
            "length": length,
            "evol_direction": evol_dir,
        },
    )


def _heuristic_synthesize_query(nodes: list[KGNode]) -> tuple[str, str]:
    """LLM 미사용/실패 시 폴백. 단일홉은 기존 단일홉 폴백과 동일한 방식,
    멀티홉은 두 청크의 주제를 이어붙인 나열형 질문으로 대체한다(정교하지
    않음 - [구현 포인트]로 남김, LLM 경로가 정상 동작하면 쓰이지 않음)."""
    if len(nodes) == 1:
        topic = _topic_of(nodes[0].text)
        return f"{topic}에 대해 설명해줘.", nodes[0].text.strip()
    topics = [_topic_of(n.text) for n in nodes]
    question = " 그리고 ".join(topics) + "의 관계를 설명해줘."
    ground_truth = "\n".join(n.text.strip() for n in nodes)
    return question, ground_truth


def _llm_synthesize_query(
    context: str,
    quadrant: str,
    subtype: str | None,
    persona: str,
    style: str,
    length: str,
    evol_dir: str,
) -> tuple[str, str] | None:
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
                    "질문과 정답 모두 컨텍스트 문장을 그대로 베끼지 말고 자기 말로 다시 구성하되, "
                    "정답은 컨텍스트에 있는 사실에서 벗어나면 안 된다. "
                    "반드시 {\"question\": str, \"ground_truth\": str} 형태의 JSON으로만 답하라."),
            user=f"[컨텍스트]\n{context}",
        )
        question = (data.get("question") or "").strip()
        ground_truth = (data.get("ground_truth") or "").strip()
        if not question or not ground_truth:
            return None
        return question, ground_truth
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
