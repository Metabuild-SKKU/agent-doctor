"""
agents/eval/probe_gen.py
STEP1: Probe(진단용 질문) 생성

설계 문서 'STEP 1: Probe 생성' 구현.
Probe = RAG 파이프라인을 진단하기 위한 테스트 질문 집합.

Probe 소스별 신뢰도 (설계 §3):
    user_log(실사용 쿼리)  >  taxonomy(사람 작성)  >  llm_generated(자동 생성)

생성 우선순위:
    1) state.user_questions 가 있으면 → user_log Probe
    2) 없으면 → 청크 기반 자동 생성(llm_generated)

현재 구현: Single-Hop Specific(단일 청크·사실 기반) 질문만 LLM으로 생성한다
(청크 내용을 그대로 베끼지 않도록 질문·정답을 LLM이 새로 구성 — retrieval_temp.py의
_llm_generate 와 동일한 폴백 규칙: OPENAI_API_KEY 없거나 호출 실패 시 휴리스틱 추출로 대체).

[구현 포인트]  (다음 단계로 남겨둠)
    - Single-Hop Abstract / Multi-Hop(bridge·comparison·aggregation) 추가.
    - DataMorgana 20%, 무응답(Held-out·False Premise) 5% 비중 혼합.
    - RAGAS TestsetGenerator 식 지식그래프(청크 간 관계) 기반 시나리오 생성으로 확장.
    - eval_probes.json 영속화 + 문서 diff 기반 증분 생성(골든 테스트셋 재사용).
    - gold_doc_id/gold_char_span 스키마 추가(재청킹에도 안 깨지는 기준).
"""
from __future__ import annotations

import json
import os
import random
import re

from core.schema import Chunk, Document, Probe
from core.state import AgentDoctorState
from agents.eval import knowledge_graph
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
)

# 자동 생성 기본 개수 (설계: testset_size=5~10 으로 시작해 비용 확인 후 확대)
DEFAULT_TESTSET_SIZE = 5


def generate_probes(state: AgentDoctorState) -> list[Probe]:
    """
    state 를 보고 Probe 리스트를 생성한다.

    읽기: state.user_questions, state.chunks
    """
    """
    user log 기반 프로브 생성
    if state.user_questions:
        probes = _from_user_questions(state.user_questions)
        print(f"[Eval] STEP1: user_log Probe {len(probes)}개 생성")
        return probes
    """
        
    probes = _from_chunks(state.chunks, DEFAULT_TESTSET_SIZE)
    print(f"[Eval] STEP1: llm_generated(폴백) Probe {len(probes)}개 생성")
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
    OpenAI 로 Single-Hop Specific (질문, 정답) 쌍을 생성한다.
    청크 문장을 그대로 베끼지 않도록 질문·정답 모두 새로 구성하게 지시한다.
    키/라이브러리 없거나 호출·파싱 실패 시 None(호출부가 휴리스틱으로 대체).
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI()
        model = os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                    "너는 RAG 파이프라인 평가용 테스트 질문을 설계하는 평가자다. "
                    "주어진 문서 조각(컨텍스트) 하나만으로 답할 수 있는, 실제 사용자가 물어볼 법한 "
                    "구체적인 사실 기반 질문(Single-Hop Specific) 하나와 그 정답을 만들어라. "
                    "질문과 정답 모두 컨텍스트 문장을 그대로 베끼지 말고 자기 말로 다시 구성하되, "
                    "정답은 컨텍스트에 있는 사실에서 벗어나면 안 된다. "
                    "반드시 {\"question\": str, \"ground_truth\": str} 형태의 JSON으로만 답하라."},
                {"role": "user", "content": f"[컨텍스트]\n{chunk_text}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
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


# ── gold span / 위치-인덱스 유틸 (다음 단계에서 사용 예정, 아직 미배선) ──
#
# RAGAS 스타일 재구현(STEP1)에서 gold_char_span/gold_spans 로 정답 위치를
# 원문(Document.content) 절대 좌표로 저장해두면, Optimize→Index 재청킹 후에도
# (chunk_size/overlap이 달라져도) probe.gold_chunk_ids 를 다시 맞출 수 있다.
# 여기서는 그 순수 계산 로직만 둔다 — generate_probes() 는 아직 이 함수들을
# 호출하지 않는다(호출부 배선은 이후 단계에서 실측 생성기와 함께 연결).
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


# ── RAGAS 스타일 시나리오 합성 (다음 단계에서 배선 예정, 아직 미배선) ──
#
# knowledge_graph.build_graph()가 만든 그래프를 입력으로 받아 단일홉(구체/추상)
# + 멀티홉(bridge/comparison/aggregation) Probe를 생성한다. generate_probes()는
# 아직 이 함수들을 호출하지 않는다 — 실제 배선(state.documents 유무에 따른 분기,
# gold_spans 채우기, probe_store 영속화)은 이후 단계에서 처리한다.
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
    RAGAS 시나리오(quadrant/subtype/persona/style/length/evol_direction)를 OpenAI
    호출 1번으로 합성한다. 기존 _llm_generate_single_hop과 동일한 폴백 규칙:
    키/라이브러리 없거나 호출·파싱 실패 시 None(호출부가 휴리스틱으로 대체).
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    try:
        client = OpenAI()
        model = os.getenv("EVAL_GEN_MODEL", "gpt-4o-mini")
        instruction = _quadrant_instruction(quadrant, subtype)
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                    "너는 RAG 파이프라인 평가용 테스트 질문을 설계하는 평가자다. "
                    f"{instruction} "
                    f"질문자의 페르소나는 '{persona}', 어투는 '{style}', 길이는 '{length}' 이며 "
                    f"'{evol_dir}' 방향으로 질문의 난이도·범위를 조정해라. "
                    "질문과 정답 모두 컨텍스트 문장을 그대로 베끼지 말고 자기 말로 다시 구성하되, "
                    "정답은 컨텍스트에 있는 사실에서 벗어나면 안 된다. "
                    "반드시 {\"question\": str, \"ground_truth\": str} 형태의 JSON으로만 답하라."},
                {"role": "user", "content": f"[컨텍스트]\n{context}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
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
