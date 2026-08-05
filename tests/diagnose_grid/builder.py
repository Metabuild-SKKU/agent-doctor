"""
tests/diagnose_grid/builder.py
진단 격자 케이스 빌더 — 1차 입력만 기술하면 나머지는 실제 계산 함수가 만든다.

케이스는 코퍼스 기하(문서 길이·청크 크기·겹침), gold span 좌표, 검색기가 돌려준 랭킹,
답변 텍스트, 심판 판정만 적는다. recall·f1·청크 경계·근거 밀도·순위 맵은 전부
metrics_* 의 실제 함수가 계산한다(signals 주입 없음, _compute_metrics 패치 없음).

주입은 set_context 스텁으로만 한다 — 검색기와 심판 LLM 은 진단기 입장에서 블랙박스라
그 출력이 원시 관측이다.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.schema import Chunk, Document, Probe
from agents.eval import metrics_common
from agents.eval.types import EvalRecord, Mode
from agents.index.agent import CHUNK_STRATEGIES


# ── 1차 입력 ──────────────────────────────────────────────────────

@dataclass
class Doc:
    """격자용 문서. text 를 직접 주거나 length 로 생성한다.

    space_every>0 이면 그 주기로 공백을 섞는다 — Index 의 청킹이 청크마다 앞뒤 공백을
    떼면서 좌표를 당기므로(_trimmed_slice), 공백이 있어야 청크 사이 '틈'이 재현된다.
    그 틈이 gold span 을 boundary_split 이 아니라 uncovered 로 만든다.
    """
    id: str
    length: int = 0
    text: Optional[str] = None
    space_every: int = 0


class Answer(str, Enum):
    """답변 텍스트 생성 규칙. 정확한 F1 값은 지정하지 않고 계산에 맡긴다."""
    GOLD_FULL = "gold_full"          # ground_truth 그대로
    GOLD_PARTIAL = "gold_partial"    # 앞 절반만
    WRONG = "wrong"                  # 무관한 답
    ABSTAIN = "abstain"              # 기권 마커
    EMPTY = "empty"                  # 빈 문자열


_ABSTAIN_TEXT = "제공된 정보로는 알 수 없습니다."
_WRONG_TEXT = "질문과 무관한 다른 사항을 설명하는 문장입니다."


@dataclass
class Case:
    id: str

    # 코퍼스 기하
    docs: list[Doc] = field(default_factory=lambda: [Doc("d1", 1200)])
    chunk_strategy: str = "fixed"             # Index 의 CHUNK_STRATEGIES 키
    chunk_size: int = 200
    chunk_overlap: int = 0
    duplicates: list[tuple[int, int]] = field(default_factory=list)  # (사본 인덱스, 원본 인덱스)
    corpus_exclude: list[int] = field(default_factory=list)          # 코퍼스에서 뺄 청크 인덱스

    # probe
    gold_spans: list[tuple[str, int, int]] = field(default_factory=list)
    span_grounding: Optional[str] = None      # None(=exact) | "exact" | "chunk_fallback" | "partial"
    ground_truth: str = "1972년 12월 27일에 제7차 개정 헌법이 공포되었다"
    qtype: Optional[str] = None
    answer_exists: Optional[bool] = None

    # 검색기 출력 (청크 인덱스)
    retrieved: list[int] = field(default_factory=list)
    wide_ranking: Optional[list[int]] = None
    dense_ranking: Optional[list[int]] = None
    lexical_ranking: Optional[list[int]] = None
    pre_rerank: Optional[list[int]] = None
    search_mode: str = "dense"
    reranked: bool = False
    mmr_applied: bool = False

    # 생성
    answer: Answer = Answer.WRONG
    oracle_answer: Optional[Answer] = Answer.GOLD_FULL

    # 심판 (없으면 그 트랙은 미측정)
    judge_real: dict = field(default_factory=dict)
    judge_oracle: dict = field(default_factory=dict)
    judge_abstention: Optional[bool] = None
    judge_reasoning_mode: Optional[str] = None
    # True 면 심판 LLM 을 실제로 태워 RAGAS 를 계산한다. 위 judge_* 에 적은 키는 계산값을
    # 덮어써서 그 지표만 고정할 수 있다(나머지는 계산). 기본 False = LLM 0회.
    # 켜면 케이스마다 LLM 호출이 나가고 값이 실행마다 흔들린다 — 골든 비교가 아니라
    # '실제 심판으로도 같은 라벨이 나오나'를 볼 때만 쓴다.
    compute_ragas: bool = False

    # 검증
    assert_derived: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)


# ── 빌드 ──────────────────────────────────────────────────────────

def doc_text(doc: Doc) -> str:
    """문서 원문. 위치마다 다른 음절이 나오게 해서 near-duplicate 오탐을 막는다."""
    if doc.text is not None:
        return doc.text
    seed = sum(ord(c) for c in doc.id)
    chars = [chr(0xAC00 + ((i * 7919 + seed * 131) % 11172)) for i in range(doc.length)]
    if doc.space_every > 0:
        for i in range(doc.space_every - 1, len(chars), doc.space_every):
            chars[i] = " "
    return "".join(chars)


def build_chunks(case: Case) -> list[Chunk]:
    """Index 의 실제 청킹 전략으로 Chunk 리스트를 만든다.

    청킹 규칙을 여기서 다시 구현하지 않는다 — 격자가 재려는 게 청킹 라벨이라, 구현을
    흉내내면 정작 재야 할 것(공백 트림으로 생기는 청크 사이 틈, 전략별 청크 모양)이 빠진다.
    chunk_id 만 격자 규약(c0, c1, ...)으로 붙인다 — 케이스가 인덱스로 청크를 참조하므로.
    """
    strategy = CHUNK_STRATEGIES[case.chunk_strategy]
    chunks: list[Chunk] = []
    for doc in case.docs:
        document = Document(doc_id=doc.id, source="grid", format="md", content=doc_text(doc))
        for draft in strategy(document, case.chunk_size, case.chunk_overlap):
            chunks.append(Chunk(
                chunk_id=f"c{len(chunks)}",
                doc_id=doc.id,
                text=draft.text,
                section=draft.section,
                char_span=(draft.start, draft.end),
            ))
    for copy_idx, origin_idx in case.duplicates:      # 사본은 원본 텍스트를 그대로 갖는다
        chunks[copy_idx].text = chunks[origin_idx].text
    return chunks


def gold_chunk_ids(case: Case, chunks: list[Chunk]) -> list[str]:
    """gold span 과 좌표가 겹치는 청크 = 정답 청크 라벨."""
    ids = []
    for chunk in chunks:
        c_start, c_end = chunk.char_span
        for doc_id, s_start, s_end in case.gold_spans:
            if chunk.doc_id == doc_id and c_start < s_end and c_end > s_start:
                ids.append(chunk.chunk_id)
                break
    return ids


def _answer_text(kind: Optional[Answer], ground_truth: str) -> Optional[str]:
    if kind is None:
        return None
    if kind is Answer.GOLD_FULL:
        return ground_truth
    if kind is Answer.GOLD_PARTIAL:
        return ground_truth[: len(ground_truth) // 2] + " " + _WRONG_TEXT
    if kind is Answer.WRONG:
        return _WRONG_TEXT
    if kind is Answer.ABSTAIN:
        return _ABSTAIN_TEXT
    return ""


def _hits(indices: Optional[list[int]], chunks: list[Chunk]) -> Optional[list[dict]]:
    if indices is None:
        return None
    return [{"chunk_id": chunks[i].chunk_id, "score": 1.0 - n * 0.001}
            for n, i in enumerate(indices)]


def _fixed_judge(case: Case, track: str) -> dict:
    """케이스가 못 박은 심판 값. 미지정 키는 넣지 않는다(계산 모드에서 계산값을 살리려고)."""
    if track == "real":
        return dict(case.judge_real)
    if track == "oracle":
        return dict(case.judge_oracle)
    if track == "abstention":
        return {} if case.judge_abstention is None else {"abstention": case.judge_abstention}
    if track == "reasoning_mode":
        return ({} if case.judge_reasoning_mode is None
                else {"reasoning_mode": case.judge_reasoning_mode})
    return {}


def _require_judge_llm() -> None:
    """compute_ragas 전제 확인. 못 갖췄으면 조용한 미측정 대신 즉시 실패시킨다 —
    _ragas_track 은 비활성·키없음에 {} 를 돌려주므로, 그대로 두면 케이스가 의도한
    지표 없이 통과해 '심판으로도 같은 라벨'을 검증한 척하게 된다."""
    from agents.eval import llm_provider
    from agents.eval.types import llm_eval_enabled
    if not llm_eval_enabled():
        raise RuntimeError("compute_ragas=True 인데 EVAL_ENABLE_LLM 이 꺼져 있다")
    if not llm_provider.has_key():
        raise RuntimeError("compute_ragas=True 인데 심판 LLM 키가 없다")


def build(case: Case) -> tuple[EvalRecord, list[Chunk]]:
    """케이스 → (EvalRecord, 코퍼스 청크). set_context 까지 마친 상태로 돌려준다."""
    if case.compute_ragas:
        _require_judge_llm()
    chunks = build_chunks(case)
    corpus = [c for i, c in enumerate(chunks) if i not in set(case.corpus_exclude)]

    probe_metadata: dict = {}
    if case.span_grounding is not None:
        probe_metadata["span_grounding"] = {"status": case.span_grounding}

    probe = Probe(
        probe_id=case.id,
        question="이 문서에서 묻는 사실은 무엇인가",
        source="taxonomy",
        answer_exists=case.answer_exists,
        ground_truth=case.ground_truth,
        gold_chunk_ids=gold_chunk_ids(case, chunks),
        qtype=case.qtype,
        gold_spans=[{"doc_id": d, "start": s, "end": e} for d, s, e in case.gold_spans],
        metadata=probe_metadata,
    )

    retrieved_ids = [chunks[i].chunk_id for i in case.retrieved]
    details: dict = {
        "search_mode": case.search_mode,
        "reranked": case.reranked,
        "mmr_applied": case.mmr_applied,
    }
    if case.pre_rerank is not None:
        details["pre_rerank_ids"] = [chunks[i].chunk_id for i in case.pre_rerank]

    gold_ids = set(probe.gold_chunk_ids)
    record = EvalRecord(
        probe=probe,
        retrieved_context=[chunks[i].text for i in case.retrieved],
        retrieved_chunk_ids=retrieved_ids,
        retrieval_details=details,
        generated_answer=_answer_text(case.answer, case.ground_truth) or "",
        oracle_answer=_answer_text(case.oracle_answer, case.ground_truth),
        oracle_context=[c.text for c in chunks if c.chunk_id in gold_ids],
    )

    wide = _hits(case.wide_ranking, chunks)
    dense = _hits(case.dense_ranking, chunks)
    lexical = _hits(case.lexical_ranking, chunks)

    def ragas_fn(rec, track):
        fixed = _fixed_judge(case, track)
        if not case.compute_ragas:
            return fixed
        from agents.eval.agent import _ragas_track      # LLM 경로일 때만 import
        computed = dict(_ragas_track(rec, track) or {})
        computed.update(fixed)                          # 케이스에 적은 키가 계산값을 이긴다
        return computed

    metrics_common.set_context(
        client=object(),
        chunks=corpus,
        retrieve_fn=(lambda *_a, **_k: wide) if wide is not None else None,
        dense_fn=(lambda *_a, **_k: dense) if dense is not None else None,
        keyword_fn=(lambda *_a, **_k: lexical) if lexical is not None else None,
        ragas_fn=ragas_fn,
        wide_n=100,
    )
    metrics_common.set_mode(Mode.DEEP)
    return record, chunks
