"""
agents/eval/replay_labels.py
리플레이 모드 전용 ext_ 라벨 - 외부 RAG 로그에 대한 원인 '소견' (미니 세트 4개)

기존 diagnose.py 의 30개 라벨은 gold 청크·우리 인덱스 재검색을 전제해 외부
로그에서 전부 침묵한다(docs/external_rag_log_intake.md §4). 이 모듈은 리플레이
모드에서만 쓰는 별도 라벨 집합을 정의한다 - 기존 라벨과 이름공간을 분리해
(ext_ 접두어) 확정 진단(내부)과 소견(외부)의 증거 수준 차이를 이름에 남긴다.

원칙 (docs §5):
- diagnose.py 무변경. 같은 Finding 데이터클래스를 생산해 build_report 로 합류.
- 지표가 실측된 경우에만 발동(미측정 → 침묵, 기존 폴백 철학).
- 권고는 자동 적용 없이 EXT_RECOMMENDATIONS 로 기존 rules 처방을 '재참조'만
  한다 - 기존 LABEL_TO_PRESCRIPTIONS 에 섞지 않는다(그 테이블은 내부 자동
  적용 루프의 소비물이라, 섞으면 언젠가 내부 모드가 ext_ 를 주울 위험).
- 검색축 신호는 gold_contexts(정답 근거 문단 텍스트)와 검색 결과의 '텍스트
  겹침'으로 판정한다 - 청크 ID 정합이 필요 없어 외부 로그에서도 결정적으로
  계산된다. gold_contexts 가 없으면 환각 소견은 예비(confirmed=False)로
  강등된다(검색 탓/생성 탓을 못 가르므로).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional

from core.schema import Finding

from agents.eval.types import CONTEXT_CHARS_MAX, EvalRecord
from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS

# ── 발동 문턱 ─────────────────────────────────────────────────────
# RAGAS 점수는 0~1. 0.5 는 "판정이 좋다/나쁘다로 기운 쪽"의 중립 경계로,
# 내부 라벨처럼 캘리브레이션된 값이 아니다(소견 수준의 보수적 기본값).
EXT_FAITH_LOW = 0.5        # 이 밑이면 답이 컨텍스트에 근거하지 않음
EXT_REL_LOW = 0.5          # 이 밑이면 답이 질문에 대한 답이 아님
EXT_CORRECTNESS_LOW = 0.5  # 이 밑이면 정답과 불일치
GOLD_OVERLAP_FOUND = 0.6   # gold 문단의 이 비율 이상이 검색 결과에 덮이면 "근거를 찾아왔다"

_SEVERITY = {
    "ext_generation_hallucination": "critical",
    "ext_answer_off_topic": "critical",
    "ext_context_overflow": "warning",
    "ext_grounded_but_wrong": "warning",   # 원인이 코퍼스/근거 쪽 - 사람 개입 계열
}

# ext_ 라벨 → 기존 rules 처방 재참조 (복사 아님 - 처방 문구의 원본은 하나로 유지)
EXT_RECOMMENDATIONS: dict[str, dict] = {
    "ext_generation_hallucination": LABEL_TO_PRESCRIPTIONS["generation_hallucination"],
    "ext_answer_off_topic": LABEL_TO_PRESCRIPTIONS["generation_misinterpretation"],
    "ext_context_overflow": LABEL_TO_PRESCRIPTIONS["too_long_context"],
    "ext_grounded_but_wrong": LABEL_TO_PRESCRIPTIONS["corpus_gap"],
}


def recommendation_ids(label: str) -> list[str]:
    """라벨의 권고(처방 id) 목록 - 리포트/CLI 의 권고 카드 문구용."""
    entry = EXT_RECOMMENDATIONS.get(label)
    if not entry:
        return []
    return [p["id"] for p in entry.get("prescriptions", [])]


# ── 검색축 결정적 신호: gold 문단 텍스트 겹침 ─────────────────────

_WS = re.compile(r"\s+")


def _squash(text: str) -> str:
    """공백 정규화 - 줄바꿈/들여쓰기 차이로 겹침이 깨지지 않게."""
    return _WS.sub(" ", (text or "").strip())


# 우연 일치 차단 문턱: 이 길이 미만의 매칭 블록(낱글자·짧은 조각)은 근거로 안 친다.
# 없으면 '700만원' 같은 짧은 gold 가 무관한 문장의 흩어진 숫자·글자와 1.0 으로
# 매칭되는 거짓 양성이 난다(→ 환각을 '확정'으로 오판). gold 가 이 길이보다
# 짧으면 gold 전체 길이를 문턱으로 쓴다(정확 포함이면 여전히 점수가 나오게).
_MIN_MATCH_BLOCK = 4


def _coverage(gold: str, haystack: str) -> float:
    """gold 문단이 haystack(검색 결과 연결본)에 덮이는 비율(0~1).

    SequenceMatcher 매칭 블록 중 _MIN_MATCH_BLOCK 이상 길이의 합 / gold 길이.
    부분 일치(청킹 경계로 잘린 gold, 공백 미세 차이)에는 비례 점수가 나오고,
    낱글자 우연 일치는 걸러진다."""
    if not gold or not haystack:
        return 0.0
    threshold = min(_MIN_MATCH_BLOCK, len(gold))
    sm = SequenceMatcher(None, gold, haystack, autojunk=False)
    matched = sum(block.size for block in sm.get_matching_blocks()
                  if block.size >= threshold)
    return matched / len(gold)


# 이보다 짧은 gold 는 '근거 문단'이 아니라 정답 조각이다("700만원" 류). 무관한
# 문맥에 통째로 우연 포함될 수 있어, 높은 겹침으로 환각을 '확정'으로 뒤집는
# 근거가 못 된다 - 측정에서 제외한다(전부 짧으면 None = 판정 재료 없음).
_MIN_GOLD_CHARS = 10


def gold_context_recall(record: EvalRecord) -> Optional[float]:
    """정답 근거 문단이 검색 결과에 얼마나 덮였나(0~1). 재료 없으면 None.

    gold_contexts 는 build_replay_records 가 probe.metadata 에 실어둔다 -
    Probe/EvalRecord 스키마를 바꾸지 않기 위해서다(공유 계약 무변경)."""
    golds = [_squash(g) for g in record.probe.metadata.get("gold_contexts", [])]
    golds = [g for g in golds if len(g) >= _MIN_GOLD_CHARS]
    if not golds or not record.retrieved_context:
        return None
    haystack = _squash(" ".join(record.retrieved_context))
    return sum(_coverage(g, haystack) for g in golds) / len(golds)


# ── ext_ 라벨 판정 ────────────────────────────────────────────────

def _context_char_total(record: EvalRecord) -> int:
    return sum(len(text or "") for text in record.retrieved_context)


def _finding(record: EvalRecord, label: str, ftype: str, confirmed: bool, reason: str) -> Finding:
    prefix = "" if confirmed else "[예비] "
    return Finding(
        finding_id=f"{record.probe.probe_id}:{label}",
        type=ftype,
        severity=_SEVERITY[label],
        description=f"{prefix}[리플레이 소견] {label} - {reason}",
        label=label,
        confirmed=confirmed,
        affected_probes=[record.probe.probe_id],
        metadata={"group": "EXT", "reason": reason},
    )


def diagnose_replay_record(record: EvalRecord) -> list[Finding]:
    """외부 로그 레코드 1건의 ext_ 소견. 실측된 지표만 근거로 쓴다.

    LLM 미실행(ragas 비어 있음)이면 빈 리스트 - "판정 불가 = 침묵"이
    기존 폴백 철학이다. 켜지는 라벨:
      ext_answer_off_topic          relevancy 낮음 (동문서답)
      ext_generation_hallucination  faithfulness 낮음 + 질문엔 답함.
                                    gold 문단 겹침이 높으면 확정(근거를 줬는데 안 씀),
                                    겹침 미측정/낮음이면 예비(검색 탓일 수 있음)
      ext_context_overflow          컨텍스트 총길이 과다 + faithfulness 낮음
      ext_grounded_but_wrong        근거엔 충실한데 정답과 불일치 (GT 필요)
    """
    findings: list[Finding] = []
    faith = record.ragas.get("faithfulness")
    rel = record.ragas.get("response_relevancy")
    corr = record.ragas.get("answer_correctness")
    overlap = gold_context_recall(record)

    if rel is not None and rel < EXT_REL_LOW:
        findings.append(_finding(
            record, "ext_answer_off_topic", "generation", True,
            f"answer relevancy {rel:.2f} - 답이 질문을 비껴감"))

    # rel 미측정(None)은 통과로 본다 - faithfulness 가 실측된 이상 "동문서답으로
    # 판명되지 않음"이면 환각 소견을 낼 근거는 충분하다(완전 미측정 침묵과 구분).
    if faith is not None and faith < EXT_FAITH_LOW and (rel is None or rel >= EXT_REL_LOW):
        if overlap is not None and overlap >= GOLD_OVERLAP_FOUND:
            confirmed, why = True, (
                f"faithfulness {faith:.2f}, gold 근거 겹침 {overlap:.2f} - "
                "검색은 근거를 찾아왔는데 답이 근거를 쓰지 않음(환각 확정)")
        elif overlap is not None:
            confirmed, why = False, (
                f"faithfulness {faith:.2f}, gold 근거 겹침 {overlap:.2f} - "
                "검색이 근거를 못 찾아왔을 가능성(검색 실패 의심)")
        else:
            confirmed, why = False, (
                f"faithfulness {faith:.2f} - 근거 없는 답변. "
                "검색 탓/생성 탓 분리는 정답 근거 문단(gold_contexts) 필요")
        findings.append(_finding(
            record, "ext_generation_hallucination", "generation", confirmed, why))

    if (faith is not None and faith < EXT_FAITH_LOW
            and _context_char_total(record) > CONTEXT_CHARS_MAX):
        findings.append(_finding(
            record, "ext_context_overflow", "context", True,
            f"컨텍스트 {_context_char_total(record)}자(상한 {CONTEXT_CHARS_MAX}) + "
            f"faithfulness {faith:.2f} - 과다 컨텍스트가 근거 사용을 방해"))

    if (corr is not None and corr < EXT_CORRECTNESS_LOW
            and faith is not None and faith >= EXT_FAITH_LOW):
        findings.append(_finding(
            record, "ext_grounded_but_wrong", "data", True,
            f"correctness {corr:.2f}인데 faithfulness {faith:.2f} - "
            "답은 근거에 충실하나 근거 자체가 틀렸거나 부족(코퍼스/검색 문제)"))

    return findings


def apply_ext_labels(records: list[EvalRecord]) -> None:
    """리플레이 레코드 전체에 ext_ 소견을 채운다(record.findings). run_replay 전용."""
    for record in records:
        record.findings = diagnose_replay_record(record)
