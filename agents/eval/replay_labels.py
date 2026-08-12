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

from agents.eval.metrics_basic import is_abstention
from agents.eval.types import CONTEXT_CHARS_MAX, EvalRecord
from agents.optimize.rules import LABEL_TO_PRESCRIPTIONS

# ── 발동 문턱 ─────────────────────────────────────────────────────
# RAGAS 점수는 0~1. 0.5 는 "판정이 좋다/나쁘다로 기운 쪽"의 중립 경계로,
# 내부 라벨처럼 캘리브레이션된 값이 아니다(소견 수준의 보수적 기본값).
EXT_FAITH_LOW = 0.5        # 이 밑이면 답이 컨텍스트에 근거하지 않음
EXT_REL_LOW = 0.5          # 이 밑이면 답이 질문에 대한 답이 아님
EXT_CORRECTNESS_LOW = 0.5  # 이 밑이면 정답과 불일치
GOLD_OVERLAP_FOUND = 0.6   # gold 문단의 이 비율 이상이 검색 결과에 덮이면 "근거를 찾아왔다"
# 검색축 문턱. context_precision 은 "가져온 컨텍스트 중 정답에 쓸모 있는 비율"이라
# 다른 지표보다 낮게 나오는 게 정상이다(top_k 를 넉넉히 잡으면 무관한 청크가 섞인다).
# 0.5 를 그대로 쓰면 정상 검색까지 잡히므로 더 보수적으로 둔다 — 실측(offtopic):
# 정상 검색 1.00 / 어긋난 검색 0.20 이라 그 사이가 넉넉하다.
EXT_CTX_PRECISION_LOW = 0.34

_SEVERITY = {
    "ext_generation_hallucination": "critical",
    "ext_answer_off_topic": "critical",
    "ext_context_overflow": "warning",
    "ext_grounded_but_wrong": "warning",   # 원인이 코퍼스/근거 쪽 - 사람 개입 계열
    "ext_retrieval_irrelevant": "critical",
    "ext_wrongful_abstention": "critical",  # 답할 수 있었는데 안 했다 - 사용자 체감 손실
    "ext_retrieval_starved_abstention": "warning",
}

# ext_ 라벨 → 기존 rules 처방 재참조 (복사 아님 - 처방 문구의 원본은 하나로 유지)
EXT_RECOMMENDATIONS: dict[str, dict] = {
    "ext_generation_hallucination": LABEL_TO_PRESCRIPTIONS["generation_hallucination"],
    "ext_answer_off_topic": LABEL_TO_PRESCRIPTIONS["generation_misinterpretation"],
    "ext_context_overflow": LABEL_TO_PRESCRIPTIONS["too_long_context"],
    "ext_grounded_but_wrong": LABEL_TO_PRESCRIPTIONS["corpus_gap"],
    # 검색이 질문과 무관한 걸 가져왔다 → 검색 설정 점검(임베딩·청킹 계열).
    # 순위/리랭커 수준의 세부 원인은 상대 인덱스 없이 못 가르므로(docs §4)
    # 여기까지가 상한이다.
    "ext_retrieval_irrelevant": LABEL_TO_PRESCRIPTIONS["retrieval_semantic_mismatch"],
    "ext_wrongful_abstention": LABEL_TO_PRESCRIPTIONS["generation_wrongful_abstention"],
    # 검색이 굶어서 기권했다 → 더 가져오게 하거나(top_k·겹침) 코퍼스를 보강한다.
    "ext_retrieval_starved_abstention": LABEL_TO_PRESCRIPTIONS["retrieval_missing_gold"],
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


def _retrieval_axis(record: EvalRecord) -> Optional[tuple[bool, str]]:
    """검색이 질문의 근거를 가져왔나 — (찾았나, 근거 문구). 미측정이면 None.

    두 재료 중 있는 것을 쓴다. 둘 다 정답지(골드셋) 계열이라, 골드셋 없는 로그에서는
    None 이 되어 검색축 라벨 세 개가 통째로 침묵한다(docs §5 - 이 경우를 열려면
    reference-free context relevance judge 가 필요하다. 미구현).

      1) gold_contexts 텍스트 겹침 — 결정적. 청크 ID 정합이 필요 없고 LLM 도 안 쓴다.
      2) context_precision — ground_truth 가 있을 때 RAGAS 가 실측하는 값
         (metrics_ragas 의 want_prec = ... and has_ref).

    둘 다 있고 판정이 엇갈리면 **나쁜 쪽**을 믿는다. 겹침은 "gold 문단이 컨텍스트
    안에 있나"만 보므로, top_k 를 넉넉히 잡아 무관한 청크가 잔뜩 섞여도 gold 하나만
    걸리면 1.00 이 된다(실측 offtopic: 겹침 1.00 인데 precision 0.20 — 검색이
    엉뚱한 걸 가져왔는데 겹침만 보면 정상으로 읽혔다). 검색 품질을 묻는 자리에서는
    '근거가 섞여 있다'보다 '쓸모없는 것이 대부분이다'가 더 정확한 신호다.
    """
    signals: list[tuple[bool, str]] = []
    overlap = gold_context_recall(record)
    if overlap is not None:
        signals.append((overlap >= GOLD_OVERLAP_FOUND, f"gold 근거 겹침 {overlap:.2f}"))
    prec = record.ragas.get("context_precision")
    if prec is not None:
        signals.append((prec >= EXT_CTX_PRECISION_LOW, f"context precision {prec:.2f}"))
    if not signals:
        return None
    failed = [s for s in signals if not s[0]]
    return failed[0] if failed else signals[0]


def _evidence_reached(record: EvalRecord) -> Optional[tuple[bool, str]]:
    """기권 분기 전용 — '근거가 컨텍스트에 도달했나'. _retrieval_axis 와 다른 질문이다.

    _retrieval_axis 는 "검색 품질이 전반적으로 좋은가"를 물어 나쁜 쪽 신호를 우선한다
    (top_k 를 넉넉히 잡아 무관한 청크가 섞여도 겹침만 보면 정상으로 읽히는 함정 방지).
    여기서 묻는 건 다르다 — "gold 문단이 컨텍스트 안에 있긴 했나"이므로, 있었다면
    주변에 무관한 청크가 얼마나 섞였든 기권의 정오는 이미 정해진다(있는데 기권 = wrongful).
    그래서 겹침이 실측되면 겹침을 믿고, 겹침이 없을 때만 precision 으로 폴백한다."""
    overlap = gold_context_recall(record)
    if overlap is not None:
        return overlap >= GOLD_OVERLAP_FOUND, f"gold 근거 겹침 {overlap:.2f}"
    prec = record.ragas.get("context_precision")
    if prec is not None:
        return prec >= EXT_CTX_PRECISION_LOW, f"context precision {prec:.2f}"
    return None


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

    기권한 답변은 소견을 내지 않는다(아래 주석 참고).
    """
    findings: list[Finding] = []
    faith = record.ragas.get("faithfulness")
    rel = record.ragas.get("response_relevancy")
    corr = record.ragas.get("answer_correctness")
    overlap = gold_context_recall(record)

    # "자료에서 확인되지 않습니다"류는 실패가 아니라 **올바른 동작**이다. 근거가
    # 없을 때 지어내지 않는 것이 우리가 권고하는 처방(strengthen_abstention /
    # require_citation)의 목표이기도 하다.
    # 그런데 기권문은 질문을 되풀이하지 않으므로 relevancy 가 0에 가깝게 나오고,
    # 그대로 두면 동문서답으로 오진된다 — 실측에서 처방 적용본의 기권 5건이 전부
    # ext_answer_off_topic 으로 잡혀, 처방 효과가 오히려 나빠진 것처럼 보였다.
    #
    # 기권이 '잘한 것'인지 '잘못한 것'인지는 검색축(질문 vs 컨텍스트)을 봐야 갈린다.
    #   근거가 있었는데 기권  → ext_wrongful_abstention (겁먹음 - 답할 수 있었다)
    #   근거가 없어서 기권    → ext_retrieval_starved_abstention (검색/코퍼스 탓)
    #   검색축 미측정         → 소견 없음 (놓치는 것이 오진보다 낫다)
    if is_abstention(record.generated_answer, extended=True):
        axis = _evidence_reached(record)
        if axis is None:
            return findings
        found, why = axis
        if found:
            findings.append(_finding(
                record, "ext_wrongful_abstention", "generation", True,
                f"{why} - 근거는 있었는데 답하지 않음(기권)"))
        else:
            findings.append(_finding(
                record, "ext_retrieval_starved_abstention", "retrieval", True,
                f"{why} - 검색이 근거를 못 가져와 기권"))
        return findings

    # 검색이 질문과 무관한 걸 가져왔다(기권하지 않고 답한 경우). 답변축 지표만 보면
    # 그 여파가 생성 문제로 보이므로, 검색축이 실측될 때는 원인을 검색으로 돌린다.
    # 실측(offtopic): 정상 검색 1.00 vs 어긋난 검색 0.20.
    axis = _retrieval_axis(record)
    if axis is not None and not axis[0]:
        findings.append(_finding(
            record, "ext_retrieval_irrelevant", "retrieval", True,
            f"{axis[1]} - 검색 결과가 질문과 무관"))

    # 검색축이 실측되고 실패했으면(axis[0] is False), 그 여파로 낮아진 답변축
    # 지표(rel·corr)를 독립된 생성/데이터 문제로 확정하지 않는다 - 환각과 같은 규칙.
    retrieval_failed = axis is not None and not axis[0]

    if rel is not None and rel < EXT_REL_LOW:
        reason = f"answer relevancy {rel:.2f} - 답이 질문을 비껴감"
        if retrieval_failed:
            reason += f" ({axis[1]} - 검색 실패의 여파일 수 있음)"
        findings.append(_finding(
            record, "ext_answer_off_topic", "generation", not retrieval_failed, reason))

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

    # rel 게이트: 동문서답(rel 낮음)이면 correctness 가 낮은 건 그 결과이지, 근거
    # 자체(코퍼스)가 틀렸다는 뜻이 아니다 - 환각(244행)과 같은 게이트.
    if (corr is not None and corr < EXT_CORRECTNESS_LOW
            and faith is not None and faith >= EXT_FAITH_LOW
            and (rel is None or rel >= EXT_REL_LOW)):
        reason = (f"correctness {corr:.2f}인데 faithfulness {faith:.2f} - "
                  "답은 근거에 충실하나 근거 자체가 틀렸거나 부족(코퍼스/검색 문제)")
        if retrieval_failed:
            reason += f" ({axis[1]} - 검색 실패의 여파일 수 있음)"
        findings.append(_finding(
            record, "ext_grounded_but_wrong", "data", not retrieval_failed, reason))

    return findings


def apply_ext_labels(records: list[EvalRecord]) -> None:
    """리플레이 레코드 전체에 ext_ 소견을 채운다(record.findings). run_replay 전용."""
    for record in records:
        record.findings = diagnose_replay_record(record)
