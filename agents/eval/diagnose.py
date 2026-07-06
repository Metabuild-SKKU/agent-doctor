"""
agents/eval/diagnose.py
STEP4: 원인 판정 (Finding 생성)

설계 문서 'STEP 4: 원인 판정' 구현.
브랜치(STEP3-1)와 지표(STEP3-1/3-2)를 근거로 실패 원인을 Finding 으로 만든다.
각 Finding 은 Optimize Agent 가 소비하므로 prescription(처방)을 함께 담는다(설계 §8).

Finding.type 은 공유 스키마의 제한된 집합만 사용한다:
    "gap" | "contradiction" | "duplicate" | "staleness"
    | "retrieval_failure" | "generation_failure"
설계의 세분화 라벨(retrieval_low_rank, generation_abstention_failure 등)은
Finding.metadata["label"] 에 담아 확장성을 확보한다(설계 STEP4 '모듈화').

[구현 포인트]
    - 설계의 비용별 4-tier 판정(규칙조회 → 코퍼스탐색 → LLM호출 → 재실행)을
      단계별 모듈로 확장. 지금은 tier1(규칙) 수준의 브랜치→라벨 매핑만 구현.
"""
from __future__ import annotations

from core.schema import Finding
from agents.eval.types import (
    Branch, EvalRecord,
    RAGAS_FAITHFULNESS_MIN, RAGAS_CONTEXT_PRECISION_MIN,
    RAGAS_CONTEXT_RECALL_MIN, RAGAS_RESPONSE_RELEVANCY_MIN,
)


def _finding(record: EvalRecord, ftype: str, severity: str, label: str,
             desc: str, prescription: str) -> Finding:
    """공통 Finding 생성 헬퍼."""
    probe = record.probe
    return Finding(
        finding_id=f"{probe.probe_id}:{label}",
        type=ftype,
        severity=severity,
        description=desc,
        affected_chunks=list(probe.gold_chunk_ids),
        affected_probes=[probe.probe_id],
        prescription=prescription,
        metadata={"label": label, "branch": record.branch},
    )


# ── 브랜치 → 규칙 기반 Finding 매핑 (tier1) ──────────────────────

# branch: (finding들의 스펙 리스트)  각 스펙 = (type, severity, label, desc, prescription)
_BRANCH_RULES: dict[str, list[tuple]] = {
    Branch.SUCCESS: [],
    Branch.NO_ANSWER_OK: [],

    Branch.RETRIEVAL_FAIL: [
        ("retrieval_failure", "critical", "retrieval_missing_gold",
         "gold 청크가 top-k 검색 결과에 없음(검색 실패). Oracle 은 통과 → 검색이 병목.",
         "chunk_overlap 확대 또는 임베딩 모델 교체(BGE-M3 등), 필요 시 Hybrid/Reranker 도입."),
    ],
    Branch.RETRIEVAL_GEN_FAIL: [
        ("retrieval_failure", "critical", "retrieval_missing_gold",
         "검색 실패 + Oracle 도 실패. 검색과 생성 모두 결함 가능.",
         "검색: 임베딩/Hybrid 개선. 우선 검색 병목부터 해소."),
        ("generation_failure", "warning", "generation_defect",
         "gold context 를 줘도(Oracle) 답을 못 만듦 → 순수 생성 결함.",
         "생성 프롬프트 개선·temperature 낮추기, 생성 모델 점검."),
    ],
    Branch.RETRIEVAL_PARTIAL: [
        ("retrieval_failure", "warning", "retrieval_incomplete_enumeration",
         "멀티홉/나열형에서 gold 청크 일부만 검색됨(부분 실패).",
         "chunk_overlap 확대·재귀적 문장 경계 청킹으로 chunking_context_mismatch 완화, top_k 상향."),
    ],
    Branch.RETRIEVAL_PARTIAL_GEN_FAIL: [
        ("retrieval_failure", "warning", "retrieval_incomplete_enumeration",
         "gold 청크 부분 검색 + 답변 실패.",
         "청킹/overlap 조정 + top_k 상향."),
        ("generation_failure", "warning", "generation_defect",
         "부분 검색 상황에서 Oracle 도 실패 → 생성 결함도 동반.",
         "생성 프롬프트 개선."),
    ],
    Branch.AMBIGUOUS_CONTEXT: [
        ("generation_failure", "warning", "context_noise_interference",
         "gold 는 검색됐으나 답이 부정확. Oracle 은 통과 → 노이즈 청크 간섭 의심.",
         "Reranker/다양성(MMR) 도입으로 상위 노이즈 제거, top_k 축소 검토."),
    ],
    Branch.AMBIGUOUS_GEN: [
        ("generation_failure", "critical", "generation_misinterpretation",
         "gold 검색·Oracle 모두에서 답이 부정확 → 질문 조건 오독 등 생성 결함.",
         "답변 전 질문 재진술 강제, 프롬프트에 조건 재확인 지시 추가."),
    ],
    Branch.NO_ANSWER_VIOLATION: [
        ("generation_failure", "critical", "generation_abstention_failure",
         "답할 수 없는(무응답) 질문인데 기권하지 않고 답을 생성함(할루시네이션 위험).",
         "기권 기준 프롬프트 강화: 근거 없으면 '모른다'고 답하도록."),
    ],
}


def diagnose(record: EvalRecord) -> list[Finding]:
    """
    한 record 의 브랜치와 RAGAS 점수로 Finding 리스트를 생성.
    """
    findings: list[Finding] = []

    # 1) 브랜치 규칙 (tier1)
    for spec in _BRANCH_RULES.get(record.branch, []):
        ftype, severity, label, desc, presc = spec
        findings.append(_finding(record, ftype, severity, label, desc, presc))

    # 2) RAGAS 점수 기반 보강 (STEP3-2 활성 시에만 채워져 있음)
    findings.extend(_from_ragas(record))

    # 3) 커스텀 AspectCritic → staleness/contradiction Finding
    findings.extend(_from_aspect(record))

    return findings


def _from_ragas(record: EvalRecord) -> list[Finding]:
    """RAGAS 4지표 임계값 미달 → 대응 Finding (설계 STEP4 표)."""
    out: list[Finding] = []
    r = record.ragas
    if not r:
        return out

    if _below(r, "faithfulness", RAGAS_FAITHFULNESS_MIN):
        out.append(_finding(record, "generation_failure", "warning", "generation_hallucination",
                            f"Faithfulness {r['faithfulness']:.2f} < {RAGAS_FAITHFULNESS_MIN}: 컨텍스트에 없는 내용 생성.",
                            "그라운딩 강제 프롬프트, temperature 낮추기."))
    if _below(r, "context_precision", RAGAS_CONTEXT_PRECISION_MIN):
        out.append(_finding(record, "retrieval_failure", "warning", "retrieval_low_precision",
                            f"Context Precision {r['context_precision']:.2f} < {RAGAS_CONTEXT_PRECISION_MIN}: 노이즈 청크가 상위.",
                            "Reranker 도입 또는 top_k 축소."))
    if _below(r, "context_recall", RAGAS_CONTEXT_RECALL_MIN):
        out.append(_finding(record, "gap", "warning", "corpus_or_retrieval_gap",
                            f"Context Recall {r['context_recall']:.2f} < {RAGAS_CONTEXT_RECALL_MIN}: 필요한 정보 누락.",
                            "문서 추가 수집 또는 chunk_overlap 확대."))
    if _below(r, "response_relevancy", RAGAS_RESPONSE_RELEVANCY_MIN):
        out.append(_finding(record, "generation_failure", "info", "generation_off_topic",
                            f"Response Relevancy {r['response_relevancy']:.2f} < {RAGAS_RESPONSE_RELEVANCY_MIN}: 동문서답.",
                            "질문 핵심을 다루도록 프롬프트 개선."))
    return out


def _from_aspect(record: EvalRecord) -> list[Finding]:
    """AspectCritic(이진 1=문제 있음) → staleness / contradiction Finding."""
    out: list[Finding] = []
    a = record.aspect
    if a.get("staleness") == 1:
        out.append(_finding(record, "staleness", "info", "staleness",
                            "답변/컨텍스트에 오래된 정보 포함(AspectCritic).",
                            "최신 문서 재수집, 날짜 메타데이터 기준 필터링."))
    if a.get("contradiction") == 1:
        out.append(_finding(record, "contradiction", "warning", "contradiction",
                            "답변이 컨텍스트와 모순(AspectCritic).",
                            "충돌 청크 정리, 근거 인용 강제."))
    return out


def _below(scores: dict, key: str, threshold: float) -> bool:
    return key in scores and scores[key] is not None and scores[key] < threshold
