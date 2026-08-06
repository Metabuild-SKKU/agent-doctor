"""
agents/eval/log_intake.py
외부 RAG 실행 로그(JSONL) 적재 + 진단 가능 수준 판정

"다른 RAG 실행 로그 연결" 조사(docs/external_rag_log_intake.md)의 1차 구현물.
다른 팀이 만든 RAG가 남긴 실행 로그를 표준 스키마로 읽어들여, 그 로그만으로
어느 깊이의 진단이 가능한지 판정한다. Eval 본체 연결(로그 리플레이 모드)은
후속 PR - 이 모듈은 적재·검증·판정까지만 담당하고 core/agents 어디에도
의존하지 않는다(순수 stdlib).

스키마 v0 (한 줄 = 요청 1건, JSON Lines):
    {"question": str,                                   # 필수
     "answer":   str,                                   # 필수
     "contexts": [str | {"text": str, "chunk_id"?, "score"?, "rank"?, "source_doc"?}],
                                                        # 권장 - 있어야 검색/생성 원인 분리
     "config":   {"top_k"?, "chunk_size"?, ...},        # 선택 - 있어야 처방에 현재값 반영
     "feedback": str,  "latency_ms": int,  "timestamp": str}   # 선택

CLI: python -m agents.eval.log_intake <log.jsonl>
     → 적재 결과와 "이 로그로 가능한 진단 수준"을 요약 출력한다.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


# ── 스키마 정의 ───────────────────────────────────────────────────

REQUIRED_KEYS = ("question", "answer")

# assess_capability 가 판정하는 수준. 숫자가 클수록 깊은 진단이 가능하다.
TIER_NONE = "none"          # 유효 레코드 없음 → 평가 불가
TIER_QA_ONLY = "qa_only"    # 질문+답변만 → answer relevancy(동문서답)만
TIER_TRIAD = "triad"        # +컨텍스트 → faithfulness/relevancy(환각·동문서답) 실측 가능


@dataclass
class ExternalLogRecord:
    """외부 로그 한 줄(요청 1건)의 정규화 결과.

    contexts 는 문자열/딕셔너리 혼용 입력을 {"text", "chunk_id", "score",
    "rank", "source_doc"} 로 통일한다(없는 키는 None). raw 는 원본 보존용 -
    후속 리플레이 모드가 스키마 밖 필드(프롬프트 등)를 쓸 수 있게 남긴다."""
    question: str
    answer: str
    contexts: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    feedback: Optional[str] = None
    latency_ms: Optional[int] = None
    timestamp: Optional[str] = None
    raw: dict = field(default_factory=dict)


# ── 적재 ─────────────────────────────────────────────────────────

def _normalize_context(entry: Any) -> Optional[dict]:
    """contexts 항목 1개를 표준 딕셔너리로. 문자열이면 text 만 채운다.
    text 가 비면 진단 재료가 못 되므로 버린다(None)."""
    if isinstance(entry, str):
        text = entry.strip()
        if not text:
            return None
        return {"text": text, "chunk_id": None, "score": None,
                "rank": None, "source_doc": None}
    if isinstance(entry, dict):
        text = str(entry.get("text") or "").strip()
        if not text:
            return None
        score = entry.get("score")
        return {
            "text": text,
            "chunk_id": entry.get("chunk_id"),
            "score": float(score) if isinstance(score, (int, float)) else None,
            "rank": entry.get("rank"),
            "source_doc": entry.get("source_doc"),
        }
    return None


def parse_record(obj: Any) -> ExternalLogRecord:
    """JSON 오브젝트 1개 → ExternalLogRecord. 스키마 위반은 ValueError 로 알린다."""
    if not isinstance(obj, dict):
        raise ValueError("JSON 오브젝트가 아님")
    missing = [k for k in REQUIRED_KEYS if not str(obj.get(k) or "").strip()]
    if missing:
        raise ValueError(f"필수 필드 누락/공백: {', '.join(missing)}")

    raw_contexts = obj.get("contexts")
    contexts: list[dict] = []
    if isinstance(raw_contexts, list):
        contexts = [c for c in (_normalize_context(e) for e in raw_contexts) if c]

    config = obj.get("config")
    latency = obj.get("latency_ms")
    feedback = obj.get("feedback")
    return ExternalLogRecord(
        question=str(obj["question"]).strip(),
        answer=str(obj["answer"]).strip(),
        contexts=contexts,
        config=config if isinstance(config, dict) else {},
        feedback=str(feedback) if feedback not in (None, "") else None,
        latency_ms=int(latency) if isinstance(latency, (int, float)) else None,
        timestamp=str(obj["timestamp"]) if obj.get("timestamp") else None,
        raw=obj,
    )


def load_external_log(path: str) -> tuple[list[ExternalLogRecord], list[str]]:
    """JSONL 파일을 적재한다.

    반환: (정상 레코드 리스트, 줄 단위 오류 메시지 리스트).
    깨진 줄은 건너뛰고 오류로 집계한다 - 실사용 로그는 일부 줄이 깨져 있는 게
    보통이라, 한 줄 때문에 전체 적재를 실패시키지 않는다(기존 폴백 철학).
    파일 자체가 없으면 FileNotFoundError 를 그대로 낸다(호출자가 경로 실수를
    빈 로그로 오인하지 않게)."""
    records: list[ExternalLogRecord] = []
    errors: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(parse_record(json.loads(line)))
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{lineno}행: {exc}")
    return records, errors


# ── 진단 가능 수준 판정 ───────────────────────────────────────────

def assess_capability(records: list[ExternalLogRecord]) -> dict:
    """이 로그만으로 가능한 진단 수준을 판정한다.

    tier 기준: 컨텍스트가 절반 이상 레코드에 있어야 TRIAD - 일부만 있는
    로그로 전체를 triad 취급하면 검색/생성 원인 분리가 표본 편향된다.
    config/feedback/score 는 tier 와 별개 가산 정보로 함께 보고한다."""
    n = len(records)
    with_contexts = sum(1 for r in records if r.contexts)
    with_scores = sum(1 for r in records if any(c["score"] is not None for c in r.contexts))
    with_config = sum(1 for r in records if r.config)
    with_feedback = sum(1 for r in records if r.feedback is not None)

    if n == 0:
        tier = TIER_NONE
    elif with_contexts * 2 >= n:
        tier = TIER_TRIAD
    else:
        tier = TIER_QA_ONLY

    notes: list[str] = []
    if tier == TIER_NONE:
        notes.append("유효 레코드가 없어 평가 불가")
    if tier == TIER_QA_ONLY:
        notes.append("검색 컨텍스트가 없거나 부족 - answer relevancy(동문서답)만 평가 가능, "
                     "faithfulness(환각)·검색/생성 원인 분리 불가")
    if tier == TIER_TRIAD:
        notes.append("질문·컨텍스트·답변 확보 - faithfulness(환각)·answer relevancy(동문서답) 실측 가능. "
                     "검색축 지표(context precision/recall)와 원인 라벨은 정답 텍스트 확보 시 확장")
        if with_scores == 0:
            notes.append("유사도 score 없음 - 랭킹 문제 vs 커버리지 문제 세부 진단은 제한")
    if with_config == 0:
        notes.append("config 없음 - 증상 진단은 되지만 처방(예: chunk_size 512→768)에 현재값을 못 쓴다")
    if with_feedback:
        notes.append(f"사용자 피드백 {with_feedback}건 - 불만족 케이스 우선 진단 표본으로 활용 가능")

    return {
        "records": n,
        "with_contexts": with_contexts,
        "with_scores": with_scores,
        "with_config": with_config,
        "with_feedback": with_feedback,
        "tier": tier,
        "notes": notes,
    }


# ── CLI ──────────────────────────────────────────────────────────

def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("사용법: python -m agents.eval.log_intake <log.jsonl>")
        return 2
    records, errors = load_external_log(argv[0])
    cap = assess_capability(records)
    print(f"적재: 정상 {cap['records']}건 / 오류 {len(errors)}건")
    for err in errors[:10]:
        print(f"  ! {err}")
    if len(errors) > 10:
        print(f"  ! ... 외 {len(errors) - 10}건")
    print(f"컨텍스트 포함 {cap['with_contexts']}건 (score 포함 {cap['with_scores']}건) · "
          f"config {cap['with_config']}건 · feedback {cap['with_feedback']}건")
    print(f"진단 가능 수준: {cap['tier']}")
    for note in cap["notes"]:
        print(f"  - {note}")
    return 0 if cap["tier"] != TIER_NONE else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
