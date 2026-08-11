"""
tools/score_ragec.py
우리 진단을 RAGEC 사람 라벨과 대조해 정확도를 낸다 (로드맵 8-d).

대조 근거는 `docs/ragec_label_mapping.md` 다. 그 문서가 정본이고 여기 표는 그 구현이다 —
둘이 어긋나면 문서를 고치고 여기를 맞춘다.

## 채점을 '포함' 으로 하는 이유

RAGEC 은 **질의당 라벨 1개**다(377 query_id / 377행, 중복 0). 단계 분포가 생성 이전에 75%
쏠려 있어, 검색이 실패하면 생성은 보지도 않는 **최초 실패 단계** 정책이다. 사람이 손으로 한
`k†` 다.

우리는 반대다 — 생성 라벨이 오라클 트랙을 보므로 "골드를 통째로 쥐여줬는데도 틀렸다" 가
검색 실패와 **독립적으로** 성립한다(로드맵이 `k†` 를 명시적으로 거부한 이유).

그래서 우리가 `retrieval_missing_gold` + `generation_hallucination` 을 내고 RAGEC 이 `E4` 만
적어 뒀다면 **우리가 틀린 게 아니라 더 말한 것**이다. top-1 비교는 이 정책 차이를 통째로
오차로 집계한다. 포함(그들 라벨이 우리 findings 안에 있나)으로 재고, 단계 정확도를 병기한다.

## 쓰는 법

    # 1) 파이프라인을 RAGEC 코퍼스로 돌린 뒤 findings 를 덤프한다
    #    (report 객체에서 뽑는 헬퍼가 이 파일에 있다 — findings_from_report)
    # 2) 채점
    python tools/score_ragec.py \\
        --findings output/ragec_findings.jsonl \\
        --key data/ragec_answer_key.jsonl

`--findings` 형식 (JSONL):

    {"qa_id": "2135", "labels": ["retrieval_missing_gold", "generation_hallucination"]}
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

# ── 대조표 (docs/ragec_label_mapping.md 의 구현) ──────────────────
#
# 값이 여러 개인 것은 우리가 그 실패를 **원인별로 쪼개 처방을 붙였기** 때문이다. 포함
# 채점에서는 그중 하나만 맞아도 통과다. 빈 집합은 대응 라벨이 없다는 뜻이고, 그 카테고리는
# 정확도 계산에서 빠진다(맞출 수단이 없는 것을 오답으로 세면 수치가 거짓이 된다).
RAGEC_TO_OURS: dict[str, set[str]] = {
    # Chunking
    "E1 Overchunking": {"chunking_overchunking", "retrieval_incomplete_enumeration"},
    "E2 Underchunking": {"chunking_underchunking"},
    "E3 Context Mismatch": {"chunking_context_mismatch"},
    # Retrieval
    "E4 Missed Retrieval": {"retrieval_missing_gold"},
    "E5 Low Relevance": {"retrieval_low_rank", "retrieval_semantic_mismatch"},
    "E6 Semantic Drift": {"retrieval_semantic_mismatch"},
    # Reranking
    "E7 Low Recall": {"retrieval_rerank_candidate_miss", "retrieval_reranker_demotion",
                      "retrieval_reranker_ineffective"},
    "E8 Low Precision": {"reranker_low_precision"},
    # Generation
    "E9 Abstention Failure": {"generation_abstention_failure"},
    "E10 Fabricated Content": {"generation_hallucination"},
    "E11 Parametric Overreliance": {"generation_parametric_overreliance"},
    "E12 Incomplete Answer": {"generation_partial_answer"},
    "E13 Misinterpretation": {"generation_misinterpretation"},
    # E14 는 우리가 별도 라벨을 두지 않기로 했다 — 처방이 restate_question 으로
    # generation_misinterpretation 과 같아서다(로드맵 3번).
    "E14 Contextual Misalignment": {"generation_misinterpretation"},
    # E15 는 도입이 결정됐으나 라벨이 아직 없다. 신설되면 여기에 적는다.
    "E15 Chronological Inconsistency": set(),
    "E16 Numerical Error": {"generation_numerical_error"},
}

# 단계 대조. RAGEC 4단계 중 셋(Chunking·Retrieval·Reranking)이 우리 A그룹 안에 있어,
# 우리 그룹(A/B/C/D)으로 재면 A 정확도가 거저 높아진다. 그래서 **우리 라벨 이름에서
# RAGEC 단계를 직접 유도**해 같은 해상도로 맞춘다.
def stage_of(label: str) -> str | None:
    """우리 라벨 → RAGEC 단계. 대응 개념이 없으면 None(C그룹·D그룹)."""
    if label.startswith("chunking_"):
        return "Chunking"
    if label.startswith("reranker_") or "rerank" in label:
        return "Reranking"
    if label.startswith("retrieval_"):
        return "Retrieval"
    if label.startswith("generation_"):
        return "Generation"
    return None


# 우리 진단이 "평가셋이 틀렸다" 고 말하는 라벨. RAGEC 은 사람이 검수한 정답지라 이런
# 사례가 없지만, DragonBall 정답지에도 오류가 있을 수 있어 두 가능성이 갈리지 않는다.
#   · 우리 진단의 오탐
#   · 실제로 DragonBall 정답이 틀림
# 정확도에 섞지 않고 따로 세어 사람이 표본을 확인하게 한다.
GOLD_ERROR_LABELS = {"bad_gold_answer", "bad_gold_chunk"}


def findings_from_report(report) -> list[dict]:
    """DiagnosticReport → 채점기 입력(JSONL 행 목록).

    probe_id 가 `probe_qa_<qa_id>` 라 접두를 떼어 qa_id 로 맞춘다(korquad 로더 규약).
    """
    by_probe: dict[str, set[str]] = collections.defaultdict(set)
    for finding in getattr(report, "findings", []) or []:
        if not finding.label:
            continue
        for probe_id in finding.affected_probes:
            by_probe[probe_id].add(finding.label)
    rows = []
    for probe_id, labels in by_probe.items():
        qa_id = probe_id[len("probe_qa_"):] if probe_id.startswith("probe_qa_") else probe_id
        rows.append({"qa_id": qa_id, "labels": sorted(labels)})
    return rows


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def score(findings_rows: list[dict], key_rows: list[dict]) -> dict:
    ours = {str(r["qa_id"]): set(r.get("labels") or []) for r in findings_rows}

    per_category: dict[str, list[bool]] = collections.defaultdict(list)
    stage_hit = stage_total = 0
    no_diagnosis = gold_error = unmappable = 0

    for key in key_rows:
        qa_id = str(key["qa_id"])
        category = key["ragec_category"].strip()
        expected = RAGEC_TO_OURS.get(category)
        got = ours.get(qa_id, set())

        if got & GOLD_ERROR_LABELS:
            # 평가셋 결함 주장 — 정확도에 섞지 않는다.
            gold_error += 1
            continue
        if expected is None:
            unmappable += 1          # 대조표에 없는 카테고리(데이터가 바뀐 것)
            continue
        if not expected:
            unmappable += 1          # 대응 라벨이 아직 없다(E15)
            continue
        if not got:
            no_diagnosis += 1        # 우리가 아무 라벨도 안 냈다 = 실패로 센다
            per_category[category].append(False)
            continue

        per_category[category].append(bool(expected & got))

        # 단계는 우리가 낸 라벨 중 **하나라도** 그 단계면 맞은 것으로 본다(포함과 같은 취지).
        stages = {stage_of(label) for label in got} - {None}
        if stages:
            stage_total += 1
            if key["ragec_stage"].strip() in stages:
                stage_hit += 1

    scored = [hit for hits in per_category.values() for hit in hits]
    return {
        "per_category": {k: (sum(v), len(v)) for k, v in sorted(per_category.items())},
        "total_hit": sum(scored),
        "total": len(scored),
        "stage_hit": stage_hit,
        "stage_total": stage_total,
        "no_diagnosis": no_diagnosis,
        "gold_error": gold_error,
        "unmappable": unmappable,
    }


def format_report(result: dict) -> str:
    total, hit = result["total"], result["total_hit"]
    lines = [
        "=" * 62,
        "  RAGEC 대조 — 진단 정확도 (포함 채점)",
        "=" * 62,
    ]
    if not total:
        lines.append("  채점 대상이 없습니다.")
        return "\n".join(lines)

    lines.append(f"  라벨 포함 정확도  {hit}/{total}  ({hit / total * 100:.1f}%)")
    if result["stage_total"]:
        sh, st = result["stage_hit"], result["stage_total"]
        lines.append(f"  단계 정확도       {sh}/{st}  ({sh / st * 100:.1f}%)")
    lines.append("")
    lines.append(f"  {'카테고리':<32}{'맞음':>8}{'전체':>8}{'정확도':>9}")
    for category, (ok, n) in result["per_category"].items():
        lines.append(f"  {category:<32}{ok:>8}{n:>8}{ok / n * 100:>8.0f}%")

    lines.append("")
    if result["no_diagnosis"]:
        lines.append(f"  · 우리가 아무 라벨도 안 낸 probe   {result['no_diagnosis']}건 (오답으로 셈)")
    if result["gold_error"]:
        lines.append(f"  · bad_gold_* 를 낸 probe          {result['gold_error']}건 "
                     f"(정확도에서 제외 — 사람이 표본 확인 필요)")
    if result["unmappable"]:
        lines.append(f"  · 대응 라벨이 없는 카테고리        {result['unmappable']}건 (제외)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="우리 진단 ↔ RAGEC 사람 라벨 대조")
    ap.add_argument("--findings", required=True,
                    help='JSONL: {"qa_id": "...", "labels": [...]}')
    ap.add_argument("--key", default="data/ragec_answer_key.jsonl")
    args = ap.parse_args()

    for path in (args.findings, args.key):
        if not pathlib.Path(path).exists():
            print(f"[오류] 파일이 없습니다: {path}", file=sys.stderr)
            return 1

    result = score(_read_jsonl(args.findings), _read_jsonl(args.key))
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
