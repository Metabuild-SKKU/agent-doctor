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
        --findings output/ragec/findings.jsonl \\
        --key data/ragec_answer_key.jsonl

probe 별 대조(질문·답변·정답 + RAGEC 정답 라벨 + 우리 진단)가 **기본으로** 나온다.
총계만 보려면 `--no-detail`.

`--findings` 형식 (JSONL). qa_id/labels/failed 가 채점에 쓰이고, 나머지는 대조 출력용이라
없으면 그 줄만 비어 나온다:

    {"qa_id": "2135", "labels": ["retrieval_missing_gold"], "failed": true,
     "question": "...", "answer": "...", "gold_answer": "..."}
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import textwrap

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
    #
    # E4 는 "검색 결과에 정답 문서가 없다" 는 **롤업 수준**의 서술이다. 우리 사전도
    # retrieval_missing_gold 를 "어느 단계에서 놓쳤는지는 모른다" 로 정의하고, 나머지 A
    # 라벨들이 같은 현상을 원인별로 쪼갠 것이다. 우리가 원인을 더 짚었으면 틀린 게 아니라
    # 더 말한 것이라 전부 인정한다(포함 채점의 취지).
    #
    # 처음엔 retrieval_missing_gold 하나로 좁혔다가 10건 스모크에서 E4 3건을 전부 놓쳤다
    # (우리는 retrieval_low_rank 를 냈다). 대조표 참고.
    "E4 Missed Retrieval": {
        "retrieval_missing_gold", "retrieval_low_rank",
        "retrieval_rerank_candidate_miss", "retrieval_reranker_demotion",
        "retrieval_reranker_ineffective",
        "retrieval_lexical_mismatch", "retrieval_semantic_mismatch",
    },
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


def _qa_id(probe_id: str) -> str:
    """`probe_qa_<qa_id>` → `<qa_id>` (korquad 로더 규약)."""
    prefix = "probe_qa_"
    return probe_id[len(prefix):] if probe_id.startswith(prefix) else probe_id


def findings_from_report(report, probes: list | None = None) -> list[dict]:
    """DiagnosticReport → 채점기 입력(JSONL 행 목록).

    **성공한 probe 도 실어야 한다.** RAGEC 377건은 *그들* RAG 시스템이 실패한 질문이고,
    우리 파이프라인은 같은 질문에서 성공할 수 있다(검색기·생성 모델이 다르다). 실패한
    probe 만 실으면 우리가 잘한 것이 채점기에서 '미진단' 으로 보여 오답이 된다.

    probes 를 주면 그 목록 전체를 싣고 findings 가 없는 것은 `failed=False` 로 표시한다.
    안 주면 findings 가 있는 probe 만 실린다(구버전 호환 — 그 경우 채점기가 '우리는 성공'
    과 '실패했는데 원인을 못 짚음' 을 구분하지 못한다).

    probes 는 `Probe` 객체 목록이거나 probe_id 문자열 목록이다. 객체를 주면 질문·정답
    원문까지 실어서 대조표(format_detail)가 **왜 어긋났는지**를 같이 보여준다 — 라벨만
    보면 대조표를 고쳐야 할지 진단을 고쳐야 할지 갈리지 않는다.

    [한계] 생성 답변은 실패 probe 만 실린다. EvalRecord 는 Eval 이 끝나면 사라지고,
    report 가 답변 원문을 남기는 곳은 `failed_questions` 뿐이라서다(성공 probe 는 애초에
    '실패 목록' 이 아니라 빠진다). 성공 probe 는 채점 대상도 아니므로 질문·정답만으로
    충분하다고 보고 여기서 넓히지 않는다.
    """
    by_probe: dict[str, set[str]] = collections.defaultdict(set)
    for finding in getattr(report, "findings", []) or []:
        if not finding.label:
            continue
        for probe_id in finding.affected_probes:
            by_probe[probe_id].add(finding.label)

    # 실패 판정은 report 가 소유한다 — findings 유무로 추론하지 않는다. 골드 오류 probe 는
    # findings 는 있지만 failed_questions 에서 빠지므로(평가셋 결함이라 '실패한 검증 질문'
    # 이 아니다) 둘이 다르다.
    answers = {
        row.get("probe_id", ""): row
        for row in (getattr(report, "failed_questions", []) or [])
    }

    if probes is None:
        entries: list[tuple[str, object]] = [(pid, None) for pid in sorted(by_probe)]
    else:
        entries = [
            (p, None) if isinstance(p, str) else (getattr(p, "probe_id", ""), p)
            for p in probes
        ]

    rows = []
    for probe_id, probe in entries:
        answer_row = answers.get(probe_id, {})
        row = {
            "qa_id": _qa_id(probe_id),
            "labels": sorted(by_probe.get(probe_id, ())),
            "failed": probe_id in answers or bool(by_probe.get(probe_id)),
        }
        question = getattr(probe, "question", "") or answer_row.get("question", "")
        gold = (
            getattr(probe, "ground_truth", None)
            or answer_row.get("expected_answer", "")
        )
        answer = answer_row.get("actual_answer", "")
        if question:
            row["question"] = question
        if gold:
            row["gold_answer"] = gold
        if answer:
            row["answer"] = answer
        rows.append(row)
    return rows


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def score(findings_rows: list[dict], key_rows: list[dict]) -> dict:
    ours = {str(r["qa_id"]): set(r.get("labels") or []) for r in findings_rows}
    # "failed" 가 없는 덤프(구버전)는 라벨 유무로 추론한다 — 그 경우 '우리는 성공' 과
    # '실패했는데 원인을 못 짚음' 이 구분되지 않으므로 리포트가 그렇다고 밝힌다.
    has_status = any("failed" in r for r in findings_rows)
    failed = {
        str(r["qa_id"]): bool(r.get("failed", bool(r.get("labels"))))
        for r in findings_rows
    }

    per_category: dict[str, list[bool]] = collections.defaultdict(list)
    stage_hit = stage_total = 0
    no_diagnosis = gold_error = unmappable = we_passed = not_run = 0

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
        if qa_id not in failed:
            not_run += 1             # 덤프에 없다 = 이 probe 를 안 돌렸다
            continue
        if not failed[qa_id]:
            # **우리 파이프라인은 이 질문에 성공했다.** RAGEC 377건은 *그들* 시스템이
            # 실패한 질문이라, 검색기·생성 모델이 다른 우리가 성공하는 건 정상이다.
            # 이걸 오답으로 세면 정확도가 진단 품질이 아니라 "얼마나 그들과 비슷하게
            # 실패하나" 를 재게 된다.
            we_passed += 1
            continue
        if not got:
            no_diagnosis += 1        # 실패했는데 원인을 못 짚었다 = 진짜 미진단
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
        "we_passed": we_passed,
        "not_run": not_run,
        "has_status": has_status,
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
    if result.get("we_passed"):
        lines.append(f"  · 우리 파이프라인이 성공한 질문    {result['we_passed']}건 "
                     f"(진단할 게 없어 제외 — 그들 시스템만 실패한 건이다)")
    if result["no_diagnosis"]:
        lines.append(f"  · 실패했는데 원인을 못 짚음        {result['no_diagnosis']}건 (오답으로 셈)")
    if result.get("not_run"):
        lines.append(f"  · 덤프에 없는 probe               {result['not_run']}건 (제외)")
    if not result.get("has_status", True):
        lines.append("  ⚠ 덤프에 failed 필드가 없어 '우리는 성공' 과 '원인을 못 짚음' 이")
        lines.append("    구분되지 않습니다 — findings_from_report(report, probe_ids) 로 다시 덤프하세요.")
    if result["gold_error"]:
        lines.append(f"  · bad_gold_* 를 낸 probe          {result['gold_error']}건 "
                     f"(정확도에서 제외 — 사람이 표본 확인 필요)")
    if result["unmappable"]:
        lines.append(f"  · 대응 라벨이 없는 카테고리        {result['unmappable']}건 (제외)")
    return "\n".join(lines)


_MARK_LEGEND = "  O 맞음 · X 틀림 · 성공=우리가 성공(제외) · - 대응 라벨 없음(제외)"


def _verdict(row: dict, category: str) -> str:
    """이 probe 의 채점 판정 한 글자. score() 의 분기와 같은 순서로 본다."""
    labels = set(row.get("labels") or [])
    if labels & GOLD_ERROR_LABELS:
        return "gold"         # 평가셋 결함 주장 — 정확도에서 제외
    expected = RAGEC_TO_OURS.get(category)
    if not expected:
        return "-"            # 대응 라벨 없음(E15) 또는 대조표에 없는 카테고리 — 제외
    if not row.get("failed", bool(labels)):
        return "성공"          # 우리 파이프라인이 이 질문에 성공 — 채점 제외
    return "O" if labels & expected else "X"


def _wrap(label: str, text: str, width: int = 88) -> list[str]:
    """`  Q  본문…` 꼴로 접어 쓴다. 이어지는 줄은 본문 열에 맞춰 들여쓴다."""
    body = " ".join((text or "").split()) or "(없음)"
    head = f"  {label:<4}"
    indent = " " * len(head)
    wrapped = textwrap.wrap(body, width=width) or [""]
    return [head + wrapped[0]] + [indent + line for line in wrapped[1:]]


def format_detail(findings_rows: list[dict], key_rows: list[dict]) -> str:
    """probe 별 대조 — 질문·답변·정답(QAR)과 'RAGEC 정답 라벨 ↔ 우리 진단' 을 한자리에.

    정확도 숫자만 보면 **어디서 어긋났는지** 를 알 수 없다. 대조표를 고칠 근거도, 진단을
    고칠 근거도 여기서 나온다 — 실측 두 건이 다 그랬다:
      · E4 3건에 우리가 전부 retrieval_low_rank 를 낸 걸 보고 E4 대응이 너무 좁다는 걸 알았다
      · 영어 질문에 한국어 답변이 붙은 걸 보고 f1=0 의 원인이 진단이 아니라 프롬프트임을 알았다
    두 번째는 **라벨만 봤으면 못 찾는다** — 답변 원문이 붙어 있어야 보인다.

    블록(probe 하나씩)을 먼저 내고, 마지막에 한눈에 보는 압축 표를 붙인다.
    """
    ours = {str(r["qa_id"]): r for r in findings_rows}
    lines = ["", "=" * 92, "  probe 별 대조 (질문·답변·정답 ↔ RAGEC 정답 라벨 ↔ 우리 진단)", "=" * 92]

    shown = 0
    for key in key_rows:
        qa_id = str(key["qa_id"])
        row = ours.get(qa_id)
        if row is None:
            continue          # 덤프에 없다 = 안 돌린 probe
        shown += 1
        category = key["ragec_category"].strip()
        mark = _verdict(row, category)
        stage = key.get("ragec_stage", "").strip()
        qtype = key.get("query_type", "").strip()

        lines.append("")
        lines.append(f"── qa_id {qa_id}  [{mark}]  {category}"
                     f"{f' ({stage})' if stage else ''}"
                     f"{f' · {qtype}' if qtype else ''}")
        lines += _wrap("Q", row.get("question", ""))
        lines += _wrap("A", row.get("answer", "")
                       or ("(성공 — 답변 원문은 실패 probe 만 보존됩니다)"
                           if not row.get("failed") else ""))
        lines += _wrap("R", row.get("gold_answer", ""))
        lines += _wrap("진단", ", ".join(row.get("labels") or []) or "(라벨 없음)")

    if not shown:
        lines.append("")
        lines.append("  덤프와 정답지에 공통으로 있는 probe 가 없습니다.")
        return "\n".join(lines)

    lines += ["", "=" * 92, "  요약 표", "=" * 92,
              f"  {'qa_id':<8}{'판정':<6}{'RAGEC 정답':<30}우리 진단"]
    for key in key_rows:
        qa_id = str(key["qa_id"])
        row = ours.get(qa_id)
        if row is None:
            continue
        lines.append(f"  {qa_id:<8}{_verdict(row, key['ragec_category'].strip()):<6}"
                     f"{key['ragec_category']:<30}"
                     f"{', '.join(row.get('labels') or []) or '(라벨 없음)'}")
    lines.append("")
    lines.append(_MARK_LEGEND + " · gold=정답지 오류 주장(제외)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="우리 진단 ↔ RAGEC 사람 라벨 대조")
    ap.add_argument("--findings", required=True,
                    help='JSONL: {"qa_id": "...", "labels": [...], "failed": bool}')
    ap.add_argument("--key", default="data/ragec_answer_key.jsonl")
    # 기본 ON. 정확도 숫자만으로는 대조표를 고칠지 진단을 고칠지 못 정한다 — 실제로 두 번
    # 다 probe 별 원문을 보고서야 원인을 찾았다(format_detail 참고). 끄는 쪽을 플래그로 둔다.
    ap.add_argument("--detail", action="store_true", default=True,
                    help="probe 별 대조(기본 ON)")
    ap.add_argument("--no-detail", dest="detail", action="store_false",
                    help="정확도 표만 출력")
    args = ap.parse_args()

    for path in (args.findings, args.key):
        if not pathlib.Path(path).exists():
            print(f"[오류] 파일이 없습니다: {path}", file=sys.stderr)
            return 1

    findings_rows = _read_jsonl(args.findings)
    key_rows = _read_jsonl(args.key)
    # probe 별 대조를 먼저, 정확도 표를 마지막에 — 스크롤이 끝난 자리에 총계가 남는다.
    if args.detail:
        print(format_detail(findings_rows, key_rows))
        print()
    print(format_report(score(findings_rows, key_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
