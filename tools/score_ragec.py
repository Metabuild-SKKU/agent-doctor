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
import unicodedata

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
    # E15 는 main(#132)에서 라벨이 신설돼 대응이 생겼다. 다만 RAGEC 정답지에 E15 가 **0건**
    # 이라(docs/ragec_label_mapping.md) 이 대조로는 검증되지 않는다 — 대응이 있다는 것과
    # 검증된다는 건 다르다.
    "E15 Chronological Inconsistency": {"generation_chronological_error"},
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


# ── 우리 검색이 성공하면 성립하지 않는 카테고리 ──────────────────
#
# RAGEC 라벨은 **그들** 시스템의 실패 원인이다. 검색기·청커·생성 모델이 다른 우리는 같은
# 질문에서 다른 지점에서 넘어질 수 있고, 그때 우리 진단이 그들 라벨과 다른 건 **오진이
# 아니라 시스템이 다른 것**이다. 실측 예(DragonBall 10건):
#
#   qa_id 2205 — RAGEC: E4 Missed Retrieval("검색이 정답 문서를 못 찾았다")
#     우리: recall@k(span)=1.00 · gold청크 2/2 검색 → 답변에 "Bali, Paris, and New York"
#     이 그대로 들어 있는데 "June 20 이라고 명시되진 않았다"며 기권했다.
#     우리 진단 generation_wrongful_abstention 은 **우리 파이프라인 기준으로 정확하다**.
#     그런데 라벨이 다르다는 이유로 오답으로 집계됐다.
#
# 그래서 "gold 가 생성기까지 도달하지 못했다" 고 주장하는 카테고리에 한해, recall 이 그
# 주장을 정면으로 반박하면 채점에서 뺀다.
#
# **단계(stage)가 아니라 카테고리 단위인 이유.** 같은 Reranking 단계라도 E7 Low Recall 은
# 정의상 recall 주장이라 반박되지만, E8 Low Precision 은 "쓰레기가 섞였다" 라서 gold 를 다
# 가져와도 성립한다. Chunking(E1~E3)도 마찬가지다 — gold 조각을 전부 검색해 recall 이 1.0
# 이어도 경계가 잘못 잘린 건 그대로다. 단계로 뭉치면 반박되지 않는 주장까지 빠져나간다.
RECALL_REFUTES = {
    "E4 Missed Retrieval",     # "검색 결과에 정답 문서가 없다"
    "E5 Low Relevance",        # "검색된 문서의 관련성이 낮다"
    "E6 Semantic Drift",       # "의미가 다른 문서를 가져왔다"
    "E7 Low Recall",           # 정의상 recall 주장
}

# 반박으로 인정하는 값. **정확히 1.0** 만 쓴다 — gold 를 빠짐없이 덮었을 때만이다.
# 문턱을 낮추면 "얼마나 낮춰야 하나" 라는 손잡이가 생기고, 그 손잡이로 정확도를 올릴 수
# 있게 된다(실측에서 부분 검색은 0.16·0.58 이라 1.0 과 뚜렷이 갈린다).
RECALL_FULL = 1.0


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
    # **순서를 보존한다.** report.findings 는 diagnose 의 처방 우선순위(확정 우선, 그룹
    # D→A→C→B)로 정렬돼 있고, 그 첫 원소를 두 곳이 '진단 1순위'로 읽는다:
    #   · score_human_labels 의 top-1 정확도 (라벨 남발 감시)
    #   · make_label_sheet 의 층화추출 키
    # set + sorted() 로 담으면 그게 **알파벳 순**이 된다 — generation_* 이 retrieval_* 보다
    # 항상 앞이라 top-1 이 진단 품질이 아니라 사전순을 재게 되고, 층화도 엉뚱한 축으로 쏠린다.
    # (실측: 라벨 2개인 7건이 전부 generation_* 을 1순위로 달고 있었다)
    by_probe: dict[str, list[str]] = collections.defaultdict(list)
    for finding in getattr(report, "findings", []) or []:
        if not finding.label:
            continue
        for probe_id in finding.affected_probes:
            if finding.label not in by_probe[probe_id]:   # 순서 보존 dedup
                by_probe[probe_id].append(finding.label)

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
            "labels": list(by_probe.get(probe_id, ())),   # 진단 순서 그대로(위 주석 참고)
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
        # recall·observations 는 실패 probe 에만 있다(성공 probe 는 failed_questions 에
        # 안 남는다). 채점 대상도 라벨링 대상도 실패 probe 뿐이라 그걸로 충분하다.
        #
        # observations 를 여기서 안 옮기면 라벨 시트가 **지표 없이** 나가고, 사람이 검색 계열
        # 라벨을 구분할 수 없게 된다. 실제로 30건 실측에서 그렇게 나갔다 — report 는 제대로
        # 실었는데 이 줄이 없어 덤프에서 끊겼고, 스모크 테스트를 손으로 만든 덤프로 해서
        # 이 구간이 한 번도 안 돌았다.
        if "recall_at_k" in answer_row:
            row["recall_at_k"] = answer_row["recall_at_k"]
            row["recall_basis"] = answer_row.get("recall_basis", "")
        if answer_row.get("observations"):
            row["observations"] = answer_row["observations"]
        rows.append(row)
    return rows


def _read_jsonl(path: str) -> list[dict]:
    """덤프 리더 **정본**. 다른 도구도 이걸 import 한다 — 세 곳에 복사해 두면 한쪽에만
    개선(BOM 허용·깨진 줄 안내)이 들어갔을 때 같은 파일이 도구에 따라 읽히거나 죽는다."""
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def probe_failed(row: dict) -> bool:
    """이 probe 를 실패로 볼 것인가. **구버전 덤프 호환 추론이 여기 한 곳에만 있다.**

    `failed` 필드가 없던 덤프는 라벨 유무로 추론한다. 이 규칙이 세 도구 네 곳에 복사돼
    있었는데, 하나만 놓치면 도구마다 성공/실패 버킷이 달라져 '우리는 성공' 집계와
    층화 표본이 서로 다른 모집단을 보게 된다."""
    return bool(row.get("failed", bool(row.get("labels"))))


# ── 표 정렬 (한글은 터미널에서 두 칸을 먹는다) ────────────────────
#
# f"{'카테고리':<32}" 는 **글자 수**로 채운다. 한글 4글자는 8칸을 차지하는데 4칸으로 세니
# 그 줄만 4칸씩 밀린다. 헤더에만 한글이 있으면 헤더와 본문이 어긋나 표가 무너진다.

def _width(text: str) -> int:
    """터미널 표시 폭. 동아시아 W/F 문자는 2칸."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    """표시 폭 기준 좌측 정렬. 이미 넘치면 최소 한 칸을 띄워 열이 붙지 않게 한다."""
    return text + " " * max(1, width - _width(text))


def _rpad(text: str, width: int) -> str:
    """표시 폭 기준 우측 정렬."""
    return " " * max(1, width - _width(text)) + text


def score(findings_rows: list[dict], key_rows: list[dict]) -> dict:
    ours = {str(r["qa_id"]): set(r.get("labels") or []) for r in findings_rows}
    # "failed" 가 없는 덤프(구버전)는 라벨 유무로 추론한다 — 그 경우 '우리는 성공' 과
    # '실패했는데 원인을 못 짚음' 이 구분되지 않으므로 리포트가 그렇다고 밝힌다.
    has_status = any("failed" in r for r in findings_rows)
    failed = {str(r["qa_id"]): probe_failed(r) for r in findings_rows}

    recalls = {
        str(r["qa_id"]): r["recall_at_k"]
        for r in findings_rows
        if isinstance(r.get("recall_at_k"), (int, float))
    }
    has_recall = bool(recalls)

    per_category: dict[str, list[bool]] = collections.defaultdict(list)
    stage_hit = stage_total = 0
    no_diagnosis = gold_error = unmappable = we_passed = not_run = 0
    retrieval_ok = 0

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
        if category in RECALL_REFUTES and recalls.get(qa_id, -1.0) >= RECALL_FULL:
            # **우리 검색은 성공했다** — 이 카테고리의 주장(gold 가 생성기에 도달하지 못했다)이
            # 우리 실행에서는 성립하지 않는다. 우리 진단이 틀린 게 아니라 실패 지점이 다르다.
            # 순서 주의: no_diagnosis 앞이어야 한다. 뒤로 가면 '검색은 됐는데 원인을 못 짚은'
            # probe 가 오답으로 먼저 세어져 이 분기가 영영 안 걸린다.
            retrieval_ok += 1
            continue
        if not got:
            no_diagnosis += 1        # 실패했는데 원인을 못 짚었다 = 진짜 미진단
            per_category[category].append(False)
            stage_total += 1         # 단계도 못 짚은 것 — 아래 주석 참고
            continue

        per_category[category].append(bool(expected & got))

        # 단계는 우리가 낸 라벨 중 **하나라도** 그 단계면 맞은 것으로 본다(포함과 같은 취지).
        #
        # 분모를 라벨 정확도와 **같게** 맞춘다. 미진단(라벨 0개)이나 단계 개념이 없는 라벨만
        # 낸 경우를 분모에서 빼면, 단계 수치가 '우리가 단계 주장을 했을 때의 조건부 정확도'가
        # 되어 체계적으로 위로 편향된다 — 그 값을 논문 57.8% 옆에 나란히 찍으면 비교가
        # 어긋난다. 말 안 한 것도 못 맞힌 것으로 센다.
        stage_total += 1
        stages = {stage_of(label) for label in got} - {None}
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
        "retrieval_ok": retrieval_ok,
        "has_status": has_status,
        "has_recall": has_recall,
    }


def format_report(result: dict) -> str:
    total, hit = result["total"], result["total_hit"]
    lines = [
        "=" * 62,
        "  RAGEC 대조 — 진단 정확도 (포함 채점)",
        "=" * 62,
    ]
    # 채점 대상이 0건이어도 **제외 내역은 반드시 찍는다.** 여기서 일찍 돌아가면 "몇 건을
    # 왜 뺐나" 가 통째로 사라져, 제외 규칙이 정확도를 조용히 올리는 장치가 된다.
    if not total:
        lines.append("  채점 대상이 없습니다 — 아래 제외 내역을 보세요.")
    else:
        lines.append(f"  라벨 포함 정확도  {hit}/{total}  ({hit / total * 100:.1f}%)")
        if result["stage_total"]:
            sh, st = result["stage_hit"], result["stage_total"]
            lines.append(f"  단계 정확도       {sh}/{st}  ({sh / st * 100:.1f}%)")
        lines.append("")
        lines.append("  " + _pad("카테고리", 32) + _rpad("맞음", 8)
                     + _rpad("전체", 8) + _rpad("정확도", 9))
        for category, (ok, n) in result["per_category"].items():
            lines.append("  " + _pad(category, 32) + _rpad(str(ok), 8)
                         + _rpad(str(n), 8) + _rpad(f"{ok / n * 100:.0f}%", 9))

    lines.append("")
    if result.get("we_passed"):
        lines.append(f"  · 우리 파이프라인이 성공한 질문    {result['we_passed']}건 "
                     f"(진단할 게 없어 제외 — 그들 시스템만 실패한 건이다)")
    if result.get("retrieval_ok"):
        lines.append(f"  · 검색 단계 라벨인데 우리 검색은 성공  {result['retrieval_ok']}건 "
                     f"(recall=1.0 — 실패 지점이 달라 채점 불가)")
    if not result.get("has_recall", False):
        # 이 줄이 없으면 "이번엔 0건" 과 "애초에 못 잰다" 가 구분되지 않는다.
        lines.append("  ⚠ 덤프에 recall 이 없어 '우리 검색은 성공' 판정을 하지 못했습니다")
        lines.append("    — 그만큼 정확도가 실제보다 낮게 나올 수 있습니다(파이프라인 재실행 필요).")
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


_MARK_LEGEND = ("  O 맞음 · X 틀림 · 성공=우리가 성공(제외) · 검색OK=우리 검색은 성공(제외)"
                " · - 대응 라벨 없음(제외)")


def _verdict(row: dict, category: str) -> str:
    """이 probe 의 채점 판정. score() 의 분기와 **같은 순서**로 본다.

    순서가 어긋나면 표가 정확도와 다른 이야기를 해서, 표를 근거로 진단을 고칠 수 없게 된다.
    """
    labels = set(row.get("labels") or [])
    if labels & GOLD_ERROR_LABELS:
        return "gold"         # 평가셋 결함 주장 — 정확도에서 제외
    expected = RAGEC_TO_OURS.get(category)
    if not expected:
        return "-"            # 대응 라벨 없음 또는 대조표에 없는 카테고리 — 제외
    if not probe_failed(row):
        return "성공"          # 우리 파이프라인이 이 질문에 성공 — 채점 제외
    recall = row.get("recall_at_k")
    if (category in RECALL_REFUTES
            and isinstance(recall, (int, float)) and recall >= RECALL_FULL):
        return "검색OK"        # 검색 단계 주장인데 우리 검색은 성공 — 채점 제외
    return "O" if labels & expected else "X"


def _wrap(label: str, text: str, width: int = 88) -> list[str]:
    """`  Q   본문…` 꼴로 접어 쓴다. 이어지는 줄은 본문 열에 맞춰 들여쓴다.

    textwrap 은 **글자 수**로 접어서, 한국어 답변이 폭 88 을 넘어 176칸까지 늘어난다
    (실측 로그에서 답변 줄만 화면 밖으로 나갔다). 표시 폭으로 접는다.
    """
    body = " ".join((text or "").split()) or "(없음)"
    head = "  " + _pad(label, 6)
    indent = " " * len(head)

    lines: list[str] = []
    current = ""
    for word in body.split(" "):
        if not current:
            current = word
        elif _width(current) + 1 + _width(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    # 공백 없는 긴 토큰(한국어 문장은 어절이 길다)은 위에서 안 잘리므로 폭으로 다시 자른다.
    out: list[str] = []
    for line in lines + [current]:
        while _width(line) > width:
            cut = len(line)
            while cut > 1 and _width(line[:cut]) > width:
                cut -= 1
            out.append(line[:cut])
            line = line[cut:]
        out.append(line)

    return [head + out[0]] + [indent + line for line in out[1:]]


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
        diagnosis = ", ".join(row.get("labels") or []) or "(라벨 없음)"
        recall = row.get("recall_at_k")
        if isinstance(recall, (int, float)) and recall >= 0:
            basis = row.get("recall_basis") or "?"
            diagnosis += f"    · recall@k({basis})={recall:.2f}"
        lines += _wrap("진단", diagnosis)

    if not shown:
        lines.append("")
        lines.append("  덤프와 정답지에 공통으로 있는 probe 가 없습니다.")
        return "\n".join(lines)

    lines += ["", "=" * 92, "  요약 표", "=" * 92,
              "  " + _pad("qa_id", 8) + _pad("판정", 6) + _pad("RAGEC 정답", 30) + "우리 진단"]
    for key in key_rows:
        qa_id = str(key["qa_id"])
        row = ours.get(qa_id)
        if row is None:
            continue
        lines.append("  " + _pad(qa_id, 8)
                     + _pad(_verdict(row, key["ragec_category"].strip()), 6)
                     + _pad(key["ragec_category"], 30)
                     + (", ".join(row.get("labels") or []) or "(라벨 없음)"))
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
    # --detail 은 두지 않는다 — 기본이 ON 이라 동작을 못 바꾸는 죽은 플래그가 되고,
    # 나중에 기본값을 뒤집으면 도움말이 거짓이 된다.
    ap.add_argument("--no-detail", dest="detail", action="store_false", default=True,
                    help="probe 별 대조 없이 총계만 출력")
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
