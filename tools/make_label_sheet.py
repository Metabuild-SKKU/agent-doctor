"""
tools/make_label_sheet.py
사람이 정답 라벨을 채울 **라벨 시트**를 만든다 (진단 유효성 검증 1단계).

`run_ragec_validation.py` 가 실행 끝에 자동으로 부른다. 표본만 다시 뽑고 싶을 때는
파이프라인을 다시 돌리지 말고 이 파일을 단독으로 실행하면 된다(100분·$5 를 아낀다).

## 왜 사람이 다시 라벨을 붙이나

RAGEC 정답지를 그대로 쓰면 **관측과 라벨의 출처가 어긋난다.** 그 라벨은 사람이 *그들*
시스템의 관측을 보고 붙인 것이라, 검색기가 다른 우리 관측에는 성립하지 않을 수 있다.
실측(qa_id 2205):

    RAGEC   E4 Missed Retrieval ("검색이 정답 문서를 못 찾았다")
    우리     recall=1.00 · gold청크 2/2 검색 · 답변에 정답이 그대로 들어 있음

우리 검색은 성공했으니 E4 는 우리에게 성립하지 않는다. **라벨이 다른 게 오진의 증거가
아니다.** RAGEC 논문(arXiv:2510.13975)의 저자들도 자기 시스템의 오류를 손으로 라벨링했다 —
출처를 맞춘 것이 그 검증이 성립한 이유다.

반대로 상황을 지어내고 기대 라벨을 `rules.py` 에서 베끼면 순환이 된다. 그래서 이 시트는
둘 다 피한다:

    관측  우리 파이프라인의 **실제 실행 결과** (합성 아님)
    라벨  그 관측을 보고 **사람이** 판단 (우리 진단은 시트에 없음)

## 무엇을 싣고 무엇을 빼나

우리가 낸 라벨과 finding 근거 문구를 **뺀다** — 라벨러가 그걸 보면 '판단' 이 아니라
'동의하는지' 를 답하게 되어 검증이 통째로 무의미해진다. 지표(recall·f1·faithfulness)는
남긴다. 그건 진단이 아니라 관측이고, 없으면 검색 계열 라벨을 사람도 구분할 수 없다.

## 쓰는 법

    python tools/make_label_sheet.py --findings output/ragec/findings.jsonl --limit 60
    → output/ragec/label_sheet.json 의 '정답라벨' 칸을 채운 뒤

    python tools/score_human_labels.py
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.score_ragec import _read_jsonl, probe_failed

LABELS_MD = "tests/diagnose_grid/LABELS.md"

# 라벨러가 32개 라벨 밖을 표현할 통로. 둘 다 없으면 억지로 하나를 고르게 되어, 택소노미
# 구멍과 자료 부족이 전부 '오답'으로 뭉개진다. 많이 나오면 그 자체가 결과다.
NO_LABEL = "해당없음"      # 32개 중 맞는 게 없다 → 택소노미 구멍 신호
UNSURE = "판단불가"        # 주어진 자료로는 못 정하겠다 → 시트가 부실하다는 신호

# 사람이 채우는 칸. 각 항목 **끝**에 붙는다(읽고 나서 바로 적도록).
#
# 한 칸만 둔다 — 2지망·확신도 같은 보조 칸은 채점에 안 쓰이면서 라벨링 시간을 늘리고,
# 안 채워진 칸이 섞이면 '무응답' 과 '해당 없음' 이 흐려진다.
FILL_FIELDS = ("정답라벨",)
PRIMARY_FIELD = FILL_FIELDS[0]

GUIDE = [
    "각 항목은 우리 RAG 파이프라인의 **실제 실행 결과**입니다.",
    f"상황을 보고 '이 실패의 원인' 이라고 판단되는 라벨을 '{PRIMARY_FIELD}' 칸에 적어주세요.",
    f"라벨 목록과 정의: {LABELS_MD}",
    f"32개 중 맞는 게 없으면 '{NO_LABEL}', 자료로 판단이 안 되면 '{UNSURE}' 라고 적어주세요.",
    "  (억지로 고르지 마세요 — 이 둘이 많이 나오는 것도 의미 있는 결과입니다)",
    "우리 시스템의 진단은 일부러 뺐습니다. 보고 나면 '맞는지' 가 아니라 '동의하는지' 를",
    "  판단하게 되어 검증이 성립하지 않습니다.",
]


def stratified_sample(rows: list[dict], limit: int, seed: int = 0) -> list[dict]:
    """우리 예측 라벨 기준으로 **골고루** 뽑는다.

    무작위로 뽑으면 실측처럼 표본이 한 라벨로 쏠린다(10건 중 5건이 retrieval_low_rank).
    그러면 희귀 라벨은 검증 표본에 아예 안 들어와 영영 못 잰다.

    예측 라벨로 층을 나누는 게 라벨러에게 힌트가 되지는 않는다 — 표본을 고르는 데만 쓰고
    시트에는 싣지 않는다. '우리가 X 라고 본 것들' 을 골고루 담을 뿐 정답이 X 라는 뜻이 아니다.
    """
    scoreable = [r for r in rows if probe_failed(r)]
    if limit <= 0 or len(scoreable) <= limit:
        return sorted(scoreable, key=lambda r: str(r["qa_id"]))

    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in scoreable:
        labels = row.get("labels") or []
        buckets[labels[0] if labels else "(라벨 없음)"].append(row)

    rng = random.Random(seed)
    for group in buckets.values():
        rng.shuffle(group)

    # 라운드로빈으로 한 개씩 — 표본이 적은 라벨이 먼저 소진돼도 남은 자리를 다른 라벨이 채운다.
    picked: list[dict] = []
    order = sorted(buckets, key=lambda k: len(buckets[k]))
    while len(picked) < limit:
        added = False
        for key in order:
            if buckets[key] and len(picked) < limit:
                picked.append(buckets[key].pop())
                added = True
        if not added:
            break
    return sorted(picked, key=lambda r: str(r["qa_id"]))


def _rank_note(obs: dict) -> str:
    """정답 청크가 검색 결과의 몇 위였나. 없으면 '검색안됨'.

    이게 없으면 사람이 retrieval_low_rank(순위가 밀림)와 retrieval_missing_gold(아예 없음)를
    구분할 수 없다 — 지표 중 가장 중요한 항목이다.
    """
    gold = obs.get("gold_chunk_ids") or []
    retrieved = obs.get("retrieved_chunk_ids") or []
    if not gold:
        return "-"
    return ", ".join(
        f"{cid}={retrieved.index(cid) + 1}위" if cid in retrieved else f"{cid}=검색안됨"
        for cid in gold
    )


def _round(value, digits: int = 3):
    return round(value, digits) if isinstance(value, (int, float)) else value


def to_entry(row: dict) -> dict:
    """덤프 한 줄 → 시트 항목 하나. **우리 라벨은 담지 않는다.**"""
    obs = row.get("observations") or {}
    recall = row.get("recall_at_k")
    entry = {
        "qa_id": row["qa_id"],
        "질문": row.get("question", ""),
        "정답": row.get("gold_answer", ""),
        "우리_답변": row.get("answer", ""),

        "검색_recall": _round(recall) if isinstance(recall, (int, float)) and recall >= 0 else None,
        "검색_recall_기준": row.get("recall_basis", ""),
        "정답청크_검색됨": f"{obs.get('gold_chunk_hit', '-')}/{obs.get('gold_chunk_total', '-')}",
        "정답청크_순위": _rank_note(obs),
        "검색된_청크수": len(obs.get("retrieved_chunk_ids") or []),
        "검색방식": obs.get("search_mode", "-"),
        "리랭커": obs.get("reranker_status", "-"),
        "MMR": "적용" if obs.get("mmr_applied") else "미적용",
        "근거밀도": _round(obs.get("span_precision")),

        "f1": _round(obs.get("f1")),
        "완전일치": obs.get("exact_match"),
        "종합점수": _round(obs.get("answer_score")),
        "정답요소_커버리지": _round(obs.get("gold_coverage")),
        "의미점수": _round(obs.get("answer_semantic")),
        "oracle_f1": _round(obs.get("oracle_f1")),
    }
    for name in ("faithfulness", "context_precision", "context_recall", "response_relevancy"):
        if name in obs:
            entry[name] = _round(obs[name])
    for field in FILL_FIELDS:
        entry[field] = ""
    return entry


def build_sheet(rows: list[dict]) -> dict:
    """JSON 은 주석을 못 달아서 안내를 데이터로 싣는다(`_` 로 시작하는 키는 채점기가 무시)."""
    return {
        "_안내": GUIDE,
        "_채울_칸": PRIMARY_FIELD,
        "항목": [to_entry(row) for row in rows],
    }


def write_sheet(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_sheet(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize(rows: list[dict]) -> str:
    dist = collections.Counter((r.get("labels") or ["(라벨 없음)"])[0] for r in rows)
    lines = ["표본 구성(우리 예측 라벨 기준 — 시트에는 안 실림):"]
    lines += [f"  {label:<40}{n:>3}" for label, n in dist.most_common()]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="사람이 정답 라벨을 채울 시트 생성(블라인드)")
    ap.add_argument("--findings", default="output/ragec/findings.jsonl",
                    help="실행 덤프. run_ragec_validation.py 가 만든다")
    ap.add_argument("--out", default="output/ragec/label_sheet.json")
    ap.add_argument("--limit", type=int, default=60,
                    help="표본 수(0=전체). 1건당 2~3분 소요를 감안할 것")
    ap.add_argument("--seed", type=int, default=0, help="층화 추출 시드(재현용)")
    args = ap.parse_args()

    if not pathlib.Path(args.findings).exists():
        print(f"[오류] 파일이 없습니다: {args.findings}", file=sys.stderr)
        return 1

    rows = _read_jsonl(args.findings)
    picked = stratified_sample(rows, args.limit, args.seed)
    if not picked:
        print("[오류] 라벨링할 probe 가 없습니다(실패 probe 0건).", file=sys.stderr)
        return 1
    if not any(r.get("observations") for r in picked):
        print("[경고] 덤프에 observations 가 없어 지표 없이 시트가 나갑니다 — "
              "파이프라인을 다시 실행하세요.", file=sys.stderr)

    out = pathlib.Path(args.out)
    write_sheet(picked, out)
    print(f"라벨 시트 {len(picked)}건 → {out}")
    print(summarize(picked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
