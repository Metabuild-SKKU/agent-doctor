"""
tools/score_human_labels.py
사람이 붙인 정답 라벨과 우리 진단을 대조한다 (진단 유효성 검증 2단계).

## RAGEC 대조(score_ragec.py)와 뭐가 다른가

    score_ragec        우리 관측 + **다른 시스템의 관측**을 본 사람 라벨  → 출처 불일치
    score_human_labels 우리 관측 + **그 관측을 본** 사람 라벨            → 출처 일치

라벨 대응표도 없다. 사람이 우리 32개 라벨 중에서 직접 고르므로 대응이 항등이다.
그래서 "대조표가 좁아서 틀렸다" 는 오차가 아예 없다.

## 채점 규칙

  · 포함 — 사람 라벨이 우리 findings 안에 있으면 맞음. 우리가 더 말한 건 오답이 아니다
           (우리 생성 라벨은 오라클 트랙을 보므로 검색 실패와 독립으로 성립한다)
  · 우리가 그 라벨을 **첫 번째로** 냈는지는 따로 센다(top-1) — 포함만 보면 라벨을 남발할수록
    점수가 오르므로, 남발 여부를 함께 봐야 수치를 해석할 수 있다
  · 해당없음/판단불가는 정확도에서 빼고 따로 센다. 전자는 택소노미 구멍, 후자는 시트 부실
  · 미진단(실패인데 라벨 0개)은 **오답**이다. 빼면 "말 안 하면 안 틀린다" 가 된다

## 쓰는 법

    python tools/score_human_labels.py
    python tools/score_human_labels.py --sheet output/ragec/label_sheet.json \\
        --findings output/ragec/findings.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
from datetime import datetime
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.make_label_sheet import NO_LABEL, PRIMARY_FIELD, UNSURE
from tools.score_ragec import (
    _pad, _read_jsonl, _rpad, _width, _wrap, probe_failed, stage_of,
)

# LABELS.md 의 그룹 구분. 그룹 정확도는 라벨보다 거칠어 표기 흔들림에 강하다.
_GROUPS = {"chunking_": "A", "retrieval_": "A", "reranker_": "A",
           "generation_": "B", "corpus_gap": "D", "bad_gold_": "D"}
_C_GROUP = {"too_long_context", "lost_in_the_middle", "context_noise_interference"}


def group_of(label: str) -> str | None:
    if label in _C_GROUP:
        return "C"
    for prefix, group in _GROUPS.items():
        if label.startswith(prefix):
            return group
    return None


def read_sheet(path: str) -> list[dict]:
    """채워진 라벨 시트를 읽는다.

    make_label_sheet 가 낸 `{"_안내": [...], "항목": [...]}` 형태와, 사람이 항목 배열만
    떼어 저장한 형태를 둘 다 받는다. 편집 중 BOM 이 붙는 경우가 있어 utf-8-sig 로 읽는다.
    """
    text = pathlib.Path(path).read_text(encoding="utf-8-sig")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # 사람이 손으로 채우는 파일이라 문법 오류가 흔하다 — 어디를 고칠지 알려준다.
        raise SystemExit(
            f"[오류] {path} 의 JSON 형식이 깨졌습니다 — {exc.lineno}번째 줄 근처:\n"
            f"       {exc.msg}\n"
            f"       (값은 큰따옴표로 감싸고, 마지막 항목 뒤에는 쉼표를 남기지 마세요)"
        ) from exc
    items = data.get("항목", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise SystemExit(f"[오류] {path} 에서 '항목' 배열을 찾지 못했습니다.")
    return items


def score(label_rows: list[dict], findings_rows: list[dict]) -> dict:
    ours = {str(r["qa_id"]): r for r in findings_rows}

    hit = top1 = total = 0
    stage_hit = stage_total = group_hit = group_total = 0
    no_diagnosis = 0
    excluded: collections.Counter = collections.Counter()
    per_label: dict[str, list[bool]] = collections.defaultdict(list)
    confusion: collections.Counter = collections.Counter()
    rows: list[dict] = []

    for row in label_rows:
        qa_id = str(row.get("qa_id") or "").strip()
        gold = str(row.get(PRIMARY_FIELD) or "").strip()
        if not qa_id or not gold:
            excluded["미기입"] += 1
            continue
        if gold == NO_LABEL:
            excluded[NO_LABEL] += 1        # 택소노미 구멍 — 우리 잘못이 아니라 라벨이 없는 것
            continue
        if gold == UNSURE:
            excluded[UNSURE] += 1          # 패킷이 부실 — 라벨러가 판단할 자료가 부족했다
            continue
        found = ours.get(qa_id)
        if found is None:
            excluded["덤프에 없음"] += 1
            continue
        if not probe_failed(found):
            excluded["우리는 성공"] += 1   # 실패가 없으니 진단할 것도 없다
            continue

        predicted = list(found.get("labels") or [])
        total += 1
        per_label[gold].append(gold in predicted)
        if gold in predicted:
            hit += 1
            if predicted[0] == gold:
                top1 += 1
        else:
            if not predicted:
                no_diagnosis += 1
            for label in predicted or ["(라벨 없음)"]:
                confusion[(gold, label)] += 1

        # 분모를 라벨 정확도와 **같게** 맞춘다. 우리가 아무 라벨도 못 냈거나 단계 개념이
        # 없는 라벨(C·D그룹)만 낸 경우를 분모에서 빼면 '단계 주장을 했을 때의 조건부
        # 정확도'가 되어 위로 편향된다 — 논문 수치(57.8%) 옆에 병기하는 값이라 어긋나면 안 된다.
        # 사람 라벨 쪽에 단계 개념이 없을 때만(C·D그룹 정답) 채점 대상에서 제외한다.
        gold_stage = stage_of(gold)
        if gold_stage:
            stage_total += 1
            stage_hit += gold_stage in ({stage_of(p) for p in predicted} - {None})
        gold_group = group_of(gold)
        if gold_group:
            group_total += 1
            group_hit += gold_group in ({group_of(p) for p in predicted} - {None})

        rows.append({"qa_id": qa_id, "gold": gold,
                     "predicted": predicted, "hit": gold in predicted})

    return {
        "total": total, "hit": hit, "top1": top1,
        "stage_hit": stage_hit, "stage_total": stage_total,
        "group_hit": group_hit, "group_total": group_total,
        "no_diagnosis": no_diagnosis,
        "excluded": dict(excluded),
        "per_label": {k: (sum(v), len(v)) for k, v in sorted(per_label.items())},
        "confusion": confusion, "rows": rows,
    }


def format_report(result: dict) -> str:
    total, hit = result["total"], result["hit"]
    lines = ["=" * 72, "  사람 라벨 대조 — 진단 유효성", "=" * 72]
    if not total:
        lines.append("  채점 대상이 없습니다 — 아래 제외 내역을 보세요.")
    else:
        # 이름 열 폭을 고정하지 않고 제일 긴 항목에 맞춘다 — 사람이 쉼표로 라벨 둘을 적으면
        # 45자가 나와 고정 폭(40)을 넘고, 그 줄부터 숫자 열이 통째로 밀린다(실측).
        rows = [("라벨 포함 정확도", hit, total,
                 "우리 findings 안에 사람 라벨이 있나"),
                ("라벨 top-1 정확도", result["top1"], total,
                 "그걸 **1순위로** 냈나 — 포함만 보면 라벨 남발이 점수를 올린다"),
                ("단계 정확도", result["stage_hit"], result["stage_total"],
                 "검색/생성 중 어느 단계인지는 맞췄나"),
                ("그룹 정확도", result["group_hit"], result["group_total"],
                 "A/B/C/D 그룹은 맞췄나 — 가장 거친 축")]
        name_w = max(_width(r[0]) for r in rows) + 2
        for name, ok, n, note in rows:
            if not n:
                continue
            lines.append("  " + _pad(name, name_w)
                         + _rpad(f"{ok}/{n}", 8) + _rpad(f"({ok / n * 100:.1f}%)", 10)
                         + "   " + note)
        lines.append("")
        lines.append("  참고 — RAGEC 논문(arXiv:2510.13975)의 자체 분류기:"
                     " 단계 57.8% · 유형 40.3%")
        lines.append("        (채점 규칙이 달라 직접 비교는 불가. 수준 감각용)")

        label_w = max([_width("사람이 붙인 라벨")]
                      + [_width(k) for k in result["per_label"]]) + 2
        lines += ["", "  " + _pad("사람이 붙인 라벨", label_w)
                  + _rpad("맞음", 6) + _rpad("전체", 6)]
        for label, (ok, n) in result["per_label"].items():
            lines.append("  " + _pad(label, label_w)
                         + _rpad(str(ok), 6) + _rpad(str(n), 6))

    lines.append("")
    if result["no_diagnosis"]:
        lines.append(f"  · 실패인데 우리가 라벨을 못 냄   {result['no_diagnosis']}건 (오답으로 셈)")
    for name, n in sorted(result["excluded"].items()):
        note = {NO_LABEL: " — 택소노미에 없는 실패 유형",
                UNSURE: " — 시트 자료가 부족했다는 신호"}.get(name, "")
        lines.append(f"  · 제외: {name:<12}{n:>4}건{note}")

    if result["confusion"]:
        lines += ["", "  자주 어긋난 쌍 (사람 → 우리)"]
        for (gold, got), n in result["confusion"].most_common(12):
            lines.append(f"    {n:>3}×  {gold}  →  {got}")
    return "\n".join(lines)


def format_detail(result: dict) -> str:
    lines = ["", "=" * 88, "  probe 별 대조", "=" * 88]
    for row in result["rows"]:
        mark = "O" if row["hit"] else "X"
        lines.append("")
        lines.append(f"── qa_id {row['qa_id']}  [{mark}]")
        lines += _wrap("사람", row["gold"])
        lines += _wrap("우리", ", ".join(row["predicted"]) or "(라벨 없음)")
    return "\n".join(lines)


def save_result(result: dict, out_dir: pathlib.Path, text: str) -> tuple[str, str]:
    """채점 결과를 파일로 남긴다.

    사람 시간이 몇 시간 들어간 라벨링의 산출물이라 콘솔에만 두면 안 된다 — 창을 닫으면
    사라지고, 나중에 "그때 몇 % 였지" 를 확인하려면 라벨 시트를 다시 찾아야 한다.
    사람이 읽을 텍스트와 기계가 읽을 JSON 을 함께 남긴다(전자는 보고용, 후자는 실행 간
    비교용 — 개선 전후를 나란히 놓으려면 파싱 가능한 형태가 필요하다).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt = out_dir / f"human_labels_{stamp}.txt"
    txt.write_text(text, encoding="utf-8")

    payload = {k: v for k, v in result.items() if k != "confusion"}
    # 혼동 쌍은 튜플 키라 JSON 이 못 담는다 — 어디가 어긋났는지가 제일 쓸모 있는 정보라
    # 리스트로 펴서 싣는다.
    payload["confusion"] = [
        {"사람": gold, "우리": got, "건수": n}
        for (gold, got), n in result["confusion"].most_common()
    ]
    payload["측정시각"] = stamp
    js = out_dir / f"human_labels_{stamp}.json"
    js.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(txt), str(js)


def main() -> int:
    ap = argparse.ArgumentParser(description="사람 라벨 ↔ 우리 진단 대조")
    ap.add_argument("--sheet", default="output/ragec/label_sheet.json",
                    help=f"'{PRIMARY_FIELD}' 을 채운 라벨 시트")
    ap.add_argument("--findings", default="output/ragec/findings.jsonl")
    ap.add_argument("--out", default="output/ragec/scores",
                    help="채점 결과 저장 위치")
    ap.add_argument("--no-detail", dest="detail", action="store_false", default=True,
                    help="probe 별 대조 없이 총계만 출력")
    ap.add_argument("--no-save", dest="save", action="store_false", default=True,
                    help="파일로 저장하지 않고 콘솔에만 출력")
    args = ap.parse_args()

    for path in (args.sheet, args.findings):
        if not pathlib.Path(path).exists():
            print(f"[오류] 파일이 없습니다: {path}", file=sys.stderr)
            return 1

    result = score(read_sheet(args.sheet), _read_jsonl(args.findings))
    blocks = ([format_detail(result), ""] if args.detail else []) + [format_report(result)]
    text = "\n".join(blocks)
    print(text)

    if args.save:
        txt, js = save_result(result, pathlib.Path(args.out), text)
        print(f"\n저장 → {txt}")
        print(f"     → {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
