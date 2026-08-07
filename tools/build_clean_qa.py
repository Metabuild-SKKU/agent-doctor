"""
tools/build_clean_qa.py
KorQuAD 골든 QA 정제 — 골드가 '정답이 실제로 있는 좁은 구간'을 가리키게 다시 만든다.

## 왜 필요한가

KorQuAD 2.1 원본은 SQuAD 형식이라 정답의 정확한 위치(answer_start)를 준다. 그런데 원문이
HTML 이라 그 좌표는 태그를 포함한 기준이고, 우리 corpus.jsonl 은 태그를 벗긴 평문이라
좌표계가 어긋난다. 그래서 최초 전처리가 answer_start 를 버리고 **정답 텍스트를 문서에서
다시 찾아** 그게 든 청크 id 를 positive_chunk_ids 로 적었다. 여기서 두 결함이 생겼다.

**결함 1 — 여러 곳에 나오는 정답에서 엉뚱한 곳을 골랐다.**

    질문: 파스칼레 소틸레의 스파이크 높이는 몇 cm인가?   정답: "332cm"
    문서: 배구 선수 명단 표. "332" 가 8곳에 나온다(선수마다 스파이크 높이가 있다).
    골드: 1188~2028   ← 다른 선수의 332cm
    실제: 5268        ← "소틸레"(문서에 1회)가 있는 행

**결함 2 — 골드가 정답보다 70배 넓다.**

    정답 "332cm" 5자  vs  골드(청크 통째) 중앙값 497자

span_recall_at_k 는 골드 구간을 **빈틈없이 덮어야** 1점을 주는 이진 판정이다(부분 점수
없음). 497자를 덮으려면 재청킹된 청크 2~3개를 정확히 다 가져와야 하는데, corpus 의 청크
경계와 Index 의 재청킹 경계가 달라 거의 성립하지 않는다.

둘이 곱해져서 **정답을 맞혀도 '검색 실패'로 집계된다.** 실측(corpus_20260804_103059):
30문항 중 4건(13%)이 answer=1.00 인데 recall=0.00 이었고, 그중 하나는 RAGAS 흔들림과
겹쳐 `retrieval_rerank_candidate_miss` 로 오분류돼 실제 처방(rerank_candidates 20→22)까지
만들어 최종 config 에 남았다.

## 이 스크립트가 하는 일

정답 텍스트가 문서에 **정확히 한 번만** 나오는 QA 만 남기고, 골드를 그 한 곳으로 좁힌다.

    ① 1회만 등장  → 어느 곳이 정답인지 모호하지 않다 = 골드가 틀릴 수 없다
    ② 그 위치로 좁힘 → 청크 하나만 가져와도 덮인다 = 이진 판정이 아프지 않다
    ③ 정답 길이 상한 → ①이 긴 서술형 정답을 남기는 쪽으로 치우치는 걸 막는다

②는 ①이 있어야 가능하다(여러 곳이면 어디로 좁힐지 모른다). ③은 ①의 부작용을 막는다 —
짧은 정답일수록 문서에 여러 번 나와 걸러지므로, ① 만 적용하면 남는 집합의 정답 길이
중앙값이 5자에서 11자로 오르고 100자 초과가 3%→31% 로 뛴다. 긴 서술형 정답은 char-F1 이
요약을 오답으로 깎아 bad_gold_answer 를 만든다.

**오라클 트랙은 좁혀도 안전하다.** gold_spans 는 recall 판정에만 쓰이고, 오라클 컨텍스트는
`_resync_gold_chunk_ids` 가 고른 **청크 전체**(agent.py::gold_ctx)다. 그 함수는 "span 전체를
포함하는 청크가 있으면 문맥 낭비가 가장 작은 하나"를 고르므로, 좁은 span 은 오히려 정확히
한 청크로 수렴한다. 지금은 골드가 엉뚱해서 오라클도 엉뚱한 청크를 받아 못 맞히고 있다
(위 사례 oracle_f1=0.19).

## 사용법

    python tools/build_clean_qa.py
    python tools/build_clean_qa.py --max-answer-chars 80 --out data/qa_pairs_clean80.jsonl

원본 `data/qa_pairs.jsonl` 은 건드리지 않는다. 새 파일을 쓰려면 Eval 설정만 바꾼다.

    EVAL_TAXONOMY_QA=data/qa_pairs_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_CORPUS = "data/corpus.jsonl"
DEFAULT_QA = "data/qa_pairs.jsonl"
DEFAULT_OUT = "data/qa_pairs_clean.jsonl"

# 정답 길이 상한 — 실측 분포에서 고른 값이다. 이 코퍼스에서 50자는 1회 등장 616건 중
# 549건(89%)을 남기고, 그 위(80·100·150자)로 올려도 3%p 씩만 늘어난다. 즉 50자가
# "채점이 안정적인 짧은 사실형"과 "요약이 오답으로 깎이는 서술형"이 갈리는 지점이다.
DEFAULT_MAX_ANSWER = 50
# 1~2자 정답("2", "가")은 문서 아무 데나 우연히 일치해 1회 판정 자체가 못 믿을 값이 된다.
DEFAULT_MIN_ANSWER = 2


def load_documents(corpus_path: str) -> dict[str, tuple[str, list[tuple[str, int, int]]]]:
    """doc_id → (원문 전체, [(chunk_id, start, end)]).

    **복원은 파이프라인과 같은 함수(`korquad._stitch`)를 쓴다.** 이 도구가 계산한 좌표를
    파이프라인이 gold_spans 로 읽으므로, 두 좌표계가 한 글자라도 어긋나면 골드가 엉뚱한
    곳을 가리킨다 — 이 도구가 고치려던 결함을 그대로 다시 만드는 셈이다.

    처음에는 `"".join(청크 본문)` 으로 이어붙였는데 틀린 방식이었다(리뷰 지적).
    `_stitch` 는 각 청크를 **`char_start` 위치에 놓고** 빈 자리를 공백으로 남기는 반면,
    이어붙이기는 좌표를 무시하고 본문을 차례로 붙인다. 둘은 청크 구간이 딱 맞닿을 때만
    같고, **겹치거나 틈이 있으면 갈라진다.**

    이 코퍼스는 청크가 겹친다(overlap). 실측:

        문서 1,000개 중 구간이 겹치는 문서      1,000개
        join 길이 - stitch 길이  중앙값 1,327자 (최대 38,910자)
        그 결과 정제본 549건 중 좌표가 어긋난 것  539건(98%)

    겹친 만큼 뒤쪽 좌표가 통째로 밀려서, 사실상 전부 틀린 골드를 싣고 있었다.
    """
    from agents.eval.datasets.korquad import _stitch

    by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            by_doc[row["doc_id"]].append(row)

    out: dict[str, tuple[str, list[tuple[str, int, int]]]] = {}
    for doc_id, rows in by_doc.items():
        rows.sort(key=lambda r: r["char_start"])
        text = _stitch([
            (int(r.get("char_start", 0)), int(r.get("char_end", 0)), r.get("text", "") or "")
            for r in rows
        ])
        spans = [(r["chunk_id"], r["char_start"], r["char_end"]) for r in rows]
        out[doc_id] = (text, spans)
    return out


def covering_chunk_ids(spans: list[tuple[str, int, int]], start: int, end: int) -> list[str]:
    """좌표 구간과 겹치는 corpus 청크 id. 하위호환용 positive_chunk_ids 를 다시 만든다."""
    return [cid for cid, s, e in spans if s < end and e > start]


def build(corpus_path: str, qa_path: str, out_path: str,
          max_answer: int, min_answer: int) -> dict[str, int]:
    docs = load_documents(corpus_path)
    stats: dict[str, int] = defaultdict(int)
    kept: list[dict] = []

    with open(qa_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            qa = json.loads(line)
            stats["total"] += 1

            doc_id = qa.get("doc_id")
            answer = (qa.get("answer_text") or "").strip()
            entry = docs.get(doc_id)
            if entry is None:
                stats["문서 없음"] += 1
                continue
            text, spans = entry

            if len(answer) < min_answer:
                stats[f"정답이 {min_answer}자 미만"] += 1
                continue
            if len(answer) > max_answer:
                stats[f"정답이 {max_answer}자 초과"] += 1
                continue

            # 완전일치만 센다. 앞 N자 매칭은 서로 다른 정답을 같은 것으로 세어
            # '1회 등장'을 거짓으로 만든다.
            hits = [m.start() for m in re.finditer(re.escape(answer), text)]
            if not hits:
                stats["정답이 문서에 없음"] += 1
                continue
            if len(hits) > 1:
                stats["정답이 여러 곳(모호)"] += 1
                continue

            start = hits[0]
            end = start + len(answer)
            kept.append({
                "qa_id": qa["qa_id"],
                "question": qa["question"],
                "answer_text": answer,
                "doc_id": doc_id,
                # 정본. 로더가 이 값을 그대로 gold_spans 로 쓴다.
                "gold_spans": [{"doc_id": doc_id, "start": start, "end": end}],
                # 하위호환 — gold_spans 를 못 읽는 옛 경로가 있어도 최소한 '정답이 든
                # 청크'를 가리키게 둔다. 원본의 엉뚱한 값을 그대로 물려주지 않는다.
                "positive_chunk_ids": covering_chunk_ids(spans, start, end),
                "source_qa": {"dataset": "korquad2.1", "cleaned_by": "tools/build_clean_qa.py"},
            })
            stats["남김"] += 1

    with open(out_path, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--qa", default=DEFAULT_QA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-answer-chars", type=int, default=DEFAULT_MAX_ANSWER)
    ap.add_argument("--min-answer-chars", type=int, default=DEFAULT_MIN_ANSWER)
    args = ap.parse_args()

    for path in (args.corpus, args.qa):
        if not pathlib.Path(path).exists():
            print(f"[오류] 파일이 없습니다: {path}", file=sys.stderr)
            print("       data/ 는 gitignore 라 각자 준비합니다 — data/README.md 참고",
                  file=sys.stderr)
            return 1

    stats = build(args.corpus, args.qa, args.out,
                  args.max_answer_chars, args.min_answer_chars)

    total = stats.pop("total", 0)
    keep = stats.pop("남김", 0)
    print(f"입력 {total:,}건 → 남김 {keep:,}건 ({keep / total * 100:.0f}%)")
    print("제외 사유:")
    for reason, n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<24} {n:>6,}건")
    print(f"\n기록: {args.out}")
    print("사용: EVAL_TAXONOMY_QA=" + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
