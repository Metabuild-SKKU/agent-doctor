"""
tools/build_ragec_dataset.py
RAGEC(사람 라벨 377건) + DragonBall(원본 코퍼스) → 우리 파이프라인 입력으로 변환.

## 왜 필요한가

RAGEC 은 사람이 라벨링한 RAG 실패 377건을 준다. 진단 정확도를 재려면 이게 정답지인데,
CSV 에는 **검색된 청크도 원본 문서도 없다**(질문/정답/RAG답변/라벨뿐). 우리 `diagnose()` 는
순위·recall@k·오라클 트랙이 필요하므로 "CSV 넣고 채점" 이 불가능하다.

원본 DragonBall 을 받아 우리 코퍼스 형식으로 바꾸면 파이프라인을 그대로 태울 수 있다.
조인은 확인됐다 — RAGEC `query_id` → DragonBall 영어 질의 **377/377**, 질문 텍스트 100% 일치.

## 내는 것 (셋)

    corpus   문서 108개 → data/README.md 의 corpus.jsonl 스키마
    qa       377건 → qa_pairs.jsonl 스키마(gold_spans 명시)
    key      정답지 — qa_id → RAGEC 라벨. **파이프라인은 이걸 보지 않는다**

key 를 따로 두는 이유: 파이프라인이 정답 라벨을 보면 채점이 아니라 커닝이 된다. 진단은
corpus·qa 만 보고 돌고, 채점기가 나중에 key 와 대조한다.

## gold_spans 를 어떻게 만드나

DragonBall `ground_truth.references` 는 근거 **문장 텍스트**다(좌표가 아니다). 문서에서
그 문장을 찾아 좌표로 바꾼다 — `tools/build_clean_qa.py` 가 KorQuAD 에 한 일과 같다.

다만 KorQuAD 보다 훨씬 쉽다. 실측 1,517개 중:

    문서에 그대로 있음   1,489 (98%)
    앞부분만 일치            27 (2%)   ← 공백·따옴표 차이. 접두 매칭으로 흡수
    못 찾음                   1 (0%)

KorQuAD 는 정답 텍스트가 문서 여러 곳에 나와 '어디가 골드인지' 가 모호했지만(그래서 1회
등장만 남기는 정제가 필요했다), 여기 references 는 근거 문장 자체라 그 문제가 거의 없다.

## 멀티문서 질문

377건 중 **233건(62%)** 이 골드 문서 2개다. `Probe.gold_doc_id` 는 단일이지만 **채점 경로는
`gold_spans` 를 본다**(`metrics_basic._gold_coverage_context` 가 span 마다 doc_id 를 읽는다).
그래서 span 을 문서별로 싣고 `doc_id` 에는 첫 문서만 적는다(표시용).

## 사용법

    # 1) 원본 셋을 받는다 (둘 다 공개·MIT/Apache)
    #    RAGEC     https://github.com/layer6ai-labs/rag-error-classification
    #                annotation/RAGEC_annotations.csv
    #    DragonBall https://github.com/OpenBMB/RAGEval
    #                dragonball_dataset/dragonball_{docs,queries}.jsonl
    python tools/build_ragec_dataset.py \\
        --ragec RAGEC_annotations.csv \\
        --docs dragonball_docs.jsonl \\
        --queries dragonball_queries.jsonl

    # 2) 파이프라인을 이 코퍼스로 돌린다
    SOURCE_TYPE=korquad SOURCE_URL=data/ragec_corpus.jsonl \\
    EVAL_TAXONOMY_QA=data/ragec_qa.jsonl python run_local_pipeline.py

`SOURCE_TYPE=korquad` 를 그대로 쓰는 이유는 스키마가 같기 때문이다 — 어댑터가 KorQuAD 형식으로
내므로 로더를 새로 만들 필요가 없다.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import Counter

DEFAULT_OUT_CORPUS = "data/ragec_corpus.jsonl"
DEFAULT_OUT_QA = "data/ragec_qa.jsonl"
DEFAULT_OUT_KEY = "data/ragec_answer_key.jsonl"

# DragonBall doc_id 는 정수이고 언어별로 따로 매겨진다(영어 108 / 중국어 108, 겹침 0).
# 그래도 접두를 붙여 둔다 — 다른 코퍼스와 섞였을 때 doc_id 하나로 출처를 알 수 있어야 한다.
DOC_PREFIX = "ragec_"

LANGUAGE = "en"     # RAGEC 은 영어 DragonBall 만 라벨링했다

# DragonBall 이 '답할 수 없는 질문'(Irrelevant Unsolvable Question)의 정답으로 쓰는 문구.
# 이 질문들은 근거(references)가 없는 게 정상이라 무응답 probe 로 실어야 한다.
_NO_ANSWER = "unable to answer"


def _norm(text: str) -> str:
    """공백만 접은 비교용 문자열. 좌표는 원문에서 다시 잡으므로 여기서는 매칭에만 쓴다."""
    return re.sub(r"\s+", " ", text or "").strip()


def load_docs(path: str) -> dict[int, dict]:
    """영어 문서만 {doc_id: row}. 언어를 섞으면 좌표계가 무의미해진다."""
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("language") == LANGUAGE:
                out[row["doc_id"]] = row
    return out


def load_queries(path: str) -> dict[int, dict]:
    """영어 질의만 {query_id: row}."""
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("language") == LANGUAGE:
                out[row["query"]["query_id"]] = row
    return out


def locate(reference: str, content: str) -> tuple[int, int] | None:
    """근거 문장이 문서에서 차지하는 좌표 [start, end). 못 찾으면 None.

    3단계로 시도한다. 원문 좌표가 목적이므로 **정규화한 문자열에서 찾은 위치를 쓰지 않는다** —
    공백을 접으면 좌표가 밀린다. 정규화는 '있는지' 판단에만 쓰고 좌표는 원문에서 잡는다.
    """
    ref = (reference or "").strip()
    if not ref or not content:
        return None

    at = content.find(ref)
    if at >= 0:
        return at, at + len(ref)

    # 공백 차이만 다른 경우 — 원문의 공백을 유연하게 받는 정규식으로 다시 찾는다.
    flexible = r"\s+".join(re.escape(tok) for tok in ref.split())
    match = re.search(flexible, content)
    if match:
        return match.start(), match.end()

    # 접두만 일치(뒤쪽이 잘리거나 따옴표가 다른 경우). 너무 짧은 접두는 우연히 맞을 수
    # 있으므로 40자 이상일 때만 인정한다.
    head = ref[:40]
    if len(head) >= 40:
        flexible_head = r"\s+".join(re.escape(tok) for tok in head.split())
        match = re.search(flexible_head, content)
        if match:
            return match.start(), match.start() + len(ref)
    return None


def build(ragec_path: str, docs_path: str, queries_path: str,
          out_corpus: str, out_qa: str, out_key: str) -> dict:
    docs = load_docs(docs_path)
    queries = load_queries(queries_path)
    with open(ragec_path, encoding="utf-8") as fh:
        annotations = list(csv.DictReader(fh))

    stats: Counter = Counter()

    # ── corpus: 문서 하나 = 행 하나 ──────────────────────────────
    # KorQuAD 는 원본이 청크 단위로 배포돼 여러 행이었지만, DragonBall 은 본문을 통째로
    # 준다. 인위적으로 쪼개면 우리가 만든 경계가 좌표에 섞이므로 그대로 한 행에 싣는다.
    # Ingest 가 _stitch 로 복원할 때 char_start=0 이면 원문과 동일하다.
    pathlib.Path(out_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(out_corpus, "w", encoding="utf-8") as fh:
        for doc_id, row in sorted(docs.items()):
            content = row.get("content") or ""
            fh.write(json.dumps({
                "doc_id": f"{DOC_PREFIX}{doc_id}",
                "chunk_id": f"{DOC_PREFIX}{doc_id}_0",
                "title": row.get("company_name") or "",
                "text": content,
                "char_start": 0,
                "char_end": len(content),
            }, ensure_ascii=False) + "\n")
            stats["문서"] += 1

    # ── qa + key ────────────────────────────────────────────────
    qa_rows, key_rows = [], []
    for ann in annotations:
        qid = int(ann["query_id"])
        query = queries.get(qid)
        if query is None:
            stats["질의 못 찾음"] += 1
            continue
        truth = query.get("ground_truth") or {}
        gold_doc_ids = [d for d in (truth.get("doc_ids") or []) if d in docs]
        if not gold_doc_ids:
            stats["골드 문서 없음"] += 1
            continue

        spans = []
        for reference in (truth.get("references") or []):
            placed = False
            for doc_id in gold_doc_ids:
                found = locate(reference, docs[doc_id].get("content") or "")
                if found:
                    spans.append({"doc_id": f"{DOC_PREFIX}{doc_id}",
                                  "start": found[0], "end": found[1]})
                    placed = True
                    break
            stats["근거 좌표화" if placed else "근거 못 찾음"] += 1

        # 답할 수 없는 질문은 근거가 없는 게 **정상**이다. 이걸 "span 없음" 으로 버리면
        # E9 Abstention Failure 가 통째로 날아간다(실측: 23건 중 17건이 무응답 질문이라
        # 6건만 남았다 — 그 라벨은 사실상 못 재게 된다).
        #
        # 우리 파이프라인에는 이미 그 경로가 있다 — answer_exists=False 면 diagnose 가
        # generation_abstention_failure·generation_wrongful_abstention 을 연다. recall 도
        # gold 가 없으면 -1(계산 불가)로 빠져 A 슬롯이 무응답 probe 를 실패로 세지 않는다.
        # **포함이 아니라 완전일치**로 본다. 포함으로 잡으면 본문에 그 문구가 스쳐 지나가는
        # 정상 정답까지 무응답으로 뒤집힌다(실측 qa_id=2399 — 자산 재편을 설명하는 긴 정답이
        # 무응답으로 분류됐다). 무응답 표기는 정답 칸이 그 문구 **하나뿐**일 때다.
        answer_text = _norm(truth.get("content") or "")
        answerable = answer_text.lower().rstrip(".") != _NO_ANSWER
        if not answerable and spans:
            # 데이터 자체의 모순 — '답할 수 없음'인데 근거가 달려 있다(실측 1건, qa_id=3537).
            # 무응답 판정을 우선하고 골드를 버린다. 남겨 두면 "기권이 정답인데 골드도 있다"가
            # 되어 recall 이 계산되고, 올바른 기권이 검색 실패로 집계된다.
            stats["무응답인데 근거 있음(버림)"] += 1
            spans = []

        if not spans and answerable:
            # 답이 있는 질문인데 근거 좌표를 못 잡은 경우만 제외한다. 이건 어댑터가
            # 실패한 것이라 채점에 넣으면 우리 진단이 아니라 변환을 재게 된다.
            stats["근거 못 잡아 제외"] += 1
            continue

        qa_rows.append({
            "qa_id": str(qid),
            "question": ann["question"],
            "answer_text": truth.get("content") or ann.get("answer") or "",
            "doc_id": f"{DOC_PREFIX}{gold_doc_ids[0]}",   # 표시용. 채점은 gold_spans 를 본다
            "gold_spans": spans,
            # 무응답 질문에는 **골드를 싣지 않는다.** 로더는 gold_spans 가 비면
            # positive_chunk_ids 로 폴백하는데(`korquad._gold_spans_of`), 여기서 문서 id 를
            # 실어 두면 폴백이 **문서 통째**를 골드로 만든다. 그러면 recall 이 계산돼
            # 무응답 경로(gold 없음 → recall=-1, 판정에서 별도 처리)를 못 타고, 올바른
            # 기권까지 '검색 실패'로 집계된다.
            "positive_chunk_ids": (
                [f"{DOC_PREFIX}{d}_0" for d in gold_doc_ids] if answerable else []
            ),
            "answer_exists": answerable,
            "source_qa": {"dataset": "dragonball-en", "cleaned_by": "tools/build_ragec_dataset.py"},
        })
        stats["무응답 질문" if not answerable else "답 있는 질문"] += 1
        key_rows.append({
            "qa_id": str(qid),
            "ragec_stage": ann["error_stage"].strip(),
            "ragec_category": ann["error_category"].strip(),
            "query_type": ann["query_type"].strip(),
        })
        stats["QA"] += 1

    for path, rows in ((out_qa, qa_rows), (out_key, key_rows)):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["span"] = sum(len(r["gold_spans"]) for r in qa_rows)
    stats["멀티문서 QA"] = sum(1 for r in qa_rows if len({s["doc_id"] for s in r["gold_spans"]}) > 1)
    return dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description="RAGEC + DragonBall → 파이프라인 입력")
    ap.add_argument("--ragec", required=True, help="RAGEC_annotations.csv")
    ap.add_argument("--docs", required=True, help="dragonball_docs.jsonl")
    ap.add_argument("--queries", required=True, help="dragonball_queries.jsonl")
    ap.add_argument("--out-corpus", default=DEFAULT_OUT_CORPUS)
    ap.add_argument("--out-qa", default=DEFAULT_OUT_QA)
    ap.add_argument("--out-key", default=DEFAULT_OUT_KEY)
    args = ap.parse_args()

    for path in (args.ragec, args.docs, args.queries):
        if not pathlib.Path(path).exists():
            print(f"[오류] 파일이 없습니다: {path}", file=sys.stderr)
            print("       받는 곳은 이 파일 상단 docstring 참고", file=sys.stderr)
            return 1

    stats = build(args.ragec, args.docs, args.queries,
                  args.out_corpus, args.out_qa, args.out_key)
    if not stats.get("QA"):
        print("[오류] 만들어진 QA 가 0건입니다.", file=sys.stderr)
        return 1

    print(f"문서 {stats.get('문서', 0):,}개 → {args.out_corpus}")
    print(f"QA   {stats.get('QA', 0):,}건 → {args.out_qa}")
    print(f"정답지 {stats.get('QA', 0):,}건 → {args.out_key}")
    print(f"\n  gold span {stats.get('span', 0):,}개 "
          f"(멀티문서 QA {stats.get('멀티문서 QA', 0):,}건)")
    for reason in ("답 있는 질문", "무응답 질문", "근거 좌표화", "근거 못 찾음",
                   "근거 못 잡아 제외", "무응답인데 근거 있음(버림)", "골드 문서 없음", "질의 못 찾음"):
        if stats.get(reason):
            print(f"  {reason:<16} {stats[reason]:>6,}")
    print(f"\n사용: SOURCE_TYPE=korquad SOURCE_URL={args.out_corpus} "
          f"EVAL_TAXONOMY_QA={args.out_qa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
