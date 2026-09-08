"""
tools/gen_nps_questions.py
data/nps_corpus.json(국민연금 사업장 실무안내) → 골든 QA 100건.
gen_tax_guide_questions.py 의 **다양성 확장 판본**.

## 왜 생성기를 또 만드나

시험지의 생성 조건은 벤치마크의 정의다. 한 파일에 코퍼스 스위치를 달면 세금가이드
실행이 어떤 프롬프트로 뽑혔는지가 사라진다. 그래서 코퍼스마다 생성기를 남긴다.

## 세금가이드 판본과 다른 점 — 질문 유형을 섞는다

세금가이드 골든셋은 전부 단일홉 사실 질문이었다. 그러면 두 시스템이 **검색 한 번으로
끝나는 문제만** 풀게 되어, 여러 근거를 모아야 하는 실패 유형이 시험지에서 통째로 빠진다.
우리 파이프라인이 probe 를 만들 때 쓰는 4분면 구성(agents/eval/probe_gen.py)을 골든셋에도
가져온다.

    단일홉 구체  60건   숫자·기한·요건·요율 하나 (기존 방식)
    멀티홉       30건   bridge 12 / comparison 10 / aggregation 8
    추상형       10건   개념·절차 요약. 정답 길이를 죄어 채점이 흐려지지 않게 한다

**aggregation 을 반드시 넣는 이유**: 진단 라벨 retrieval_incomplete_enumeration 은
qtype=aggregation 일 때만 확정되고, 아니면 예비로 강등돼 retrieval_low_rank 에 슬롯을
뺏긴다. 그 둘은 처방이 정반대다("검색 개수를 늘려라" vs "리랭커를 켜라"). 나열형 문항이
없으면 그 레버가 벤치마크에서 죽는다.

**무응답형(답이 없는 질문)은 넣지 않는다.** gold_contexts 가 비면 bench_golden_to_probes 와
bench_make_autorag_dataset 이 **둘 다 그 문항을 버린다** — AutoRAG 의 retrieval_gt 는
정의상 근거 doc_id 를 요구하기 때문이다. 시험지에 실을 수 없는 유형이라 제외한다.

## 검증 (영어 코퍼스 실험에서 실측으로 얻은 것들)

  1) **0회차 청킹에 들어가는 발췌만 채택**  — AutoRAG 의 corpus.parquet 은 0회차 청킹으로
     고정되고 retrieval_gt 는 "발췌를 통째로 담은 청크"로 표현된다. 경계에 걸친 발췌는
     변환에서 문항째 버려진다(영어 실측: 100건 중 20건 손실 → test 가 권장 하한 미달).
  2) **코퍼스 전체 1회 등장만 채택** — 좌표 변환이 첫 등장을 잡으므로, 반복되는 상투구를
     발췌로 쓰면 엉뚱한 위치가 골드가 되고 recall 이 **조용히** 틀린다.
  3) **공백 무시 매칭 후 원문 구간 반환** — PDF 는 문장 중간에서 줄을 바꾸는데 LLM 은 한
     줄로 인용한다. 글자 그대로 비교하면 멀쩡한 발췌가 대량 탈락한다.
  4) **출처 참조 질문 배제** — "위 자료에서" 류는 색인 전체를 상대로 단독으로 던져지면
     가리킬 대상이 없어 답이 정해지지 않는다.

사용법:
    python -m tools.gen_nps_questions --target=100
    python -m tools.gen_nps_questions --limit-windows=6 --target=4   # 프롬프트 점검용
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from core.llm_clients import openai_chat, OPENROUTER_BASE_URL  # noqa: E402

DEFAULT_CORPUS = "data/nps_corpus.json"
DEFAULT_OUT = "tools/nps_questions.json"

# 목차가 끝나고 제1장 본문이 시작되는 지점 — 실측 약 1,750.
# 앞부분(표지·CONTENTS·목차)은 사실 진술이 아니라 시험지로 부적합하다. 색인에는 그대로
# 남아 방해 문서 역할을 하므로 검색 난이도에는 오히려 도움이 된다.
SKIP_PREFIX_CHARS = 1750
# 윈도우가 넓을수록 문항 하나당 재료는 좋아지지만 후보 수가 줄어든다. 1400자에서는
# 후보가 147개뿐이라 100건을 채우기 전에 소진될 위험이 있었다(실측: 1차 68건에서 소진).
# 발췌는 어차피 512자 청크 안에 들어가야 하므로 1000자로도 재료는 충분하다.
WINDOW_CHARS = 1000

# 파이프 줄 비율이 이보다 높으면 표 덩어리로 보고 문항 출처에서 뺀다. 셀 구분자가 낀
# 문자열은 사람이 읽기에도 근거 문장이 아니다(코퍼스에는 남긴다 — 양쪽에 같은 노이즈다).
PIPE_LINE_RATIO_MAX = 0.3

GEN_MODEL = os.getenv("EXT_RAG_QA_GEN_MODEL", "openai/gpt-4o-mini")
# 동시 호출 수. 임베딩 경로(요청당 64건·동시 8)와 같은 수준으로 둔다.
GEN_CONCURRENCY = int(os.getenv("QA_GEN_CONCURRENCY", "8"))

# 정답 길이 상한. 서술형 정답은 채점 단위를 흐린다 — 문단을 정답으로 두면 어느 시스템이
# 맞혔는지 판정 자체가 주관적이 된다. 추상형은 더 죈다.
GROUND_TRUTH_MAX = 120
GROUND_TRUTH_MAX_ABSTRACT = 60

# 출처를 가리키는 질문. 문항은 색인 전체를 상대로 **단독으로** 던져지므로 "위 자료"가
# 가리킬 대상이 없다 — 답이 정해지지 않는 문항이 된다.
# 두 갈래를 잡는다.
#   (a) 출처 지칭   "위 자료에서", "본문에 따르면"
#   (b) **본문 라벨 지칭** "본문 A와 B 중 어느 쪽이", "두 본문에서"
#       (b)는 멀티홉 프롬프트에서만 나오고 실제로 나왔다(실측: "국민연금 관련 서식의
#       종류는 본문 A와 B 중 어느 쪽이 더 많은가?"). 시험지에서는 A·B 라는 구분 자체가
#       존재하지 않으므로 답이 정의되지 않는다.
_META_REF_RE = re.compile(
    r"(위 자료|아래 자료|본문에|이 자료|해당 자료|위 표|아래 표|제시된 자료|주어진 자료"
    r"|본문\s*[AB]|[AB]\s*와\s*[AB]|두 본문|첫 번째 본문|두 번째 본문|각 본문)")

# 질문 문장의 변주 축. 우리 probe_gen 이 쓰는 것과 같은 값을 쓴다(agents/eval/types.py).
# 시험지가 한 가지 말투로만 차면 "그 말투에 맞춘 검색"을 재게 된다.
PERSONAS = ["신입 사업장 담당자", "국민연금 실무 담당자"]
STYLES = {
    "web_search": "검색창에 치듯 짧은 명사 위주",
    "conversational": "동료에게 묻듯 자연스러운 구어체",
    "imperative": "'~를 알려줘' 처럼 지시하는 말투",
}

# 멀티홉은 **뺐다.** 코드는 남겨 둔다(gen_multi·multihop_pairs·_needs_both) — 재료가 좋은
# 코퍼스에서는 다시 켤 수 있고, 아래 실측이 그 판단의 근거다.
#
# 실측 성립률(국민연금 코퍼스):
#   초기(무작위 쌍)                     ~17%
#   KG 연결 쌍으로 교체                  변화 미미 — 쌍은 문제가 아니었다
#                                       (쌍 텍스트 자카드 중앙 0.073, 중복은 4%뿐)
#   검사기에서 정답을 가림                39%
#
# 39% 라도 100건 중 30건을 채우려면 쌍을 80회 가까이 소비해야 하고, 통과한 것 중에도
# 약한 문항이 섞였다("2025.12.1. 와 2026.6.21. 중 어느 입사일이 더 빠른가?" — 문서를
# 안 봐도 답한다). 들이는 비용 대비 시험지 품질 이득이 낮아 단일홉+추상형으로 간다.
#
# 세금가이드 실험이 단일홉만으로 유의미한 결과를 냈다는 점도 근거다(종합 88 대 79,
# 갈린 문항 7건). 멀티홉은 있으면 좋은 것이지 벤치마크 성립 조건이 아니다.
MIX = [
    ("single", None, 85),
    ("abstract", None, 15),
]

SYSTEM = (
    "당신은 국민연금 사업장 실무안내 책자에서 평가용 질문(QA)을 만드는 도구입니다. "
    "반드시 주어진 본문에 명시적으로 나온 사실만 사용하세요. 지어내지 마세요."
)

# 출력 순서가 곧 모델이 생각하는 순서다. 발췌를 **먼저** 쓰게 해야 원문에 닻을 내린 뒤
# 질문을 만든다. 질문부터 쓰게 하면 모델이 답을 먼저 정하고 그에 맞춰 문장을 **새로 짓는다**
# — 실측(100건 생성)에서 최다 탈락 사유가 "발췌 불일치" 36건이었고, 사례를 까 보니
# 원문의 숫자만 가져오고 나머지는 자기 문장이었다("2025년도 소득총액 신고 기간은 …입니다").
_COMMON_RULES = """작성 순서를 반드시 지키세요. gold_context 를 **먼저** 정하고, 그 문장만
보고 정답과 질문을 만듭니다.

1) gold_context: 본문에서 **연속된 구간을 글자 그대로 복사**합니다.
   - 한 글자도 바꾸지 마세요. 요약·의역·문장 다듬기·어미 변경 전부 금지입니다.
   - 새 문장을 쓰지 마세요. 본문에 그대로 있는 연속 구간이어야 합니다.
   - 나쁜 예: 본문이 "2025.01.01. ~ 2025.12.31. 기간의 소득총액" 인데
     "신고 기간은 2025.01.01.~2025.12.31.입니다" 라고 쓰는 것(문장을 새로 지었음).
   - 좋은 예: "2025.01.01. ~ 2025.12.31. 기간의 소득총액" (그대로 복사)
2) ground_truth: 그 발췌에서 뽑은 간결한 정답. 값·명칭·기한이며 문단이 아닙니다.
3) question: 그 정답을 묻는 질문. **다른 구절로는 답할 수 없을 만큼** 구체적일 것.
   질문만 따로 떼어 검색창에 넣어도 뜻이 통해야 합니다 — "위 자료에서", "본문에" 같은
   출처 지칭은 절대 쓰지 마세요(가리킬 대상이 없습니다).

말투: {style_desc}
독자: {persona}
"""

PROMPT_SINGLE = """아래는 국민연금 사업장 실무안내의 한 부분입니다. 이 본문에서 확인 가능한
구체적 사실(금액·기한·요율·일수·요건) 하나를 골라 질문 1개를 만드세요.

""" + _COMMON_RULES + """
본문에 위 조건을 만족하는 사실이 없으면 {{"skip": true}} 만 반환하세요.

본문:
---
{chunk}
---

JSON으로만 답하세요(이 순서 그대로): {{"gold_context": "...", "ground_truth": "...", "question": "..."}}
"""

_MULTI_GUIDE = {
    "bridge": "두 본문을 **연결해야만** 답이 나오는 질문. A에서 얻은 값·조건을 근거로 B의 사실을 묻는다.",
    "comparison": "두 본문의 사실을 **비교**해야 답이 나오는 질문(어느 쪽이 더 긴가/짧은가/높은가, 무엇이 다른가).",
    "aggregation": ("두 본문에 흩어진 항목을 **모두 모아야** 답이 완성되는 나열형 질문. "
                    "단, 정답은 **이 두 본문만으로 확정**되어야 한다 — 책자 전체를 세야 "
                    "답이 나오는 질문(예: '문서에 나온 총 개수')은 만들지 말 것."),
}

PROMPT_MULTI = """아래는 국민연금 사업장 실무안내의 서로 다른 두 부분입니다.

유형: {subtype} — {subtype_guide}

**두 본문이 모두 필요한 질문**을 1개 만드세요. 한쪽만 읽어도 답할 수 있으면 실패입니다.

질문은 이 두 본문을 못 본 사람에게 던져집니다. "본문 A", "두 본문 중" 같은 표현은
가리킬 대상이 없으므로 절대 쓰지 말고, 내용 자체로 지칭하세요
(예: "사업장 신규적용 시…와 상실 신고 시… 중 어느 기한이 더 짧은가?").

""" + _COMMON_RULES + """gold_context_a 는 본문 A에서, gold_context_b 는 본문 B에서 각각 그대로 복사합니다.

두 본문을 엮을 만한 접점이 없으면 {{"skip": true}} 만 반환하세요.

본문 A:
---
{chunk_a}
---

본문 B:
---
{chunk_b}
---

JSON으로만 답하세요:
{{"gold_context_a": "...", "gold_context_b": "...", "ground_truth": "...", "question": "..."}}
"""

PROMPT_ABSTRACT = """아래는 국민연금 사업장 실무안내의 한 부분입니다. 이 본문이 설명하는
**절차나 개념**에 대해 질문 1개를 만드세요(단일 수치를 묻지 말 것).

""" + _COMMON_RULES + f"""
- ground_truth 는 {GROUND_TRUTH_MAX_ABSTRACT}자 이내로. 길어질 답만 가능하면 {{{{"skip": true}}}}.

본문:
---
{{chunk}}
---

JSON으로만 답하세요(이 순서 그대로): {{{{"gold_context": "...", "ground_truth": "...", "question": "..."}}}}
"""


def _locate_verbatim(quote: str, chunk: str) -> str | None:
    """인용문을 청크에서 찾아 **원문 그대로의 구간**을 돌려준다. 못 찾으면 None.

    공백을 무시하고 찾는 이유: PDF 는 문장 중간에서 줄을 바꾸는데 LLM 은 자연스러운 한
    줄로 인용한다. 글자 그대로 비교하면 멀쩡한 발췌가 대량 탈락한다(영어 코퍼스 실측:
    12청크 스모크에서 탈락 8건이 전원 이 사유였고, 고친 뒤 유효율 2/12 → 10/12).

    **정규화본이 아니라 원문 구간을 돌려주는 것이 핵심이다.** gold_contexts 는
    bench_golden_to_probes(text.find) 와 bench_make_autorag_dataset(gold in chunk)
    양쪽에서 **정확한 부분 문자열**로 쓰인다. 정규화한 문자열을 실으면 두 도구가 전부
    그 문항을 버린다."""
    flat, index_map = [], []
    prev_space = True
    for i, ch in enumerate(chunk):
        if ch.isspace():
            if not prev_space:
                flat.append(" ")
                index_map.append(i)
            prev_space = True
        else:
            flat.append(ch)
            index_map.append(i)
            prev_space = False
    flat_quote = " ".join((quote or "").split())
    if not flat_quote:
        return None
    at = "".join(flat).find(flat_quote)
    if at < 0:
        return None
    return chunk[index_map[at]:index_map[at + len(flat_quote) - 1] + 1]


def _is_table_heavy(text: str) -> bool:
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return True
    piped = sum(1 for line in lines if line.strip().startswith("|"))
    return piped / len(lines) > PIPE_LINE_RATIO_MAX


def multihop_pairs(path: str, limit: int) -> tuple[list[tuple[str, str]], int]:
    """멀티홉용 청크 쌍 — **우리 probe_gen 과 같은 방식**으로 고른다.

    처음엔 윈도우를 무작위로 짝지었다. 그러면 접점이 없는 두 구절이 붙고, LLM 은 스키마를
    채우려고 단일홉 질문 + 무관한 두 번째 근거를 만들어낸다(실측: 스모크 멀티홉 3건 중
    2건이 그랬다). 원인이 프롬프트가 아니라 **쌍 선택**이었다.

    knowledge_graph.build_graph 는 각 청크를 top-k 최근접 이웃하고만 연결한다. 그 파일
    주석에 근거가 있다 — 절대 임계값을 쓰면 "무관 쌍끼리도 cos 중앙값이 0.46 이라 후보의
    78%가 노이즈"였고 그것이 억지 멀티홉의 근원이었다. 같은 함수를 그대로 쓴다.

    0회차 청크(512자)를 쓰는 이유가 하나 더 있다: 발췌가 그 청크 안에서 나오므로 "0회차
    청킹에 들어가는가" 검사를 자동으로 통과한다(경계에 걸릴 수가 없다).
    """
    from core.schema import Chunk
    from agents.eval import knowledge_graph
    from agents.index.qdrant_store import embed_batch
    from tools.bench_make_autorag_dataset import (
        build_chunks, DEFAULT_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    )

    # 단일홉 윈도우와 **같은 기준으로** 후보를 좁힌다. 안 하면 앞부분 목차 청크가 딸려
    # 들어오는데, 목차끼리는 서로 매우 비슷해서 KG 가 강하게 연결한다 — 실측에서 상위
    # 쌍이 전부 "CONTENTS ↔ 부록 목록"이었다. 질문을 만들 수 없는 재료다.
    rows = build_chunks(path, DEFAULT_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    kept = [r for r in rows
            if r["start_end_idx"][0] >= SKIP_PREFIX_CHARS
            and not _is_table_heavy(r["contents"])]
    texts = [r["contents"] for r in kept]
    print(f"[gen-qa] 멀티홉 후보 청크 {len(texts)}개"
          f" (전체 {len(rows)} · 목차/표 제외 {len(rows) - len(texts)})")
    print(f"[gen-qa] 임베딩 계산 중…")
    vectors = embed_batch(texts)
    chunks = [
        Chunk(chunk_id=f"c{i:05d}", doc_id="nps", text=t, embedding=v)
        for i, (t, v) in enumerate(zip(texts, vectors))
    ]
    graph = knowledge_graph.build_graph(chunks)
    by_id = {c.chunk_id: c.text for c in chunks}
    pairs = [(by_id[a], by_id[b]) for a, b in knowledge_graph.connected_pairs(graph, n=2)]
    return pairs[:limit] if limit else pairs, len(texts)


def baseline_chunks(path: str) -> list[str]:
    """0회차 설정으로 자른 청크 — 발췌가 시험지가 될 수 있는지 판정하는 기준.

    청킹 파라미터를 여기서 다시 적지 않고 bench_make_autorag_dataset 에서 가져오는 이유:
    한쪽만 바뀌면 "생성기는 통과시켰는데 변환기는 버리는" 상태가 된다."""
    from tools.bench_make_autorag_dataset import (
        build_chunks, DEFAULT_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    )
    rows = build_chunks(path, DEFAULT_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
    return [row["contents"] for row in rows]


def load_windows(path: str, size: int, skip: int) -> tuple[list[str], str, dict]:
    """코퍼스 → (후보 윈도우, 전문, 통계). 전문은 발췌 유일성 검사에 쓴다."""
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not docs:
        sys.exit(f"[gen-qa] 코퍼스가 비었습니다: {path}")
    full = "\n".join(d["text"] for d in docs)

    windows, stats = [], {"total": 0, "short": 0, "table": 0}
    for doc in docs:
        text = doc["text"][skip:]
        for i in range(0, len(text), size):
            piece = text[i:i + size].strip()
            stats["total"] += 1
            if len(piece) <= 200:
                stats["short"] += 1
            elif _is_table_heavy(piece):
                stats["table"] += 1
            else:
                windows.append(piece)
    return windows, full, stats


def _validate(question: str, ground_truth: str, excerpts: list[str],
              corpus: str, base_chunks: list[str], max_gt: int) -> str:
    """공통 검증. 통과하면 "" 를 돌려준다."""
    if _META_REF_RE.search(question):
        return "출처 참조"
    # 정답이 질문 안에 이미 들어 있으면 시험 문항이 아니다. 표에서 뽑을 때 자주 나온다
    # (실측: "소득총액이 2,000,000인 경우의 금액은?" → 정답 "2,000,000",
    #  "소급분 연금보험료 분할 납부에 대한 법적 근거는?" → 정답 "소급분 연금보험료 분할 납부").
    # 검색이 필요 없는 문항이라 두 시스템을 가르지 못하고, 점수만 부풀린다.
    flat_q = " ".join(question.split())
    flat_a = " ".join(ground_truth.split())
    if flat_a and flat_a in flat_q:
        return "정답이 질문에 포함됨"
    if len(ground_truth) > max_gt:
        return "정답 서술형"
    for gc in excerpts:
        if corpus.count(gc) != 1:
            return f"발췌 {corpus.count(gc)}회 등장"
        if not any(gc in base for base in base_chunks):
            return "청크 경계에 걸림"
    return ""


# 멀티홉 진위 검사 프롬프트. **필수다** — 실측(10건 스모크)에서 멀티홉 3건 중 2건이
# 가짜였다. LLM 이 단일홉 질문을 만든 뒤 스키마를 채우려고 무관한 문장을 두 번째 근거로
# 붙인다(예: "협정발효일이 가장 빠른 나라?" 의 근거1 이 "대상자와 소득총액 신고대상자 구축").
#
# 왜 그냥 두면 안 되나: gold_contexts 는 AutoRAG 의 retrieval_gt 로 변환될 때 발췌마다
# 내부 리스트 하나 = **AND 조건**이 된다. 무관한 근거가 끼면 두 시스템 모두 "찾을 이유가
# 없는 청크"를 못 찾아 검색 지표가 깎인다 — 시험지가 약해지는 게 아니라 **틀린다**.
_VERIFY_SYSTEM = "당신은 질문과 근거를 대조해 답변 가능 여부만 판정하는 도구입니다."
# **정답을 보여주지 않는다.** 처음엔 정답을 같이 넘겼는데, 그러면 비교형 질문
# ("A와 B 중 어느 기한이 더 짧은가?" · 정답 "탈퇴일")에서 한쪽 근거에 그 단어만 있어도
# 모델이 "충분하다"고 답한다 — 실제로는 비교 대상이 없어 답할 수 없는데도. 그 결과
# 멀쩡한 멀티홉이 대량 탈락했다(실측: 가짜 멀티홉 45건 중 상당수가 이 오판이었다).
# 정답을 가리고 "이 근거만 보고 답해보라"고 시켜야 진짜 판별이 된다.
_VERIFY_PROMPT = """아래 근거 **하나만** 보고 다음 질문에 답할 수 있습니까?

질문: {question}

근거:
---
{excerpt}
---

판정 기준:
- 이 근거 안에 질문이 요구하는 정보가 **전부** 있으면 true.
- 비교·합산·연결이 필요한데 한쪽 값만 있으면 false(다른 쪽을 모르면 답할 수 없다).
- 근거가 질문과 무관하면 false.

JSON으로만: {{"sufficient": true 또는 false}}
"""


def _needs_both(question: str, ground_truth: str, excerpts: list[str],
                api_key: str) -> bool:
    """각 근거를 **혼자** 줬을 때 답이 나오는지 본다. 하나라도 혼자 충분하면 멀티홉이 아니다.

    한쪽씩 따로 묻는 이유: 둘을 함께 보여주고 "둘 다 필요한가" 를 물으면 모델이 순응해서
    거의 항상 예라고 답한다. 혼자 줬을 때 답하게 만들어야 판별력이 생긴다."""
    for excerpt in excerpts:
        try:
            raw = openai_chat(
                _VERIFY_SYSTEM,
                _VERIFY_PROMPT.format(question=question, excerpt=excerpt),
                GEN_MODEL, json_mode=True, api_key=api_key,
                base_url=OPENROUTER_BASE_URL, max_output_tokens=60, tag="qa-verify")
        except Exception:                        # noqa: BLE001 — 판정 불가는 보수적으로 버린다
            return False
        if not raw:
            return False                      # 판정 불가 — 보수적으로 버린다
        try:
            if json.loads(raw).get("sufficient") is True:
                return False
        except json.JSONDecodeError:
            return False
    return True


# 100건 생성은 15분쯤 걸리고 그동안 LLM 호출이 수백 번이다. 네트워크가 한 번만 끊겨도
# 예외가 위로 올라가 **그때까지 만든 문항이 통째로 날아간다**(실측: DNS 실패로 22건에서
# 죽었고 파일에 아무것도 안 써졌다). 전송 계층 실패는 시험지 품질과 무관하므로 삼킨다.
_CHAT_RETRIES = 3
_CHAT_BACKOFF_SEC = 10


def _chat(prompt: str, api_key: str) -> dict | None:
    for attempt in range(1, _CHAT_RETRIES + 1):
        try:
            raw = openai_chat(SYSTEM, prompt, GEN_MODEL, json_mode=True, api_key=api_key,
                              base_url=OPENROUTER_BASE_URL, max_output_tokens=700,
                              tag="qa-gen")
        except Exception as exc:                 # noqa: BLE001 — provider 예외 종류가 다양하다
            if attempt == _CHAT_RETRIES:
                print(f"    ! 호출 실패({exc.__class__.__name__}) — 이 문항 건너뜀")
                return None
            print(f"    재시도 {attempt}/{_CHAT_RETRIES - 1} ({exc.__class__.__name__})")
            time.sleep(_CHAT_BACKOFF_SEC * attempt)
            continue
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


def gen_single(window: str, kind: str, style: str, persona: str, api_key: str,
               corpus: str, base_chunks: list[str]) -> tuple[dict | None, str]:
    tmpl = PROMPT_ABSTRACT if kind == "abstract" else PROMPT_SINGLE
    max_gt = GROUND_TRUTH_MAX_ABSTRACT if kind == "abstract" else GROUND_TRUTH_MAX
    obj = _chat(tmpl.format(chunk=window, style_desc=STYLES[style], persona=persona), api_key)
    if obj is None:
        return None, "무응답/파싱실패"
    if obj.get("skip"):
        return None, "skip"
    q, gt, gc = obj.get("question"), obj.get("ground_truth"), obj.get("gold_context")
    if not (q and gt and gc):
        return None, "필드 결손"
    located = _locate_verbatim(gc, window)
    if located is None:
        return None, "발췌 불일치"
    why = _validate(q.strip(), gt.strip(), [located], corpus, base_chunks, max_gt)
    if why:
        return None, why
    return {"question": q.strip(), "ground_truth": gt.strip(),
            "gold_contexts": [located], "qtype": None,
            "metadata": {"kind": kind, "style": style, "persona": persona}}, ""


def gen_multi(win_a: str, win_b: str, subtype: str, style: str, persona: str,
              api_key: str, corpus: str, base_chunks: list[str]) -> tuple[dict | None, str]:
    obj = _chat(PROMPT_MULTI.format(
        chunk_a=win_a, chunk_b=win_b, subtype=subtype,
        subtype_guide=_MULTI_GUIDE[subtype],
        style_desc=STYLES[style], persona=persona), api_key)
    if obj is None:
        return None, "무응답/파싱실패"
    if obj.get("skip"):
        return None, "skip"
    q, gt = obj.get("question"), obj.get("ground_truth")
    ga, gb = obj.get("gold_context_a"), obj.get("gold_context_b")
    if not (q and gt and ga and gb):
        return None, "필드 결손"
    la, lb = _locate_verbatim(ga, win_a), _locate_verbatim(gb, win_b)
    if la is None or lb is None:
        return None, "발췌 불일치"
    if la == lb:
        return None, "두 발췌가 같음"       # 한쪽만 읽어도 답이 되는 문항
    why = _validate(q.strip(), gt.strip(), [la, lb], corpus, base_chunks, GROUND_TRUTH_MAX)
    if why:
        return None, why
    if not _needs_both(q.strip(), gt.strip(), [la, lb], api_key):
        return None, "한쪽 근거로 충분(가짜 멀티홉)"
    return {"question": q.strip(), "ground_truth": gt.strip(),
            "gold_contexts": [la, lb], "qtype": subtype,
            "metadata": {"kind": "multi", "style": style, "persona": persona}}, ""


def main() -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()

    ap = argparse.ArgumentParser(description="국민연금 실무안내 → 골든 QA(유형 혼합)")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--skip-prefix", type=int, default=SKIP_PREFIX_CHARS)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--limit-windows", type=int, default=0, help="점검용 소량 실행")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("[gen-qa] OPENROUTER_API_KEY 가 없습니다 — .env 를 확인하세요.")

    rng = random.Random(args.seed)
    windows, corpus, stats = load_windows(args.corpus, WINDOW_CHARS, args.skip_prefix)
    base_chunks = baseline_chunks(args.corpus)
    if args.limit_windows:
        windows = windows[:args.limit_windows]

    print(f"[gen-qa] 코퍼스 {args.corpus} · 본문 시작 {args.skip_prefix}")
    print(f"[gen-qa] 0회차 청킹 기준 {len(base_chunks)}청크 (발췌 수용 판정용)")
    print(f"[gen-qa] 윈도우 {stats['total']}개 → 후보 {len(windows)}개"
          f" (짧아서 제외 {stats['short']} · 표 덩어리 제외 {stats['table']})")

    # 목표 비율을 target 에 맞춰 스케일한다.
    plan: list[tuple[str, str | None]] = []
    for kind, subtype, count in MIX:
        n = max(1, round(count * args.target / 100))
        plan += [(kind, subtype)] * n
    plan = plan[:args.target]
    rng.shuffle(plan)
    print(f"[gen-qa] 목표 {len(plan)}건 — " + " · ".join(
        f"{k}{'/'+s if s else ''} {plan.count((k, s))}" for k, s, _ in MIX))

    # 멀티홉 쌍은 KG 로 미리 뽑아 둔다(임베딩 1회). 필요 수의 4배를 잡는 이유: 가짜
    # 멀티홉 검사와 발췌 검증에서 상당수가 탈락하므로 여유가 없으면 유형별 목표를 못 채운다.
    n_multi = sum(1 for k, _ in plan if k == "multi")
    pair_pool, n_base = ([], 0)
    if n_multi:
        pair_pool, n_base = multihop_pairs(args.corpus, limit=n_multi * 4)
        rng.shuffle(pair_pool)
        print(f"[gen-qa] KG 연결 쌍 {len(pair_pool)}개 확보 (멀티홉 목표 {n_multi}건)")

    # 병렬로 던진다. 문항 하나가 LLM 왕복 1회(멀티홉은 검증까지 3회)라 순차로 돌면
    # 100건에 10분이 넘는다 — 임베딩 경로가 동시 8로 도는 것과 같은 이유로 여기도 푼다.
    #
    # 윈도우를 **미리** 문항 수보다 넉넉히 배정하는 이유: 탈락률이 20~30% 라 목표만큼만
    # 던지면 모자란다. 남는 결과는 목표 수에서 잘라낸다(순서는 아래에서 다시 고정한다).
    from concurrent.futures import ThreadPoolExecutor

    results: list[dict] = []
    seen: set[str] = set()
    rejects: dict[str, int] = {}
    order = list(range(len(windows)))
    rng.shuffle(order)
    cursor = 0
    pair_cursor = 0

    def _one(job):
        kind, subtype, widx, style, persona = job
        if kind == "multi":
            win_a, win_b = pair_pool[widx]
            return gen_multi(win_a, win_b, subtype, style, persona,
                             api_key, corpus, base_chunks)
        return gen_single(windows[widx], kind, style, persona,
                          api_key, corpus, base_chunks)

    jobs = []
    for idx, (kind, subtype) in enumerate(plan * 2):        # 2배로 던져 탈락분을 흡수
        if kind == "multi":
            if pair_cursor >= len(pair_pool):
                continue
            widx = pair_cursor
            pair_cursor += 1
        else:
            if cursor >= len(order):
                continue
            widx = order[cursor]
            cursor += 1
        jobs.append((kind, subtype, widx, rng.choice(list(STYLES)), rng.choice(PERSONAS)))

    print(f"[gen-qa] 병렬 {GEN_CONCURRENCY} 로 {len(jobs)}회 시도")
    with ThreadPoolExecutor(max_workers=GEN_CONCURRENCY) as pool:
        for (kind, subtype, *_), (item, why) in zip(jobs, pool.map(_one, jobs)):
            if len(results) >= len(plan):
                break
            if item and item["question"] not in seen:
                seen.add(item["question"])
                results.append(item)
                tag = f"{kind}/{subtype}" if subtype else kind
                print(f"  [{len(results)}/{len(plan)}] ({tag}) {item['question'][:52]}")
            else:
                rejects[why or "질문 중복"] = rejects.get(why or "질문 중복", 0) + 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[gen-qa] {len(results)}건 → {args.out}")
    kinds: dict[str, int] = {}
    for r in results:
        k = r["qtype"] or r["metadata"]["kind"]
        kinds[k] = kinds.get(k, 0) + 1
    print("[gen-qa] 유형 분포:", kinds)
    if rejects:
        print("[gen-qa] 탈락 사유:")
        for why, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
            print(f"    {why}: {n}건")
    print()
    print("다음: 20건쯤 눈으로 검수한 뒤")
    print(f"  python -m tools.bench_split_golden --input={args.out} --outdir=bench/nps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
