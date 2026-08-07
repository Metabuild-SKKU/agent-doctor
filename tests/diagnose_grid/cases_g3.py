"""
tests/diagnose_grid/cases_g3.py
G3 청킹 부분격자.

케이스는 상황에서 출발한다. 현실에서 있을 법한 실패 상황을 적고, LABELS.md 를 보며
'이 상황에서 무엇을 고쳐야 하나'로 기대 라벨을 정한다. 진단 코드의 판별 조건이나 라벨 사이
우선순위는 보지 않는다 — 그걸 보고 기대를 적으면 코드가 자기 자신을 채점하게 된다.
코드가 다른 라벨을 내면 기대를 고치지 않고 known_gap 으로 남긴다.

문서·질문·정답은 KorQuAD(data/corpus.jsonl, data/qa_pairs.jsonl)에서 가져온다. 진단
파이프라인이 실제로 쓰는 데이터라 좌표계가 그대로 맞고, 질문·정답을 사람이 만들었다.
케이스는 qa_id 만 고르고 답변·검색 결과만 정한다.

gold_span_mode
  exact : 정답 텍스트가 원문에서 차지하는 구간. 청킹 기하를 재려면 이쪽이어야 한다.
  chunk : positive_chunk 전체 범위. KorQuAD 원래 라벨이고 실제 파이프라인이 쓰는 값이다.

answer 는 검색 결과로 생성한 답, oracle_answer 는 골드 컨텍스트만 주고 생성한 답이다.
가급적 실제 문장으로 적는다 — 정답을 그대로 복사하면 oracle_f1 이 1.0 으로 굳어 생성 실패(B)가
안 열린다. 다만 정답이 긴 KorQuAD 항목은 오라클이 요약형이면 char-F1 이 문턱 아래로 떨어져
(실측: 324자 정답에 55자 요약 → 0.29) 오라클 실패로 잡힌다. 그 경우만 Answer.GOLD_FULL 로 둔다.
심판 의미축이 붙으면 요약형도 통과하므로 그때 실제 문장으로 바꿀 수 있다.

심판 값(RAGAS·AspectCritic)은 케이스에 적지 않는다. 손으로 적으면 답변과 어긋난 값을
넣을 수 있고, 그러면 진단의 임계값 비교만 검사하게 된다. 파이프라인이 심판 LLM 으로
뽑아 채운다.

config 는 실제 파이프라인 기본값(BASELINE)을 쓰고, 상황이 요구하는 값만 벗어난다.
"""
from tests.diagnose_grid.builder import Answer, Case, Doc

# 실제 파이프라인 기본값 — agents/index/agent.py (chunk_size 600 / overlap 80 / top_k 5)
BASELINE = dict(chunk_strategy="fixed", chunk_size=600, chunk_overlap=80,
                search_mode="dense", reranked=False, mmr_applied=False)

NIMITZ = Doc("doc_ace9d8c1ce5d", korquad="doc_ace9d8c1ce5d")   # 체스터 니미츠 (5053자)
P45 = Doc("doc_11e4d4a95225", korquad="doc_11e4d4a95225")      # P45 (2634자)
CHEOLSAN = Doc("doc_faadd815d063", korquad="doc_faadd815d063")  # 철산군 (2440자)


CASES = [
    Case(
        id="g3_chunk_much_larger_than_evidence",
        situation="니미츠 문서에서 임관 계급을 묻는다. 정답은 두 글자인데 청크가 600자라 "
                  "생애 전반이 통째로 딸려온다. 검색은 그 청크를 가져왔고 답은 틀렸다.",
        # 판단: 청크가 근거보다 훨씬 크다 → 청크를 줄여야 한다
        docs=[NIMITZ], korquad_qa="61137", gold_span_mode="exact", **BASELINE,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[3, 2, 4, 1, 5],
        answer="니미츠는 1907년에 중위로 임관했다",          # 계급이 틀렸다
        oracle_answer="1907년 소위로 임관하였다",              # 근거만 주면 맞힌다
        assert_derived={"recall_at_k": 1.0, "evidence_density": "<0.2"},
        expect={"A": "chunking_underchunking"},
        needs_judge="chunking_underchunking 만 심판의 context_precision 을 함께 요구한다 — "
                    "심판이 없으면 근거 밀도가 아무리 낮아도 C 슬롯 롤업으로 떨어진다.",
    ),

    Case(
        id="g3_evidence_split_at_boundary",
        situation="P45 문서에서 이 낱말이 사전에 실린 경위를 묻는다. 설명이 300자 남짓인데 "
                  "청크 경계에 걸쳐 잘렸다. 양쪽 청크 다 검색됐고 답은 앞부분만 말했다.",
        # 판단: 근거가 경계에서 잘렸다 → 겹침이나 청크 크기를 늘려야 한다
        docs=[P45], korquad_qa="19495", gold_span_mode="exact", **BASELINE,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[1, 2, 0, 3, 4],
        answer="장난으로 만들어진 낱말이라 실용 용어가 아니다",   # 사전 수록 경위가 빠졌다
        oracle_answer=Answer.GOLD_FULL,          # 정답이 324자라 요약하면 char-F1 이 문턱 아래
        assert_derived={"recall_at_k": 1.0, "boundary_split": ">0", "oversized_count": 0},
        expect={"A": "chunking_context_mismatch"},
    ),

    Case(
        id="g3_evidence_longer_than_chunk",
        situation="철산군 문서에서 연혁 전체를 묻는다. 근거가 800자에 걸쳐 있어 600자 청크 "
                  "하나에 담기지 않는다. 검색은 관련 청크를 다 가져왔고 답은 일부 시기를 빠뜨렸다.",
        # 판단: 근거가 청크 하나에 안 들어간다 → 청크 크기를 키워야 한다
        docs=[CHEOLSAN], korquad_qa="85143", gold_span_mode="chunk", **BASELINE,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[1, 2, 0, 3, 4],
        answer="고구려와 발해를 거쳐 고려의 영역이 되었다",       # 고려 이후 연혁이 빠졌다
        oracle_answer=Answer.GOLD_FULL,          # 정답이 492자라 요약하면 char-F1 이 문턱 아래
        assert_derived={"recall_at_k": 1.0, "oversized_count": ">0"},
        expect={"A": "chunking_overchunking"},
    ),

    Case(
        id="g3_long_evidence_partially_retrieved",
        situation="같은 철산군 연혁을 묻는데, 근거가 걸쳐 있는 청크 중 뒷부분만 검색됐다. "
                  "답은 앞 시기를 통째로 빠뜨렸다.",
        # 판단: 근거가 청크 하나에 안 들어가서 조각났다 → 청크 크기를 키워야 한다.
        #       검색 개수를 늘려도 잘린 조각을 더 가져올 뿐이다.
        docs=[CHEOLSAN], korquad_qa="85143", gold_span_mode="chunk", **BASELINE,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[2, 3, 4],
        wide_ranking=[2, 3, 4, 1, 0],
        answer="1413년에 철산군으로 이름이 바뀌었다",           # 앞 시기가 통째로 빠졌다
        oracle_answer=Answer.GOLD_FULL,          # 정답이 492자라 요약하면 char-F1 이 문턱 아래
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "chunking_overchunking"},
        known_gap="현재 진단은 retrieval_low_rank 를 낸다. A 슬롯에서 청킹 라벨은 "
                  "qtype=bridge 일 때만 채택되고, 그 외에는 순위·코퍼스 라벨이 선점한다.",
    ),

    Case(
        id="g3_evidence_across_markdown_sections",
        situation="P45 문서를 마크다운 섹션 단위로 청킹했다. 근거가 섹션 경계에 걸쳐 있고 "
                  "양쪽 청크가 다 검색됐는데 답은 틀렸다.",
        # 판단: 근거가 섹션 경계에서 잘렸다 → 겹침이나 청크 크기를 늘려야 한다
        docs=[P45], korquad_qa="19495", gold_span_mode="exact",
        chunk_strategy="markdown_recursive", chunk_size=600, chunk_overlap=80,
        search_mode="dense", reranked=False, mmr_applied=False,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[0, 1, 2, 3, 4],
        answer="사전 편집자가 실수로 수록한 것이다",             # 근거와 어긋난다
        oracle_answer=Answer.GOLD_FULL,          # 정답이 324자라 요약하면 char-F1 이 문턱 아래
        assert_derived={"missed_count": 0},
        expect={"A": "chunking_context_mismatch"},
        known_gap="섹션 경계의 좌표 틈 때문에 gold 청크를 다 집었는데도 span_recall 이 0 이 "
                  "되고 구체 라벨이 전부 침묵한다 (#100).",
    ),

    Case(
        id="g3_evidence_document_not_collected",
        situation="니미츠의 임관 계급을 묻는데 그 대목이 담긴 문서 조각을 아직 수집하지 않았다. "
                  "검색은 무관한 청크만 가져왔고 모델은 근거 없이 답을 지어냈다.",
        # 판단: 자료가 없다 → 문서를 수집해야 한다. 근거가 없으면 기권했어야 한다.
        docs=[NIMITZ], korquad_qa="61137", gold_span_mode="exact", **BASELINE,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[5, 6, 7],
        corpus_exclude=[2, 3],                   # 근거가 걸친 청크를 코퍼스에서 제외
        answer="니미츠는 대위로 임관했다",                      # 근거가 없는데 지어냈다
        oracle_answer="1907년 소위로 임관하였다",
        assert_derived={"recall_at_k": "<1"},
        expect={"D": "corpus_gap", "B": "generation_abstention_failure"},
    ),

    Case(
        id="g3_evidence_fits_and_answer_correct",
        situation="니미츠 문서에서 진주만 공습 날짜를 묻는다. 근거가 청크 하나에 온전히 담겼고 "
                  "검색이 그 청크를 가져왔으며 답도 맞았다.",
        # 판단: 고칠 게 없다
        docs=[NIMITZ], korquad_qa="73135", gold_span_mode="exact", **BASELINE,
        question="", gold_spans=[], span_grounding=None, ground_truth="",
        qtype=None, answer_exists=None,
        retrieved=[4, 3, 5, 2, 6],
        answer="1941년 12월 7일에 일어났다",                    # 정답
        oracle_answer="일본의 진주만 공습은 1941년 12월 7일에 일어났다",
        assert_derived={"recall_at_k": 1.0},
        expect={},
    ),
]
