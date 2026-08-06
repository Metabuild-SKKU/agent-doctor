"""
tests/diagnose_grid/cases_g3.py
G3 청킹 부분격자.

케이스는 상황에서 출발한다. 현실에서 있을 법한 실패 상황을 적고, LABELS.md 를 보며
'이 상황에서 무엇을 고쳐야 하나'로 기대 라벨을 정한다. 진단 코드의 판별 조건이나 라벨 사이
우선순위는 보지 않는다 — 그걸 보고 기대를 적으면 코드가 자기 자신을 채점하게 된다.
코드가 다른 라벨을 내면 기대를 고치지 않고 known_gap 으로 남긴다.

문서는 docs/ 아래 실제 글이고, 정답 근거는 `[[gold:이름]]` 마커로 표시한다. 케이스는 좌표를
적지 않고 마커 이름만 고른다 — 문서를 고치거나 청크 크기를 바꿔도 좌표가 따라간다.
config 는 실제 파이프라인 기본값(BASELINE)을 쓰고, 상황이 요구하는 값만 벗어난다.
"""
from tests.diagnose_grid.builder import Answer, Case, Doc

# 실제 파이프라인 기본값 — agents/index/agent.py (chunk_size 600 / overlap 80 / top_k 5)
BASELINE = dict(chunk_strategy="fixed", chunk_size=600, chunk_overlap=80,
                search_mode="dense", reranked=False, mmr_applied=False)

HR = Doc("hr", file="hr_policy.md")            # 마커: annual_days / carryover / apply
MANUAL = Doc("manual", file="device_manual.md")  # 마커: filter_steps / warranty


CASES = [
    Case(
        id="g3_chunk_much_larger_than_evidence",
        situation="인사 규정에서 연차 일수를 묻는다. 답은 한 문장인데 청크가 600자라 총칙·"
                  "이월 규정까지 통째로 딸려온다. 검색은 그 청크를 가져왔고 답은 틀렸다.",
        # 판단: 청크가 근거보다 훨씬 크다 → 청크를 줄여야 한다
        docs=[HR], gold_marks=["annual_days"], **BASELINE,
        question="연차 휴가는 며칠이 부여되나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="입사 1년이 지난 시점부터 15일이 부여된다",
        qtype=None, answer_exists=None,
        retrieved=[0, 1, 2],
        answer="입사와 동시에 연차 15일이 부여되며 매년 자동으로 갱신된다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.2, "context_precision": 0.3},
        assert_derived={"recall_at_k": 1.0, "evidence_density": "<0.2"},
        expect={"A": "chunking_underchunking"},
    ),

    Case(
        id="g3_evidence_split_at_boundary",
        situation="인사 규정에서 미사용 연차 처리를 묻는다. 이월·소멸·정산이 한 덩어리로 설명돼 "
                  "있는데 그 설명이 청크 경계에 걸쳐 잘렸다. 양쪽 청크 다 검색됐고 답은 틀렸다.",
        # 판단: 근거가 경계에서 잘렸다 → 겹침이나 청크 크기를 늘려야 한다
        docs=[HR], gold_marks=["carryover"], **BASELINE,
        question="쓰지 않은 연차는 어떻게 되나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="다음 해 3월 31일까지 이월할 수 있고 그 뒤에는 소멸하나, 회사가 사용을 촉진하지 않았으면 수당으로 정산한다",
        qtype=None, answer_exists=None,
        retrieved=[0, 1, 2],
        answer="미사용 연차는 다음 해 3월 31일까지 이월할 수 있다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.25},      # 소멸·정산 조건이 빠졌다
        assert_derived={"recall_at_k": 1.0, "boundary_split": ">0", "oversized_count": 0},
        expect={"A": "chunking_context_mismatch"},
    ),

    Case(
        id="g3_evidence_longer_than_chunk",
        situation="정수기 매뉴얼에서 필터 교체 절차를 묻는다. 절차 설명이 600자를 넘어 청크 "
                  "하나에 담기지 않는다. 검색은 관련 청크를 다 가져왔고 답은 일부 단계를 빠뜨렸다.",
        # 판단: 근거가 청크 하나에 안 들어간다 → 청크 크기를 키워야 한다
        docs=[MANUAL], gold_marks=["filter_steps"], **BASELINE,
        question="정수기 필터는 어떻게 교체하나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="전원을 뽑고 급수 밸브를 잠근 뒤 커버를 분리해 필터를 돌려 빼고 새 필터를 끼운 다음 누수를 확인하고 물을 흘려보낸다",
        qtype=None, answer_exists=None,
        retrieved=[0, 1, 2],
        answer="커버를 열고 필터를 돌려서 빼낸 뒤 새 필터를 끼우면 된다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.3},       # 전원·밸브·누수 확인 단계가 빠졌다
        assert_derived={"recall_at_k": 1.0, "oversized_count": ">0"},
        expect={"A": "chunking_overchunking"},
    ),

    Case(
        id="g3_long_evidence_partially_retrieved",
        situation="같은 매뉴얼에서 필터 교체 절차를 묻는데, 절차가 걸쳐 있는 두 청크 중 "
                  "뒷부분만 검색됐다. 답은 앞 단계를 통째로 빠뜨렸다.",
        # 판단: 근거가 청크 하나에 안 들어가서 조각났다 → 청크 크기를 키워야 한다.
        #       검색 개수를 늘려도 잘린 조각을 더 가져올 뿐이다.
        docs=[MANUAL], gold_marks=["filter_steps"], **BASELINE,
        question="정수기 필터는 어떻게 교체하나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="전원을 뽑고 급수 밸브를 잠근 뒤 커버를 분리해 필터를 돌려 빼고 새 필터를 끼운 다음 누수를 확인하고 물을 흘려보낸다",
        qtype=None, answer_exists=None,
        retrieved=[1, 2],                        # c0(0~600) 을 놓쳐 앞 단계가 빠진다
        wide_ranking=[1, 2, 0],
        answer="새 필터를 끼운 뒤 누수를 확인하고 물을 흘려보낸다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.25},      # 앞 단계가 통째로 빠졌다
        assert_derived={"recall_at_k": "<1", "oversized_count": ">0"},
        expect={"A": "chunking_overchunking"},
        known_gap="현재 진단은 retrieval_low_rank 를 낸다. A 슬롯에서 청킹 라벨은 "
                  "qtype=bridge 일 때만 채택되고, 그 외에는 순위·코퍼스 라벨이 선점한다.",
    ),

    Case(
        id="g3_evidence_across_markdown_sections",
        situation="인사 규정을 마크다운 섹션 단위로 청킹했다. 연차 이월 설명이 장 경계에 걸쳐 "
                  "있고 양쪽 청크가 다 검색됐는데 답은 틀렸다.",
        # 판단: 근거가 섹션 경계에서 잘렸다 → 겹침이나 청크 크기를 늘려야 한다
        docs=[HR], gold_marks=["carryover"],
        chunk_strategy="markdown_recursive", chunk_size=600, chunk_overlap=80,
        search_mode="dense", reranked=False, mmr_applied=False,
        question="쓰지 않은 연차는 어떻게 되나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="다음 해 3월 31일까지 이월할 수 있고 그 뒤에는 소멸하나, 회사가 사용을 촉진하지 않았으면 수당으로 정산한다",
        qtype=None, answer_exists=None,
        retrieved=[0, 1, 2],
        answer="미사용 연차는 전액 수당으로 정산된다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.15},      # 이월·소멸을 빼고 정반대로 단정했다
        assert_derived={"missed_count": 0},
        expect={"A": "chunking_context_mismatch"},
        known_gap="섹션 경계의 좌표 틈 때문에 gold 청크를 다 집었는데도 span_recall 이 0 이 "
                  "되고 구체 라벨이 전부 침묵한다 (#100).",
    ),

    Case(
        id="g3_evidence_document_not_collected",
        situation="복지 제도 중 학자금 지원을 묻는데 그 내용을 담은 문서를 아직 수집하지 않았다. "
                  "검색은 무관한 청크만 가져왔고 모델은 근거 없이 답을 지어냈다.",
        # 판단: 자료가 없다 → 문서를 수집해야 한다. 근거가 없으면 기권했어야 한다.
        docs=[HR], gold_marks=["annual_days"], **BASELINE,
        question="자녀 학자금은 얼마까지 지원되나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="자녀 학자금은 연 200만원까지 지원한다",
        qtype=None, answer_exists=None,
        retrieved=[1, 2],
        corpus_exclude=[0],                      # 근거 청크를 코퍼스에서 제외
        answer="자녀 학자금은 연 500만원까지 지원된다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.1},       # 금액이 틀렸다
        assert_derived={"recall_at_k": "<1"},
        expect={"D": "corpus_gap", "B": "generation_abstention_failure"},
    ),

    Case(
        id="g3_evidence_fits_and_answer_correct",
        situation="인사 규정에서 휴가 신청 방법을 묻는다. 근거가 청크 하나에 온전히 담겼고 "
                  "검색이 그 청크를 가져왔으며 답도 맞았다.",
        # 판단: 고칠 게 없다
        docs=[HR], gold_marks=["apply"], **BASELINE,
        question="휴가는 어떻게 신청하나요?",
        gold_spans=[], span_grounding=None,
        ground_truth="사내 포털의 근태 메뉴에서 신청하고 팀장 승인을 받아야 확정된다",
        qtype=None, answer_exists=None,
        retrieved=[1, 0, 2],
        answer="사내 포털 근태 메뉴에서 신청한 뒤 팀장 승인을 받아야 확정된다",
        oracle_answer=Answer.GOLD_FULL,
        judge_real={"answer_correctness": 0.95, "faithfulness": 0.9},
        assert_derived={"recall_at_k": 1.0},
        expect={},
    ),
]
