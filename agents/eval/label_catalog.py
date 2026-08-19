"""진단 라벨 카탈로그.

웹 리포트는 "이번 실행에서 발견된 라벨"만이 아니라 "검사 대상이 되는 전체 라벨"
위에 실제 발생 수를 채워야 한다. 그 축이 report.html 안에만 박혀 있으면 라벨이 추가될
때 화면과 진단 코드가 쉽게 갈라진다. 이 모듈을 서버 응답의 정본으로 사용한다.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


DIAGNOSIS_LABEL_CATALOG: dict[str, list[dict[str, str]]] = {
    "retrieval": [
        {"code": "retrieval_low_rank", "name": "정답 순위가 낮음"},
        {"code": "retrieval_rank_fusion_loss", "name": "검색 융합 손실"},
        {"code": "retrieval_duplicate_crowding", "name": "중복 청크 쏠림"},
        {"code": "retrieval_rerank_candidate_miss", "name": "리랭커 후보 누락"},
        {"code": "retrieval_reranker_demotion", "name": "리랭커 강등"},
        {"code": "retrieval_reranker_ineffective", "name": "리랭커 효과 부족"},
        {"code": "retrieval_lexical_mismatch", "name": "키워드 검색 필요"},
        {"code": "retrieval_semantic_mismatch", "name": "의미 검색 불일치"},
        {"code": "retrieval_missing_gold", "name": "정답 근거 미검색"},
        {"code": "retrieval_incomplete_enumeration", "name": "나열형 근거 누락"},
        {"code": "retrieval_missing_bridge_dependency", "name": "다단계 연결 근거 누락"},
        {"code": "chunking_overchunking", "name": "근거가 여러 청크로 쪼개짐"},
        {"code": "chunking_context_mismatch", "name": "청크 경계에서 근거 분리"},
        {"code": "chunking_underchunking", "name": "청크가 너무 큼"},
        {"code": "reranker_low_precision", "name": "리랭커 낮은 정밀도"},
        {"code": "retrieval_failure", "name": "검색 실패"},
    ],
    "generation": [
        {"code": "generation_hallucination", "name": "근거 없는 생성"},
        {"code": "generation_partial_answer", "name": "부분 답변"},
        {"code": "generation_contradiction", "name": "근거와 모순"},
        {"code": "generation_misinterpretation", "name": "질문 해석 오류"},
        {"code": "generation_numerical_error", "name": "수치 오류"},
        {"code": "generation_hop_binding_error", "name": "근거 연결 오류"},
        {"code": "generation_chronological_error", "name": "시간 순서 오류"},
        {"code": "generation_abstention_failure", "name": "기권 실패"},
        {"code": "generation_wrongful_abstention", "name": "잘못된 기권"},
        {"code": "generation_parametric_overreliance", "name": "모델 지식 과의존"},
        {"code": "generation_failure", "name": "생성 실패"},
    ],
    "context": [
        {"code": "too_long_context", "name": "컨텍스트 과다"},
        {"code": "lost_in_the_middle", "name": "중간 근거 손실"},
        {"code": "context_noise_interference", "name": "노이즈 간섭"},
        {"code": "context_failure", "name": "컨텍스트 실패"},
    ],
    "data": [
        {"code": "corpus_gap", "name": "코퍼스 근거 없음"},
        {"code": "corpus_gap_partial_hop", "name": "일부 hop 문서 누락"},
        {"code": "bad_gold_answer", "name": "골드 정답 오류"},
        {"code": "bad_gold_chunk", "name": "골드 청크 오류"},
    ],
}


LABEL_DETAILS: dict[str, dict[str, str]] = {
    "retrieval_low_rank": {
        "situation": "정답 청크가 후보에는 있지만 순위가 낮아 상위 k개 밖으로 밀렸습니다.",
        "prescription": "리랭커를 켭니다.",
    },
    "retrieval_rank_fusion_loss": {
        "situation": "벡터 검색이나 키워드 검색 중 한쪽은 정답을 상위에 뒀지만, 결합 과정에서 순위가 깎였습니다.",
        "prescription": "하이브리드 가중치를 유리한 쪽으로 옮깁니다.",
    },
    "retrieval_duplicate_crowding": {
        "situation": "거의 같은 내용의 청크들이 상위 자리를 차지해 정답이 밀렸습니다.",
        "prescription": "중복을 접는 MMR을 사용합니다.",
    },
    "retrieval_rerank_candidate_miss": {
        "situation": "정답 청크가 리랭커 후보 목록에 없어 리랭커가 보지도 못했습니다.",
        "prescription": "리랭커 후보 목록을 넓힙니다.",
    },
    "retrieval_reranker_demotion": {
        "situation": "리랭커가 정답 청크를 후보로 받고도 아래로 떨어뜨렸습니다.",
        "prescription": "리랭커를 되돌리거나 모델을 바꿉니다.",
    },
    "retrieval_reranker_ineffective": {
        "situation": "리랭커가 정답 청크를 봤지만 상위로 끌어올리지 못했습니다.",
        "prescription": "리랭커 모델을 바꿉니다.",
    },
    "retrieval_lexical_mismatch": {
        "situation": "벡터 검색은 놓쳤지만 키워드로는 찾아지는 표현 차이 문제입니다.",
        "prescription": "키워드 검색을 함께 쓰는 하이브리드 검색을 켭니다.",
    },
    "retrieval_semantic_mismatch": {
        "situation": "벡터·키워드 검색이 모두 질문과 근거를 연결하지 못했습니다.",
        "prescription": "임베딩 모델이나 청킹 전략을 바꿉니다.",
    },
    "retrieval_missing_gold": {
        "situation": "정답 청크가 코퍼스에는 있지만 검색 결과에 들어오지 않았습니다.",
        "prescription": "검색 개수를 늘리거나 청크 겹침·크기를 조정합니다.",
    },
    "retrieval_incomplete_enumeration": {
        "situation": "여러 근거가 필요한 질문인데 일부 근거만 가져왔습니다.",
        "prescription": "질문에 따라 검색 개수를 늘립니다.",
    },
    "retrieval_missing_bridge_dependency": {
        "situation": "2단계 질문에서 두 번째 근거가 첫 단계 답을 알아야 검색됩니다.",
        "prescription": "질문을 단계별로 쪼개 검색합니다.",
    },
    "chunking_overchunking": {
        "situation": "정답 근거가 청크 하나에 담기기엔 너무 길어 여러 조각으로 쪼개졌습니다.",
        "prescription": "청크 크기를 키웁니다.",
    },
    "chunking_context_mismatch": {
        "situation": "정답 근거가 청크 경계에 걸쳐 잘렸습니다.",
        "prescription": "청크 겹침이나 크기를 늘리고 문장 경계로 자릅니다.",
    },
    "chunking_underchunking": {
        "situation": "청크가 근거보다 훨씬 커서 무관한 내용이 함께 들어왔습니다.",
        "prescription": "청크 크기를 줄입니다.",
    },
    "reranker_low_precision": {
        "situation": "리랭커가 질문과 무관한 청크를 상위로 올렸습니다.",
        "prescription": "리랭커 모델을 바꾸거나 문턱을 조입니다.",
    },
    "retrieval_failure": {
        "situation": "구체 원인은 못 짚었지만 검색 단계 실패로 분류됐습니다.",
        "prescription": "검색 설정과 코퍼스 범위를 우선 확인합니다.",
    },
    "generation_hallucination": {
        "situation": "정답 근거를 줬는데도 근거에 없는 내용을 지어냈습니다.",
        "prescription": "온도를 낮추고 인용을 요구하거나 생성 모델을 올립니다.",
    },
    "generation_partial_answer": {
        "situation": "답의 일부만 말하고 나머지 요소를 빠뜨렸습니다.",
        "prescription": "빠짐없이 답하도록 프롬프트를 고칩니다.",
    },
    "generation_contradiction": {
        "situation": "근거와 정면으로 어긋나는 답을 냈습니다.",
        "prescription": "답을 근거와 대조하는 검증 단계를 넣습니다.",
    },
    "generation_misinterpretation": {
        "situation": "질문의 관계나 조건을 잘못 읽었습니다.",
        "prescription": "질문을 다시 진술하게 합니다.",
    },
    "generation_numerical_error": {
        "situation": "수치를 잘못 읽거나 잘못 계산했습니다.",
        "prescription": "수치 인용을 요구하고 계산을 검산합니다.",
    },
    "generation_hop_binding_error": {
        "situation": "여러 근거를 찾았지만 서로 잘못 엮었습니다.",
        "prescription": "단계별 근거를 답에 묶어 쓰게 합니다.",
    },
    "generation_chronological_error": {
        "situation": "날짜·사건은 옮겼지만 무엇이 먼저인지 뒤바꿨습니다.",
        "prescription": "답에 나온 시점을 근거의 시간축과 대조합니다.",
    },
    "generation_abstention_failure": {
        "situation": "근거가 없는 질문인데 모른다고 하지 않고 답을 지어냈습니다.",
        "prescription": "근거 없을 때 기권하도록 강화하고 인용을 요구합니다.",
    },
    "generation_wrongful_abstention": {
        "situation": "근거가 검색됐는데도 알 수 없다고 회피했습니다.",
        "prescription": "기권 조건을 완화합니다.",
    },
    "generation_parametric_overreliance": {
        "situation": "답은 맞았지만 검색 근거가 아니라 모델 지식에 기대어 답했습니다.",
        "prescription": "인용을 요구하고 온도를 낮춥니다.",
    },
    "generation_failure": {
        "situation": "구체 원인은 못 짚었지만 생성 단계 실패로 분류됐습니다.",
        "prescription": "프롬프트·기권 조건·생성 모델을 확인합니다.",
    },
    "too_long_context": {
        "situation": "컨텍스트가 너무 길어 모델이 근거를 제대로 못 썼습니다.",
        "prescription": "검색 개수를 줄이거나 컨텍스트를 압축합니다.",
    },
    "lost_in_the_middle": {
        "situation": "정답 청크가 컨텍스트 가운데 묻혀 참조되지 않았습니다.",
        "prescription": "검색 개수를 줄이거나 정답을 양끝으로 재배치합니다.",
    },
    "context_noise_interference": {
        "situation": "정답이 아닌 청크의 상충 정보에 끌려 답이 틀어졌습니다.",
        "prescription": "노이즈를 걸러내거나 충돌 처리를 지시합니다.",
    },
    "context_failure": {
        "situation": "구체 원인은 못 짚었지만 컨텍스트 구성 실패로 분류됐습니다.",
        "prescription": "컨텍스트 길이·순서·노이즈를 확인합니다.",
    },
    "corpus_gap": {
        "situation": "답에 필요한 문서가 코퍼스에 없습니다.",
        "prescription": "해당 주제 문서를 수집해 인덱싱합니다.",
    },
    "corpus_gap_partial_hop": {
        "situation": "여러 단계 질문에서 중간 단계를 뒷받침하는 문서가 빠졌습니다.",
        "prescription": "빠진 단계의 문서를 수집합니다.",
    },
    "bad_gold_answer": {
        "situation": "파이프라인이 아니라 정답셋이 틀렸거나 모호합니다.",
        "prescription": "검증 질문과 정답을 다시 만듭니다.",
    },
    "bad_gold_chunk": {
        "situation": "정답 텍스트는 맞지만 근거로 지정된 청크가 엉뚱한 곳입니다.",
        "prescription": "골드 청크를 다시 지정합니다.",
    },
}


def diagnosis_label_catalog() -> dict[str, list[dict[str, Any]]]:
    """웹/리포트 소비처에 넘기는 라벨 카탈로그 사본."""
    catalog = deepcopy(DIAGNOSIS_LABEL_CATALOG)
    for rows in catalog.values():
        for row in rows:
            row.update(LABEL_DETAILS.get(row.get("code", ""), {}))
    return catalog
