"""
agents/optimize/rules.py
라벨 → 처방 규칙 테이블 (선언적 데이터)

[이 파일의 역할]
  Eval이 확정한 Finding.label 을 받아서, 어떤 처방을 어떤 순서로
  시도할지 정의한 "룩업 테이블"이다. 실행 로직(planner)이나 우선순위
  계산(schemas 기반)은 여기 들어오지 않는다. 이 파일은 순수 데이터.

[설계 원칙 — 방식2 유지]
  1. 처방은 항상 "순서 있는 리스트". 가벼운 것(런타임) → 무거운 것(재색인) 순.
     planner가 맨 앞부터 하나씩 꺼내 적용하고, 실패 시 다음 후보로 순차검증.
     (동시 적용 금지 = 방식2. 한 라벨의 여러 config를 한꺼번에 바꾸면 방식1로 후퇴)
  2. 각 처방에 (reindex, cost) 메타데이터를 박는다.
     - reindex: 재색인 필요 여부. True면 그래프가 Index 노드를 경유해야 함.
     - cost:    처방비용. 우선순위 공식(빈도×신뢰도÷비용)의 분모. 런타임=1, 재색인=3.
  3. 미확정 라벨은 지우지 말고 status="draft"로 남긴다.


"""
from __future__ import annotations


# ── 처방 상태 상수 ────────────────────────────────────────────────
# ready       : 처방 로직 확정, planner가 실행 가능
# draft       : 라벨은 있으나 처방 미확정 (신호/스키마 합의 대기)
# manual      : config로 못 고침, 사람 개입 필요 (D그룹)

# ── config 키 주의 ────────────────────────────────────────────────
# 현재 state.index_config 에는 chunk/embedding/search 설정과 reranker 기본값,
# 그리고 생성(B그룹) 설정(temperature·grounding_strict 등)이 함께 담긴다.
# Optimize 내부 patch는 flat key 대신 canonical path를 사용하고 config_mapper를 거친다.
#
# ⚠️ 단일 축 규칙: optimizer는 한 처방이 config 키 하나만 바꾸게 강제한다(효과 귀속).
#   patch에 키가 2개 이상이면 거부(multi_axis_search_space)되므로, 한 처방 = 한 키.
#
# 생성(B그룹 Tier1): generation.* 처방은 index_config의 생성 키로 매핑되고 generator가
#   프롬프트/온도로 소비한다(Eval이 index_config를 generate_answer로 전달). 실제 프롬프트
#   문구는 generator(_build_prompt)가 플래그로 조립한다 — rules는 스위치만 든다.
#   Tier2(verifier/checklist/calculation 노드)·Tier3(모델 교체)는 소비처 부재로 draft.

# swap_embedding_model이 바꿔 끼울 실제 임베딩 모델명.
# TODO(embedding-후보-합의): 품질 기준으로 검증된 실제 업그레이드 후보가 아직 정해지지
#   않았다. 지금은 sentence-transformers로 바로 로드 가능한 다른 차원(384차원)의 모델을
#   임시로 지정해, "차원이 바뀌는 임베딩 모델 교체 → recreate_collection_on_dimension_mismatch로
#   Qdrant 컬렉션 재생성" 경로가 실제로 동작하는지 검증하는 용도다. 기본 모델(bge-m3, 1024차원)
#   보다 검색 품질이 낫다는 근거는 없으므로, 실제 품질 개선용 후보가 정해지면 교체해야 한다.
_EMBEDDING_MODEL_UPGRADE_CANDIDATE = "sentence-transformers/all-MiniLM-L6-v2"


LABEL_TO_PRESCRIPTIONS: dict[str, dict] = {

    # ═══════════════════════════════════════════════════════════════
    #  A그룹 — 검색 실패 (Oracle Test 통과)
    # ═══════════════════════════════════════════════════════════════

    "retrieval_low_rank": {
        "group": "A",

        "status": "ready",
        "diagnosis_confidence": None,  # 숫자 튜닝 필요
        "target_metrics": ["context_precision"],  # gold는 검색됨, 순위 품질이 문제
        "prescriptions": [
            {
                "id": "enable_reranker",
                "patch": {"reranker.enabled": True},
                "reindex": False,
                "cost": None, # 숫자 튜닝 필요
            },
            {
                # 이미 reranker가 켜져 있는데도 low-rank가 남으면 더 넓은
                # 1차 후보군을 재정렬해 gold가 reranker 입력에 들어올 기회를 늘린다.
                "id": "widen_rerank_candidates",
                "patch": {"reranker.candidate_count": "increase"},
                "reindex": False,
                "cost": None,
            },
        ],
        # NOTE: baseline에서는 먼저 켜고, 문제가 남으면 후보 수를 한 단계 넓힌다.
    },

    "retrieval_lexical_mismatch": {
        "group": "A",
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # dense가 놓친 gold를 검색결과에 포함시킴
        "prescriptions": [
            {
                "id": "enable_hybrid",
                # canonical 경로로 하이브리드 검색을 켠다. flat "use_hybrid": True 를 쓰면
                # config_mapper 가 retriever.search_type 으로 정규화하면서 값을 문자열 "hybrid"
                # 와 비교(str(True).lower() != "hybrid")해 오히려 use_hybrid=False 로 꺼버렸다.
                # 값은 반드시 "hybrid"/"dense" 문자열이어야 매핑이 올바로 켠다.
                "patch": {"retriever.search_type": "hybrid"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: baseline이 이미 hybrid면 후보가 no-op으로 필터돼 발동하지 않는다(dense-only 전용).
    },

    "retrieval_semantic_mismatch": {
        "group": "A",
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # dense·BM25 둘 다 놓친 gold를 검색되게

        #   토픽클러스터 분석은 Eval 소관 → finding.metadata["topic_cluster"]로 넘어옴.
        #   rules는 후보만 나열 + applies_when 태그, 실제 선택은 planner가 수행.
        #     "spread"       → Case3(임베딩 모델 자체 약함) → 임베딩 교체
        #     "concentrated" → Case2(특정 도메인 약함)      → 임베딩 교체(도메인특화/파인튜닝)
        #     "none"         → Case1(청크 희석)             → 청킹 조정
        
        #   신호가 없으면(MVP) planner가 리스트 순서대로 순차 시도(fallback).
        
        # TODO(eval-합의): topic_cluster 신호 키/값을 Eval과 확정.
        
        "prescriptions": [
            {
                # 임베딩 모델 바꾸기 case 3 2에 해당
                "id": "swap_embedding_model",
                "patch": {"embedding_model": _EMBEDDING_MODEL_UPGRADE_CANDIDATE},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
                "applies_when": {"topic_cluster": ["spread", "concentrated"]},
                # 차원이 바뀌는 임베딩 모델 교체이므로, optimizer._run_rules가 이 patch를
                # search_space에서 골라 최종 ConfigPatch를 만들 때
                # recreate_collection_on_dimension_mismatch=True를 자동으로 함께 실어 보낸다
                # (agents/optimize/optimizer.py 참고 — 여기서 직접 넣지 않는 이유는
                # planner의 단일 축 search_space 계산을 깨지 않기 위함).

                # TODO: Case2(도메인약함)는 범용 upgrade가 아니라 도메인특화/파인튜닝 모델이
                # 이상적 → adapter 단계에서 세분화. MVP는 upgrade로 통합.
            },
            {
                # 청크 크기 축소 case 1에 해당
                "id": "shrink_chunk_size",
                "patch": {"chunk_size": "decrease"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
                "applies_when": {"topic_cluster": ["none"]},
            },
            {
                # 청킹 전략 교체 case 1에 해당 (초안 누락분 보강)
                "id": "switch_chunking_strategy",
                "patch": {"chunker.strategy": "recursive_sentence"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
                "applies_when": {"topic_cluster": ["none"]},
                # NOTE: chunking_context_mismatch와 동일 처방(Case1: 청크 경계 의미 희석).
                # Index가 recursive_sentence 전략을 CHUNK_STRATEGIES에 등록해 실행 가능하다
                # (chunker.strategy → chunk_strategy 매핑, chunking_strategy capability=True).
            },
        ],
    },

    "retrieval_missing_gold": {
        "group": "A",
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # 후보에 아예 없는 gold를 가져오게
        "prescriptions": [
            {
                "id": "increase_top_k",
                "patch": {"top_k": "increase"},
                "reindex": False,       # 제일 가벼움, 먼저 시도
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "increase_chunk_overlap",
                "patch": {"chunk_overlap": "increase"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "increase_chunk_size",
                "patch": {"chunk_size": "increase"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
              
                "id": "expand_query",
                "patch": {"query_rewrite": "expand"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # TODO: BLOCKER: query_rewrite 필드가 AgentDoctorState/index_config에 없음
        #   (retrieval_missing_bridge_dependency의 query_rewrite:"decompose"와 같은 필드,
        #    값만 다름 — 필드 자체는 이미 그쪽에서 스키마 합의 대기 중).
    },

    "retrieval_incomplete_enumeration": {
        "group": "A",
        # dynamic_top_k 하나가 실행 가능해져 ready 로 승격(나머지 2개는 여전히 스키마 미정).
        # top_k 는 STATE_MAPPABLE_PATHS 에 있고 Eval 이 index_config["top_k"] 를 실제로
        # 읽어 검색에 쓴다. mmr/adaptive_retrieval 은 매핑 불가라 optimizer 가 후보
        # 단계에서 자동으로 걸러내므로, 그 둘 때문에 라벨 전체를 막아둘 이유는 없다.
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # 나열형 gold 일부 누락(recall@k 부분) 보완
        "prescriptions": [
            {
                "id": "dynamic_top_k",
                "patch": {"top_k": "increase"},   # 나열형은 gold 개수 > 고정 top_k
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                # 관련성만이 아니라 다양성까지 고려해 top-k 안 쏠림을 줄임
                "id": "enable_mmr",
                "patch": {"retriever.mmr": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
                # 공통 Retriever가 use_mmr로 후보풀을 MMR 재정렬해 실행 가능하다
                # (retriever.mmr → use_mmr 매핑, mmr capability=True).
            },
            {
                # top-k 고정 대신, 검색 도중 "더 필요한지" 판단해 반복 검색
                "id": "enable_adaptive_retrieval",
                "patch": {"adaptive_retrieval": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요

                #  # TODO(index-합의) BLOCKER: adaptive_retrieval 필드가 없음 + 단순 config 값이 아니라
                #   검색 제어흐름 자체를 바꾸는 처방이라 구현 난이도 제일 높음.
                #   지금 당장 개발 대상은 아니고, 후보로만 남겨둠(추후 발전 여지).
            },
        ],
    },

    "retrieval_missing_bridge_dependency": {
        "group": "A",
        "status": "draft",              # multi-hop query rewrite / max_hops 스키마 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # 미검색된 hop2 gold를 가져오게
        "prescriptions": [
            {
                #baseline이 single-shot 검색이라는 전제 하에 유효한 처방
                "id": "enable_query_decomposition",
                "patch": {
                    "query_rewrite": "decompose",
                    "max_hops": "increase",
                    "sub_query_generator_prompt": "bridge_entity_aware",
                },
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                # 초안 외 확장: 브릿지 엔티티를 명시적으로 추출해 다음 hop 검색어에 강조.
                #   초안 판별신호("hop1 답을 쿼리에 추가해 재검색")에서 파생된 보조 기법.
                "id": "expand_bridge_entity_query",
                "patch": {"bridge_entity_expansion": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: query_rewrite/max_hops/sub_query_generator_prompt가 index_config가 아니라
        #   generation_config 소속인데, 그 네임스페이스 자체가 AgentDoctorState에 없음.
        #   B그룹 전체를 막는 것과 동일 원인(파일 상단 35-38번 줄 참고) → 이 필드가
        #   추가되면 B그룹뿐 아니라 이 라벨도 같이 풀림.
    },

    "chunking_context_mismatch": {
        "group": "A",
        # gold span/청크 절대좌표로 경계 분할을 확정하고 overlap 후보를
        # 사전검증할 수 있으므로 실행 가능한 라벨로 승격한다.
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # 청크 경계에서 잘린 gold를 온전히 검색되게
        "prescriptions": [
            {
                "id": "increase_chunk_overlap",
                "patch": {"chunk_overlap": "increase"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                # overlap 안전 상한으로 회복할 수 없는 긴 정답은 다음 단계에서
                # chunk_size를 늘려 검증한다.
                "id": "increase_chunk_size",
                "patch": {"chunk_size": "increase"},
                "reindex": True,
                "cost": None,
            },
            {
                "id": "switch_chunking_strategy",
                "patch": {"chunker.strategy": "recursive_sentence"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: 세 처방 모두 현재 index_config로 실행 가능하다.
        # (recursive_sentence 전략이 Index CHUNK_STRATEGIES에 등록됨 — chunker.strategy 매핑.)
    },

    "chunking_overchunking": {
        "group": "A",
        # Eval 이 기하(span 길이 > 최장 청크)로 확정하고 planner 가 span 길이로 chunk_size 를
        # 근거화하므로 실행 가능한 라벨로 승격한다(chunking_context_mismatch 와 같은 기준).
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_recall"],  # 과편화된 청크 → 완전한 gold 맥락 확보
        "prescriptions": [
            {
                "id": "increase_chunk_size",
                "patch": {"chunk_size": "increase"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "chunking_underchunking": {
        "group": "A",
        "status": "draft",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_precision"],  # 큰 청크에 섞인 무관 내용 → 유용성 개선
        "prescriptions": [
            {
                "id": "decrease_chunk_size",
                "patch": {"chunk_size": "decrease"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    # ── 폐기(보류) 라벨: reranker_low_recall ────────────────────────
    # 2026-07-27 Eval팀 라벨 합의에서 도입하지 않기로 결정. 완전 삭제하지 않고
    # 주석으로 남겨 근거를 보존한다(필요 시 아래 정의를 되살릴 수 있음).
    # [이유] recall 은 retriever 단계 속성이고 리랭커는 후보 "재정렬"만 한다 →
    #   후보 집합에 없는 gold 는 리랭커가 못 살린다. "리랭커가 gold 를 컷오프
    #   아래로 밀어냈다(리랭킹 전>후 recall)"로 좁게 정의하고 pre/post delta 를
    #   신호로 잡을 때만 성립하며, 그 외엔 retrieval_low_rank /
    #   retrieval_incomplete_enumeration 과 처방이 겹쳐 이중 라벨링이 된다.
    #   → 처방이 기존 라벨과 겹치면 새 라벨이 아니다(라벨 도입 원칙).
    #
    # "reranker_low_recall": {
    #     "group": "A",
    #     "assigned": "권성우",
    #     "status": "draft",              # 이 튜닝 라벨과 threshold 소비 경로는 아직 없음
    #     "diagnosis_confidence": None,   # 숫자 튜닝 필요
    #     "target_metrics": ["context_recall"],  # 재랭커가 걸러낸 gold를 다시 살림
    #     "prescriptions": [
    #         {
    #             "id": "widen_rerank_candidates",
    #             "patch": {"rerank_candidates": "increase"},
    #             "reindex": False,
    #             "cost": None,           # 숫자 튜닝 필요
    #         },
    #         {
    #             "id": "relax_reranker_threshold",
    #             "patch": {"reranker_threshold": "decrease"},
    #             "reindex": False,
    #             "cost": None,           # 숫자 튜닝 필요
    #         },
    #     ],
    #     # BLOCKER: Eval 라벨 생성과 reranker_threshold 소비 경로가 아직 없음.
    # },

    "reranker_low_precision": {
        "group": "A",
        "status": "draft",              # 모델 교체 후보와 threshold 소비 경로는 아직 없음
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_precision"],  # 재랭커가 상위로 올린 무관 청크 억제
        "prescriptions": [
            {
                "id": "swap_reranker_model",
                "patch": {"reranker_model": "upgrade"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "tighten_reranker_threshold",
                "patch": {"reranker_threshold": "increase"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: Eval 라벨 생성, 모델 교체 후보, threshold 소비 경로가 아직 없음.
    },



    # ═══════════════════════════════════════════════════════════════
    #  B그룹 — 생성 실패 (Oracle Test 실패)
    # ═══════════════════════════════════════════════════════════════

    "generation_hallucination": {
        "group": "B",
        "status": "ready",              # 프롬프트/온도 플래그가 generator에 소비됨(Tier1)
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["faithfulness"],  # context에 없는 내용 지어냄 → 근거성
        "prescriptions": [
            {
                # 온도만 낮추는 게 제일 가벼운 독립 레버. baseline이 0이면 no-op으로 필터.
                "id": "lower_temperature",
                "patch": {"generation.temperature": "decrease"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                # 근거가 없으면 지어내지 말고 강하게 기권하도록. (문구는 generator 소유,
                # 여기선 스위치만 — grounding_strict는 기본 True라 abstention_strict가 실질 레버.)
                "id": "strengthen_abstention",
                "patch": {"generation.abstention_strict": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                # 근거 번호 인용 강제(기본 True라 대개 no-op이나, 꺼진 경우 복구용).
                "id": "require_citation",
                "patch": {"generation.require_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "upgrade_generation_model",
                "patch": {"generation.model": "upgrade"},  # 프롬프트로 안 되면 최후 수단
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
                # BLOCKER(Tier3): 검증된 교체 후보 없음 → capability generation_model=False.
            },
        ],
    },

    "generation_partial_answer": {
        "group": "B",
        "status": "ready",              # completeness_mode 플래그가 generator에 소비됨(Tier1)
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["answer_relevancy"],  # 질문 요구 일부만 충족 → 완결성
        "prescriptions": [
            {
                # 여러 항목·하위 질문에 빠짐없이 답하도록 유도(문구는 generator 소유).
                "id": "completeness_prompt",
                "patch": {"generation.completeness_mode": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "checklist_review_step",
                "patch": {"answer_checklist_review": True},  # 답변 누락 점검 단계 추가
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
                # BLOCKER(Tier2): 답변 점검 노드가 아직 없음(소비처 부재).
            },
        ],
    },

    "generation_contradiction": {
        "group": "B",
        "status": "draft",              # 재실행형(LLM 재검증 패스), 실행 방식도 별도 확정 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["faithfulness"],  # 청크 간 모순을 못 풀고 답변 → 근거성
        "prescriptions": [
            {
                "id": "llm_verification_pass",
                "patch": {"verifier_on": True, "verifier_type": "faithfulness"},
                "reindex": False,
                "cost": None,            # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 없음 + verifier 노드 자체가 아직 미구현
        #   (B그룹 공통노드: evidence_mapper/generation_verifier/revision, 설계 초안 단계)
    },

    "generation_misinterpretation": {
        "group": "B",
        "status": "ready",              # restate_question 플래그가 generator에 소비됨(Tier1)
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["answer_relevancy"],  # 질문 조건 오독 → 질문 의도 반영도
        "prescriptions": [
            {
                "id": "restate_question",
                "patch": {"generation.restate_question": True},  # 답변 전 질문 재진술 강제
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "generation_abstention_failure": {
        "group": "B",
        "status": "ready",              # abstention_strict 플래그가 generator에 소비됨(Tier1)
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["faithfulness"],  # 모른다고 해야 하는데 지어냄 → 근거성
        "prescriptions": [
            {
                "id": "strengthen_abstention",
                "patch": {"generation.abstention_strict": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "require_citation",
                "patch": {"generation.require_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "generation_parametric_overreliance": {
        "group": "B",
        "status": "ready",              # abstention_strict 플래그가 generator에 소비됨(Tier1)
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["faithfulness"],  # 맞았지만 context 근거 없음(파라미터 의존) → 근거성
        "prescriptions": [
            {
                # 정답이라도 context 근거가 없으면 기권하도록 강화(파라미터 기억 의존 억제).
                "id": "strengthen_abstention",
                "patch": {"generation.abstention_strict": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "require_citation",
                "patch": {"generation.require_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "lower_temperature",
                "patch": {"generation.temperature": "decrease"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "generation_numerical_error": {
        "group": "B",
        "status": "draft",              # generation_config 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["faithfulness"],  # 숫자 계산/집계 오류 → 원문 근거성
        "prescriptions": [
            {
                "id": "require_numeric_citation",
                "patch": {"numeric_citation_required": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "enable_calculation_check",
                "patch": {"calculation_check": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 필드 및 calculation checker 단계 없음.
    },

    "generation_hop_binding_error": {
        "group": "B",
        "status": "draft",              # multi-hop answer planning 스키마 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        # 각 fact는 근거 있어 faithfulness는 안 낮음 → 대신 결합 오류를 잡는
        # answer_relevancy를 대표 지표로. 이상적으론 List-Component F1 같은 커스텀 채점.
        "target_metrics": ["answer_relevancy"],
        "prescriptions": [
            {
                "id": "force_hop_evidence_binding",
                "patch": {"answer_format": "cot_chained", "require_hop_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "enable_bridge_entity_verifier",
                "patch": {"bridge_entity_verifier": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 필드 및 verifier 단계 없음.
    },

    # ═══════════════════════════════════════════════════════════════
    #  C그룹 — context 구조 문제
    # ═══════════════════════════════════════════════════════════════

    "too_long_context": {
        "group": "C",
        "status": "ready",              # top_k 축소는 기존 키로 바로 실행 가능
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["noise_sensitivity"],  # 과다 context의 잡음에 답변이 흔들림
        "prescriptions": [
            {
                "id": "decrease_top_k",
                "patch": {"top_k": "decrease"},
                "reindex": False,       # 가장 가벼움, 먼저 시도
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "context_compression",
                "patch": {"context_compression": True},  # 관련도 낮은 청크 필터링/압축
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
                # TODO(index-합의): context_compression 필드 index_config에 없음
            },
            {
                "id": "shrink_chunk_size",
                "patch": {"chunk_size": "decrease"},
                "reindex": True,        # 마지막 수단, 재색인 필요
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "lost_in_the_middle": {
        "group": "C",
        # decrease_top_k 하나가 실행 가능해져 ready 로 승격(retrieval_incomplete_enumeration
        # 과 동일 선례). context 재정렬(reorder_context_edges)은 정렬 서브시스템이 없어
        # optimizer 가 후보 단계에서 자동으로 걸러내므로, 그 하나 때문에 라벨 전체를 막지 않는다.
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["context_utilization"],  # 검색은 됐으나 중간 청크를 답변에 못 씀
        "prescriptions": [
            {
                # 검색 결과를 짧게 줄이면 "중간"에서 잃을 구간 자체가 줄어드는 정당한 완화책.
                # top_k 는 STATE_MAPPABLE 이고 Eval 이 index_config["top_k"] 를 실제로 소비한다.
                "id": "decrease_top_k",
                "patch": {"top_k": "decrease"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                # 가장 관련도 높은 청크를 컨텍스트 양끝에 배치해 lost-in-the-middle 을 직접
                # 완화하는 본래 처방. 컨텍스트 정렬 서브시스템 부재로 아직 막힘(후보로만 남김).
                "id": "reorder_context_edges",
                "patch": {"context_ordering": "most_relevant_edges"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: reorder_context_edges 는 context_ordering 소비 경로가 없어 자동 필터된다.
    },

    "context_noise_interference": {
        "group": "C",
        # enable_mmr 이 실행 가능해져 ready 로 승격. 다양성 재정렬은 중복·잡음 근접청크를
        # 줄여 노이즈 오염을 완화하는 정당한 처방이다. noise_filter/conflict_prompt 는
        # 소비 노드가 없어 optimizer 가 후보 단계에서 자동으로 걸러낸다.
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "target_metrics": ["noise_sensitivity"],  # 비-gold 상충 청크가 답변을 오염
        "prescriptions": [
            {
                # 관련성+다양성 균형으로 후보풀을 재정렬해 중복·잡음 청크 쏠림을 억제.
                # 값 키는 A그룹과 통일한다 — flat "mmr":True 대신 canonical retriever.mmr.
                # 공통 Retriever 가 use_mmr 로 소비해 실행 가능(retriever.mmr → use_mmr 매핑).
                "id": "enable_mmr",
                "patch": {"retriever.mmr": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "enable_noise_filter",
                "patch": {"noise_filter": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "strict_conflict_prompt",
                "patch": {"conflict_resolution_prompt": "prefer_high_confidence_evidence"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: noise_filter/conflict_resolution_prompt 는 소비 경로가 없어 자동 필터된다.
    },

    # ═══════════════════════════════════════════════════════════════
    #  D그룹 — 데이터 문제 (config로 처방 불가, 사람 개입)
    # ═══════════════════════════════════════════════════════════════
    #
    # [매뉴얼 처방 규약]
    #   A/B/C 처방은 config patch(자동 적용)지만, D그룹 처방은 사람이 수행할
    #   "매뉴얼 스텝"이다. 각 항목은 patch 대신 다음 필드를 든다:
    #     - manual: True   → planner/optimizer가 절대 자동 적용하지 않는다는 표식.
    #                        (status="manual" 이라 is_actionable 도 계속 False)
    #     - action        : 사용자가 할 한 줄 조치(명령형).
    #     - detail        : 어떻게 하는지 구체 설명.
    #     - show          : "어디가 문제인지"를 사용자에게 보여줄 때 쓸 finding/probe
    #                       필드 목록(reporter가 렌더). 실제 배선은 Eval의 finding.metadata
    #                       계약 확정(PR #55, missing_gold_ids 등) 후 붙인다 — 그전엔
    #                       reporter가 manual_action 헤드라인만 읽는다(동작 불변).
    #   manual_action 은 라벨 전체를 한 줄로 요약한 헤드라인으로 계속 유지한다
    #   (스텝들은 그 아래에 번호로 렌더될 예정).

    "corpus_gap": {
        "group": "D",
        "status": "manual",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        # config로 못 고침 → 사람이 수행할 매뉴얼 스텝. planner는 이 라벨을 manual로
        # 분리해 reporter로만 넘긴다(자동처방 루프에서 제외).
        "prescriptions": [
            {
                "id": "locate_missing_evidence",
                "manual": True,
                "action": "코퍼스에서 빠진 근거를 특정",
                "detail": "해당 질문의 정답 근거가 담긴 원본 문서 중 코퍼스에 없는 것을 확인한다. "
                          "누락 gold의 원본 문서(gold_doc_id)와 질문을 함께 제시한다.",
                # reporter가 "어디가 문제인지" 보여줄 때 쓸 위치정보(배선은 Eval 계약 확정 후):
                "show": ["question", "missing_gold_ids", "gold_doc_id", "corpus_membership_ratio"],
            },
            {
                "id": "collect_and_reindex",
                "manual": True,
                "action": "그 문서를 수집·추가한 뒤 재색인",
                "detail": "특정된 주제/문서를 소스에서 추가 수집해 코퍼스에 넣고 재색인한 뒤 "
                          "다시 진단을 실행한다.",
            },
        ],
        # reporter가 사용자에게 보여줄 헤드라인. (config로 못 고치는 D그룹 전용)
        "manual_action": "질문에 답할 근거 문서가 코퍼스에 없습니다. 해당 주제를 다루는 문서를 추가로 수집·인덱싱해 주세요.",
    },

    "corpus_gap_partial_hop": {
        "group": "D",
        "status": "manual",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "locate_missing_hop",
                "manual": True,
                "action": "끊긴 hop의 근거를 특정",
                "detail": "다단계 질문에서 어느 hop의 근거가 코퍼스에 없는지 특정한다. "
                          "누락 hop의 원본 문서(gold_spans[i].doc_id)와 질문을 함께 제시한다.",
                # 멀티홉은 hop별 위치(gold_spans)로 어느 단계가 빠졌는지까지 보여준다:
                "show": ["question", "missing_gold_ids", "gold_spans", "corpus_membership_ratio"],
            },
            {
                "id": "collect_bridge_docs",
                "manual": True,
                "action": "빠진 hop 문서를 수집·추가한 뒤 재색인",
                "detail": "특정된 hop을 뒷받침하는 문서를 추가 수집해 코퍼스에 넣고 재색인한 뒤 "
                          "다시 진단을 실행한다.",
            },
        ],
        "manual_action": "다단계(multi-hop) 질문의 중간 단계를 뒷받침하는 문서가 일부 누락됐습니다. 빠진 hop과 관련된 문서를 추가로 수집해 주세요.",
    },

    "bad_gold_answer": {
        "group": "D",
        "status": "manual",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        # RAG 결함이 아니라 평가셋(정답) 문제 → 해당 probe 자체를 재생성하도록 요청한다.
        "prescriptions": [
            {
                "id": "regenerate_probe",
                "manual": True,
                "action": "이 라벨이 붙은 probe를 재생성",
                "detail": "affected_probes에 해당하는 각 probe의 질문·정답(ground_truth)을 "
                          "다시 생성하거나 검수해 교체한다. 재생성으로도 타당한 정답을 얻지 "
                          "못하면 해당 probe를 평가셋에서 제외한다.",
                # 어느 probe를 재생성할지 사용자에게 명시:
                "show": ["probe_id", "question", "ground_truth"],
            },
        ],
        "manual_action": "RAG 파이프라인이 아니라 평가셋(정답) 문제로 보입니다. 해당 probe를 재생성(질문·정답 재작성)하거나 평가셋에서 제외해 주세요.",
    },
}


# ── 편의 조회 함수 ────────────────────────────────────────────────

def get_rule(label: str) -> dict | None:
    """라벨에 해당하는 규칙 반환. 없으면 None."""
    return LABEL_TO_PRESCRIPTIONS.get(label)


def is_actionable(label: str) -> bool:
    """planner가 실제로 처방을 실행해도 되는 라벨인지.
    ready 상태 + 처방이 비어있지 않아야 True.
    draft/unassigned/manual 은 아직 실행 금지."""
    rule = LABEL_TO_PRESCRIPTIONS.get(label)
    if not rule:
        return False
    return rule.get("status") == "ready" and bool(rule.get("prescriptions"))


def is_manual(label: str) -> bool:
    """D그룹처럼 config 처방 불가 → 사람 개입 라벨인지."""
    rule = LABEL_TO_PRESCRIPTIONS.get(label)
    return bool(rule) and rule.get("status") == "manual"
