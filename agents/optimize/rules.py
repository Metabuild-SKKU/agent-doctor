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

[협업 구조 — 라벨 단위 소유권, 그룹은 taxonomy 정리용]
  taxonomy 그룹(A/B/C/D)은 표의 분류를 그대로 유지해 읽기 편하게 하되,
  실제 담당은 그룹 단위가 아니라 "라벨 단위"로 나뉜다.
  각자 자기 담당 라벨의 dict "값"만 채우고, 남의 라벨 항목은 절대 건드리지 않는다
  (status="unassigned" 껍데기 그대로 둠). 이러면 서로 다른 줄을 고치는 게 되어
  git merge 시 충돌이 거의 발생하지 않는다.

  이승준 담당(12개):
    retrieval_low_rank, retrieval_lexical_mismatch, retrieval_semantic_mismatch,
    retrieval_missing_gold, retrieval_incomplete_enumeration,
    generation_hallucination, generation_partial_answer, generation_contradiction,
    generation_misinterpretation, too_long_context, corpus_gap, corpus_gap_partial_hop

  나머지 13개는 unassigned 스켈레톤만 존재 (다른 담당자가 자기 항목만 채울 것).
"""
from __future__ import annotations


# ── 처방 상태 상수 ────────────────────────────────────────────────
# ready       : 처방 로직 확정, planner가 실행 가능
# draft       : 라벨은 있으나 처방 미확정 (신호/스키마 합의 대기)
# unassigned  : 아직 담당자가 채우지 않은 스켈레톤 (건드리지 않을 것)
# manual      : config로 못 고침, 사람 개입 필요 (D그룹)

# ── config 키 주의 ────────────────────────────────────────────────
# 현재 state.index_config 에 존재하는 키: chunk_size, chunk_overlap,
#   embedding_model, use_hybrid  (core/state.py 기준)
# use_reranker, top_k, rerank_candidates 등은 index_config에 아직 없음
#   → Index 팀과 합의해 키 추가 필요.  # TODO(index-합의)
#
# ⚠️ 더 큰 블로커: generation_config 필드 자체가 core/state.py의
#   AgentDoctorState 에 아예 없음 (index_config만 존재). B그룹 처방은
#   전부 이 필드가 생겨야 실행 가능 → schema/state 합의가 선행돼야 함.
#   # TODO(state-스키마-확장): generation_config: dict 필드 추가 필요


LABEL_TO_PRESCRIPTIONS: dict[str, dict] = {

    # ═══════════════════════════════════════════════════════════════
    #  A그룹 — 검색 실패 (Oracle Test 통과)
    # ═══════════════════════════════════════════════════════════════

    "retrieval_low_rank": {
        "group": "A",

        "assigned": "이승준",
        "status": "ready",
        "diagnosis_confidence": None,  # 숫자 튜닝 필요 
        "prescriptions": [
            {
                "id": "enable_reranker",
                "patch": {"use_reranker": True},
                "reindex": False,
                "cost": None, # 숫자 튜닝 필요
            },
            {
                "id": "widen_rerank_candidates",
                "patch": {"rerank_candidates": "increase"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "retrieval_lexical_mismatch": {
        "group": "A",
        "assigned": "이승준",
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "enable_hybrid",
                "patch": {"use_hybrid": True},
                "reindex": False,       # 검색 시점 융합, 색인 재생성 불필요
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: baseline이 이미 hybrid면 발생 안 함. naive(dense-only) MVP 전용.
    },

    "retrieval_semantic_mismatch": {
        "group": "A",
        "assigned": "이승준",
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "swap_embedding_model",
                "patch": {"embedding_model": "upgrade"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
                # WARN: VECTOR_DIM 변경 시 Qdrant 컬렉션 재생성 필요 (qdrant_store.py)
            },
            {
                "id": "shrink_chunk_size",
                "patch": {"chunk_size": "decrease"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: 세 갈래(모델약함/도메인약함/청크희석) 구분은 Eval 토픽클러스터 신호로
        #   갈린다. rules는 후보만 나열, 선택은 planner가 finding.metadata 보고 결정.
    },

    "retrieval_missing_gold": {
        "group": "A",
        "assigned": "이승준",
        "status": "ready",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
                "id": "adjust_chunk_size",
                "patch": {"chunk_size": "adjust"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "retrieval_incomplete_enumeration": {
        "group": "A",
        "assigned": "이승준",
        "status": "draft",              # top_k 처방만 실행 가능, MMR 옵션은 스키마 미정
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "dynamic_top_k",
                "patch": {"top_k": "increase"},   # 나열형은 gold 개수 > 고정 top_k
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            # TODO(index-합의): MMR 다양성 옵션 필드가 index_config에 없음.
            #   {"id": "enable_mmr", "patch": {"mmr": True}, ...}
        ],
    },

    "retrieval_missing_bridge_dependency": {
        "group": "A",
        "assigned": "권성우",
        "status": "draft",              # multi-hop query rewrite / max_hops 스키마 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "enable_query_decomposition",
                "patch": {"query_rewrite": "decompose", "max_hops": "increase"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "expand_bridge_entity_query",
                "patch": {"bridge_entity_expansion": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: query_rewrite/max_hops 필드가 AgentDoctorState/index_config에 없음.
    },

    "chunking_context_mismatch": {
        "group": "A",
        "assigned": "권성우",
        "status": "draft",              # chunking_strategy 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "increase_chunk_overlap",
                "patch": {"chunk_overlap": "increase"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "switch_chunking_strategy",
                "patch": {"chunking_strategy": "recursive_sentence"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # NOTE: overlap 증가는 현재 index_config에 존재하지만 chunking_strategy는 추가 합의 필요.
    },

    "chunking_overchunking": {
        "group": "A",
        "assigned": "권성우",
        "status": "draft",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
        "assigned": "권성우",
        "status": "draft",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "decrease_chunk_size",
                "patch": {"chunk_size": "decrease"},
                "reindex": True,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
    },

    "reranker_low_recall": {
        "group": "A",
        "assigned": "권성우",
        "status": "draft",              # reranker 필드가 아직 index_config에 없음
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "widen_rerank_candidates",
                "patch": {"rerank_candidates": "increase"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "relax_reranker_threshold",
                "patch": {"reranker_threshold": "decrease"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: reranker 관련 config 필드와 실제 reranker 단계가 아직 없음.
    },

    "reranker_low_precision": {
        "group": "A",
        "assigned": "권성우",
        "status": "draft",              # reranker 필드가 아직 index_config에 없음
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
        # BLOCKER: reranker 관련 config 필드와 실제 reranker 단계가 아직 없음.
    },

    # ═══════════════════════════════════════════════════════════════
    #  B그룹 — 생성 실패 (Oracle Test 실패)
    # ═══════════════════════════════════════════════════════════════

    "generation_hallucination": {
        "group": "B",
        "assigned": "이승준",
        "status": "draft",              # 로직은 확정, generation_config 필드 부재로 블로킹
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "strict_grounding_prompt",
                "patch": {"system_prompt": "context에 없으면 모른다고 답하라",
                          "require_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "upgrade_generation_model",
                "patch": {"generation_model": "upgrade"},  # 프롬프트로 안 되면 최후 수단
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: core/state.py 에 generation_config 필드 없음. 추가 전까지 draft.
    },

    "generation_partial_answer": {
        "group": "B",
        "assigned": "이승준",
        "status": "draft",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "completeness_prompt",
                "patch": {"system_prompt": "모든 하위 질문에 빠짐없이 답하라"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "checklist_review_step",
                "patch": {"answer_checklist_review": True},  # 답변 누락 점검 단계 추가
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 필드 없음.
    },

    "generation_contradiction": {
        "group": "B",
        "assigned": "이승준",
        "status": "draft",              # 재실행형(LLM 재검증 패스), 실행 방식도 별도 확정 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
        "assigned": "이승준",
        "status": "draft",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "restate_question",
                "patch": {"restate_question": True},  # 답변 전 질문 재진술 강제
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 없음.
    },

    "generation_abstention_failure": {
        "group": "B",
        "assigned": "권성우",
        "status": "draft",              # generation_config 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "strengthen_abstention_prompt",
                "patch": {"abstention_prompt": "strengthen", "grounding_strict": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "require_citation",
                "patch": {"require_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 필드 없음.
    },

    "generation_parametric_overreliance": {
        "group": "B",
        "assigned": "권성우",
        "status": "draft",              # generation_config 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "strict_grounding_prompt",
                "patch": {"grounding_strict": True, "require_citation": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "lower_temperature",
                "patch": {"temperature": "decrease"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: generation_config 필드 없음.
    },

    "generation_numerical_error": {
        "group": "B",
        "assigned": "권성우",
        "status": "draft",              # generation_config 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
        "assigned": "권성우",
        "status": "draft",              # multi-hop answer planning 스키마 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
        "assigned": "이승준",
        "status": "ready",              # top_k 축소는 기존 키로 바로 실행 가능
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
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
        "assigned": "권성우",
        "status": "draft",              # context ordering 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "reorder_context_edges",
                "patch": {"context_ordering": "most_relevant_edges"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "decrease_top_k",
                "patch": {"top_k": "decrease"},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
        ],
        # BLOCKER: context_ordering/top_k 필드가 현재 index_config에 없음.
    },

    "context_noise_interference": {
        "group": "C",
        "assigned": "권성우",
        "status": "draft",              # filtering/MMR/reranker 필드 합의 필요
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [
            {
                "id": "enable_noise_filter",
                "patch": {"noise_filter": True},
                "reindex": False,
                "cost": None,           # 숫자 튜닝 필요
            },
            {
                "id": "enable_mmr",
                "patch": {"mmr": True},
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
        # BLOCKER: noise_filter/mmr/generation_config 필드가 아직 없음.
    },

    # ═══════════════════════════════════════════════════════════════
    #  D그룹 — 데이터 문제 (config로 처방 불가, 사람 개입)
    # ═══════════════════════════════════════════════════════════════

    "corpus_gap": {
        "group": "D",
        "assigned": "이승준",
        "status": "manual",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [],            # config 처방 없음. 튜닝 루프에서 제외, 리포트만.
        # 처방: 사용자에게 관련 문서 추가 수집 요청. Optimize 우회.
    },

    "corpus_gap_partial_hop": {
        "group": "D",
        "assigned": "이승준",
        "status": "manual",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [],
        # corpus_gap과 동일 처리 + 어느 hop이 빠졌는지 리포트에 구체적으로 명시.
    },

    "bad_gold_answer": {
        "group": "D",
        "assigned": "권성우",
        "status": "manual",
        "diagnosis_confidence": None,   # 숫자 튜닝 필요
        "prescriptions": [],
        # RAG pipeline 결함이 아니라 평가셋 문제. Probe ground_truth 수정/제거 또는 사람 검수 큐로 전달.
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


def my_labels(name: str = "이승준") -> list[str]:
    """특정 담당자가 맡은 라벨 목록. 진행상황 체크용."""
    return [k for k, v in LABEL_TO_PRESCRIPTIONS.items() if v.get("assigned") == name]
