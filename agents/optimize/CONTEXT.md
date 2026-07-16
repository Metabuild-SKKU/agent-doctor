# Optimize 모듈 설계 컨텍스트 (Claude Code용)

> 이 파일은 Optimize 파트를 작업하는 AI 에이전트가 **먼저 읽어야 하는** 설계 맥락이다.
> 여기 적힌 원칙과 충돌하는 코드를 생성하지 말 것. 새 코드가 아래 "거부된 패턴"을
> 다시 끌어들이면 멈추고 사람에게 확인할 것.

---

## 1. 프로젝트 한 줄 요약

Agent Doctor = 기존 RAG 파이프라인을 입력받아 **"왜 실패하는지 진단"**하고 처방하는
메타 에이전트. 산학협력 프로젝트(SKKU).

**유일한 방어선 = 진단이지 최적화가 아니다.** AutoRAG는 brute-force 탐색이고,
우리는 인과 진단으로 탐색 범위를 좁힌다. 이 구분이 흐려지는 코드는 프로젝트의 존재
이유를 무너뜨린다.

---

## 2. 방식1 vs 방식2 (가장 중요)

- **방식1 (거부됨):** 문제 상황에 정해진 config 세트를 한꺼번에 수정. 효과 귀속 불가.
- **방식2 (채택됨):** 원인을 진단한 뒤 **필요한 것만 하나씩** 수정 → 순차 검증. 효과 설명 가능.

### 방식2를 지키는 규칙 (코드에서 반드시)
1. 한 라벨의 여러 config를 **동시에 바꾸지 않는다.** 처방은 순서 있는 리스트로 하나씩 적용.
2. 처방을 적용하면 **검증 → 유지/롤백** 판단이 따라와야 한다.
3. **라벨은 진단 신호로 이름 짓는다. 처방(고치는 방법)으로 이름 짓지 않는다.**
   - 예: `chunking_overchunking`(처방 이름)은 방식1 후퇴. `retrieval_missing_gold`(신호)가 옳음.

---

## 3. 담당 경계 (엄격히 지킬 것)

- **진단(라벨 판별)은 Eval 팀이 한다.** Optimize는 **라벨을 받아서 처방만** 한다.
- Optimize 코드 안에 "이 라벨이 맞는지 판별하는 로직"(코퍼스 재조회, LLM 재검증 등)을
  넣지 말 것. 그건 Eval 소관이다. Optimize는 `Finding`을 신뢰하고 처방 매핑만 수행.
- Optimize가 만지는 파트: `agents/optimize/` 전체 + (합의 후) `core/schema.py`,
  `core/state.py`의 optimize 관련 필드.

담당자: A그룹(retrieval_*, chunking_*) = 이승준 / B·C·D그룹 = 권성우

---

## 4. 우선순위 공식

```
우선순위점수 = (빈도 × 진단신뢰도) ÷ 처방비용
```

- **빈도** = `len(finding.affected_probes)` (측정 쉬움)
- **진단신뢰도(diagnosis_confidence)** = 라벨별 판별신호 명확도 **상수** (lexical 0.9, semantic 0.6…)
  - 주의: "진단이 맞을 확률"이지 "고치면 좋아지는 정도"가 아니다. 후자는 impact이며 별개.
- **처방비용** = 재색인 등급 (런타임=1, 재색인=3, 데이터=5)
- **영향도(impact)** 는 공식에서 제외. 처방 전 예측 불가 → 검증에서 결과로 실측.
- MVP는 빈도÷비용만으로 시작 가능.

### 용어 주의 (혼동 금지)
- `diagnosis_confidence` = 진단 확신도 (상수, rules.py에 박음)
- `impact` = 처방 기대효과 (검증으로 학습, 나중 단계)
- 이 둘을 같은 변수로 쓰지 말 것.

---

## 5. 롤백/유지 = 검증 결과가 결정 (greedy + backtrack)

- **성공(점수↑):** 유지하고 다음 진단으로 진행
- **실패(점수↓):** 롤백 + 다음 후보 + [라벨,처방] 블랙리스트 (무한반복 방지)
- **부분성공:** ① 하한선(guardrail) 위반이면 무조건 롤백 → ② 사용자목적 가중치로 순이득 판단
  → ③ 애매하면 보수적 롤백

### 단일 점수 시스템 (사용자 표시용)
- 지표별 달성도(현재값/임계값, 낮을수록 좋은 건 뒤집기) → 가중평균 → ×100
- 모든 임계값 달성 = 100점
- **표시는 반올림, 롤백 판단은 원값 사용.** "참고용 추정치"임을 명시.

---

## 6. 처방 비용 = 재색인 여부 (구조적으로 중요)

각 처방은 `reindex` 플래그를 가진다:
- **재색인형(reindex=True, cost=3):** `embedding_model`, `chunk_size`, `chunk_overlap`,
  `chunking_strategy`. corpus 전체 재임베딩·재저장 필요 → **그래프가 Index 노드를 경유해야 함.**
- **런타임형(reindex=False, cost=1):** `use_reranker`, `use_hybrid`, `top_k`,
  `context_ordering` 등. 검색/생성 시점 파라미터 → Eval 직행 가능.

⚠️ 재색인형 처방을 Index 없이 바로 Eval로 넘기면 안 된다. config만 바뀌고 실제 색인은
안 바뀌어서 처방이 반영되지 않는다. `graph.py`의 `add_edge("optimize","index")`가 이 때문에 존재.

---

## 7. rules.py 구조 (이미 초안 존재)

`agents/optimize/rules.py` 의 `LABEL_TO_PRESCRIPTIONS` 는 **선언적 데이터 테이블**이다.
로직(실행/우선순위 계산)을 여기 넣지 말 것.

각 라벨 항목:
- `group`: A/B/C/D
- `status`: ready(실행가능) / draft(처방·신호 미확정) / todo(미담당) / manual(사람개입)
- `diagnosis_confidence`: 상수 (미확정이면 None)
- `prescriptions`: **순서 있는 리스트** (가벼운 것 먼저). 각 원소는
  `{id, patch, reindex, cost, guardrail}`

`is_actionable(label)` 이 True인 라벨만 planner가 실행. draft/todo/manual은 실행 금지.

---

## 8. 거부된 패턴 (다시 끌어들이지 말 것)

- ❌ 라벨을 처방 이름으로 짓기 (`chunking_overchunking` 류) → 방식1 후퇴
- ❌ 한 라벨에 여러 config 동시 적용 → 방식1 후퇴
- ❌ Optimize 안에서 라벨 판별(진단) 수행 → Eval 담당 침범
- ❌ 재색인형 처방을 Index 경유 없이 Eval로 직행
- ❌ 신호 검증 없이 라벨 추가 → 모든 신규 라벨은 "Eval이 고유·저비용 신호를 계산 가능한가?"
  게이트를 통과해야 함
- ❌ `diagnosis_confidence`(상수)와 `impact`(학습값)를 혼용

---

## 9. 미해결 (사람 확인 필요, 임의로 결정하지 말 것)

1. `chunking_context_mismatch / overchunking / underchunking` 3개의 운명:
   Eval이 "gold 청크 경계 걸침" 신호를 저비용으로 계산 가능한지 미확인.
   → 가능하면 context_mismatch만 missing_gold 하위로, 나머지 2개는 폐기. 지금은 전부 draft.
2. index_config에 추가할 키들(`use_reranker`, `top_k`, `rerank_candidates`,
   `reranker_model`, `chunking_strategy`, `mmr`): Index 팀 합의 대기.
3. `missing_bridge_dependency`의 `query_rewrite`가 generation_config 쪽이라 B그룹과 경계 겹침.

---

## 10. 파일 구조 (계획)

```
agents/optimize/
  agent.py         # 진입점: state 읽고 → planner 호출 → state.index_config 수정 + iteration++
  planner.py       # 라벨 읽고 우선순위 정렬 → 처방 후보(search space) 생성
  rules.py         # 라벨→처방 테이블 (선언적 데이터) [초안 완성]
  schemas.py       # OptimizationRequest, ConfigPatch, Guardrail 등 데이터 모델
  optimizer.py     # 어댑터 호출 상위 래퍼
  config_mapper.py # RAGBuilder 결과 → state.index_config 형식 변환
  history.py       # optimization_history + 롤백/블랙리스트 관리
  reporter.py      # 사용자용 처방 리포트 생성
  adapters/
    ragbuilder_adapter.py  # 추상 요청 → RAGBuilder 구체 config 변환
    internal_adapter.py    # 도구 없이 직접 patch (일단 구조만)
```

구현 순서: rules.py(완) → schemas.py → planner.py → agent.py 껍데기 →
adapters → 검증·롤백 → 라벨 확장.