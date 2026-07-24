# 최적화 파트 점검 결과와 구현 로드맵

> 작성 기준: 2026-07-23  
> 대상 브랜치: `feature/optimization-label-coverage`  
> 목적: 현재 최적화가 실제로 동작하지 않는 지점과, 구현 틀만 있고 연결되지 않은
> 기능을 구분해 다음 개발 순서를 결정한다.

> 범위 조정: reranker 구현·튜닝·전용 진단은 다른 팀 담당이므로 이 문서의 우리 작업
> 목록에서 제외한다. 현재 문제는 외부 의존성으로만 기록하고, 해당 팀 결과가 들어올 때
> 연동 계약을 검증한다.

## 1. 결론

현재 Optimize 파트의 핵심 반복 구조는 이미 구현되어 있다.

```text
진단 결과
  → 처방 선택
  → 설정 변경
  → Index/Eval 재실행
  → 점수 비교
  → 유지 또는 롤백
```

문제는 `rules.py`의 처방 수가 부족해서가 아니다. 처방이 있어도 다음 중 한 곳에서
끊어진다.

1. Eval이 그 라벨을 만들지 않는다.
2. Planner가 조건을 제대로 해석하지 않는다.
3. Optimizer가 capability/path를 막는다.
4. Mapper가 실제 state 키로 변환하지 못한다.
5. RAG/Generator가 바뀐 값을 읽지 않는다.
6. Eval에서는 바뀌지만 Serve 재시작 후 설정이 사라진다.

따라서 `rules.py`의 `draft`를 일괄적으로 `ready`로 바꾸는 방식은 피해야 한다.
라벨마다 **진단 → 처방 → 설정 변환 → 실제 소비 → 재평가 → Serve 보존**의 전 경로를
완성해야 한다.

## 2. 현재 실제로 구현된 부분

다음은 코드와 테스트가 존재하는 구현이다.

- 확정된 Finding만 자동 처방 대상으로 삼는 Planner
- 그룹/빈도/비용 기반 우선순위
- top-k, chunk size, chunk overlap 후보 계산
- 한 번에 한 설정 축만 바꾸는 안전장치
- 설정 적용 후 다음 Eval에서 유지/롤백 판정
- 실패 처방 blacklist
- 여러 top-k 후보를 순차 평가하는 internal adapter
- chunk 후보의 저비용 사전검증
- RAGBuilder adapter와 optimizer dispatch
- D그룹의 수동 조치 리포트

기존 `AGENTS.md`, `PROGRESS.md`, `README.md`에는 위 일부를 아직 “미구현”이라고 적은
과거 기록이 섞여 있다. 현재 판단은 실제 코드와 테스트를 우선해야 한다.

## 3. 지금 제대로 작동하지 않는 부분

### 3.1 Hybrid search

현재 RAG 검색기는 hybrid를 지원하지만 Optimize에서는 capability가 꺼져 있어 처방이
거절된다.

더 위험한 점은 `rules.py`가 `use_hybrid=True`를 사용한다는 것이다. Planner가 이를
canonical 경로로 바꾼 뒤 Mapper에 전달하면 boolean `True`가 문자열 `"hybrid"`가
아니므로 오히려 `use_hybrid=False`로 저장될 수 있다.

또한 기본 `index_config`는 이미 hybrid가 켜져 있는데 규칙은 dense baseline에서 hybrid를
켜는 상황을 전제로 한다. 이미 hybrid인 상태에서 동일한 lexical mismatch가 발생하면
“hybrid 켜기”가 아니라 fusion 비율이나 후보 수를 조정해야 한다.

개선안:

- 규칙을 `retriever.search_type="hybrid"` 형태로 통일
- `"dense" ↔ "hybrid"` 왕복 변환 테스트 추가
- 현재 검색 모드를 Finding metadata에 포함
- 이미 hybrid라면 반복 enable을 금지
- `hybrid_dense_weight`와 lexical 후보 수를 정식 최적화 축으로 등록

### 3.2 Reranker — 타 팀 담당, 우리 구현 범위 제외

Reranker 모델 실행 코드는 이미 있다. 하지만 Optimize는 다음 이유로 접근할 수 없다.

- `reranker.enabled`를 state에 쓰는 mapper가 없음
- optimizer capability가 꺼져 있음
- rules backend 지원 경로에서 제외됨
- candidate count는 설정값이 아니라 `top_k × 4`로 고정
- threshold 처방은 있지만 threshold를 읽는 실행 코드가 없음

또한 모델 로드가 실패해 원래 순서를 그대로 반환해도 결과에는 `reranked=True`가 찍힌다.
이 상태로는 reranker 품질 진단을 신뢰할 수 없다.

위 항목은 이 작업에서 구현하지 않는다. 담당 팀 결과를 받을 때 다음 계약만 확인한다.

- canonical state mapping과 안전한 기본값
- 실제 적용 여부와 fallback 이유
- 리랭킹 전후 후보/rank 관측 정보
- `reranker_model=None`이 문자열 `"None"`으로 해석되지 않는지
- Optimize→Eval→재시작된 Serve까지 반영되는 통합 테스트

해당 계약이 들어오기 전에는 reranker 관련 처방을 `external_dependency`로 두고 자동
실행하지 않는다.

### 3.3 `applies_when`

`retrieval_semantic_mismatch`는 `topic_cluster` 값에 따라 임베딩 교체와 청킹 조정을
나누도록 설계되어 있다. 그러나 현재 Planner는 `applies_when`을 candidate에 복사만 하고
Finding metadata와 비교하지 않는다.

그 결과 조건이 서로 다른 처방도 단순히 리스트 순서대로 시도된다.

개선안:

- scalar/list 조건의 일치 규칙 정의
- 여러 Finding을 묶었을 때 `any`/`all` 집계 규칙 정의
- 필수 신호가 없으면 해당 처방을 선택하지 않도록 처리
- `topic_cluster`를 Eval에서 실제 생산하거나 해당 조건을 제거

### 3.4 실제 적용 처방과 사용자 리포트 불일치

Optimizer는 첫 후보가 막히면 다음 후보를 고를 수 있다. 하지만 최초 적용 리포트는 실제
선택 결과가 아니라 항상 `request.candidates[0]`을 표시한다.

예를 들어 임베딩 교체가 capability에서 막혀 chunk size 축소가 실제 적용되어도 사용자에게
임베딩 교체를 적용했다고 보일 수 있다.

Agent는 Mapper가 반환한 `ConfigDiff`도 버린다. 따라서 무시된 키나 실제 변경 없음이
리포트에 정확히 나타나지 않는다.

개선안:

- `OptimizationResult.selected_candidate`를 Reporter에 전달
- 실제 concrete patch와 ConfigDiff를 리포트의 기준으로 사용
- 예상한 key가 바뀌지 않으면 `applied`가 아니라 실패/불확실 상태로 처리

### 3.5 Eval에서 성공한 설정이 Serve에서 사라지는 문제

top-k, hybrid, reranker처럼 재색인이 필요 없는 설정은 Index가 기존 청크를 그대로
재사용한다. Eval은 `state.index_config`를 직접 넘기므로 변경이 반영된다.

하지만 Serve는 청크만 JSON으로 저장하고 최종 config는 저장하지 않는다. 새 API
프로세스는 오래된 chunk metadata로 설정을 복원할 수 있다. `/search`와 `/answer`의
기본 `top_k=3`도 최적화된 값을 덮어쓴다.

이미 API 서버가 떠 있으면 새 chunks/config를 reload하지 않는 문제도 있다.

개선안:

- chunks와 최종 retrieval/generation config를 하나의 버전된 serving artifact로 저장
- API 요청의 top_k를 optional로 바꾸고 미지정 시 최적화값 사용
- artifact version 기반 reload 또는 안전한 서버 재시작 구현
- 동일 프로세스 테스트가 아니라 API 재시작 후 통합 테스트 추가
- MCP가 항상 넘기는 `top_k=3`도 제거

### 3.6 롤백 판정의 신뢰성

현재 before/after report가 없으면 “판정 불가이므로 유지”한다. 검증하지 못한 변경이
성공처럼 남을 수 있다.

RAGAS 하한선도 임시값이며, `answer_relevancy`와 Eval의 실제 키인
`response_relevancy`가 다르다. 없는 지표는 통과한 것처럼 건너뛴다.

개선안:

- report가 없으면 `inconclusive`로 두고 롤백하거나 pending 유지
- Eval과 Optimize가 공통 metric alias/threshold 정책 사용
- 필수 guardrail 지표 누락과 통과를 구분
- 작은 확률적 흔들림을 개선으로 오인하지 않도록 공통 `min_delta` 적용

### 3.7 진단 깊이 때문에 actionable 라벨이 거의 없는 문제

기본 진단 모드는 FAST이고 Planner는 `confirmed=False`인 예비 Finding을 모두 버린다.
정상 기본 실행에서 확정되면서 ready인 라벨은 사실상
`retrieval_incomplete_enumeration` 하나뿐이다.

- low-rank/lexical/semantic: STANDARD 필요
- generation 원인: DEEP 필요
- context 원인: FULL 필요
- missing-gold/chunk boundary: FAST에서는 주로 예비

따라서 “진단 리포트에는 문제가 있는데 처방 가능한 finding 없음”으로 Serve하는 상황이
쉽게 발생한다.

개선안:

- 최적화를 시작할 때 필요한 진단 tier까지 한 단계씩 제한적으로 승급
- 또는 `needs_deeper_diagnosis`를 명시적으로 보고
- 비용을 아끼기 위해 예비 라벨을 바로 자동 적용하지는 않음

### 3.8 검색 라벨 우선순위가 일부 ready 라벨을 가리는 문제

STANDARD에서는 keyword가 gold를 찾으면 lexical, 못 찾고 corpus에 gold가 있으면
semantic이 `retrieval_missing_gold`보다 먼저 확정된다. 따라서 missing-gold의 ready
처방은 정상 자원이 모두 있는 경로에서 거의 선택되지 않는다.

`retrieval_incomplete_enumeration`도 질문이 실제 나열형인지 확인하지 않고 gold 개수와
검색 결과 개수의 비율만으로 확정되며, 더 정밀한 bridge/low-rank/semantic 원인을
선점할 수 있다.

개선안:

- missing-gold를 설명되지 않은 검색 실패 fallback으로 재정의하거나 고유 신호 추가
- Probe에 나열형 여부와 기대 항목 수를 명시
- top-k counterfactual에서 실제로 항목이 회복될 때만 enumeration 확정
- 원인이 겹치는 fixture로 precedence 테스트

### 3.9 Target metric이 실제 유지 판정에 사용되지 않는 문제

Rules는 `context_recall`, `context_precision` 같은 target metric을 갖고 있지만 Planner가
모든 요청의 primary metric을 `overall_score`로 고정한다. History도 target metric 개선을
확인하지 않고 전체 점수가 조금이라도 오르면 유지할 수 있다.

또한 internal sweep의 마지막 후보가 best면 `verified` 상태로 바로 Serve할 수 있어,
아직 임계값 미달이고 다른 처방이 남아 있어도 최적화를 끝낼 가능성이 있다.

개선안:

- target metric을 primary objective로 사용
- overall score는 전역 guardrail로 사용
- target metric 누락은 inconclusive
- target metric `min_delta`와 비열화 제한 적용
- sweep 완료 후 gate 미통과·예산 잔여면 일반 Planner 루프로 복귀

### 3.10 폴백이 최적화 성공처럼 평가되는 문제

현재 다음 폴백이 구조화된 실행 정보 없이 발생한다.

- 임베딩 모델 실패 → deterministic hash embedding
- Qdrant/벡터 검색 실패 → keyword 검색
- reranker 로드 실패 → 원래 순서
- LLM/provider 실패 → 첫 context 추출식 답변
- 일부 문서 Index 실패 → 축소된 corpus

요청한 component가 실제로 실행되지 않았는데도 정상 trial처럼 점수를 비교하고 처방을
blacklist할 수 있다.

개선안:

- retrieval/generation/index execution details를 공통 구조로 기록
- target stage 미적용 또는 partial corpus면 `inconclusive`
- 인프라/모델 다운로드 실패는 처방 품질 실패와 분리

### 3.11 RAGBuilder가 성공해도 적용되지 않는 문제

RAGBuilder adapter와 optimizer dispatch는 구현되어 있지만 성공 결과는 `best_config`만
가지고 `config_patch`가 없다. Agent는 `config_patch`가 없으면 state에 적용하지 않는다.

정상 Planner도 backend를 `rules` 또는 `internal`로만 고르므로 RAGBuilder는 일반
파이프라인에서 선택되지 않는다.

개선안:

- 모든 backend의 proposed 결과에 concrete `ConfigPatch`를 필수화
- RAGBuilder best config를 검증 후 같은 mapper/diff 경로로 적용
- backend 선택 입력과 필요한 input source/eval dataset 계약 추가
- “adapter 단위 테스트”가 아니라 Agent state 변경 통합 테스트 추가

### 3.12 Generation 진단 자체도 아직 거친 문제

현재 live generation 세부 라벨은 hallucination, partial answer, hop binding 정도다.
하지만 판정 기준이 아직 원인을 충분히 분리하지 못한다.

- `generation_partial_answer`는 실제 하위 요구사항 누락이 아니라 response relevancy 저하로
  판정
- `generation_hop_binding_error`는 multi-hop + 높은 faithfulness라는 넓은 조건
- oracle `bad_gold_answer`가 generation 원인보다 먼저 검사되어 실제 추론 오류를 정답셋
  문제로 보낼 가능성
- 답은 맞지만 context 근거가 약한 parametric overreliance는 성공 조기 종료 때문에
  진단 슬롯까지 가지 못할 수 있음

개선안:

- partial answer는 질문의 요구 component/subquestion coverage로 판정
- hop binding은 hop별 evidence→claim→bridge entity 연결을 검사
- bad-gold는 자동 확정이 아니라 다른 생성 원인을 배제한 뒤 수동 검수 후보로 처리
- correct-but-ungrounded는 성공 여부와 별개로 additive Finding 허용
- generation 원인이 서로 배타적이지 않다면 primary/secondary 라벨 구조 도입

## 4. 구현 틀은 있지만 정상 경로에 연결되지 않은 부분

| 기능 | 현재 있는 틀 | 빠진 연결 |
| --- | --- | --- |
| `propose_only` | schema와 reporter 분기 | 사용자 입력, planner 선택, graph route 계약 |
| RAGBuilder | 큰 adapter와 optimizer dispatch, 테스트 | planner가 선택하지 않음, 실제 input/eval dataset 공급 없음 |
| AutoRAG | backend 이름과 오류 분기 | adapter 전체 |
| generation config | `generate_answer(config=...)` 인자 | state 필드, Eval 전달, provider/prompt 소비, Serve 저장 |
| contradiction | AspectCritic helper와 record 필드 | 실제 Eval 호출과 diagnose 슬롯 |
| query decomposition | rule과 FULL ablation | 공용 query planner와 Serve 실행 경로 |
| MMR/adaptive retrieval | rule 선언 | Retriever 구현과 관측 정보 |
| context compression/order/filter | 일부 진단 ablation과 rule | 실제 RAG/Serve 변환 단계 |

이 기능들은 “코드가 조금 있으니 거의 완성”으로 보면 안 된다. 정상 파이프라인에서
선택되고, 실제 소비되며, Eval과 Serve가 같은 동작을 해야 구현 완료다.

특히 `generation_abstention_failure`는 비교적 작은 단위로 먼저 구현할 수 있다.
`answer_exists=False`인 신뢰 가능한 no-answer Probe에서 시스템이 답을 지어냈는지는
규칙 기반으로 확정할 수 있다.

## 5. 라벨 전체 현황

현재 `rules.py`에는 25개 라벨이 있고 `diagnose.py`가 실제로 만들 수 있는 라벨은
17개다.

### 5.1 현재 일부라도 자동 실행 가능한 라벨

- `retrieval_missing_gold`
  - top-k, chunk overlap, chunk size는 실행 가능
  - query expansion은 미구현
- `retrieval_incomplete_enumeration`
  - top-k는 실행 가능
  - MMR, adaptive retrieval은 미구현
- `chunking_context_mismatch`
  - overlap과 size는 실행 가능
  - chunk strategy는 계약 불일치
- `too_long_context`
  - top-k와 chunk size는 실행 가능
  - context compression은 미구현
- `retrieval_semantic_mismatch`
  - chunk size fallback은 실행 가능
  - embedding capability, 조건 분기, chunk strategy는 불완전

단, 기본 FAST에서는 이 라벨 대부분이 `confirmed=True`가 되지 않으므로 실제 자동
실행 가능성과 규칙의 `ready` 표시는 구분해서 보아야 한다.

### 5.2 진단과 규칙은 있지만 intended 처방이 차단된 라벨

- `retrieval_low_rank`: reranker 담당 팀의 연동 결과 대기
- `retrieval_lexical_mismatch`: hybrid capability 차단 및 boolean 변환 오류

### 5.3 Eval이 만들지만 draft라 자동 처방되지 않는 라벨

- `retrieval_missing_bridge_dependency`
- `generation_hallucination`
- `generation_partial_answer`
- `generation_hop_binding_error`
- `lost_in_the_middle`
- `context_noise_interference`

이 중 `lost_in_the_middle`의 `decrease_top_k`처럼 일부 처방은 이미 지원되는 축이다.
현재 status가 라벨 단위라 지원되는 처방까지 함께 막힌다. 처방별 readiness가 필요하다.

### 5.4 rules에만 있고 실제 진단 함수가 없는 라벨

- `chunking_overchunking`
- `chunking_underchunking`
- `reranker_low_recall`
- `reranker_low_precision`
- `generation_contradiction`
- `generation_misinterpretation`
- `generation_abstention_failure`
- `generation_parametric_overreliance`
- `generation_numerical_error`

`generation_contradiction`은 AspectCritic helper와 `EvalRecord.aspect` 필드는 있지만 실제
Eval 호출과 diagnose 함수가 연결되지 않았다.

### 5.5 자동 최적화 대상이 아닌 라벨

- `corpus_gap`
- `corpus_gap_partial_hop`
- `bad_gold_answer`

이 세 라벨은 문서 추가나 평가셋 수정이 필요한 문제이므로 수동 조치가 올바른 종착점이다.

다만 현재 corpus-gap 판정은 주로 gold chunk ID가 현재 corpus에 존재하는지를 본다.
일반 사용자 질문에 필요한 지식이 의미적으로 없는지까지 판정하는 기능은 아니다.
`corpus_gap_partial_hop`도 실제로 빠진 gold ID/hop을 metadata에 기록하도록 보강해야
사용자에게 구체적인 수집 조치를 안내할 수 있다.

### 5.6 규칙이 없는 예비 롤업

`generation_failure`는 낮은 진단 모드에서 세부 원인을 확정하지 못했을 때 사용하는
`confirmed=False` 롤업이다. 여기에 임의 처방을 붙이기보다 진단 모드를 높여
hallucination/partial/hop-binding 등으로 세분화해야 한다.

## 6. 추천 구조 개선

### 6.1 파라미터 레지스트리

현재 config 경로 정보가 `rules.py`, `config_mapper.py`, `optimizer.py`에 분산되어 있다.
다음 정보를 하나의 레지스트리로 모으는 것이 좋다.

- canonical path
- 실제 state target/key
- 값 타입과 허용 범위
- 필요한 capability
- 재색인 여부
- 실제 consumer

예:

```python
ParameterSpec(
    canonical_path="retriever.search_type",
    target="index_config",
    state_key="use_hybrid",
    value_type=str,
    allowed_values=("dense", "hybrid"),
    capability="hybrid_search",
    reindex_required=False,
)
```

Rules는 canonical path만 사용하고 Mapper와 Optimizer는 같은 레지스트리를 읽게 한다.

### 6.2 처방별 readiness

라벨 전체에 `ready`/`draft`를 붙이는 대신 처방마다 다음을 기록한다.

```python
{
    "id": "enable_hybrid",
    "status": "ready",
    "requires": ["hybrid_search", "dense_hybrid_rank_evidence"],
    "search_space": {"retriever.search_type": ["dense", "hybrid"]},
}
```

라벨은 실행 가능한 처방이 하나 이상일 때 actionable이 된다.

### 6.3 Hybrid 토글도 counterfactual 평가

Hybrid는 무조건 켜는 방식보다 현재 상태의 반대값을 한 번 적용하고 기존 history
rollback으로 판단하는 것이 안전하다.

```text
dense → hybrid → Eval → 좋아지면 유지, 아니면 dense 복원
```

단, 어떤 stage가 실제 실행됐는지와 fallback 여부를 Eval이 반드시 알아야 한다.

### 6.4 실행 결과 계약 통일

Rules, internal, RAGBuilder 등 어떤 backend를 쓰더라도 Agent가 받는 성공 결과는 다음을
항상 포함해야 한다.

- 실제 선택된 prescription
- concrete `ConfigPatch`
- 예상 reindex 여부
- 실행/폴백 provenance
- 적용 후 `ConfigDiff`

이 계약이 없으면 backend별로 “성공했지만 state는 안 바뀜” 같은 예외가 생긴다.

## 7. 구현 우선순위

### 1단계: 계약 테스트부터 추가

- ready 처방은 canonical path만 사용하는지
- path가 mapper/optimizer/consumer에 모두 등록되어 있는지
- rules 라벨과 diagnose 라벨이 닫혀 있는지
- `applies_when`이 실제 metadata를 거르는지
- Reporter가 실제 선택 candidate와 diff를 보여주는지
- report 누락이 성공 유지로 처리되지 않는지
- target metric 누락과 fallback 실행이 inconclusive로 처리되는지

### 2단계: Hybrid 최적화 완성

- state 기본값
- mapper/optimizer capability
- dense/hybrid/fallback truthful details
- Eval dense↔hybrid counterfactual 진단
- Serve config 저장/reload
- dense↔hybrid 통합 테스트

이 단계는 지금 가장 적은 변경으로 눈에 보이는 최적화 효과를 만들 수 있다.

### 3단계: 청킹과 검색 진단 라벨 완성

- over/underchunking 신호
- chunk strategy 이름/매핑 통일
- `topic_cluster` 계약 또는 제거
- 처방별 readiness
- retrieval cause precedence와 enumeration 신호 개선

Reranker 전용 라벨은 담당 팀 결과가 들어올 때까지 draft/external로 유지한다.

### 4단계: 고급 검색 기능

- query expansion
- multi-hop decomposition
- MMR
- adaptive retrieval
- context compression/order/noise filter

각 기능을 Eval과 Serve가 같은 구현으로 공유해야 한다.

### 5단계: Generation 최적화

- `generation_config` 상태 추가
- Eval/RAG/Serve 전달
- provider temperature/model과 prompt policy 소비
- 독립 처방은 한 축씩 분리
- contradiction, abstention, numerical 등 진단을 하나씩 활성화

### 6단계: 선택형 backend와 사용자 모드

- 개선안만 보기(`propose_only`) 입력/route
- rules/internal/RAGBuilder 선택 정책
- RAGBuilder 실제 데이터 입력
- AutoRAG는 adapter가 생기기 전까지 노출하지 않음
- RAGBuilder 결과를 실제 state에 적용하는 공통 patch 계약

## 8. 완료 기준

한 라벨이 “구현 완료”가 되려면 다음을 모두 만족해야 한다.

- Eval이 재현 가능한 신호로 라벨을 생산한다.
- `confirmed=True`의 근거가 metadata에 남는다.
- 실행 가능한 처방이 최소 하나 있다.
- 처방은 canonical 단일 축이다.
- Mapper와 capability 검사를 통과한다.
- 실제 RAG/Index/Generator가 값을 읽는다.
- fallback 여부까지 포함해 Eval이 변경 효과를 측정한다.
- 악화 시 history가 원래 설정으로 롤백한다.
- Reporter가 실제 처방과 실제 diff를 보여준다.
- 새 Serve 프로세스에서도 최적화 설정이 유지된다.
- 단위 테스트와 `Eval → Optimize → Index → Eval` 통합 테스트가 통과한다.

이 기준을 통과하기 전에는 `rules.py`의 status를 `ready`로 올리지 않는 것이 안전하다.

## 9. 주요 근거 파일

- 진단 라벨과 우선순위: [`agents/eval/diagnose.py`](../eval/diagnose.py)
- 진단 신호와 ablation: [`agents/eval/signals.py`](../eval/signals.py)
- Eval 실행 경로: [`agents/eval/agent.py`](../eval/agent.py)
- 처방 테이블: [`rules.py`](rules.py)
- 처방 선택: [`planner.py`](planner.py)
- capability와 backend: [`optimizer.py`](optimizer.py)
- canonical 설정 변환: [`config_mapper.py`](config_mapper.py)
- 적용/반복 제어: [`agent.py`](agent.py)
- 유지·롤백 판정: [`history.py`](history.py)
- 공통 Retriever: [`agents/rag/retriever.py`](../rag/retriever.py)
- Index 소비 경로: [`agents/index/agent.py`](../index/agent.py)
- Generator: [`agents/rag/generator.py`](../rag/generator.py)
- Serve 직렬화/API: [`agents/serve/agent.py`](../serve/agent.py),
  [`agents/serve/api.py`](../serve/api.py)
