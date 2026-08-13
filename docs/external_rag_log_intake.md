# 다른 RAG 실행 로그 연결 — 조사 결과와 연결 설계 (v0)

> 과제: "다른 사람이 만든 RAG를 그 실행 로그로 진단할 수 있는가(될지 안될지 확인)".
> 결론: **된다. 단, 얼마나 깊게 되는지는 로그에 뭐가 담겼느냐에 비례한다.**
> 1차 구현물: `agents/eval/log_intake.py` (적재 + 진단 가능 수준 판정), `tests/test_log_intake.py`.

## 1. 핵심 원리

Agent Doctor의 Eval이 자체 RAG를 돌려 만들어내는 재료가 정확히 세 가지다
(`EvalRecord.retrieved_context` / `generated_answer`, agents/eval/types.py):

**질문 · 검색된 컨텍스트 · 생성된 답변 — "RAG triad"**

외부 RAG의 로그에 이 triad가 남아 있으면, STEP2(자체 검색·생성)를 건너뛰고
로그 값을 EvalRecord에 채워 STEP3~5(지표·진단·리포트)를 재사용할 수 있다
(= **로그 리플레이 모드**). 정답(GT)이 없어도 reference-free 평가가 성립한다.

**단, 현행 코드 기준으로 gold 없이 실측되는 지표는 둘뿐이다** (§4 감사 결과):

- **faithfulness** — 답변이 자기가 찾아온 컨텍스트에 근거했나 → 환각 검출
- **answer relevancy** — 답변이 질문에 대한 답인가 → 동문서답 검출

"컨텍스트가 질문과 관련 있나"(검색 실패 검출)는 이 코드베이스에서
context_precision/recall의 **WithReference 변형**으로만 구현돼 있어
(`metrics_ragas.py`의 `want_prec = ... and has_ref`), **정답 텍스트가 있어야**
검색축 지표가 나온다. 즉 검색/생성 원인 분리를 온전히 하려면 로그에
`ground_truth`(정답 텍스트)를 함께 받는 것이 결정적으로 유리하다.

이것이 QA셋(질문+정답지)과의 결정적 차이 — QA셋은 시험지고, 로그는 답안지다.
진단에는 답안지(실행 증거)가 필요하다. QA셋(정답)은 답안지를 채점하는
보충재로서만 의미가 있다.

## 2. 로그 내용별 평가 가능 수준

| 로그에 있는 것 | 가능한 진단 (현행 코드 기준) | log_intake tier |
|---|---|---|
| 질문 + 답변 | answer relevancy만 (환각·검색 진단 불가) | `qa_only` |
| + 검색 컨텍스트 원문 | faithfulness + answer relevancy(환각·동문서답) + 점수 리포트. **원인 라벨은 0개** (§4) | `triad` |
| + 정답(GT) 텍스트 | + answer correctness, **context_precision/recall(검색축)** — 검색/생성 분리가 지표 수준에서 완성. 라벨 게이트 개방의 재료 | (강력 권장) |
| + config (top_k, chunk_size, …) | 권고문에 현재값 반영 ("512→768로 늘려라") | (가산 정보) |
| + 사용자 피드백 (👍/👎) | 불만족 케이스 우선 진단 표본 선별 | (가산 정보) |
| + 청크별 score/rank/chunk_id | **현행 diagnose는 미소비** — 미래 확장용으로만 수용 | (가산 정보) |
| + 원본 코퍼스 | recall@k, 재인덱싱 A/B 실험까지 | (후속) |

주의: gold 정답 청크 ID(recall용)는 상대 청크 네임스페이스와 정합해야 해서
사실상 요구 불가(우리 워크스페이스에서도 네임스페이스 불일치로 recall 거짓 0을
겪었다). **요구는 "정답 텍스트"까지만** — correctness/context 계열은 청크 정합이
필요 없다.

주의: **top_k·chunk_size 같은 설정값은 로그에 자동으로 남지 않는다**
(요청마다 나오는 출력이 아니라 시스템에 박힌 설정이라서). 우리도 LangSmith
트레이싱을 붙일 때 index_config를 metadata로 직접 심어야 화면에 떴다.
역추정(top_k ≈ contexts 개수, chunk_size ≈ 텍스트 길이 분포)은 근사치일 뿐이므로
로그 스키마에 config 필드를 포함시키는 것을 기본으로 요구한다.

## 3. 연결 트랙 — 상대 스택에 따라 선택

**첫 확인 질문: "무슨 프레임워크로 만들었고, 트레이싱 쓰세요?"**

### 트랙 A: LangSmith 경유 (상대가 LangChain/LangGraph 계열일 때)

```
상대 RAG ──(트레이싱 자동 전송)──▶ LangSmith 워크스페이스 ──(list_runs로 pull)──▶ Eval 리플레이
```

- 상대 부담: LangChain 컴포넌트 기반이면 **env 2줄**(`LANGSMITH_TRACING`,
  `LANGSMITH_API_KEY`)로 코드 수정 0줄. 수제 호출부가 섞여 있으면 `@traceable`
  수동 부착 필요(우리 feature/langsmith 경험과 동일).
- 우리 쪽: `Client.list_runs(project_name=..., is_root=True)`로 root run(질문·답변)
  + `run_type="retriever"` 자식 run(컨텍스트·score)을 뽑아 triad로 변환.
  langsmith SDK는 ragas 경유로 이미 설치돼 있다(0.9.7).
- 장점: 실시간 자동 수집, 파일 주고받기 불필요. 진단 결과를 feedback score로
  상대 트레이스에 되붙여 공유하는 채널로도 재사용 가능.
- 제약: 무료 5k 트레이스/월, API rate limit(≤7일 창 10req/10초), 그리고
  **질문·문서 본문이 SaaS로 전송**된다 — 상대 데이터가 민감하면 거부될 수 있다.

### 트랙 B: JSONL 파일 (프레임워크 무관, 데이터 로컬 유지)

상대 코드의 답변 반환 지점에 5줄 로깅을 요청한다. 프로그램마다 내부 변수명은
다르지만, **필드 이름(스키마)은 우리가 정하고 자기 변수를 끼워 맞추는 매핑은
상대가 1회** 하면 된다(연말정산 서류 양식과 같은 구조). LangChain 계열이면
콜백 핸들러(`on_retriever_end`/`on_llm_end`)를 우리가 만들어 건네줘서 매핑
자체를 없앨 수도 있다.

**스키마 v1** (한 줄 = 요청 1건, JSON Lines, UTF-8):

```json
{"question": "공제 한도는?",
 "contexts": [{"text": "청크 원문...", "chunk_id": "doc3_c12", "score": 0.83,
               "rank": 1, "source_doc": "세금가이드.pdf"}],
 "answer": "700만원입니다",
 "ground_truth": "700만원",
 "gold_contexts": ["연금저축 세액공제 한도는 연 700만원이다."],
 "config": {"top_k": 5, "chunk_size": 512, "embedding_model": "bge-m3", "use_reranker": false},
 "feedback": "thumbs_down", "latency_ms": 1840, "timestamp": "2026-08-06T14:02:11"}
```

| 필드 | 타입 | 수준 | 규칙 |
|---|---|---|---|
| `question` | string | **필수** | 공백뿐이면 그 줄 거부(오류 집계) |
| `answer` | string | **필수** | 사용자에게 반환된 최종 답변 원문(스트리밍이면 완성본) |
| `contexts` | (string \| object)[] | **준필수** | 줄 단위는 관용, **파일 단위 게이트**: 절반 미만이면 tier `qa_only` → 진단 거부가 기본(--allow-qa-only 옵트인). object형 `{text*, chunk_id, score, rank, source_doc}`, 문자열 항목 허용 |
| `ground_truth` | string | 선택·강력 권장 | 정답 텍스트. 지표 2개→5개(correctness, context P·R)를 가른다. v1은 단일 문자열(복수 정답 변형은 후속) |
| `gold_contexts` | string \| string[] | 선택 | 정답 근거 문단의 **원문 텍스트**(청크 ID 아님 — ID는 네임스페이스 정합 불가로 요구 금지). 텍스트 겹침으로 "검색이 정답 근거를 찾았나"를 결정적으로 판정 → 검색축 라벨을 LLM judge 없이 켤 수 있다 |
| `config` | object | 선택·권장 | top_k/chunk_size/embedding_model/use_reranker 등 자유 키. 시스템 설정이라 **1회만 받아 우리가 전 레코드에 병합해도 된다** |
| `feedback` | string | 선택 | thumbs_up/down 권장, 자유 문자열 허용. 불만족 표본 선별용 |
| `latency_ms` / `timestamp` | number / string(ISO 8601) | 선택 | 관측용 |
| 그 외 키 | — | 허용 | raw 보존, 현행 미소비 |

**요구 수준 설계 원칙**: 요구는 상대의 획득 비용에 정렬한다 — 코드에 이미
변수로 존재하는 것만 필수(question/answer/contexts), 사람이 만들어야 하는 것
(ground_truth/gold_contexts)은 절대 필수로 두지 않으며, 변환 노동이 필요한
값(score 방향 등)은 요구하느니 뺀다. 로그는 외부 전송 없이 로컬에서만
처리됨을 요청서에 명시해 보안 승인 비용도 낮춘다.

의미 규칙(형식보다 중요):

- **`contexts`는 "실제 사용분"만** — 검색 후보 전체가 아니라 답변 생성 프롬프트에
  실제로 들어간 top-k. 후보 전체를 넣으면 faithfulness/precision이 오염된다.
- **`score`는 "높을수록 관련"일 때만** — retriever가 distance(낮을수록 가까움)를
  주는 경우 방향 미명시가 랭킹 해석을 뒤집는 단골 사고다. 반전 변환이 번거로우면
  **score를 아예 빼고 달라**(선택 필드다 — 상대에게 변환 코드를 요구하지 않는다).
- **`rank`는 1부터.** `contexts` 배열 순서는 검색 순위순 권장.
- **출처 구분**: question/contexts/answer는 실행 기록(답안지), ground_truth/
  gold_contexts는 시험지 계열 — 실행 로그에 원래 존재하지 않으며 QA셋을 가진
  상대만 채울 수 있다. 없다고 상대를 조르지 말 것.

(적재기 코드는 v0 상태 — `ground_truth`/`gold_contexts` 공식 필드화, qa_only
파일 게이트, gold_contexts 겹침 판정은 다음 PR에서 구현.)

받은 로그는 `python -m agents.eval.log_intake <log.jsonl>`로 적재 검증과
진단 가능 수준(tier) 판정을 바로 확인할 수 있다.

## 4. 코드 감사 결과 — 현행 코드로 가능한 것의 정확한 경계 (2026-08-06)

diagnose/rules/metrics_ragas/scoring 전수 감사 결론:

**외부 로그(gold·우리 인덱스 없음)만으로는 진단 라벨 30개 전부 발동하지 않는다.**
개별 라벨 조건 이전에 관문 구조가 막는다:

- `diagnose.py`의 성공/실패 게이트(`_is_success`)가 `probe.ground_truth` 없으면
  판정 불가(None) → **finding 없이 즉시 반환**. 라벨 루프 진입 자체가 안 된다.
- 검색(A)/생성(B)/컨텍스트(C) 슬롯 진입 조건이 각각 recall(gold 청크) /
  oracle 트랙(gold 컨텍스트 재생성) / recall+oracle을 전제한다.
- Optimize planner는 `confirmed` 라벨만 소비 → **라벨 0개 = 처방 0개** =
  "권고 리포트"가 점수 말고는 빈 껍데기.

추가 확인 사항:

- **A그룹 세부 라벨(순위 손실·리랭커 강등·채널 불일치 등)은 원리적으로 불가** —
  우리 retriever를 재검색해서 얻는 신호(dense/BM25/wide-N 순위, `metrics_search.py`의
  `_ctx.retrieve_fn`/`dense_fn`/`keyword_fn`) 전제라, 로그에 뭘 더 담아도 안 된다.
  로그의 score/rank 필드는 현행 diagnose 어디서도 소비되지 않는다.
- **gold 없이 실측되는 RAGAS는 faithfulness + response_relevancy(+기권 critic)뿐.**
  context_precision/recall·correctness·reasoning_mode는 reference(정답 텍스트) 필요.
- **scoring은 우아하게 저하된다**: 미측정 지표는 0 취급이 아니라 분모에서 제외
  (재정규화). 외부 로그에서는 composite가 quality 성분(faithfulness+relevancy
  재정규화)만으로 산출되고 reliability는 None으로 빠진다.

따라서 리플레이 모드의 산출물 수준은 두 갈래다:

- **(a) 점수 리포트만**: 코드 수정 최소(STEP1·2 대체). "환각도 X, 동문서답도 Y".
- **(b) reference-free 라벨 세트 추가**(§5 초안): 원인 라벨→권고까지 이어진다.

## 5. 리플레이 라벨 세트 초안 (v0) — 산출물 수준 (b)의 설계

원칙: **기존 30개 라벨과 게이트는 손대지 않는다.** 리플레이 모드에서만 쓰는
`ext_` 접두어 라벨 세트를 분리 정의한다 — 기존 라벨의 gold 전제 의미를
오염시키지 않기 위해서다. 게이트는 reference-free로 재정의한다:
`failed = faithfulness < 문턱 ∨ relevancy < 문턱` (GT 있으면 correctness 추가),
지표 미측정(LLM 키 없음)이면 판정 불가 → 라벨 없음(기존 폴백 철학).

| 라벨 | 축 | 발동 조건 | 권고 (기존 rules 처방 재사용) |
|---|---|---|---|
| `ext_generation_hallucination` | 생성 | faithfulness 낮음 ∧ relevancy 정상 (질문엔 답하는데 근거가 없음) | require_citation / lower_temperature / strengthen_abstention |
| `ext_answer_off_topic` | 생성 | relevancy 낮음 (동문서답) | restate_question |
| `ext_retrieval_irrelevant` | 검색 | GT 있으면 context_precision 낮음; GT 없으면 **신규 reference-free context-relevance judge** 낮음 | 검색 설정 점검 목록 (top_k/하이브리드/임베딩 — config 있으면 현재값 명시) |
| `ext_wrongful_abstention` | 생성 | 기권(critic) ∧ 컨텍스트 관련성 정상 | relax_abstention |
| `ext_retrieval_starved_abstention` | 검색 | 기권 ∧ 컨텍스트 무관/빈약 | 검색 설정 점검 + 코퍼스 보강 권고 |
| `ext_context_overflow` | 컨텍스트 | 컨텍스트 총길이 > CONTEXT_CHARS_MAX ∧ faithfulness 낮음 | decrease_top_k / context_compression |
| `ext_grounded_but_wrong` (GT 전용) | 검색/데이터 | faithfulness 높음 ∧ correctness 낮음 → 근거 자체가 틀림/부족 (context_recall 낮음이 방증) | 코퍼스 보강 (corpus_gap 계열) |

설계 노트:

- **검색축 신호 — 구현됨(2026-08-10), 단 골든셋 전용.** `_retrieval_axis()`가
  ① `gold_contexts` 텍스트 겹침 ② `context_precision`(GT 있을 때 RAGAS 실측)을
  보고 "검색이 근거를 가져왔나"를 판정한다. 이 신호 하나로 `ext_retrieval_irrelevant`
  / `ext_wrongful_abstention` / `ext_retrieval_starved_abstention` 셋이 함께 열린다
  — 셋 다 같은 갈림길(근거가 있었나)에 서 있기 때문이다.
  두 신호가 엇갈리면 **나쁜 쪽**을 믿는다: 겹침은 gold 하나만 걸려도 1.00 이라
  무관한 청크가 잔뜩 섞인 검색을 정상으로 읽는다(실측 offtopic: 겹침 1.00 /
  precision 0.20).
  **골든셋이 없는 로그에서는 여전히 침묵한다.** 그 경우를 열려면 reference-free
  context relevance judge(질문 vs 컨텍스트, LLM 프롬프트 1개)가 필요하다 — 미구현.
- confirmed 규칙 유지: 지표가 실측된 경우만 `confirmed=True`(예비 강등 철학 그대로).
- 처방은 자동 적용 없이 **권고 카드**로만 렌더 — planner의 적용 루프를 타지 않고
  리포트가 rules의 patch를 권고 문구로 변환한다.
- `ext_retrieval_*`의 상한은 "검색이 나빴다 + 설정 점검 목록"이다. 순위/리랭커/채널
  수준의 세부 원인 분리는 상대 인덱스 없이는 불가(§4).

## 6. 후속 작업 (이 PR 범위 밖)

1. **Eval 로그 리플레이 모드** — ✅ **구현됨: `agents/eval/replay.py`**
   (`python -m agents.eval.replay <log.jsonl> [--limit=N]`, 산출물 수준 (a) 점수
   리포트, `tests/test_replay.py` 11건). STEP1·2를 코드 수정 없이 우회한다 —
   로그에서 Probe/EvalRecord를 직접 합성해 `build_report`(STEP5)로 바로 간다.
   정직성 장치: recall은 -1 센티널(기본 0.0은 "진짜 0"으로 집계됨), faithfulness
   실측 시 `retrieval_axis` seam으로 reliability의 recall(-1→0) 오염 방지,
   GT+LLM 없음 조합은 신뢰도 보수 집계 경고 출력. 원래 설계했던 seam 3곳:
   - STEP1: probe 생성(`generate_probes`)을 우회하고 **로그 레코드에서 Probe 합성**
     (probe = 로그의 질문). oracle 트랙도 gold 전제라 스킵.
   - STEP2 검색: "로그에서 해당 질문의 컨텍스트를 돌려주는 리플레이 retriever" 주입.
     duck-typed `search()` 계약은 이미 열려 있다(agent.py의 외부 주입 주석 참고).
     `metrics_common.set_context`의 재검색 자원(retrieve_fn 등)은 None으로 주입.
   - STEP2 생성: 로그 답변이 있으면 `generate_answer` 호출을 건너뛰는 분기.
2. **스키마 v0에 `ground_truth` 선택 필드 추가** — §4 감사상 정답 텍스트 유무가
   "지표 2개 vs 5개 + 라벨 게이트"를 가른다. 상대에게 "정답 아는 질문은 정답도
   같이"를 처음부터 요청.
3. **리플레이 라벨 세트 구현** — §5 초안. 산출물 수준 (a)로 1차 PR을 끊고
   (b)는 라벨 설계 리뷰 후 후속 PR 권장.
4. **파이프라인 분기** — 외부 진단 모드는 Ingest/Index를 건너뛰고
   `LoadLogs → Eval(리플레이) → Report`로 흐른다. Optimize의 처방은 남의
   인덱스에 자동 적용할 수 없으므로 **권고 리포트**로만 출력한다.
5. **LangSmith 커넥터** — 트랙 A 채택 시 `list_runs` → 스키마 v0 변환 어댑터.
6. **정답/코퍼스 확보 시 확장** — recall@k, 재인덱싱 A/B 실험
   (이때 비로소 Optimize 루프가 외부 RAG에도 의미를 가진다).

## 7. 한 줄 요약

로그에 triad(질문·컨텍스트·답변)만 있으면 정답지 없이도 증상 측정(환각·동문서답
점수)이 성립하고, 원인 라벨→권고까지 가려면 리플레이 라벨 세트(§5) 구현과 —
가능하면 — 정답 텍스트 수령이 필요하며, 수집은 상대 스택에 따라 LangSmith(자동·
클라우드) 또는 JSONL(수동·로컬) 두 트랙 중 고르되 어느 쪽이든 우리 쪽 수용구는
스키마 v0 하나로 통일된다.
