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
(= **로그 리플레이 모드**). 정답(GT)이 없어도 reference-free 평가가 성립한다:

- **faithfulness** — 답변이 자기가 찾아온 컨텍스트에 근거했나 → 환각 검출
- **answer relevancy** — 답변이 질문에 대한 답인가 → 동문서답 검출
- **context relevancy** — 컨텍스트가 질문과 관련 있나 → 검색 실패 검출

컨텍스트가 있어야 실패 원인을 **검색 쪽 vs 생성 쪽**으로 가를 수 있다.
이것이 QA셋(질문+정답지)과의 결정적 차이 — QA셋은 시험지고, 로그는 답안지다.
진단에는 답안지(실행 증거)가 필요하다.

## 2. 로그 내용별 평가 가능 수준

| 로그에 있는 것 | 가능한 진단 | log_intake tier |
|---|---|---|
| 질문 + 답변 | answer relevancy만 (환각·검색 진단 불가) | `qa_only` |
| + 검색 컨텍스트 원문 | reference-free RAGAS 풀셋, 검색/생성 원인 분리 | `triad` |
| + 청크별 score/rank/chunk_id | 랭킹 문제 vs 커버리지 문제 세부 분리 | (가산 정보) |
| + config (top_k, chunk_size, …) | 처방에 현재값 반영 ("512→768로 늘려라") | (가산 정보) |
| + 사용자 피드백 (👍/👎) | 불만족 케이스 우선 진단 표본 선별 | (가산 정보) |
| + 정답(GT) 또는 원본 코퍼스 | correctness/F1, recall@k, 재인덱싱 A/B 실험까지 | (후속) |

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

스키마 v0 (한 줄 = 요청 1건):

```json
{"question": "공제 한도는?",
 "contexts": [{"text": "청크 원문...", "chunk_id": "doc3_c12", "score": 0.83,
               "rank": 1, "source_doc": "세금가이드.pdf"}],
 "answer": "700만원입니다",
 "config": {"top_k": 5, "chunk_size": 512, "embedding_model": "bge-m3", "use_reranker": false},
 "feedback": "thumbs_down", "latency_ms": 1840, "timestamp": "2026-08-06T14:02:11"}
```

`question`/`answer`만 필수, `contexts`는 강력 권장(없으면 tier가 `qa_only`로
떨어진다), 나머지는 선택. `contexts` 항목은 문자열도 허용(원문만 있는 로그 수용).

받은 로그는 `python -m agents.eval.log_intake <log.jsonl>`로 적재 검증과
진단 가능 수준(tier) 판정을 바로 확인할 수 있다.

## 4. 후속 작업 (이 PR 범위 밖)

1. **Eval 로그 리플레이 모드** — seam 2곳 수정:
   - retriever가 내부 고정 생성(`get_retriever`, agents/eval/agent.py)이라
     "로그에서 해당 질문의 컨텍스트를 돌려주는 리플레이 retriever" 주입 훅 필요.
     duck-typed `search()` 계약은 이미 열려 있다(agent.py의 외부 주입 주석 참고).
   - 답변 생성이 `generate_answer` 직접 호출(Phase B)이라, 로그 답변이 있으면
     생성을 건너뛰는 분기 필요.
2. **파이프라인 분기** — 외부 진단 모드는 Ingest/Index를 건너뛰고
   `LoadLogs → Eval(리플레이) → Report`로 흐른다. Optimize의 처방은 남의
   인덱스에 자동 적용할 수 없으므로 **권고 리포트**로만 출력한다.
3. **LangSmith 커넥터** — 트랙 A 채택 시 `list_runs` → 스키마 v0 변환 어댑터.
4. **정답/코퍼스 확보 시 확장** — correctness/F1, 재인덱싱 A/B 실험
   (이때 비로소 Optimize 루프가 외부 RAG에도 의미를 가진다).

## 5. 한 줄 요약

로그에 triad(질문·컨텍스트·답변)만 있으면 정답지 없이도 외부 RAG 진단이 성립하고,
수집은 상대 스택에 따라 LangSmith(자동·클라우드) 또는 JSONL(수동·로컬) 두 트랙 중
고르며, 어느 쪽이든 우리 쪽 수용구는 스키마 v0 하나로 통일된다.
