# Eval Agent

Index Agent가 만든 `state.chunks`를 대상으로 RAG 파이프라인 품질을 진단하는 에이전트.
"점수를 내는 것"이 아니라 **RAG가 왜 실패하는지 원인(검색 실패 vs 생성 실패)을 구분**하는 것이 목표.

> 현재는 **골격(틀)** 구현입니다. 규칙 지표(STEP3-1)와 진단·리포트 흐름은 동작하며,
> RAGAS·LLM 생성·DataMorgana·Reranker 등은 `[구현 포인트]` 주석으로 표시된 확장 지점입니다.

---

## 역할

```
[Index Agent] → state.chunks
                     ↓
                [Eval Agent]
   Probe 생성 → 검색·생성 → 지표 → 원인 판정 → 리포트
                     ↓
      state.probes, state.report → route_after_eval()
                     ↓
        pass_threshold ? [Serve] : [Optimize]
```

---

## 처리 흐름 (설계 STEP 1~5)

```
STEP1  Probe 생성        probe_gen.py   user_log(우선) / 청크 기반 자동생성(폴백)
STEP2  검색 + 생성        retrieval_temp.py  (임시) 자체 Qdrant 검색 + LLM 생성(폴백) — Index 리트리버 개발 시 삭제 예정
STEP3-1 규칙 지표         metrics.py     Recall@k, token F1, Oracle F1 → 브랜치 판정
STEP3-2 LLM(RAGAS) 진단   ragas_eval.py  Faithfulness/ContextPrecision/Recall/Relevancy (옵션)
STEP4  원인 판정          diagnose.py    브랜치·지표 → Finding(label=처방 라벨)
STEP5  리포트            report.py      overall_score / pass_threshold 산출
```

### 브랜치 판정 (STEP3-1)

| 브랜치 | Recall@k | F1 | Oracle | 의미 |
|--------|----------|----|--------|------|
| `success` | 1 | 통과 | — | 정상 |
| `retrieval_fail` | 0 | — | 통과 | 검색이 병목 |
| `retrieval_fail_gen_fail` | 0 | — | 실패 | 검색·생성 모두 결함 |
| `retrieval_partial` | 0<x<1 | — | 통과 | 부분 검색(멀티홉/나열) |
| `retrieval_partial_gen_fail` | 0<x<1 | — | 실패 | 부분 검색 + 생성 결함 |
| `ambiguous_context` | 1 | 실패 | 통과 | 노이즈 간섭 의심 |
| `ambiguous_gen` | 1 | 실패 | 실패 | 순수 생성 결함 |
| `no_answer_ok` / `no_answer_violation` | — | — | — | 무응답 기권/위반 |

---

## 입출력 (계약)

```
읽기: state.chunks, state.user_questions, state.index_config, state.iteration
쓰기: state.probes, state.report, state.iteration, state.status, state.error
```

`DiagnosticReport`(→ `state.report`)의 핵심 필드:

```python
overall_score : float | None   # RAGAS 가중평균(있으면) / 규칙지표 폴백 / 신호없으면 None
pass_threshold: bool           # overall_score >= 0.8 (설계 §7, types.PASS_SCORE_THRESHOLD)
ragas_scores  : dict           # RAGAS 평균 + 규칙지표 평균 + 브랜치 분포
oracle_accuracy: float | None  # Oracle 트랙 통과율
findings      : list[Finding]  # 원인 + prescription (Optimize 가 소비)
```

---

## 환경 변수

| 변수 | 기본 | 설명 |
|------|------|------|
| `EVAL_ENABLE_LLM` | off | `1/true` 면 STEP3-2 RAGAS(LLM-as-Judge) 진단 활성화 |
| `OPENAI_API_KEY` | — | 있으면 답변 생성/RAGAS 를 LLM 으로, 없으면 폴백 |
| `EVAL_GEN_MODEL` | `gpt-4o-mini` | 답변 생성 모델(응답용) |
| `EVAL_JUDGE_MODEL` | `gpt-4o` | RAGAS 평가(심판) 모델(설계 원칙: 응답≠평가) |
| `EVAL_EMBED_MODEL` | `text-embedding-3-small` | Response Relevancy 코사인용 임베딩 |
| `QDRANT_URL` / `QDRANT_API_KEY` | `:memory:` | 검색 인덱스 대상 |

> 기본값만으로도 **외부 API 없이** 규칙 지표 기반 진단이 동작합니다(폴백 설계).

---

## 테스트

```bash
python tests/test_eval.py       # mock chunks 로 단독 실행 (Index 없이)
```

---

## 파일 구조

```
agents/eval/
├── agent.py       # run(state) — STEP1~5 오케스트레이션
├── types.py       # EvalRecord(내부 중간결과) · 상수 · Branch
├── probe_gen.py   # STEP1  Probe 생성
├── retrieval_temp.py  # STEP2  (임시) 자체 검색 + 답변 생성 — Index 검색 개발 후 삭제
├── metrics.py     # STEP3-1 규칙 지표(Recall@k/F1/Oracle) + 브랜치
├── ragas_eval.py  # STEP3-2 RAGAS/LLM 진단 (옵션)
├── diagnose.py    # STEP4  원인 판정 → Finding
├── report.py      # STEP5  DiagnosticReport 집계
└── README.md      # 이 파일
```

---

## 주요 확장 지점 `[구현 포인트]`

1. **Probe 자동생성** (`probe_gen.py`) — 청크 직접 추출 폴백 → **RAGAS TestsetGenerator**
   (지식그래프 + 시나리오, 75% RAGAS / 20% DataMorgana / 5% 무응답)로 교체.
2. **RAGAS 지표** (`ragas_eval.py`) — ✅ 구현됨. RAGAS 알고리즘(청구 분해·순위 가중 등)을
   OpenAI LLM-as-Judge로 직접 계산(Faithfulness/ContextPrecision/ContextRecall/ResponseRelevancy
   + staleness/contradiction AspectCritic). `EVAL_ENABLE_LLM=1`로 활성화. ragas 라이브러리는
   langchain 버전 충돌로 import가 불안정해 미사용 — 환경이 지원하면 `evaluate_*_track` 내부만 교체.
3. **LLM 생성** (`retrieval_temp.py`) — `_llm_generate` 를 실제 RAG 생성기로. (Index 검색 개발 후에도 유지될 부분)
4. **Reranker** — Bi-Encoder 후 Cross-Encoder 재정렬(2차 개선). Index 검색이 담당하게 될 영역.
5. **진단 신호 훅** (`diagnose.py`) — 처방 파일의 A/B/C/D 16개 라벨을 **라벨당 판정 함수 1개**로
   구현하고(각 함수가 자기 원인·신호를 검사해 Finding/None 반환), **브랜치별 dispatch**(`_dx_*`)가
   설계 STEP4 판정 순서대로 그 함수들을 호출한다. 다만 일부 판별 신호는 파이프라인 추가 실험이
   필요해 훅으로 남겨둠: `_bm25_hits_gold`(lexical vs semantic), `_gold_in_wider_candidates`(low_rank),
   `_gold_in_corpus`(corpus_gap), `_context_shorten_helps`/`_gold_front_helps`(too_long/lost).
   → 각 훅에 재검색/재실행 로직을 채우면 해당 라벨 함수가 자동으로 살아난다.
6. **임시 검색 제거** — Index Agent가 검색 리트리버를 제공하면 `retrieval_temp.py` 를 삭제하고
   `agent.py` 의 검색 호출을 Index 쪽으로 교체. 생성 함수만 eval 로 이관.
