# 논문 기반 업그레이드 로드맵

최신 RAG 논문 조사(2026-08-06)와 실측 로그 분석에서 나온 개선 항목 8개.
**순서가 결론이다** — 아래 "왜 이 순서인가"를 먼저 읽을 것.

근거 실행: `output/logs/corpus_20260804_103059.txt` (30문항 · 5반복 · KorQuAD 20문서)

---

## 현황

| # | 항목 | 출처 | 상태 |
|---|---|---|---|
| 1 | 평가 문항 30 → 150 | RAISE | ✅ 문서화 완료 (`.env.example`·`agents/eval/README.md`) |
| 2 | 같은 config 3회 반복해 σ 측정 | RAISE | ✅ `tools/measure_eval_noise.py` — **실측 미실행** |
| 3 | 라벨 2개 추가 검토 (E14·E15) | RAGEC | ⏸ 팀 합의 대기 |
| 4 | **채점을 부분점수로** | RAGChecker | ⬜ 미착수 — **최대 임팩트** |
| 5 | `faithfulness` 를 라벨 게이트에서 제외 | 심판 감사 | ⬜ 미착수 |
| 6 | 개선 마진을 통계로 | Noisy but Valid | 🔒 2번 실측 필요 |
| 7 | 처방 선택을 밴딧으로 | AutoRAG-HP | 🔒 4·6 필요 |
| 8 | 라벨 정확도 측정 | Doctor-RAG | 🔒 코퍼스 배관 필요 |

---

## 왜 이 순서인가

**지금 종합점수는 노이즈와 구분되지 않는다.** 그래서 무엇을 고쳐도 좋아졌는지 측정할 수 없다.

```
반복 0 : 종합 75   (리랭커 off)
반복 5 : 종합 73   (리랭커 off, rerank_candidates 만 다름)
```

리랭커가 꺼지면 `rerank_candidates` 는 검색에 관여하지 않으므로(판정창에만 쓰임) 두 반복은
기능적으로 같은 config 다. 편차 0.020 = `MIN_IMPROVEMENT_MARGIN`(0.020).

같은 실행에서 살아남은 처방 하나가 `+0.020 ≥ 마진 0.020` 턱걸이였다.

→ **1·4 를 먼저 해야 나머지를 잴 수 있다.** 7(밴딧)은 보상 신호가 노이즈면 학습 자체가 안 된다.

---

## 항목별

### 1. 평가 문항 30 → 150 ✅

RAISE 가 13개 탐색 알고리즘을 벤치마크하며 프록시 크기를 ablation 했고, **100~200 예제에서
안정**된다고 보고한다(그 아래는 시드 간 편차가 큼). 우리는 30이었다.

`KORQUAD_MAX_DOCS` 가 먼저 문서를 자르고 그 안에서 QA 를 세므로 둘을 같이 올려야 한다.
정제본 기준 QA 150 건 = 문서 267 개(실측).

### 2. σ 측정 도구 ✅ (실측 미실행)

`history.max_repeated_measurement_spread()` 가 정상 실행에서 σ 를 줍도록 설계돼 있으나,
롤백이 Eval 을 재실행하지 않고 진단 캐시를 복원해(로그의 "롤백 진단 캐시 복원") 같은 config
가 두 번 측정되는 일이 없다 → 항상 `None`. 그래서 전용 도구를 뒀다.

```powershell
python tools/measure_eval_noise.py -n 3
```

150문항 3회 ≈ 75분 / $0.55. **비용이 드는 이유**: 재려는 흔들림이 LLM 에서 나오므로 LLM 을
실제로 불러야만 관측된다.

### 3. 라벨 2개 추가 검토 ⏸

RAGEC 택소노미(16개)와 우리 31개를 대조한 결과 **이름까지 같은 게 6개**로 외부 검증됐다.
우리에게 없는 것이 둘.

| 그들 | 뜻 |
|---|---|
| E14 Contextual Misalignment | 답은 맞는데 질문에 답하지 않음 |
| E15 Chronological Inconsistency | 사건·사실의 시간 순서를 뒤바꿔 말함 |

도입 판단은 기존 원칙을 따른다 — **처방이 기존 라벨과 다르면 새 라벨, 겹치면 아니다.**
(`agents/optimize/rules.py` 의 라벨 도입 합의)

### 4. 채점을 부분점수로 ⬜ ← 다음

**우리 최대 결함.** `span_recall_at_k` 는 골드 구간을 빈틈없이 덮어야 1점인 이진 판정이라,
정답을 맞힌 실행도 recall=0 이 된다. 실측 **30문항 중 14건(47%)**.

RAGChecker(NeurIPS 2024)는 골드와 응답을 **원자 claim** 으로 쪼개고 entailment 로 대조한다.
recall 이 자연히 연속값이 된다(덮은 claim / 전체 claim). 귀속 원리도 함께 온다.

> 빠진 claim = 검색 결함 · 근거 없는 claim = 생성 결함

부수 효과: `tools/build_clean_qa.py` 가 "정답이 여러 곳이라 모호"로 버린 **1,169건**을
부분점수로는 회수할 수 있다.

설계 갈림길: 골드 분해를 **LLM claim 추출**로 할지 **규칙 기반**으로 할지.

### 5. `faithfulness` 를 라벨 게이트에서 제외 ⬜

같은 실행의 `probe_qa_4195` 를 5회 전부 추적하면, 답도 검색 결과도 같은데 **반복 3에서만**
`faithfulness` 가 1.000 → 0.000 으로 튀어 `bad_gold_chunk` 의 근거성 조건에서 탈락했다.
그 한 번의 재분류가 처방 `rerank_candidates 20→22` 를 만들었고, KEEP 판정으로 최종 config
에 남았다(`context_precision` 은 0.78 → 0.73 으로 떨어졌는데도).

심판 감사 논문(arXiv 2607.08535)이 **"같은 모델을 여러 번 돌린 배심원단은 에러 상관
ρ=0.944~0.972"** 라고 보고한다 → **반복 평균으로는 안 잡힌다.** 이종 심판을 섞거나, 결정론적
신호(`oracle_f1`·`recall`)만으로 게이트를 세우는 쪽이 맞다.

### 6. 개선 마진을 통계로 🔒

2번 실측 후. 상수를 올릴지, "회차 평균으로 판정"으로 판정 방식 자체를 바꿀지가 선택지다.
후자가 근본이지만 반복당 비용이 3배다.

Noisy but Valid(arXiv 2601.20913)가 보정셋으로 심판 TPR/FPR 을 추정해 분산 보정 임계값을
만드는 방법을 제시한다. ⚠️ **원문 미확인**(PDF 용량 초과) — 채택 전 직접 읽을 것.

### 7. 처방 선택을 밴딧으로 🔒

`rank_action_candidates` 의 첫 정렬 키가 `_tier_of`(A>C>B 하드 순서)라 **점수를 무조건 이긴다.**

| 반복 | 선택된 것 | 밀린 것 | 결과 |
|---|---|---|---|
| 1 | `reranker.enabled:disable` 1.0 | `abstention_relaxed` 2.0 | 롤백 |
| 3 | `reranker.candidate_count:increase` 1.0 | `abstention_relaxed` 3.0 | 마진 턱걸이 |
| 4 | `reranker.enabled:disable` 1.0 | `restate_question` **4.0** | 롤백 |

5회 중 3회가 이 패턴이고 전부 롤백/턱걸이다. 게다가 `reranker.enabled:disable` 은
**두 번 시도돼 두 번 다 실패**했다 — `ActionAttemptKey` 가 baseline 지문을 포함해서,
무관한 축(`rerank_candidates`) 변경이 차단을 풀었다.

AutoRAG-HP(EMNLP Findings 2024)의 2단 계층 MAB 는 상위가 **어느 모듈**을, 하위가 **어느 값**을
고른다. 우리 구조와 대응되는데 차이는 상위 결정이 **학습되느냐 고정이냐**다. 밴딧이면 실패한
팔의 기대값이 내려가 같은 처방을 두 번 뽑지 않는다.

### 8. 라벨 정확도 측정 🔒

Doctor-RAG(arXiv 2604.00865)가 라벨별 정확도 표를 싣는다(Format 94.9% ~ Search 62.0%).
**우리는 진단 정확도를 한 번도 재본 적이 없다.**

정답지 후보: [layer6ai-labs/rag-error-classification](https://github.com/layer6ai-labs/rag-error-classification)
(MIT) — 사람이 라벨링한 RAG 에러 **377건**.

⚠️ CSV 에 검색된 청크도 원본 문서도 없다(질문/정답/RAG답변/라벨뿐). 우리 `diagnose()` 는
순위·recall@k·오라클 트랙·RAGAS 가 필요하므로 **"CSV 넣고 채점"은 불가.** 원본
DragonBall 을 받아 우리 파이프라인으로 다시 돌려야 한다. 배관은 남지만 제일 비싼
것(사람 라벨링)은 끝나 있다.

---

## 채택하지 않은 것

**Doctor-RAG 의 `k†`(최초 실패 지점 국소화) — 도입하지 않는다.**

한 probe 에 검색 라벨과 생성 라벨이 같이 붙으면 하류를 감쇠시키자는 제안이었으나, **우리
생성 라벨은 오라클 트랙을 본다**(`_faith_oracle`·`_correctness_counts_oracle` →
`record.oracle_ragas`). 골드를 통째로 쥐여줬는데도 틀렸다는 뜻이라 검색 실패와 **독립적으로**
성립한다. Doctor-RAG 는 같은 판단을 모델에게 물어봐서 하고 진단 정확도가 62~81% 다.

**우리 반사실 실행이 원리적으로 더 단단하므로 유지한다.**

단, 오라클 판정의 유효성 = 골드의 정확성이다. 골드가 엉뚱하면 오라클도 엉뚱한 청크를 받아
실패하고 생성을 무고한다.

---

## 부수 확인 — KorQuAD 로는 라벨 검증이 불가능하다

RAGEC 377건과 질문 종류를 대조했다.

| 질문 종류 | RAGEC | 우리 실행 |
|---|---|---|
| 여러 문서 종합 | 120 | 0 |
| 시간 순서 | 59 | 0 |
| 멀티홉 추론 | 50 | 0 |
| 여러 문서 비교 | 47 | 0 |
| 단순 사실 | 33 (9%) | **30 (100%)** |

우리 30문항은 전부 `taxonomy·single`, `qtype=None` 이다. 31개 라벨 중 **19개가 미발화**했는데,
멀티홉 라벨은 멀티홉 질문이 0이면 **원리적으로** 못 나온다.

→ 라벨 유효성 검증이 목표라면 **코퍼스 교체가 선택이 아니라 전제다.** 다만 파이프라인이
도는지 확인하는 용도로는 KorQuAD 로 충분하다. 교체 대상은 "라벨 검증용 코퍼스"이지
"실행 확인용 코퍼스"가 아니다.

---

## 출처

| 약칭 | 논문 |
|---|---|
| Doctor-RAG | [Failure-Aware Repair Framework for Agentic RAG](https://arxiv.org/html/2604.00865) |
| RAGChecker | [Fine-grained Framework for Diagnosing RAG](https://proceedings.neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-Paper-Datasets_and_Benchmarks_Track.pdf) (NeurIPS 2024) |
| RAISE | [RAG Design as an Architecture Search Problem](https://arxiv.org/html/2605.30029) |
| AutoRAG-HP | [Automatic Online Hyper-Parameter Tuning for RAG](https://arxiv.org/abs/2406.19251) (EMNLP Findings 2024) |
| 심판 감사 | [When the Judge Changes, So Does the Measurement](https://arxiv.org/html/2607.08535v1) |
| Noisy but Valid | [Robust Statistical Evaluation of LLMs with Imperfect Judges](https://arxiv.org/pdf/2601.20913) |
| RAGEC | [Classifying and Addressing the Diversity of Errors in RAG](https://aclanthology.org/2026.eacl-long.147/) (EACL 2026) |
| 실패 택소노미 | [A Systematic Taxonomy of Failure Modes in RAG](https://aclanthology.org/2026.trustnlp-main.27.pdf) (TrustNLP 2026) |
