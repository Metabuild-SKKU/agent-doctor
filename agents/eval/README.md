# Eval Agent

Index Agent가 만든 `state.chunks`를 대상으로 RAG 파이프라인 품질을 진단하는 에이전트.
"점수를 내는 것"이 아니라 **RAG가 왜 실패하는지 원인(검색 실패 vs 생성 실패)을 구분**하는 것이 목표.

> STEP1~5 전 구간이 실제로 동작합니다(Probe 자동생성, 검색+생성, 규칙 지표, RAGAS(옵션),
> 원인 판정, 리포트). 남은 확장 지점은 맨 아래 `[구현 포인트]` 참고 — 대부분 "더 정교하게"의
> 문제고 "동작 안 함"은 아니다. STEP2 검색·생성은 공용 모듈 `agents/rag/`(retriever/generator)를
> 사용한다(과거의 임시 파일 `retrieval_temp.py`는 삭제됨).

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
STEP1  Probe 생성        probe_gen.py       user_log(최우선) / 지식그래프 기반 RAGAS 4분면
                                            + DataMorgana-lite + 무응답(Held-out·False Premise)
                                            / 그래프가 비면 단일홉 폴백. eval_probes.json 캐시(probe_store.py)
STEP2  검색 + 생성        agents/rag/        retriever.py(벡터 검색, 키워드 폴백) + generator.py(LLM 생성, 추출식 폴백)
STEP3-1 규칙 지표         metrics.py         Recall@k, token F1, Oracle F1 (diagnose 진입 시 계산)
분류(브랜치 없음)         signals.py         전제 함수(_no_diagnosis/_retrieval_failed/_generation_failed/
                                            _context_applicable)가 옛 브랜치 판정을 대체 — 아래 참고
STEP3-2 LLM(RAGAS) 진단   metrics_ragas.py   Faithfulness/ContextPrecision/Recall/Relevancy (옵션, DEEP 이상)
STEP4  원인 판정          diagnose.py        전제 함수 + 판별 신호 → Finding(label=처방 라벨)
STEP5  리포트            report.py          overall_score / pass_threshold 산출
```

### 분류 방식 — "브랜치 판정"은 폐기되고 게이팅 함수로 대체됨

과거엔 Recall@k·F1·Oracle 조합으로 `Branch` enum(success/retrieval_fail/ambiguous_gen 등)을
먼저 확정한 뒤 그 브랜치 전용 로직을 타는 구조였다. 지금은 **브랜치 자체가 없다** — `diagnose.py`가
16개 라벨 함수를 전부 시도하고, 각 라벨이 자기 전제(`signals.py`)로 스스로 범위를 좁힌다(안 맞으면
자연히 `None`). 옛 브랜치가 갈랐던 조건은 아래 전제 함수들로 흩어져 남아있다:

| 옛 브랜치 | 지금 대응 함수(`signals.py`) |
|---|---|
| `success` / `no_answer_ok` | `_no_diagnosis()` — 진단 불필요(정답셋 없음 · 올바른 기권 · recall=1&F1 통과) |
| `retrieval_fail` / `retrieval_partial` | `_retrieval_failed()` — `0 <= recall_at_k < 1` |
| `*_gen_fail` / `ambiguous_gen` | `_generation_failed()` — oracle 실패 또는 무응답인데 답을 지어냄 |
| `ambiguous_context` | `_context_applicable()` — recall=1·oracle 통과인데 실제 F1만 실패 |

`diagnose()`는 이 전제로 A(검색)/B(생성)/C(컨텍스트) 슬롯마다 `_pick()`으로 확정(confirmed) 우선
하나씩 채택하고, D(데이터, `corpus_gap` 계열)는 additive로 더 붙는다.

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
ragas_scores  : dict           # RAGAS 평균 + 규칙지표 평균 + 결과 분포(diagnosed/ok)
oracle_accuracy: float | None  # Oracle 트랙 통과율
findings      : list[Finding]  # 원인 라벨(확정 우선 정렬). 각 Finding 은 label/confirmed 보유
findings_summary: dict         # {mode, total, confirmed, preliminary, confirmed_labels, preliminary_labels}
```

각 `Finding` 은 `label`(진단명)과 **`confirmed`**(현재 모드에서 확정됐는지)를 가진다.
`confirmed=False`(예비)는 *더 깊은 모드에서 확정 가능한 의심 원인*이며, `overall_score`/`pass_threshold` 를
바꾸지 않는다(지표 기반). Optimize 는 `findings_summary.confirmed_labels` 로 확정 원인부터 처방한다.

---

## 환경 변수

| 변수 | 기본 | 설명 |
|------|------|------|
| `EVAL_MODE` | `fast` | **진단 깊이(비용 tier)**: `fast`/`standard`/`deep`/`full` 또는 `1`~`4`. 아래 표 참고 |
| `EVAL_ENABLE_LLM` | off | `1/true` 면 RAGAS(LLM-as-Judge) 진단 허용 (**+ `EVAL_MODE≥deep` 이어야 실제 실행**) |
| `EVAL_LLM_PROVIDER` | `openai` | LLM 호출 provider 선택: `openai` / `gemini` / `github` (아래 참고) |
| `OPENAI_API_KEY` | — | provider=openai 일 때 필요 |
| `GEMINI_API_KEY` | — | provider=gemini 일 때 필요(Google AI Studio 무료 티어) |
| `GITHUB_TOKEN` | — | provider=github 일 때 필요(`models:read` 권한 포함된 PAT) |
| `EVAL_GEN_MODEL` / `EVAL_GEN_MODEL_GEMINI` / `EVAL_GEN_MODEL_GITHUB` | `gpt-4o-mini` / `gemini-flash-latest` / `openai/gpt-4o-mini` | 답변·Probe 질문 생성 모델(응답용) |
| `EVAL_JUDGE_MODEL` / `EVAL_JUDGE_MODEL_GEMINI` / `EVAL_JUDGE_MODEL_GITHUB` | `gpt-4o` / `gemini-flash-latest` / `openai/gpt-4o` | RAGAS 평가(심판) 모델(설계 원칙: 응답≠평가) |
| `EVAL_EMBED_MODEL` / `EVAL_EMBED_MODEL_GEMINI` | `text-embedding-3-small` / `gemini-embedding-001` | Response Relevancy 코사인용 임베딩(github provider는 임베딩 미지원 → OpenAI 키로 폴백) |
| `QDRANT_URL` / `QDRANT_API_KEY` | `:memory:` | 검색 인덱스 대상 |

> 기본값만으로도(위 키 전부 미설정) **외부 API 없이** 규칙 지표 기반 진단이 동작합니다(폴백 설계).

### LLM Provider — `agents/eval/llm_provider.py`

OpenAI 유료 토큰이 없어도 무료 대체 provider로 STEP1(질문 생성)·STEP2(답변 생성)·STEP3-2(RAGAS
심판·임베딩)를 실제 LLM으로 돌릴 수 있게 하는 브릿지 계층. `generate_text`/`chat_json`/`embed_texts`
세 함수로 provider 차이를 감추고, `probe_gen.py`/`metrics_ragas.py`가 전부
이 계층만 호출한다(직접 `from openai import OpenAI` 하지 않음). STEP2 답변 생성은
`agents/rag/generator.py`가 담당하며, 그쪽은 자체 provider 선택 로직(`RAG_LLM_PROVIDER`)을 쓴다.

- **openai**(기본): 정식 OpenAI API.
- **gemini**: Google AI Studio 무료 티어. `google-genai` 패키지 필요(`pip install google-genai`).
  **주의**: 무료 티어가 분당 요청 수 제한(계정별로 다름, 낮으면 5회/분 수준)이 있어 청크 수가
  많으면(질문 생성 N회 + 답변 생성 N회) 429(RESOURCE_EXHAUSTED)가 잦다 — 실패해도 자동으로
  휴리스틱/추출식 폴백으로 넘어가 파이프라인은 안 죽는다.
- **github**: [GitHub Models](https://github.com/marketplace/models) — OpenAI 호환 API를
  `base_url=https://models.github.ai/inference` 로 그대로 재사용. GitHub 개인 액세스 토큰에
  **`models:read` 권한이 반드시 있어야 함**(없으면 401/403). 임베딩 엔드포인트는 제공하지
  않아 `embed_texts`는 provider 무관하게 OpenAI 키로만 동작(없으면 해당 RAGAS 지표만 스킵).

모델명·무료 티어 한도는 시점에 따라 바뀔 수 있다 — 401/403/404 가 나면 해당 콘솔에서 현재
사용 가능한 모델명을 다시 확인할 것.

---

## 진단 모드 (비용 tier)

라벨(진단명)은 판별에 필요한 **가장 비싼 자원**을 tier 로 갖는다. 사용자가 고른 모드(`EVAL_MODE`)가
그 tier 상한을 정해, 감당 못 하는 라벨은 **예비(`Finding.confirmed=False`)** 로 내보내고 상위 모드에서 확정한다.

**확정(`confirmed=True`)은 그 라벨의 '확정 신호'가 실제로 발동해야 성립한다** — 자원 미실행/미측정이면 예비다
(단순히 `mode>=tier` 라서 확정이 아님). 신호는 자기 tier의 자원을 self-gate 하고(모드 부족 시 `None`),
라벨 함수는 `확정 신호 → (미실행이고 싼 신호 있으면) 예비` 순으로 판정한다. → 거짓확정이 구조적으로 불가.

| 모드 | 값 | 추가 자원 | 확정 가능한 라벨 |
|------|----|----------|-----------------|
| `fast` | 1 | 규칙·기존 지표만(추가 쿼리 없음) | `retrieval_incomplete_enumeration` (gold수 vs top-k 순수 규칙) |
| `standard` | 2 | + 추가 검색 쿼리(top-N 재검색·BM25·코퍼스) | 검색 원인(`low_rank`/`lexical`/`semantic`/`missing_gold`), `corpus_gap(_partial_hop)` |
| `deep` | 3 | + **LLM(RAGAS/AspectCritic)** | 생성 원인(`hallucination`/`partial`/`hop_binding`/`contradiction`), `bad_gold_answer` |
| `full` | 4 | + 파이프라인 재실행(ablation) | context 원인(`too_long`/`lost_in_the_middle`/`context_noise`), `bridge_dependency` |

- **생성 원인은 전부 RAGAS(=deep) 의존** → `deep` 미만이면 하나의 예비 `generation_failure` 로 롤업된다
  (LLM 없이는 hallucination/bad_gold 를 싸게 구분할 수 없다는 정직한 한계).
- **STEP3-2 RAGAS 는 `deep` 이상에서만 실행**된다(`signals` 의 RAGAS 신호 DEEP 게이트). `EVAL_ENABLE_LLM` 과 AND 조건.
- tier2~4 판별 신호(`_gold_in_wider_candidates`/`_bm25_hits_gold`/`_gold_in_corpus`/
  `_context_shorten_helps`/`_gold_front_helps`/`_noise_removal_helps`/`_bridge_decompose_recovers`)는
  **구현·배선 완료**돼 있다 — `agent.py::run()`이 `set_diag_context(retrieve_fn=retrieve,
  keyword_fn=_keyword_search, generate_fn=generate_answer, ragas_fn=_ragas_track)` 로 실제
  자원을 주입한다. `EVAL_MODE`가 그 신호들의 self-gate 를 통과할 만큼 높아야(standard 이상,
  ablation 계열은 full) 실제로 호출된다 — "미구현"이 아니라 "모드가 낮아서 미실행"인 경우가 대부분.

### 라벨별 tier 분류 (확정에 필요한 자원)

아래는 각 라벨을 **확정**하는 데 필요한 자원 정리(설계 문서용). tier 값은 `Finding` 에 싣지 않고,
각 판별 신호가 `signals.py` 에서 자기 자원(tier)을 실행 모드로 self-gate 한다(mode 부족 시 `None`).

| 라벨 | 그룹 | tier | 확정 자원 |
|---|---|:---:|---|
| `retrieval_incomplete_enumeration` | A 검색 | 1 | gold수 vs top-k 순수 규칙 |
| `retrieval_low_rank` | A 검색 | 2 | top-N 재검색 |
| `retrieval_lexical_mismatch` | A 검색 | 2 | BM25 조회 |
| `retrieval_semantic_mismatch` | A 검색 | 2 | BM25 + 코퍼스 확인 |
| `retrieval_missing_gold` | A 검색 | 2 | 코퍼스 멤버십 조회 |
| `retrieval_missing_bridge_dependency` | A 검색 | 4 | iterative decompose 재실행 |
| `generation_hallucination` | B 생성 | 3 | RAGAS faithfulness |
| `generation_partial_answer` | B 생성 | 3 | RAGAS relevancy |
| `generation_hop_binding_error` | B 생성 | 3 | RAGAS faithfulness(+추론검증) |
| `generation_contradiction` | B 생성 | 3 | AspectCritic(LLM) |
| `generation_failure` (롤업) | B 생성 | 3 | DEEP에서 세분화 (항상 예비) |
| `too_long_context` | C context | 4 | ablation 재실행(축소) |
| `lost_in_the_middle` | C context | 4 | 재실행(재정렬) |
| `context_noise_interference` | C context | 4 | 재실행(노이즈 제거) |
| `bad_gold_answer` | D 데이터 | 3 | RAGAS 2지표(진짜 확정은 사람) |
| `corpus_gap` | D 데이터 | 2 | 코퍼스 조회 |
| `corpus_gap_partial_hop` | D 데이터 | 2 | 코퍼스 조회(hop별) |

> "확정"은 원인 검증(처방이 실제로 통함)을 뜻한다. C그룹·`bridge_dependency` 는 ablation/재실행으로만
> 확정되므로 tier4다 — 필요해지면 후에 `preliminary_tier=3(RAGAS 후보) / confirm_tier=4(재실행 확정)` 로
> 분리할 수 있다. `bad_gold_answer` 는 자동으론 tier3까지만 의심 가능하고 진짜 확정은 사람 검수 몫.

---

## 테스트

```bash
python tests/test_eval.py       # mock chunks 5개로 STEP1~5 단독 실행 (Index 없이).
                                 # Probe(질문/정답) + STEP2 검색결과·생성답변까지 콘솔에 출력한다.
python tests/test_ragas_eval.py # 실제 OpenAI API로 RAGAS 4지표 실측 검증 (키 없으면 자동 스킵)
```

`test_eval.py`는 `EVAL_LLM_PROVIDER`(`.env`)를 그대로 읽으므로, `github`/`gemini` 로 설정해두면
실제 무료 LLM이 만든 질문·답변이 출력된다(미설정 시 키 없음 폴백 경로로 결정적 동작).

첫 실행 후 프로젝트 루트에 `eval_probes.json`(Probe 캐시, STEP1 참고)이 생긴다 —
`.gitignore`의 `*.json` 에 걸려 커밋되지 않으며, 지워도 다음 실행에서 재생성된다.

---

## 파일 구조

```
agents/eval/
├── agent.py           # run(state) — STEP1~5 오케스트레이션 + tier2~4 자원 주입(set_diag_context)
├── types.py           # EvalRecord(내부 중간결과) · Mode(tier) · 상수
├── signals.py         # 판별 신호 레이어 — 옛 브랜치 판정을 대체하는 전제 함수 + tier1~4 판별 신호(memoize)
├── probe_gen.py       # STEP1  Probe 생성(user_log/RAGAS 4분면/DataMorgana-lite/무응답)
├── probe_store.py     # STEP1  eval_probes.json 영속화(코퍼스 버전 불변 시 재사용)
├── knowledge_graph.py # STEP1  청크 간 관계 그래프(RAGAS 멀티홉 후보 탐색용, 휴리스틱 전용)
├── llm_provider.py    # LLM 호출 추상화(OpenAI/Gemini/GitHub Models) — probe_gen/metrics_ragas 공용
├── metrics.py         # STEP3-1 규칙 지표(Recall@k/token F1/Oracle F1) 순수 함수
├── metrics_ragas.py   # STEP3-2 RAGAS 4지표 + AspectCritic(LLM-as-Judge, 옵션)
├── diagnose.py        # STEP4  16개 라벨 함수(A/B/C/D) → Finding, 브랜치리스 조립
├── report.py          # STEP5  DiagnosticReport 집계(overall_score/pass_threshold/findings_summary)
└── README.md          # 이 파일
```

---

## 주요 확장 지점 `[구현 포인트]`

1. **Probe 자동생성** (`probe_gen.py`, `knowledge_graph.py`, `probe_store.py`) — ✅ 배선 완료.
   `generate_probes()`가 `_allocate_budget()` 비율(75% RAGAS 4분면 / 20% DataMorgana-lite /
   5% 무응답)대로 실제로 섞어 생성하고, `probe_store.py`가 `eval_probes.json`으로 캐시해
   코퍼스가 안 바뀌면 재사용한다(매 Optimize 반복마다 LLM 재호출 방지). 세부:
   - `knowledge_graph.py` — 청크 간 관계 그래프(키워드 Jaccard + 임베딩 코사인, LLM 미사용)와
     `connected_pairs()`로 멀티홉 후보 탐색.
   - `_generate_ragas_probes`(`probe_gen.py`) — 그래프 기반 단일홉(구체/추상) + 멀티홉
     (bridge/comparison/aggregation) 질문 합성(LLM + 휴리스틱 폴백). 그래프가 비면(청크
     부족) `_from_chunks` 단일홉 폴백으로 전체 대체.
   - `_generate_datamorgana_probes` — 거친 스타일(conversational/long/breadth) 조합으로
     단일홉 질문 생성(풀 DataMorgana 대신 최소 버전).
   - `_generate_no_answer_probes`(Held-out·False Premise 절반씩) — `answer_exists=False`,
     `ground_truth=None` probe를 만들어 `_no_diagnosis`/`is_abstention` 게이팅이 "정답 없음을
     올바르게 기권"과 "무응답인데 답을 지어내는 생성 실패"를 구분해 진단할 수 있게 함.
   - `_build_doc_position_index`/`_locate_span`/`_resync_gold_chunk_ids` — `gold_spans`(원문
     절대 좌표) 기준으로 재청킹 후에도 `gold_chunk_ids`를 다시 맞추는 유틸. `_generate_ragas_probes`가
     만드는 probe는 아직 `gold_spans`를 채우지 않아 현재는 사실상 no-op(외부 taxonomy 소스가
     `gold_spans`를 채울 때 실효).
   - 남은 일: `state.user_questions` 없이 taxonomy(사람 작성) 소스 자체를 만드는 부분은 미착수.
2. **LLM Provider** (`llm_provider.py`) — ✅ 구현됨. OpenAI 토큰 승인 전 무료 대체용 브릿지.
   `EVAL_LLM_PROVIDER=openai|gemini|github` 로 전환, `probe_gen.py`/`retrieval_temp.py`/
   `metrics_ragas.py` 전부 이 계층만 통해 LLM을 호출한다. 자세한 내용은 위 "LLM Provider" 절 참고.
3. **RAGAS 지표** (`metrics_ragas.py`) — ✅ 구현됨. RAGAS 0.4.3 소스의 프롬프트·예시·조립
   형식을 그대로 옮겨 LLM-as-Judge로 직접 계산(Faithfulness/ContextPrecision/ContextRecall/
   ResponseRelevancy + contradiction AspectCritic). `EVAL_ENABLE_LLM=1`+`EVAL_MODE≥deep`로
   활성화. ragas 라이브러리는 langchain 버전 충돌로 import가 불안정해 미사용 — 환경이
   지원하면 `evaluate_*_track` 내부만 교체해도 결과가 동일하도록 설계.
4. **LLM 생성** (`agents/rag/generator.py`) — STEP2 답변 생성은 공용 RAG 모듈이 담당한다
   (`RAG_LLM_PROVIDER`로 OpenAI/Gemini/GitHub 선택, 키 없으면 추출식 폴백). 프롬프트
   엔지니어링·컨텍스트 랭킹 등 고도화는 rag 모듈 쪽 과제.
5. **Reranker** — Bi-Encoder 후 Cross-Encoder 재정렬(2차 개선). Index 검색이 담당하게 될 영역.
   README 작성 시점 기준 우선순위 낮음.
6. **STEP4 라벨 세트 확장** (`diagnose.py`) — Notion 설계 문서엔 현재 16개(A/B/C/D)보다 많은
   후보 라벨이 있다(`chunking_*`, `reranker_*`, `generation_contradiction/misinterpretation/
   abstention_failure/parametric_overreliance/numerical_error` 등). `generation_contradiction`은
   `metrics_ragas.py::evaluate_aspect_critics`가 이미 `record.aspect["contradiction"]`을 계산해둬서
   라벨 함수만 추가하면 저비용으로 확장 가능(diagnose.py 주석상 "나중에 개발"로 자리만 표시돼 있음).
   `chunking_*`/`reranker_*`는 Index Agent 쪽(청킹 전략 정보·리랭커)이 선행돼야 판별 가능.
7. **임시 검색 제거** — ✅ 완료. `retrieval_temp.py` 는 삭제됐고, `agent.py` 는
   `agents/rag/retriever.py`(`build_retriever`) 검색과 `agents/rag/generator.py`
   (`generate_answer`) 생성을 호출한다.
