# 벤치마크 #2 (RAGChecker) — 발표 근거 묶음

발표에서 인용하는 숫자의 출처. **이 폴더만 넘겨도 검증되는 자기완결 묶음**이다
(#1 의 `report_*_rr/` 와 같은 역할).

`bench/out/` 은 gitignore 대상이라 실행 산출물이 남지 않는다. 그런데 이 실험의
로그·채점은 **LLM 생성물이라 다시 뽑으면 값이 달라진다** — 재현이 아니라 보존이
필요한 자산이므로 여기에 복사해 추적한다.

## 실험 한 줄

원인을 우리가 아는 결함 로그 4종(none / starve / hallucinate / offtopic)을
RAGChecker(Amazon Science)와 우리 replay 진단기에 **똑같이** 먹여 "원인을 누가
맞히나"를 잰다. 설계·판정 기준은 노션 "벤치마킹" §8.

## 통제한 조건 (5중)

| 축 | 값 |
|---|---|
| 코퍼스 | 국세청 26년 세금가이드 PDF (`data/pdf_corpus.json`) — #1 AutoRAG 와 동일 |
| 질문 | `bench/tax_ext_questions.json` (골든 test 분할에서 결정적 추출) · none/starve/offtopic 은 동일 6문항 |
| 입력 로그 | 양쪽에 **같은 파일** (`tests/fixtures/external_rag/tax/ext_*.jsonl`) |
| 판정 규칙 | v2 대칭 규칙 — 1차 지표가 none 대비 15pt 이동 (`tools/bench_judge_symmetric.py`) |
| 심판 모델 | 양쪽 `gpt-4o-mini` (우리 심판을 내려서 통일) |

`hallucinate` 만 gap 질문 6개를 쓴다 — 코퍼스에 답이 있으면 환각이 발병하지 않는다
(0회 등장 검증 완료). RAGChecker 는 `gt_answer` 가 필수라 기권 문장으로 대체했고,
이는 RAGChecker 에 유리한 비대칭이다.

## 파일

| 파일 | 내용 |
|---|---|
| `judge_symmetric_v2.json` | **발표 숫자의 출처** — 동일 잣대 v2 판정 |
| `judge_summary.json` | v1(사전 등록) 판정 — 잣대 비대칭이 발견된 원본, 수정 이력으로 보존 |
| `ours_verdicts.json` | v1 규칙에 대한 우리 쪽 라벨 건수 판정 |
| `ragchecker_metrics.json` | RAGChecker 지표 4종 (원본 `output_*.json` 은 claim 대조 전문이라 460KB — 지표만 추림) |
| `ours_*.log` | 우리 진단기 출력 원본 (소견 + 처방 카드) |
| `ours_hallucinate_fixed.log` | **처방 A/B 짝** — 같은 질문·같은 검색, 프롬프트만 교체 |

## 재실행

```bash
python tools/bench_ragchecker_convert.py --src-dir tests/fixtures/external_rag/tax --out-dir bench/out/ragchecker/tax
.venv-ragchecker/Scripts/python.exe tools/bench_ragchecker_run.py --dir bench/out/ragchecker/tax
python -m agents.eval.replay tests/fixtures/external_rag/tax/ext_none.jsonl
python tools/bench_judge_symmetric.py --dir bench/out/ragchecker/tax
```

## 읽을 때 주의

- **"적중률 대결에서 이겼다"고 읽지 말 것.** RAGChecker 지표 11개 전체를 보면 세
  결함 모두에서 여러 지표가 크게 움직였다 — "놓쳤다"는 판정은 우리가 지정한 1차
  지표 기준이다. 살아남는 주장은 산출물의 층위 차이(숫자 대시보드 vs 병명+처방).
- **검색축 내부 특정(recall vs precision)은 심판에 따라 뒤집힐 만큼 근소하다.**
  검색/생성 구분까지만 주장할 것.
- 결함 유형을 우리가 설계했다(출제자 편향). 문항은 로그당 6개 — 통계적 우열이
  아니라 구조적 차이를 보이는 통제 실험이다.
