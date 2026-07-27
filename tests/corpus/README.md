# tests/corpus — 파이프라인 회귀 코퍼스

서버(web_api 8767)를 띄우고 브라우저로 PDF 를 업로드하지 않아도, 문서 하나로
파이프라인을 돌리고 진단서를 파일로 받아볼 수 있게 하는 자리다.

## 폴더 규약

이 폴더 하나가 코퍼스다 — 하위 폴더를 만들지 않는다.

```
tests/corpus/
  <문서>.pdf        원본 문서 (.md / .txt 도 가능). 직접 넣는다 — 이것만 필수.
  qa.json           QA셋(골든 테스트셋). 첫 실행이 만들어 저장한다.
  out/report.html   결과 진단서 (실행마다 덮어씀)
  out/report.json   진단서에 심은 원본 JSON (디버깅·diff 용)
```

`*.pdf` · `qa.json` · `out/` 은 루트 `.gitignore` 에서 제외된다 — 로컬 산출물이라
문서와 QA셋은 각자 준비해야 한다(이 README 만 커밋된다).
문서가 여러 개면 이름순 첫 번째만 쓴다(Ingest 의 `file` 소스가 파일 1개를 받는다).
이 README 는 `.md` 지만 원본 문서로 잡히지 않는다.

## 실행

```bash
python tests/run_corpus.py             # 돌리고 진단서까지 자동으로 띄움
python tests/run_corpus.py --no-open   # 브라우저 안 띄움(CI·원격 등)
python tests/run_corpus.py --regen-qa  # qa.json 버리고 새로 생성
python tests/run_corpus.py --loop      # 품질 미달 시 Optimize→재색인 반복
```

파이프라인이 끝나면 진단서가 브라우저로 자동으로 뜬다. API 서버는 필요 없다 — 실행
결과가 `out/report.html` 안에 그대로 박혀 있어서, 나중에 그 파일을 다시 열어도 된다.
실행 로그는 `output/logs/corpus_*.log` 에 쌓인다.

진단 깊이는 항상 `EVAL_MODE=full` + `EVAL_ENABLE_LLM=1` 로 고정한다(RAGAS·tier4 검증까지).
LLM 을 실제로 호출하므로 비용이 들지만, findings 가 '예비'에 머물지 않고 확정된다.

## QA셋을 왜 저장하나

첫 실행은 LLM 으로 질문을 생성한다(비용·시간이 든다). 저장해 두면 이후 실행은 그대로
재사용하므로, **같은 질문지로 채점**해 청킹 파라미터 변경 전후를 비교할 수 있다.
매번 새로 만들면 점수가 움직여도 그게 개선인지 질문이 바뀐 탓인지 알 수 없다.

문서를 교체했으면 `--regen-qa` 로 다시 만들어야 한다(질문이 옛 문서 기준으로 남는다).

## 문서 교체

```bash
rm tests/corpus/*.pdf tests/corpus/qa.json   # 기존 문서·QA셋 정리
cp ~/어딘가/새문서.pdf tests/corpus/
python tests/run_corpus.py                   # qa.json 이 새로 생성된다
```

생성된 `qa.json` 은 한 번 훑어보는 것을 권한다 — LLM 생성이 실패한 자리는 휴리스틱
폴백이 원문 문장을 이어붙인 "질문=정답" 형태로 나올 수 있다(현재 품질 게이트가
멀티홉 템플릿의 자기참조를 놓친다). 엉뚱한 질문은 손으로 지우거나 고쳐도 된다.

스캔본 PDF 는 텍스트 레이어가 없어 추출 결과가 거의 비는 경우가 있다. Ingest 가
"추출된 텍스트가 매우 적습니다" 경고를 내면 점수가 낮게 나와도 검색 품질 문제가
아니라 원본 추출 한계다.
