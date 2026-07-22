# UI 프로토타입

`index.html`(입력·진단 실행), `report.html`(진단 리포트) — 더미 데이터로 만든 정적 목업.
백엔드(FastAPI 등)와 아직 연결되어 있지 않다. 각 파일을 브라우저로 바로 열어서 확인 가능.

- `index.html` → "결과 리포트 보기" 버튼이 `report.html`로 이동
- `report.html` → "새 진단"이 `index.html`로 돌아감

실제 API와 연결할 때 대체해야 할 부분:
- `index.html`의 `stageDefs`/`runStages()` — 지금은 setTimeout으로 진행 상황을 흉내만 냄
- `report.html`의 `metrics`/`rxs`/`dxs`/`qas` — 지금은 하드코딩된 더미 배열, `state.report`/`OptimizationRequest` 결과로 교체 필요
