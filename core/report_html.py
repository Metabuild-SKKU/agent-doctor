"""
core/report_html.py
진단 결과를 **서버 없이 열리는 단독 HTML** 로 떨군다.

`web/prototype/report.html` 은 원래 실행 결과를 서버(web_api 8767)에서 fetch 해 그린다.
그 분기를 "이미 심어둔 데이터로 렌더" 로 갈아끼우면 파일 하나로 완결돼, 브라우저에서
그냥 열면 된다. 서버를 띄울 필요도, 실행이 끝난 뒤 상태를 붙들고 있을 필요도 없다.

원래 tests/run_corpus.py 안에 있던 함수다. run_local_pipeline.py 도 같은 진단서가
필요해지면서(실코퍼스 검증) 공용 위치로 옮겼다 — 실행 스크립트가 tests/ 를 import 하는
모양이 되면 의존 방향이 뒤집힌다.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_TEMPLATE = REPO_ROOT / "web" / "prototype" / "report.html"

# 템플릿의 fetch 분기를 대체한 흔적. 사람이 파일을 열어봤을 때 "이건 주입된 것" 임을
# 알 수 있게 남긴다.
_INJECT_MARKER = "/* injected by core/report_html.py */"


def write_report(state, out_dir: Path | str) -> tuple[Path, dict]:
    """build_report_view 결과를 report.html 에 심어 단독 진단서로 저장.

    반환: (저장된 파일 경로, view dict). 템플릿이 없으면 JSON 만 쓰고 그 경로를 준다.
    """
    from agents.serve.report_view import build_report_view

    view = build_report_view(state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")

    if not REPORT_TEMPLATE.exists():
        print(f"  경고: 리포트 템플릿 없음({REPORT_TEMPLATE}) → JSON 만 저장")
        return json_path, view

    html = REPORT_TEMPLATE.read_text(encoding="utf-8")

    # 템플릿은 run_id 쿼리스트링이 있으면 서버로 fetch, 없으면 더미를 그린다. 서버가
    # 없으므로 그 분기 전체(fetch 체인 + else 더미)를 "심어둔 데이터로 렌더" 한 줄로
    # 갈아끼운다. 분기를 통째로 들어내야 fetch 가 남아 실패 배너를 띄우는 일이 없다.
    start = "  var runId = new URLSearchParams(location.search).get('run_id');"
    end = "  } else {\n    renderReport({}, false);\n  }\n"
    s_at = html.find(start)
    e_at = html.find(end, s_at)
    if s_at == -1 or e_at == -1:
        raise SystemExit(
            "report.html 의 데이터 로딩 분기를 찾지 못했습니다 — 템플릿이 바뀌었으면 "
            "core/report_html.py 의 write_report() 도 같이 고쳐야 합니다."
        )
    html = (
        html[:s_at]
        + f"  {_INJECT_MARKER}\n"
        + "  renderReport(window.__AGENT_DOCTOR_REPORT__, true);\n"
        + html[e_at + len(end):]
    )

    # 데이터 블록은 반드시 렌더 스크립트보다 **앞**에 와야 한다 — 뒤에 두면 렌더 시점엔
    # 아직 undefined 라 빈 리포트가 그려진다. </script> 파싱을 깨지 않게 </ 는 이스케이프.
    payload = json.dumps(view, ensure_ascii=False).replace("</", "<\\/")
    data_script = (
        f"<script>{_INJECT_MARKER}\n"
        f"window.__AGENT_DOCTOR_REPORT__ = {payload};\n"
        "</script>\n"
    )
    main_script_at = html.rfind("<script>")
    if main_script_at == -1:
        raise SystemExit("report.html 에 <script> 가 없습니다 — write_report() 확인 필요")
    html = html[:main_script_at] + data_script + html[main_script_at:]

    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path, view
