"""
tests/report_html.py
진단서 뷰(dict) → 단독 실행 가능한 report.html 로 심는 공용 유틸.

run_corpus.py(내부 파이프라인)와 run_replay_report.py(외부 로그 리플레이)가
같은 템플릿에 각자의 뷰를 심는다. 심는 방법은 템플릿 구조에 의존하므로 —
fetch 분기를 통째로 들어내고 데이터 블록을 렌더 스크립트 앞에 끼운다 — 두 벌로
복사하면 템플릿이 바뀔 때 한쪽만 고쳐져 조용히 빈 리포트를 그리게 된다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_TEMPLATE = REPO_ROOT / "web" / "prototype" / "report.html"

INJECT_MARKER = "/* injected by tests/report_html.py */"


def inject_view(view: dict[str, Any], template_html: str) -> str:
    """템플릿의 데이터 로딩 분기를 '심어둔 데이터로 렌더'로 갈아끼운다.

    템플릿은 run_id 쿼리스트링이 있으면 서버로 fetch, 없으면 더미를 그린다.
    서버가 없으므로 그 분기 전체(fetch 체인 + else 더미)를 한 줄로 바꾼다 —
    분기를 통째로 들어내야 fetch 가 남아 실패 배너를 띄우는 일이 없다.
    """
    start = "  var runId = new URLSearchParams(location.search).get('run_id');"
    end = "  } else {\n    renderReport({}, false);\n  }\n"
    s_at = template_html.find(start)
    e_at = template_html.find(end, s_at)
    if s_at == -1 or e_at == -1:
        raise SystemExit(
            "report.html 의 데이터 로딩 분기를 찾지 못했습니다 — 템플릿이 바뀌었으면 "
            "tests/report_html.py 의 inject_view() 도 같이 고쳐야 합니다."
        )
    html = (
        template_html[:s_at]
        + f"  {INJECT_MARKER}\n"
        + "  renderReport(window.__AGENT_DOCTOR_REPORT__, true);\n"
        + template_html[e_at + len(end):]
    )

    # 데이터 블록은 반드시 렌더 스크립트보다 **앞**에 와야 한다 — 뒤에 두면 렌더
    # 시점엔 아직 undefined 라 빈 리포트가 그려진다. </script> 파싱을 깨지 않게
    # </ 는 이스케이프한다.
    payload = json.dumps(view, ensure_ascii=False).replace("</", "<\\/")
    data_script = (
        f"<script>{INJECT_MARKER}\n"
        f"window.__AGENT_DOCTOR_REPORT__ = {payload};\n"
        "</script>\n"
    )
    main_script_at = html.rfind("<script>")
    if main_script_at == -1:
        raise SystemExit("report.html 에 <script> 가 없습니다 — inject_view() 확인 필요")
    return html[:main_script_at] + data_script + html[main_script_at:]


def write_report_files(view: dict[str, Any], out_dir: Path) -> tuple[Path, dict]:
    """뷰를 out_dir 에 report.json / report.html 로 떨군다.

    템플릿이 없으면 JSON 만 남기고 그 경로를 돌려준다 — 템플릿 부재로 진단
    자체를 실패시키지 않는다(뷰는 이미 만들어졌고, 그게 디버깅의 재료다)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8")

    if not REPORT_TEMPLATE.exists():
        print(f"  경고: 리포트 템플릿 없음({REPORT_TEMPLATE}) → JSON 만 저장")
        return json_path, view

    html = inject_view(view, REPORT_TEMPLATE.read_text(encoding="utf-8"))
    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path, view
