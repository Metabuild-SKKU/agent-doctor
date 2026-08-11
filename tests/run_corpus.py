"""
tests/run_corpus.py
코퍼스 폴더 하나(원본 PDF + QA셋)로 파이프라인을 돌리고, 결과를 진단서 HTML 로 떨군다.

서버(web_api 8767 / api 8766)를 띄우지 않는다. report.html 이 fetch 로 받아가던
JSON 을 report_view.build_report_view() 로 직접 만들어 HTML 안에 심어버리기 때문에,
브라우저에서 결과 파일을 그냥 열면 된다.

코퍼스 규약 (tests/corpus/ 하나만 쓴다 — 하위 폴더 없음):
    <아무거나>.pdf      원본 문서 (.md/.txt 도 가능). 필수 — 직접 넣어야 한다.
    qa.json             QA셋(골든 테스트셋). 없으면 첫 실행이 만들어서 여기 저장한다.
    out/report.html     결과 진단서 (실행할 때마다 덮어씀)
    out/report.json     진단서에 심은 원본 JSON (디버깅·diff 용)

문서를 여러 개 넣으면 이름순 첫 번째만 쓴다(Ingest 의 file 소스가 파일 1개를 받는다).

Run:
    python tests/run_corpus.py             # 돌리고 진단서까지 자동으로 띄움
    python tests/run_corpus.py --no-open   # 브라우저 안 띄움(CI·원격 등)
    python tests/run_corpus.py --regen-qa  # qa.json 무시하고 새로 생성

진단 깊이는 항상 EVAL_MODE=full(= deep) + EVAL_ENABLE_LLM=1 로 고정한다(RAGAS 까지).
LLM 을 실제로 호출하므로 비용이 든다 — 대신 findings 가 '예비'에 머물지 않고 확정된다.

QA셋은 한 번 만들어지면 계속 재사용된다(EVAL_PROBE_SOURCE=made 와 같은 계약) —
청킹 파라미터가 바뀌어도 같은 질문으로 채점해야 최적화 전후를 비교할 수 있기 때문.
문서를 바꿨으면 --regen-qa 로 다시 만들어야 한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.console import force_utf8_stdio
force_utf8_stdio()   # cp949 콘솔에서 '—' 등이 UnicodeEncodeError 를 내지 않도록(로깅과 독립)

from core.embedding_cli import (   # noqa: E402
    add_embedding_args,
    apply_embedding_args,
    describe_embedding_route,
)

def _load_env() -> None:
    """.env 로드. import 시점이 아니라 main() 에서만 부른다.

    override=True 로 .env 를 프로세스 전역에 밀어넣기 때문에, import 만 해도 환경이
    바뀐다 — tests/test_run_corpus.py 가 이 모듈을 import 하면 EVAL_TESTSET_SIZE 같은
    값이 같은 프로세스의 다른 테스트에까지 새어 들어가 결과를 바꾼다(실측: Probe 개수가
    10→30 이 되면서 gold_spans 없는 no_answer Probe 가 섞여 무관한 테스트가 깨졌다).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass


CORPUS_ROOT = Path(__file__).resolve().parent / "corpus"

DOC_SUFFIXES = (".pdf", ".md", ".txt")
QA_FILENAME = "qa.json"

# 코퍼스 폴더에 같이 사는 문서지만 원본이 아닌 것들. 평평한 구조라 .md 인 README 가
# 이름순 첫 번째로 뽑혀 원본 문서 행세를 하는 사고를 막는다.
_NON_CORPUS_STEMS = {"readme"}



def find_source_doc(corpus_dir: Path = CORPUS_ROOT) -> Path:
    """코퍼스의 원본 문서. 여러 개면 이름순 첫 번째(현재 Ingest 가 파일 1개만 받음)."""
    if not corpus_dir.exists():
        raise SystemExit(f"코퍼스 폴더가 없습니다: {corpus_dir}")
    docs = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in DOC_SUFFIXES
        and p.stem.lower() not in _NON_CORPUS_STEMS
    )
    if not docs:
        raise SystemExit(
            f"원본 문서가 없습니다 — {corpus_dir}/ 에 "
            f"{' / '.join(DOC_SUFFIXES)} 중 하나를 넣어주세요."
        )
    if len(docs) > 1:
        names = ", ".join(p.name for p in docs)
        print(f"  주의: 문서 {len(docs)}개({names}) → 첫 번째만 사용({docs[0].name})")
    return docs[0]


def run_pipeline_for(corpus_dir: Path, *, regen_qa: bool, loop: bool):
    """코퍼스 하나에 대해 Ingest→Index→Eval→Optimize 를 돌리고 최종 state 를 반환한다.

    Serve 노드는 타지 않는다 — 이 스크립트의 목적은 진단 결과를 파일로 받는 것이고,
    Serve 는 API 서버 기동과 Claude Desktop MCP 등록이라 테스트에 불필요하다.
    """
    from core.state import AgentDoctorState

    source_doc = find_source_doc(corpus_dir)
    qa_path = corpus_dir / QA_FILENAME

    # Eval 의 Probe 영속화 기본 경로는 레포 루트 eval_probes.json 이다. 코퍼스마다
    # QA셋을 따로 들고 있어야 하므로 이 코퍼스의 qa.json 으로 돌린다
    # (probe_store.resolve_store_path 가 호출 시점에 이 환경변수를 읽는다).
    os.environ["EVAL_PROBE_STORE"] = str(qa_path)

    if regen_qa and qa_path.exists():
        qa_path.unlink()
        print(f"  --regen-qa → 기존 {QA_FILENAME} 삭제, 새로 생성합니다")

    # 진단 깊이는 항상 full(= deep 으로 접힘) — 가장 깊은 모드로 고정한다.
    # 모드만으로는 부족하고 EVAL_ENABLE_LLM 이 같이 켜져야 RAGAS(LLM-as-Judge)가
    # 실제로 돈다(types.ragas_enabled 는 별도 환경변수를 본다) — web_api 의 depth=full 과 동일 조합.
    # .env/셸에 다른 값이 있어도 이 스크립트에선 full 로 고정한다(회귀 비교 기준을 흔들지 않기 위해).
    os.environ["EVAL_MODE"] = "full"
    os.environ["EVAL_ENABLE_LLM"] = "1"

    # made: 코퍼스 버전과 무관하게 저장된 QA셋을 그대로 재사용. 파일이 없으면
    # Eval 이 자동 생성 후 같은 경로에 저장한다(첫 실행 = QA셋 생성 실행).
    os.environ["EVAL_PROBE_SOURCE"] = "made"
    if qa_path.exists():
        print(f"  QA셋 재사용: {qa_path.relative_to(REPO_ROOT)}")
    else:
        print(f"  QA셋 없음 → 이번 실행에서 생성해 {QA_FILENAME} 로 저장합니다")

    state = AgentDoctorState()
    state.source_type = "file"
    state.source_url = str(source_doc)

    if loop:
        # 품질 미달이면 Optimize→재색인→재평가를 예산까지 반복(graph.py 와 동일 라우팅).
        # Serve 는 건너뛰도록 서브그래프가 아니라 노드 직접 호출 루프를 쓴다.
        return _run_with_loop(state)
    return _run_once(state)


def _steps():
    from agents.ingest.agent import run as ingest_run
    from agents.index.agent import run as index_run
    from agents.eval.agent import run as eval_run
    from agents.optimize.agent import run as optimize_run
    return ingest_run, index_run, eval_run, optimize_run


def _run_once(state):
    """Ingest→Index→Eval→Optimize 1회 (run_local_pipeline.py 와 같은 흐름)."""
    ingest_run, index_run, eval_run, optimize_run = _steps()
    for name, fn in (
        ("Ingest", ingest_run), ("Index", index_run),
        ("Eval", eval_run), ("Optimize", optimize_run),
    ):
        print("\n" + "=" * 56)
        print(f"  {name}")
        print("=" * 56)
        state = fn(state)
        if state.error:
            raise SystemExit(f"[중단] {name} 오류: {state.error}")
    return state


def _run_with_loop(state):
    """graph.py 의 route_after_eval / route_after_optimize 를 그대로 써서 재색인 루프를
    돌되, serve 로 갈 자리에서 루프를 끝낸다(Serve 노드는 실행하지 않는다)."""
    from graph import route_after_eval, route_after_optimize
    ingest_run, index_run, eval_run, optimize_run = _steps()

    def step(name, fn, st):
        print("\n" + "=" * 56)
        print(f"  {name}")
        print("=" * 56)
        st = fn(st)
        if st.error:
            raise SystemExit(f"[중단] {name} 오류: {st.error}")
        return st

    state = step("Ingest", ingest_run, state)
    state = step("Index", index_run, state)
    while True:
        state = step("Eval", eval_run, state)
        if route_after_eval(state) == "serve":
            break
        state = step("Optimize", optimize_run, state)
        if route_after_optimize(state) != "index":
            break
        state = step("Index (재색인)", index_run, state)
    return state


def write_report(state, corpus_dir: Path) -> tuple[Path, dict]:
    """build_report_view 결과를 report.html 템플릿에 심어 단독 실행 가능한 진단서로 저장.

    심는 방법 자체는 tests/report_html.py 가 갖는다 — 외부 로그 리플레이
    (tools/run_replay_report.py)도 같은 템플릿에 심으므로, 두 벌로 복사하면 템플릿이
    바뀔 때 한쪽만 고쳐져 조용히 빈 리포트를 그린다."""
    from agents.serve.report_view import build_report_view

    from tests.report_html import write_report_files

    return write_report_files(build_report_view(state), corpus_dir / "out")


def print_summary(state, view: dict, html_path: Path):
    score = view.get("score", {})
    doc_label = Path(state.source_url).name if state.source_url else "문서"
    print("\n" + "=" * 56)
    print(f"  결과 — {doc_label}")
    print("=" * 56)
    print(f"문서/청크/질문 : {len(state.documents)} / {len(state.chunks)} / {len(state.probes)}")
    print(f"종합 점수      : {score.get('before')} → {score.get('after')} "
          f"(delta {score.get('delta')})")
    if state.report:
        gate_info = score.get("gate") or {}
        print(f"overall_score  : {state.report.overall_score}  "
              f"gate_pass={gate_info.get('pass')}")
        print(f"composite_gate : {gate_info.get('composite_total')} / "
              f"{gate_info.get('composite_threshold')} "
              f"({gate_info.get('score_source')})")
        print(f"recall_gate    : {gate_info.get('mean_recall_at_k')} / "
              f"{gate_info.get('recall_floor')} "
              f"reason={gate_info.get('reason')}")
        print(f"findings       : {state.report.findings_summary}")
    print(f"처방 keep/roll : {score.get('kept')} / {score.get('rolled')}")
    print(f"index_config   : {state.index_config}")
    print(f"\n진단서 → {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="코퍼스 폴더로 파이프라인을 돌리고 진단서 HTML 을 만든다(서버 불필요)."
    )
    parser.add_argument("--no-open", action="store_true",
                        help="끝나고 브라우저를 띄우지 않는다(기본은 띄움)")
    parser.add_argument("--regen-qa", action="store_true", help="기존 qa.json 을 버리고 재생성")
    parser.add_argument("--loop", action="store_true",
                        help="품질 미달 시 Optimize→재색인→재평가 반복(기본은 1회)")
    add_embedding_args(parser)
    args = parser.parse_args()

    _load_env()   # 실제 실행할 때만 .env 를 적용한다(import 부작용 방지)

    # .env 뒤에 적용한다 — _load_env 가 override=True 라 먼저 적용하면 덮어쓰인다.
    applied = apply_embedding_args(args)
    if applied:
        print(f"  임베딩 설정 적용: {applied}")

    # 로깅은 파이프라인 import 보다 먼저 설치한다 — 모델 로딩 경고처럼 import 시점에
    # 나오는 출력까지 로그에 담기게. (setup 이전 출력은 콘솔에만 남는다.)
    from core.run_logger import setup_run_logging
    log_path = setup_run_logging(prefix="corpus")

    print(f"  문서   : {find_source_doc().name}")
    print(f"  진단   : EVAL_MODE=full (RAGAS 포함)")
    print(f"  {describe_embedding_route()}")

    try:
        state = run_pipeline_for(CORPUS_ROOT, regen_qa=args.regen_qa, loop=args.loop)
        html_path, view = write_report(state, CORPUS_ROOT)
        print_summary(state, view, html_path)
    except BaseException:
        # 실패해도 로그 파일 위치는 알려준다 — 콘솔이 길어 앞부분이 밀려도 찾을 수 있게.
        # 트레이스백은 stderr 로 나가고 Tee 가 같은 파일에 받아 적는다.
        if log_path:
            print(f"\n[log] 실행 로그: {log_path}")
        raise

    if log_path:
        print(f"[log] 실행 로그: {log_path}")

    if not args.no_open:
        print(f"브라우저로 여는 중: {html_path}")
        webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
