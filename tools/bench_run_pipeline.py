"""
tools/bench_run_pipeline.py
벤치마크용 파이프라인 러너 — graph.py 의 루프를 그대로 돌리되 두 가지를 더 한다.

왜 graph.py 를 직접 안 쓰나:
  1) **리랭커 처방 차단** — 우리만 켜서 이기면 "상대는 못 쓰는 무기로 이겼다"가 된다.
     rules.py 를 고치면 벤치마크 때문에 제품 코드를 왜곡하는 셈이라 config 로만 막는다
     (BLOCK_RERANKER_CONFIG 주석 참고).
  2) **최종 index_config 저장** — make_our_rag.py --stage=tuned 가 이 파일을 받아야
     "튜닝 후" 로그를 뽑을 수 있다. 콘솔 출력에서 손으로 옮기면 오타가 결과를 바꾼다.

리랭커 차단에 대한 정정(실측 후):
  처음엔 "AutoRAG(non_gpu)는 torch 가 없어 리랭커를 아예 못 쓴다"고 보고 이 축을 잠갔다.
  틀렸다 — AutoRAG 에는 torch 가 필요 없는 API 리랭커 모듈이 있고, 우리 키로 닿는
  rankgpt 를 주면 상대도 이 축을 쓴다(bench/autorag_config.yaml 의 같은 정정 참고).
  그래서 **본 비교는 --allow-reranker 판본**이고, 차단 판본은 "리랭커 축을 빼면 어떤가"를
  보는 대조군으로 남았다.

run_local_pipeline.py 를 쓰지 않는 이유: 그건 루프가 없는 스모크라 처방을 **적용만 하고
검증하지 않는다**(재색인·재평가·keep/rollback 없음). 검증 안 된 설정으로 "튜닝 후" 막대를
만들면 올랐다고 말할 근거가 없다.

--allow-reranker 는 리랭커 축을 여는 실행이며, 산출물 이름이 갈려 차단 판본을 덮지 않는다.
양쪽에 리랭커를 준 것이 **본 비교**이고, 차단 판본은 그 축이 없을 때의 대조군이다.

사용법:
    SOURCE_TYPE=json_corpus SOURCE_URL=data/pdf_corpus.json \
    EVAL_PROBE_SOURCE=made EVAL_PROBE_STORE=bench/out/probes_train.json \
    EVAL_MODE=deep EVAL_ENABLE_LLM=1 \
    python -m tools.bench_run_pipeline [--allow-reranker]
"""
from __future__ import annotations

import json
import os
import sys

OUT_CONFIG = "bench/out/tuned_index_config.json"
OUT_HISTORY = "bench/out/tuned_optimization_history.json"
# --allow-reranker 실행은 차단 판본 산출물을 덮지 않는다. 두 판본을 나란히 남겨야
# "리랭커 축이 얼마나 기여했나"를 사후에 되짚을 수 있다.
OUT_CONFIG_RR = "bench/out/tuned_reranker_index_config.json"
OUT_HISTORY_RR = "bench/out/tuned_reranker_optimization_history.json"

# 리랭커 차단은 config 로 한다. eligibility._runtime_verified 가 reranker.* action 을
# "runtime capability 의 status 가 verified 인가"로 거르는데, reranker_preflight="disabled"
# 는 그 status 를 unknown 으로 만든다(agents/index/agent.py::_refresh_runtime_capabilities).
# → planner 가 reranker 계열 action 을 runtime_capability_unavailable 로 제외한다.
#
# state.blocked_action_attempts 에 (라벨, 처방id) 튜플을 넣는 방법은 쓰지 않는다. planner 는
# 튜플을 받지만 그 앞단(agent.py::_active_excluded_action_keys)이 원소마다 .action_key 를
# 읽어 AttributeError 로 죽는다 — 그 필드는 history 가 만든 객체 전용이다(실측).
BLOCK_RERANKER_CONFIG = {"reranker_preflight": "disabled"}


def main() -> int:
    from core.console import force_utf8_stdio
    force_utf8_stdio()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

    # 생성 LLM 을 AutoRAG 와 같은 값으로 고정한다(모델 차이를 비교에서 배제).
    os.environ.setdefault("RAG_LLM_PROVIDER", "openrouter")
    os.environ.setdefault("RAG_OPENROUTER_MODEL", "openai/gpt-4o-mini")

    from core.run_logger import setup_run_logging
    from core.state import AgentDoctorState
    from core.llm_usage import print_agent_table
    from graph import build_graph

    setup_run_logging(prefix="bench_pipeline")

    state = AgentDoctorState(
        source_url=os.getenv("SOURCE_URL", "data/pdf_corpus.json"),
        source_type=os.getenv("SOURCE_TYPE", "json_corpus"),
        status="running",
    )
    allow_reranker = "--allow-reranker" in sys.argv
    out_config = OUT_CONFIG_RR if allow_reranker else OUT_CONFIG
    out_history = OUT_HISTORY_RR if allow_reranker else OUT_HISTORY
    if not allow_reranker:
        state.index_config.update(BLOCK_RERANKER_CONFIG)

    print("=" * 60)
    print("벤치마크 파이프라인 — " + ("리랭커 허용(본 비교)" if allow_reranker else "리랭커 처방 차단(대조군)"))
    print(f"  소스   : {state.source_url} ({state.source_type})")
    print(f"  Probe  : {os.getenv('EVAL_PROBE_STORE', '(기본)')}")
    if allow_reranker:
        print("  주의   : AutoRAG 쪽도 리랭커(rankgpt)를 받은 실행과 짝지어야 공정하다.")
        print("           bench/autorag_config.yaml 의 passage_reranker 노드 참고.")
    else:
        print(f"  차단   : {BLOCK_RERANKER_CONFIG} → reranker.* action 이"
              " runtime_capability_unavailable 로 제외됨")
    print("=" * 60)

    final = build_graph().invoke(state)
    if isinstance(final, dict):
        final = AgentDoctorState(**final)

    print_agent_table()

    os.makedirs(os.path.dirname(out_config) or ".", exist_ok=True)
    with open(out_config, "w", encoding="utf-8") as f:
        json.dump(final.index_config, f, ensure_ascii=False, indent=2, default=str)
    # default=str 로 덤프하면 dataclass 가 repr 문자열이 되어 나중에 기계가 못 읽는다
    # (실측: "OptimizationHistoryItem(trial_id=..." 가 통째로 한 줄 문자열). asdict 로 편다.
    import dataclasses
    with open(out_history, "w", encoding="utf-8") as f:
        json.dump(
            [dataclasses.asdict(h) if dataclasses.is_dataclass(h) else h
             for h in final.optimization_history],
            f, ensure_ascii=False, indent=2, default=str)

    # 내부 진단서(30라벨 + 처방 이력)를 HTML 로 남긴다. replay 진단서(ext_ 7라벨)와 달리
    # keep/rollback 검증 이력이 실리는 유일한 산출물이라, 발표에서 "고객이 실제로 받는 것"
    # 슬라이드가 여기서 나온다. 템플릿 심기는 tools/report_html.py 가 갖는다
    # (run_corpus.py · run_replay_report.py 와 같은 경로 — 두 벌로 복사하면 한쪽만 고쳐진다).
    try:
        from pathlib import Path
        from agents.serve.report_view import build_report_view
        from tools.report_html import write_report_files

        report_dir = Path(out_config).parent / (
            "report_internal_reranker" if allow_reranker else "report_internal")
        view = build_report_view(final)
        html_path, _ = write_report_files(view, report_dir)
        print(f"내부 진단서 → {html_path}")
    except Exception as exc:                      # 진단서 실패로 실행 결과를 잃지 않는다
        print(f"! 내부 진단서 생성 실패: {exc}")

    cfg = final.index_config
    print("=" * 60)
    print(f"최종 index_config → {out_config}")
    print(f"  chunk       : {cfg.get('chunk_strategy')} {cfg.get('chunk_size')}/{cfg.get('chunk_overlap')}")
    print(f"  top_k       : {cfg.get('top_k')}")
    print(f"  use_hybrid  : {cfg.get('use_hybrid')}")
    print(f"  use_reranker: {cfg.get('use_reranker')}  (차단했으므로 False 여야 정상)")
    print(f"처방 이력 {len(final.optimization_history)}건 → {out_history}")
    if final.report and final.report.composite_score:
        print(f"내부 종합점수: {final.report.composite_score.get('total')}")
    print("=" * 60)
    print("다음: python -m tools.make_our_rag --stage=tuned"
          f" --index-config={out_config}")

    # 리랭커가 켜져 있으면 차단이 안 먹은 것이다 — 조용히 넘어가면 비교의 전제가 깨진다.
    if cfg.get("use_reranker") and not allow_reranker:
        print("! 경고: 리랭커가 켜져 있습니다. 차단이 적용되지 않았습니다.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
