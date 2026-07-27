"""
tools/langsmith_regression.py
4단계: LangSmith Dataset 기반 회귀평가 러너 (답변 F1 중심).

고정 QA셋(eval_probes.json, 정답 포함)을 LangSmith Dataset 으로 올려두고,
같은 시험지로 RAG 파이프라인을 매번 채점한다. 코드/파라미터를 바꿔 다시 돌리면
LangSmith 가 "실행(Experiment) 간 비교 + 질문별 점수 변화"를 UI 로 보여준다.

채점은 답변 F1(KorQuAD char-F1) 중심 — agents/eval/metrics_basic.char_f1 재활용.
검색 recall 은 eval_probes.json 이 어느 코퍼스로 만들어졌는지 명시가 없어 제외한다
(gold_chunk_ids 정합이 필요하므로). 코퍼스는 레포에 포함된 sample_docs 를 적재한다.

사용:
    # reranker OFF (기본)
    python tools/langsmith_regression.py
    # reranker ON — 같은 Dataset, 다른 Experiment 로 올라가 UI 에서 비교됨
    USE_RERANKER=1 python tools/langsmith_regression.py

전제: .env 에 LANGSMITH_TRACING=true / LANGSMITH_API_KEY 설정.
      (키 없으면 LangSmith 업로드 없이 로컬 채점 결과만 콘솔에 찍고 끝낸다.)
"""
from __future__ import annotations

import json
import os
import sys

# Windows 콘솔(cp949) 유니코드 출력 보호
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# .env 로드(트레이싱 env 포함) — graph import 가 load_dotenv 를 먼저 수행한다.
import graph  # noqa: F401
from core.state import AgentDoctorState
from agents.ingest.agent import run as ingest_run
from agents.index.agent import run as index_run
from agents.rag.retriever import get_retriever
from agents.rag.generator import answer_question
from agents.eval.metrics_basic import char_f1

PROBES_FILE = "eval_probes.json"
DATASET_NAME = "agent-doctor-qa"
CORPUS_TYPE = "file"
CORPUS_URL = "sample_docs/hr_policy.md"


def load_probes() -> list[dict]:
    """eval_probes.json → ground_truth 가 있는 probe 만 (F1 채점 대상)."""
    data = json.load(open(PROBES_FILE, encoding="utf-8"))
    probes = data.get("probes", data) if isinstance(data, dict) else data
    return [p for p in probes if p.get("question") and p.get("ground_truth")]


def prepare_corpus(use_reranker: bool):
    """코퍼스를 Ingest→Index 로 Qdrant 에 적재하고, 적재된 chunks+config 로 retriever 반환."""
    state = AgentDoctorState(
        source_url=CORPUS_URL, source_type=CORPUS_TYPE, status="running",
    )
    state.index_config["use_reranker"] = use_reranker
    state = ingest_run(state)
    state = index_run(state)
    if not state.chunks:
        raise RuntimeError(f"코퍼스 적재 실패: {state.error}")
    retriever = get_retriever(state.chunks, state.index_config)
    return retriever, state.index_config


def build_target(retriever, config):
    """LangSmith target: inputs({question}) → outputs({answer})."""
    def target(inputs: dict) -> dict:
        question = inputs["question"]
        result = answer_question(question, retriever, config=config)
        answer = result["answer"] if isinstance(result, dict) else str(result)
        return {"answer": answer}
    return target


def f1_evaluator(run, example) -> dict:
    """LangSmith evaluator: char_f1(생성답변, 정답) → score(0~1)."""
    answer = (run.outputs or {}).get("answer", "")
    reference = (example.outputs or {}).get("ground_truth", "")
    return {"key": "char_f1", "score": char_f1(answer, reference)}


def upload_dataset(client, probes: list[dict]):
    """eval_probes.json 을 LangSmith Dataset 으로 (이름 조회 후 없으면 생성). 멱등."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"[Dataset] 기존 '{DATASET_NAME}' 재사용 (id={ds.id})")
        return ds
    ds = client.create_dataset(dataset_name=DATASET_NAME,
                               description="Agent Doctor 고정 QA 회귀평가 시험지")
    # langsmith 0.9.x: examples=[{"inputs": {...}, "outputs": {...}}, ...] dict 리스트.
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"question": p["question"]},
                "outputs": {"ground_truth": p["ground_truth"]},
            }
            for p in probes
        ],
    )
    print(f"[Dataset] '{DATASET_NAME}' 생성 + {len(probes)}문항 업로드 (id={ds.id})")
    return ds


def local_score(target, probes: list[dict]) -> None:
    """LangSmith 키가 없을 때 폴백 — 로컬에서 채점만 하고 콘솔 요약."""
    scores = []
    for p in probes:
        out = target({"question": p["question"]})
        s = char_f1(out["answer"], p["ground_truth"])
        scores.append(s)
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"[로컬 채점] {len(scores)}문항 평균 char_f1 = {avg:.4f}")


def main() -> None:
    use_reranker = os.getenv("USE_RERANKER", "").strip().lower() in ("1", "true", "yes")
    label = "reranker=on" if use_reranker else "reranker=off"
    print("=" * 60)
    print(f"[회귀평가] {label} — 코퍼스={CORPUS_URL}, 시험지={PROBES_FILE}")
    print("=" * 60)

    probes = load_probes()
    print(f"[시험지] ground_truth 있는 probe {len(probes)}개")

    retriever, config = prepare_corpus(use_reranker)
    target = build_target(retriever, config)

    if not os.getenv("LANGSMITH_API_KEY"):
        print("[경고] LANGSMITH_API_KEY 없음 → 업로드 없이 로컬 채점만 수행")
        local_score(target, probes)
        return

    from langsmith import Client, evaluate

    client = Client()
    upload_dataset(client, probes)

    print(f"[Experiment] evaluate 실행 중... (prefix={label})")
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[f1_evaluator],
        experiment_prefix=label,       # 실험 이름에 reranker on/off 표시 → UI 비교
        metadata={"use_reranker": use_reranker, "index_config": config},
    )
    print("=" * 60)
    print(f"[완료] LangSmith Experiment 로 업로드됨 → Datasets & Experiments 탭 확인")
    print(f"       같은 Dataset 에 reranker on/off 를 각각 돌리면 실행 간 비교가 됩니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
