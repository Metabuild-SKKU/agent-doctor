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
    # reranker OFF (기본 — sample_docs + eval_probes.json)
    python tools/langsmith_regression.py
    # reranker ON — 같은 Dataset, 다른 Experiment 로 올라가 UI 에서 비교됨
    USE_RERANKER=1 python tools/langsmith_regression.py

    # tests/corpus/ 의 실제 코퍼스 + run_corpus 가 만든 qa.json 으로 회귀평가
    # (폴더만 지정 — 한글 파일명도 파이썬이 폴더에서 직접 찾아 안전):
    REGRESSION_CORPUS_DIR=tests/corpus python tools/langsmith_regression.py

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

# 기본값은 레포 포함 sample_docs + eval_probes.json. 환경변수로 다른 코퍼스/QA셋을
# 지정하면(예: tests/corpus/ 의 실제 코퍼스 + qa.json) 그 세트로 회귀평가한다.
#
#   REGRESSION_CORPUS_DIR  코퍼스 폴더(run_corpus 규약: 안의 .pdf/.md/.txt + qa.json).
#       ★ 한글 파일명 권장 방식 — 파일명을 셸/환경변수로 넘기지 않고 파이썬이 폴더에서
#         직접 glob 으로 찾으므로 cp949 인코딩 깨짐이 없다.
#   REGRESSION_CORPUS  코퍼스 문서 경로 하나(.pdf/.md/.txt) — 영문 경로일 때만 권장.
#   REGRESSION_QA      QA셋(정답 포함) 경로 — eval_probes.json / run_corpus 의 qa.json 동일 형식.
#   REGRESSION_DATASET LangSmith Dataset 이름 명시(기본은 QA 파일명 stem — 아래 참고).
#
# Dataset 이름 기본 규칙 = QA 파일명(stem). 회귀평가 분리 기준은 '시험지(QA셋)'이므로,
# QA 파일이 다르면 Dataset 도 갈린다("같은 코퍼스 + 다른 QA"까지 분리). 예:
#   qa.json → 'qa', qa_hard.json → 'qa_hard', eval_probes.json → 'eval_probes'.
import glob
from pathlib import Path

_DOC_SUFFIXES = (".pdf", ".md", ".txt")


def _dataset_name_for(qa_path: str) -> str:
    """Dataset 이름을 'QA 파일명(stem)' 기준으로 만든다 — QA셋이 다르면 Dataset 도 갈린다.

    회귀평가의 분리 기준은 '시험지(QA셋)'다. 코퍼스나 폴더가 아니라 QA 파일이 이름표 —
    그래야 '같은 코퍼스 + 다른 QA셋'(예: qa.json vs qa_hard.json)도 서로 다른 Dataset 으로
    갈려 한 표에 섞이지 않는다. REGRESSION_DATASET 으로 언제든 덮어쓸 수 있다.
    (단, run_corpus 규약상 코퍼스마다 QA 파일명이 다 'qa.json' 이면 이름이 겹칠 수 있으므로,
     그럴 땐 폴더를 나누고 REGRESSION_DATASET 으로 구분하거나 QA 파일명을 달리 둘 것.)"""
    override = os.getenv("REGRESSION_DATASET")
    if override:
        return override
    return Path(qa_path).stem


def _resolve_corpus() -> tuple[str, str, str]:
    """(코퍼스 문서 경로, QA셋 경로, Dataset 이름)을 환경변수에서 해석한다.

    REGRESSION_CORPUS_DIR 이 우선 — 폴더 안에서 문서/qa.json 을 파이썬이 직접 찾아
    한글 파일명이 셸을 거치며 깨지는 문제를 피한다."""
    corpus_dir = os.getenv("REGRESSION_CORPUS_DIR")
    if corpus_dir:
        base = Path(corpus_dir)
        docs = sorted(
            p for p in base.iterdir()
            if p.is_file() and p.suffix.lower() in _DOC_SUFFIXES
            and p.stem.lower() != "readme"
        )
        if not docs:
            raise SystemExit(f"코퍼스 문서가 없습니다: {base}/*{_DOC_SUFFIXES}")
        qa = os.getenv("REGRESSION_QA") or str(base / "qa.json")
        return str(docs[0]), qa, _dataset_name_for(qa)
    corpus = os.getenv("REGRESSION_CORPUS", "sample_docs/hr_policy.md")
    qa = os.getenv("REGRESSION_QA", "eval_probes.json")
    return corpus, qa, _dataset_name_for(qa)


CORPUS_URL, PROBES_FILE, DATASET_NAME = _resolve_corpus()
CORPUS_TYPE = "file"


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
    # 어느 Dataset 에 올라가는지 미리 알린다. 같은 이름이면 같은 Dataset 에 Experiment 가
    # 누적된다 — 'corpus' 같은 범용 폴더명을 재활용하면 서로 다른 코퍼스가 한 Dataset 에
    # 섞이므로, 코퍼스별로 폴더를 나누거나 REGRESSION_DATASET 으로 이름을 지정할 것.
    print(f"[회귀평가] LangSmith Dataset = '{DATASET_NAME}' (같은 이름이면 Experiment 누적)")
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
