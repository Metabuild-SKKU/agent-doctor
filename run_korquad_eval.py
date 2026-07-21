"""
run_korquad_eval.py
KorQuAD 2.1(전처리본, data/) 로 Eval Agent 를 단독 실행한다.

  python run_korquad_eval.py                                  # qa 30개 + gold문서, dense(bge-m3)
  KORQUAD_QA_LIMIT=100 KORQUAD_DISTRACTORS=200 python run_korquad_eval.py
  KORQUAD_EMBED=0 python run_korquad_eval.py                  # 임베딩 스킵 → keyword(BM25) 검색
  EVAL_MODE=deep EVAL_ENABLE_LLM=1 python run_korquad_eval.py # RAGAS 까지(LLM 키·비용 발생)

동작:
  corpus 는 이미 청킹돼 있어 Chunk 로 직접 주입(Ingest/Index 청킹 건너뜀)하고,
  qa 는 taxonomy Probe(정답+gold 청크)로 eval_probes.json 에 저장한 뒤
  EVAL_PROBE_SOURCE=made 로 Eval 이 그대로 쓰게 한다.

환경변수:
  KORQUAD_QA_LIMIT    평가할 qa 개수 (기본 30, "all"=전체 1718)
  KORQUAD_DISTRACTORS gold 외 코퍼스 문서 수 (기본 0, "all"=전체 코퍼스)
  KORQUAD_EMBED       1=dense 임베딩(기본) / 0=keyword 전용
  KORQUAD_EMBED_MODEL 임베딩 모델 (기본 BAAI/bge-m3)
  KORQUAD_TOP_K       검색 top_k (기본 5)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.run_logger import setup_run_logging
setup_run_logging(prefix="korquad_eval")

from core.state import AgentDoctorState
from agents.eval.datasets.korquad import load_dataset
from agents.eval.probe_store import save_probes
from agents.eval.agent import run as eval_run
from agents.index.qdrant_store import embed

EMBED_MODEL = os.getenv("KORQUAD_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("KORQUAD_EMBED_DIM", "1024"))


def _int_or_all(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip().lower()
    if raw == "all":
        return -1 if name == "KORQUAD_DISTRACTORS" else None  # -1: 전체 코퍼스 / None: 전체 qa
    return int(raw) if raw else default


def main() -> None:
    qa_limit = _int_or_all("KORQUAD_QA_LIMIT", 30)
    distractors = _int_or_all("KORQUAD_DISTRACTORS", 0)
    do_embed = os.getenv("KORQUAD_EMBED", "1").strip().lower() not in ("0", "false", "no")

    print(f"[KorQuAD] 로드: qa_limit={qa_limit} distractors={distractors} embed={do_embed}")
    chunks, probes = load_dataset(qa_limit=qa_limit, distractor_docs=(distractors or 0))
    gold_docs = {p.gold_doc_id for p in probes}
    print(f"[KorQuAD] 청크 {len(chunks)}개 / probe {len(probes)}개 (gold 문서 {len(gold_docs)}개)")

    if do_embed:
        print(f"[KorQuAD] dense 임베딩: {EMBED_MODEL} × {len(chunks)}청크 …")
        for i, c in enumerate(chunks, 1):
            c.embedding = embed(c.text, model_name=EMBED_MODEL, vector_dim=EMBED_DIM)
            if i % 200 == 0:
                print(f"  …{i}/{len(chunks)}")
    else:
        print("[KorQuAD] 임베딩 스킵 → keyword(BM25) 검색으로 진단")

    state = AgentDoctorState()
    state.chunks = chunks
    state.index_config = {
        **state.index_config,
        "embedding_model": EMBED_MODEL,
        "embedding_dimension": EMBED_DIM,
        "top_k": int(os.getenv("KORQUAD_TOP_K", "5")),
    }

    # qa 를 made 소스로 주입 (gold_chunk_ids 보존). version 값은 made 경로에서 무시된다.
    save_probes(probes, version="korquad")
    os.environ["EVAL_PROBE_SOURCE"] = "made"

    state = eval_run(state)
    if state.error:
        print(f"[중단] Eval 오류: {state.error}")
        sys.exit(1)

    r = state.report
    print("\n" + "=" * 56)
    print("  KorQuAD Eval 결과")
    print("=" * 56)
    print(f"probe: {len(state.probes)}개")
    if r:
        print(f"overall_score  = {r.overall_score}   pass = {r.pass_threshold}")
        print(f"oracle_accuracy= {r.oracle_accuracy}")
        print(f"ragas_scores   = {r.ragas_scores}")
        print(f"findings_summary = {r.findings_summary}")
    print("\nKorQuAD Eval 완료 [OK]")


if __name__ == "__main__":
    main()
