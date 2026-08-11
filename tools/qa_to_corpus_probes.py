"""
tools/qa_to_corpus_probes.py
tools/tax_guide_questions.json(question/ground_truth/gold_contexts) → tests/corpus/qa.json
(Probe 리스트, probe_store.py 포맷)으로 변환한다.

목적: 외부 RAG 시뮬레이터(make_external_rag.py)에 쓴 것과 "같은 질문·같은 정답"으로
우리 기본 파이프라인(tests/run_corpus.py)의 Eval을 돌려 직접 비교하기 위함.

Probe.gold_char_span/gold_spans는 원문 문자 오프셋이라, tests/corpus/ 에 실제로 놓일
PDF를 Ingest 와 동일한 방식(agents.ingest.agent.run, source_type=file)으로 다시
읽어서 만든 content 문자열 위에서 gold_contexts[0]의 위치를 찾는다. doc_id도
Ingest 가 그 경로에 대해 만드는 것과 같은 규칙(_stable_doc_id)으로 재계산해야
gold_doc_id가 실제 파이프라인이 만든 Document와 맞아떨어진다.

사용법:
    python -m tools.qa_to_corpus_probes
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QUESTIONS_PATH = REPO_ROOT / "tools" / "tax_guide_questions.json"
CORPUS_DIR = REPO_ROOT / "tests" / "corpus"
OUT_PATH = CORPUS_DIR / "qa.json"
DOC_SUFFIXES = (".pdf", ".md", ".txt")


def find_source_doc() -> Path:
    docs = sorted(
        p for p in CORPUS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in DOC_SUFFIXES and p.stem.lower() != "readme"
    )
    if not docs:
        sys.exit(f"[qa-to-probes] {CORPUS_DIR} 에 원본 문서가 없습니다.")
    return docs[0]


def stable_doc_id(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "|".join(("file", str(path.resolve())))))


def main() -> int:
    from core.state import AgentDoctorState
    from agents.ingest.agent import run as ingest_run

    source_doc = find_source_doc()
    print(f"[qa-to-probes] 원본 문서: {source_doc.name}")

    state = AgentDoctorState()
    state.source_type = "file"
    state.source_url = str(source_doc)
    state = ingest_run(state)
    if state.error:
        sys.exit(f"[qa-to-probes] Ingest 실패: {state.error}")

    doc = state.documents[0]
    content = doc.content
    doc_id = doc.doc_id
    expected_doc_id = stable_doc_id(source_doc)
    if doc_id != expected_doc_id:
        print(f"[qa-to-probes] 경고: doc_id 불일치 ({doc_id} != {expected_doc_id})"
              " — Ingest 규칙이 바뀐 것으로 보입니다. 그래도 실제 doc.doc_id 를 씁니다.")

    items = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"[qa-to-probes] 질문 {len(items)}건 로드")

    probes = []
    skipped = []
    for i, item in enumerate(items):
        gold_contexts = item.get("gold_contexts") or []
        span = None
        matched_text = None
        for gc in gold_contexts:
            idx = content.find(gc)
            if idx >= 0:
                span = (idx, idx + len(gc))
                matched_text = gc
                break
        if span is None:
            skipped.append(item["question"])
            continue

        probes.append({
            "probe_id": f"probe_ext_{i:04d}",
            "question": item["question"],
            "source": "llm_generated",
            "expected_difficulty": "medium",
            "answer_exists": True,
            "ground_truth": item.get("ground_truth"),
            "gold_chunk_ids": [],
            "qtype": None,
            "metadata": {"gen_method": "manual_import_ext", "matched_gold_context": matched_text},
            "gold_doc_id": doc_id,
            "gold_char_span": list(span),
            "gold_spans": [{"doc_id": doc_id, "start": span[0], "end": span[1]}],
        })

    print(f"[qa-to-probes] 변환 성공 {len(probes)}건 / 실패(gold_contexts 매칭 안 됨) {len(skipped)}건")
    for q in skipped:
        print(f"  ! 매칭 실패: {q[:60]}")

    out = {"version": "manual-ext-100", "probes": probes}
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qa-to-probes] {len(probes)}건 → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
