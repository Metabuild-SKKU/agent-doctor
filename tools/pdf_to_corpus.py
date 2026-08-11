"""
tools/pdf_to_corpus.py
PDF 1개 → data/pdf_corpus.json (tools/make_external_rag.py 가 기대하는 [{id, text, source}] 형식)

"남의 RAG 진단" 시나리오를 로컬에서 재현하기 위한 전 단계 — 우리 Ingest agent의
PDF 파싱(agents/ingest/agent.py `_ingest_file`, pdfplumber 기반 표·헤더/푸터 처리
포함)을 그대로 재사용해 문서를 텍스트로 뽑고, make_external_rag.py 가 LlamaIndex로
"남의 RAG"를 구성할 때 쓰는 코퍼스 파일로 저장한다.

사용법:
    python -m tools.pdf_to_corpus docs/문서.pdf
    python -m tools.pdf_to_corpus docs/문서.pdf --out=data/pdf_corpus.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from core.state import AgentDoctorState
from agents.ingest.agent import run as ingest_run

DEFAULT_OUT = "data/pdf_corpus.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="PDF → pdf_corpus.json 변환")
    ap.add_argument("pdf_path")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.pdf_path):
        sys.exit(f"[pdf-to-corpus] 파일 없음: {args.pdf_path}")

    state = AgentDoctorState()
    state.source_url = args.pdf_path
    state.source_type = "file"
    state = ingest_run(state)

    if state.error:
        sys.exit(f"[pdf-to-corpus] Ingest 실패: {state.error}")

    docs = [
        {"id": doc.doc_id, "text": doc.content, "source": doc.metadata.get("filename") or doc.source}
        for doc in state.documents
    ]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    for doc in docs:
        print(f"[pdf-to-corpus] '{doc['source']}' {len(doc['text'])}자")
    print(f"[pdf-to-corpus] {len(docs)}건 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
