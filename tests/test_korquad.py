"""
tests/test_korquad.py
KorQuAD 어댑터(agents/eval/datasets/korquad.py) 단위 테스트.

임시 jsonl 을 만들어 순수 변환 로직만 검증한다(실제 data/ 파일·모델·API 불필요).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.eval.datasets.korquad import (
    _stitch, _selected_doc_ids, reconstruct_documents, load_taxonomy_probes,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


# ── _stitch: 좌표 버퍼 복원 ────────────────────────────────────────

def test_stitch_single_chunk():
    assert _stitch([(0, 3, "abc")]) == "abc"


def test_stitch_adjacent():
    assert _stitch([(0, 3, "abc"), (3, 6, "def")]) == "abcdef"


def test_stitch_gap_filled_with_space():
    # 3~5 구간이 비어 공백으로 채워진다(길이 8).
    out = _stitch([(0, 3, "abc"), (5, 8, "xyz")])
    assert out == "abc  xyz"
    assert len(out) == 8


def test_stitch_overlap_same_text_overwrites():
    # 겹치는 구간("cd")은 같은 글자로 덮어써져 깨지지 않는다.
    assert _stitch([(0, 4, "abcd"), (2, 6, "cdef")]) == "abcdef"


def test_stitch_out_of_order_input():
    # 입력 순서가 뒤섞여도 좌표 기준으로 복원.
    assert _stitch([(3, 6, "def"), (0, 3, "abc")]) == "abcdef"


def test_stitch_empty():
    assert _stitch([]) == ""


# ── _selected_doc_ids: max_docs 선택 ──────────────────────────────

def test_selected_doc_ids(tmp_path):
    corpus = _write_jsonl(tmp_path / "c.jsonl", [
        {"doc_id": "d1", "chunk_id": "d1_0", "text": "a", "char_start": 0, "char_end": 1},
        {"doc_id": "d2", "chunk_id": "d2_0", "text": "b", "char_start": 0, "char_end": 1},
        {"doc_id": "d3", "chunk_id": "d3_0", "text": "c", "char_start": 0, "char_end": 1},
    ])
    assert _selected_doc_ids(corpus, None) is None      # 미지정=전체
    assert _selected_doc_ids(corpus, 0) is None         # 0=전체
    assert _selected_doc_ids(corpus, 2) == {"d1", "d2"}  # 등장 순서 앞 2개


# ── reconstruct_documents ─────────────────────────────────────────

def test_reconstruct_documents_basic(tmp_path):
    corpus = _write_jsonl(tmp_path / "c.jsonl", [
        {"doc_id": "d1", "chunk_id": "d1_0", "title": "T1", "text": "hello ", "char_start": 0, "char_end": 6},
        {"doc_id": "d1", "chunk_id": "d1_1", "title": "T1", "text": "world", "char_start": 6, "char_end": 11},
        {"doc_id": "d2", "chunk_id": "d2_0", "title": "T2", "text": "foo", "char_start": 0, "char_end": 3},
    ])
    docs = reconstruct_documents(corpus)
    assert len(docs) == 2
    by_id = {d.doc_id: d for d in docs}
    assert by_id["d1"].content == "hello world"
    assert by_id["d1"].metadata["title"] == "T1"
    assert by_id["d2"].content == "foo"


def test_reconstruct_documents_max_docs(tmp_path):
    corpus = _write_jsonl(tmp_path / "c.jsonl", [
        {"doc_id": "d1", "chunk_id": "d1_0", "text": "a", "char_start": 0, "char_end": 1},
        {"doc_id": "d2", "chunk_id": "d2_0", "text": "b", "char_start": 0, "char_end": 1},
    ])
    docs = reconstruct_documents(corpus, max_docs=1)
    assert [d.doc_id for d in docs] == ["d1"]


# ── load_taxonomy_probes ──────────────────────────────────────────

def _corpus_qa(tmp_path, corpus_rows, qa_rows):
    return (_write_jsonl(tmp_path / "c.jsonl", corpus_rows),
            _write_jsonl(tmp_path / "q.jsonl", qa_rows))


def test_taxonomy_gold_spans_mapping(tmp_path):
    corpus, qa = _corpus_qa(tmp_path,
        [{"doc_id": "d1", "chunk_id": "d1_3", "text": "x", "char_start": 10, "char_end": 20}],
        [{"qa_id": "1", "question": "Q?", "answer_text": "A", "doc_id": "d1",
          "positive_chunk_ids": ["d1_3"]}])
    probes = load_taxonomy_probes(qa, corpus)
    assert len(probes) == 1
    p = probes[0]
    assert p.source == "taxonomy" and p.ground_truth == "A" and p.gold_doc_id == "d1"
    assert p.gold_spans == [{"doc_id": "d1", "start": 10, "end": 20}]
    assert p.gold_chunk_ids == []   # resync 전이라 비어 있음


def test_taxonomy_composite_key_no_collision(tmp_path):
    """서로 다른 문서가 같은 chunk_id 를 써도 gold 는 qa 의 doc_id 문서 좌표를 써야 한다.
    (단일 chunk_id 키였다면 뒤 문서 d2 의 좌표로 오염됐을 케이스.)"""
    corpus, qa = _corpus_qa(tmp_path,
        [{"doc_id": "d1", "chunk_id": "c0", "text": "x", "char_start": 0, "char_end": 5},
         {"doc_id": "d2", "chunk_id": "c0", "text": "y", "char_start": 100, "char_end": 105}],
        [{"qa_id": "1", "question": "Q?", "answer_text": "A", "doc_id": "d1",
          "positive_chunk_ids": ["c0"]}])
    probes = load_taxonomy_probes(qa, corpus)
    assert probes[0].gold_spans == [{"doc_id": "d1", "start": 0, "end": 5}]  # d1 좌표, d2(100) 아님


def test_taxonomy_missing_positive_gives_empty_spans(tmp_path):
    corpus, qa = _corpus_qa(tmp_path,
        [{"doc_id": "d1", "chunk_id": "d1_0", "text": "x", "char_start": 0, "char_end": 5}],
        [{"qa_id": "1", "question": "Q?", "answer_text": "A", "doc_id": "d1",
          "positive_chunk_ids": ["d1_9"]}])   # 존재하지 않는 청크
    probes = load_taxonomy_probes(qa, corpus)
    assert probes[0].gold_spans == []


def test_taxonomy_qa_limit_and_max_docs(tmp_path):
    corpus_rows = [
        {"doc_id": "d1", "chunk_id": "d1_0", "text": "x", "char_start": 0, "char_end": 1},
        {"doc_id": "d2", "chunk_id": "d2_0", "text": "y", "char_start": 0, "char_end": 1},
    ]
    qa_rows = [
        {"qa_id": "1", "question": "Q1", "answer_text": "A", "doc_id": "d1", "positive_chunk_ids": ["d1_0"]},
        {"qa_id": "2", "question": "Q2", "answer_text": "B", "doc_id": "d1", "positive_chunk_ids": ["d1_0"]},
        {"qa_id": "3", "question": "Q3", "answer_text": "C", "doc_id": "d2", "positive_chunk_ids": ["d2_0"]},
    ]
    corpus, qa = _corpus_qa(tmp_path, corpus_rows, qa_rows)
    # max_docs=1 → d1 문서의 qa 만(2개), qa_limit 은 그 뒤 상한
    assert len(load_taxonomy_probes(qa, corpus, max_docs=1)) == 2
    assert len(load_taxonomy_probes(qa, corpus, limit=1)) == 1
    assert {p.gold_doc_id for p in load_taxonomy_probes(qa, corpus, max_docs=1)} == {"d1"}


def test_missing_file_raises_friendly(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        reconstruct_documents(str(tmp_path / "nope.jsonl"))
