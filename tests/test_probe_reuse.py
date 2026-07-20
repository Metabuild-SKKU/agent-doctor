# doc_id 안정성과 probe 캐시 무효화 계약 (외부 API·모델 없이).
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from agents.eval.agent import _pipeline_version
from agents.eval.probe_gen import uses_user_log
from agents.eval.probe_store import load_probes, save_probes
from agents.ingest.agent import _ingest_file
from core.schema import Chunk, Probe
from core.state import AgentDoctorState


def _write(tmp: Path, name: str, text: str) -> str:
    path = tmp / name
    path.write_text(text, encoding="utf-8")
    return str(path)


class StableDocIdTests(unittest.TestCase):
    def test_same_file_yields_same_doc_id_across_runs(self):
        """실행이 갈려도 같은 파일이면 doc_id 가 같아야 probe 캐시가 산다."""
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp), "hr.md", "# 규정\n재택근무는 주 2일까지.")

            first = _ingest_file(src)[0]
            second = _ingest_file(src)[0]

            self.assertEqual(first.doc_id, second.doc_id)

    def test_edited_file_keeps_doc_id(self):
        """본문이 바뀌어도 문서의 정체성(=파일)은 그대로 — 무효화는 청크 hash 가 맡는다."""
        with tempfile.TemporaryDirectory() as tmp:
            src = _write(Path(tmp), "hr.md", "원래 본문")
            before = _ingest_file(src)[0]
            _write(Path(tmp), "hr.md", "고쳐 쓴 본문")
            after = _ingest_file(src)[0]

            self.assertEqual(before.doc_id, after.doc_id)
            self.assertNotEqual(before.content, after.content)

    def test_different_files_yield_different_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _write(Path(tmp), "a.md", "본문 A")
            b = _write(Path(tmp), "b.md", "본문 B")

            self.assertNotEqual(_ingest_file(a)[0].doc_id, _ingest_file(b)[0].doc_id)


class PipelineVersionTests(unittest.TestCase):
    def _state(self, *chunks: Chunk) -> AgentDoctorState:
        state = AgentDoctorState()
        state.chunks = list(chunks)
        return state

    def _chunk(self, chunk_id: str, text: str, chunk_hash: str) -> Chunk:
        return Chunk(chunk_id=chunk_id, doc_id="doc-1", text=text, hash=chunk_hash)

    def test_same_corpus_yields_same_version(self):
        left = self._state(self._chunk("d_chunk_000", "본문", "h0"))
        right = self._state(self._chunk("d_chunk_000", "본문", "h0"))

        self.assertEqual(_pipeline_version(left), _pipeline_version(right))

    def test_edited_content_changes_version(self):
        """chunk_id 는 같고 본문만 바뀐 경우 — hash 를 안 넣으면 stale probe 를 쓰게 된다."""
        before = self._state(self._chunk("d_chunk_000", "원래 본문", "h0"))
        after = self._state(self._chunk("d_chunk_000", "고쳐 쓴 본문", "h1"))

        self.assertNotEqual(_pipeline_version(before), _pipeline_version(after))

    def test_index_config_change_changes_version(self):
        state = self._state(self._chunk("d_chunk_000", "본문", "h0"))
        original = _pipeline_version(state)
        state.index_config["chunk_size"] = 999

        self.assertNotEqual(original, _pipeline_version(state))


class ProbeSourcePredicateTests(unittest.TestCase):
    """agent.py STEP1 의 캐시 여부는 generate_probes 의 실제 경로와 일치해야 한다."""

    def _state(self, questions: list[str]) -> AgentDoctorState:
        state = AgentDoctorState()
        state.user_questions = questions
        return state

    def test_auto_ignores_user_questions(self):
        """EVAL_PROBE_SOURCE=auto: 질문이 있어도 LLM 생성 → 캐시 대상이어야 한다."""
        with patch.dict("os.environ", {"EVAL_PROBE_SOURCE": "auto"}):
            self.assertFalse(uses_user_log(self._state(["연차는?"])))

    def test_user_log_is_forced_when_questions_exist(self):
        with patch.dict("os.environ", {"EVAL_PROBE_SOURCE": "user_log"}):
            self.assertTrue(uses_user_log(self._state(["연차는?"])))

    def test_unset_falls_back_to_question_presence(self):
        with patch.dict("os.environ", {"EVAL_PROBE_SOURCE": ""}):
            self.assertTrue(uses_user_log(self._state(["연차는?"])))
            self.assertFalse(uses_user_log(self._state([])))


class ProbeStoreRoundTripTests(unittest.TestCase):
    def test_probes_are_reused_when_version_matches(self):
        probes = [Probe(probe_id="p0", question="연차는?", source="llm_generated")]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "probes.json")
            save_probes(probes, "v1", path=path)

            self.assertEqual(len(load_probes("v1", path=path) or []), 1)
            self.assertIsNone(load_probes("v2", path=path))


if __name__ == "__main__":
    unittest.main()
