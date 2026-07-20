import unittest

from agents.eval import diagnose, signals
from agents.eval.types import EvalRecord, Mode
from agents.optimize import planner
from agents.optimize.adapters.chunk_prescreener import run as run_prescreener
from core.schema import Chunk, DiagnosticReport, Document, Finding, Probe
from core.state import AgentDoctorState


def _overlap_policy() -> dict:
    return {
        "target_quantiles": [0.50, 0.85, 0.95],
        "rounding_step": 25,
        "candidate_count": 3,
        "min_crossing_span_count": 1,
        "max_ratio": 0.40,
        "max_overlap": 300,
    }


def _fixed_chunks(doc_id: str, length: int, size: int, overlap: int) -> list[Chunk]:
    chunks = []
    step = size - overlap
    for index, start in enumerate(range(0, length, step)):
        end = min(length, start + size)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_chunk_{index:03d}",
                doc_id=doc_id,
                text="가" * (end - start),
                char_span=(start, end),
            )
        )
        if end >= length:
            break
    return chunks


class ChunkBoundaryDiagnosisTest(unittest.TestCase):
    def tearDown(self):
        signals.set_context()
        signals.set_mode(Mode.FAST)

    def test_split_exact_span_is_confirmed_without_llm(self):
        chunks = _fixed_chunks("d1", 1000, 400, 50)
        signals.set_context(chunks=chunks)
        probe = Probe(
            probe_id="p1",
            question="정답은?",
            source="taxonomy",
            answer_exists=True,
            ground_truth="정답",
            gold_chunk_ids=[chunks[0].chunk_id, chunks[1].chunk_id],
            gold_spans=[{"doc_id": "d1", "start": 325, "end": 450}],
            metadata={"span_grounding": {"status": "exact"}},
        )
        record = EvalRecord(
            probe=probe,
            retrieved_chunk_ids=[chunks[0].chunk_id],
            generated_answer="오답",
            oracle_answer="정답",
        )

        findings = diagnose.diagnose(record, Mode.FAST)

        self.assertEqual(findings[0].label, "chunking_context_mismatch")
        self.assertTrue(findings[0].confirmed)
        self.assertEqual(findings[0].metadata["group"], "A")
        self.assertEqual(
            findings[0].metadata["boundary_analysis"]["boundary_split_count"],
            1,
        )

    def test_chunk_fallback_span_is_not_used_for_boundary_diagnosis(self):
        chunks = _fixed_chunks("d1", 1000, 400, 50)
        signals.set_context(chunks=chunks)
        probe = Probe(
            probe_id="p1",
            question="정답은?",
            source="taxonomy",
            answer_exists=True,
            ground_truth="정답",
            gold_chunk_ids=[chunks[0].chunk_id, chunks[1].chunk_id],
            gold_spans=[{"doc_id": "d1", "start": 325, "end": 450}],
            metadata={"span_grounding": {"status": "chunk_fallback"}},
        )
        record = EvalRecord(
            probe=probe,
            retrieved_chunk_ids=[chunks[0].chunk_id],
            generated_answer="오답",
            oracle_answer="정답",
        )

        findings = diagnose.diagnose(record, Mode.FAST)

        self.assertFalse(any(
            finding.label == "chunking_context_mismatch" for finding in findings
        ))


class ChunkOverlapGroundingTest(unittest.TestCase):
    def _state(self) -> AgentDoctorState:
        document = Document("d1", "memory", "txt", "가" * 1600)
        chunks = _fixed_chunks("d1", len(document.content), 400, 50)
        spans = [(325, 450), (625, 800), (900, 1150)]
        probes = [
            Probe(
                probe_id=f"p{index}",
                question=f"질문 {index}",
                source="taxonomy",
                answer_exists=True,
                gold_spans=[{"doc_id": "d1", "start": start, "end": end}],
                metadata={"span_grounding": {"status": "exact"}},
            )
            for index, (start, end) in enumerate(spans, start=1)
        ]
        findings = [
            Finding(
                finding_id=f"{probe.probe_id}:chunking_context_mismatch",
                type="retrieval_failure",
                severity="warning",
                description="청크 경계에서 정답이 나뉨",
                label="chunking_context_mismatch",
                affected_probes=[probe.probe_id],
            )
            for probe in probes
        ]
        return AgentDoctorState(
            documents=[document],
            chunks=chunks,
            probes=probes,
            report=DiagnosticReport(
                report_id="r1",
                findings=findings,
                overall_score=0.5,
                pass_threshold=False,
            ),
            index_config={
                "chunk_size": 400,
                "chunk_overlap": 50,
                "chunk_strategy": "fixed",
                "top_k": 5,
                "chunk_overlap_candidate_policy": _overlap_policy(),
            },
        )

    def test_percentiles_create_safe_overlap_candidates(self):
        request, decision = planner.plan(self._state())

        self.assertEqual(decision.mode, "apply_optimize")
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(
            request.search_space,
            {"chunker.chunk_overlap": [75, 125, 150]},
        )
        grounding = request.metadata["candidate_grounding"]
        self.assertEqual(grounding["status"], "grounded")
        self.assertEqual(grounding["p50"], 75)
        self.assertEqual(grounding["p85"], 125)
        self.assertEqual(grounding["limit_exceeded_count"], 1)
        self.assertLessEqual(max(request.search_space["chunker.chunk_overlap"]), 160)

    def test_prescreener_selects_smallest_recovering_overlap(self):
        request, _decision = planner.plan(self._state())

        result = run_prescreener(request)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config, {"chunker.chunk_overlap": 125})
        selected = next(
            item
            for item in result.metadata["candidate_metrics"]
            if item["value"] == 125
        )
        self.assertEqual(selected["boundary_recovery_rate"], 1.0)
        self.assertEqual(selected["unrecovered_cut_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
