import unittest
from unittest.mock import patch

from agents.eval.probe_gen import _SynthesizedProbe, _from_chunks, generate_probes
from agents.optimize import planner
from agents.optimize.adapters.chunk_prescreener import run as run_chunk_prescreener
from core.schema import Chunk, DiagnosticReport, Document, Finding
from core.state import AgentDoctorState


class ChunkGroundingIntegrationTest(unittest.TestCase):
    @patch("agents.eval.probe_gen._llm_synthesize_query")
    def test_generated_gold_spans_drive_chunk_prescreener(self, synthesize):
        evidence = "정답" * 100
        content = ("머리" * 50) + evidence + ("꼬리" * 600)
        document = Document("d1", "memory", "txt", content)
        chunk_text = content[:800]
        chunk = Chunk(
            "d1_chunk_000",
            "d1",
            chunk_text,
            char_span=(0, len(chunk_text)),
            metadata={"chunk_index": 0},
        )
        synthesize.return_value = _SynthesizedProbe(
            question="정답 근거는 무엇인가요?",
            ground_truth="문서에 제시된 근거입니다.",
            evidence=[{"source_index": 0, "quote": evidence}],
        )
        state = AgentDoctorState(
            documents=[document],
            chunks=[chunk],
            index_config={
                "chunk_size": 800,
                "chunk_overlap": 50,
                "chunk_strategy": "recursive",
                "chunk_candidate_policy": {
                    "target_quantile": 0.85,
                    "margin_ratio": 0.20,
                    "rounding_step": 50,
                    "path_fractions": [0.33, 0.66, 1.0],
                    "candidate_count": 3,
                    # 단일 문서 통합 fixture이므로 작은 표본을 명시적으로 허용한다.
                    "min_span_count": 1,
                },
            },
        )
        state.probes = generate_probes(state)
        finding = Finding(
            finding_id="f1",
            type="retrieval_failure",
            severity="warning",
            description="검색 context가 너무 깁니다.",
            label="too_long_context",
            affected_probes=[probe.probe_id for probe in state.probes],
        )
        state.report = DiagnosticReport(
            report_id="r1",
            findings=[finding],
            overall_score=60.0,
            ragas_scores={"context_recall": 0.6},
            pass_threshold=False,
        )

        request, _decision = planner.plan(
            state,
            blacklist={
                ("too_long_context", "decrease_top_k"),
                ("too_long_context", "context_compression"),
            },
        )
        result = run_chunk_prescreener(request)

        self.assertTrue(all(probe.gold_spans for probe in state.probes))
        self.assertEqual(request.optimizer, "internal")
        self.assertGreater(len(request.search_space["chunker.chunk_size"]), 1)
        self.assertEqual(result.status, "completed")
        self.assertIn(
            result.best_config["chunker.chunk_size"],
            request.search_space["chunker.chunk_size"],
        )

    @patch("agents.eval.probe_gen._llm_generate_single_hop", return_value=None)
    def test_llm_free_heuristic_spans_drive_multiple_chunk_candidates(self, _generate):
        documents = [
            Document(f"d{i}", "memory", "txt", str(i) + ("가" * 799))
            for i in range(3)
        ]
        chunks = [
            Chunk(
                f"d{i}_chunk_000",
                f"d{i}",
                document.content,
                char_span=(0, len(document.content)),
                metadata={"chunk_index": 0},
            )
            for i, document in enumerate(documents)
        ]
        state = AgentDoctorState(
            documents=documents,
            chunks=chunks,
            index_config={
                "chunk_size": 800,
                "chunk_overlap": 50,
                "chunk_strategy": "recursive",
                "chunk_candidate_policy": {
                    "target_quantile": 0.85,
                    "margin_ratio": 0.20,
                    "rounding_step": 50,
                    "path_fractions": [0.33, 0.66, 1.0],
                    "candidate_count": 3,
                    "min_span_count": 3,
                },
            },
        )
        state.probes = _from_chunks(
            chunks,
            3,
            {document.doc_id: document for document in documents},
        )
        finding = Finding(
            finding_id="f1",
            type="retrieval_failure",
            severity="warning",
            description="검색 context가 너무 깁니다.",
            label="too_long_context",
            affected_probes=[probe.probe_id for probe in state.probes],
        )
        state.report = DiagnosticReport(
            report_id="r1",
            findings=[finding],
            overall_score=60.0,
            ragas_scores={"context_recall": 0.6},
            pass_threshold=False,
        )

        request, _decision = planner.plan(
            state,
            blacklist={
                ("too_long_context", "decrease_top_k"),
                ("too_long_context", "context_compression"),
            },
        )

        self.assertEqual(
            [span["end"] - span["start"] for probe in state.probes for span in probe.gold_spans],
            [240, 240, 240],
        )
        self.assertTrue(all(
            probe.metadata["span_grounding"]["status"] == "exact"
            for probe in state.probes
        ))
        self.assertEqual(request.optimizer, "internal")
        self.assertEqual(
            request.search_space["chunker.chunk_size"],
            [650, 450, 300],
        )
        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "grounded",
        )


if __name__ == "__main__":
    unittest.main()
