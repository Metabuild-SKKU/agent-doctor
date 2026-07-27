import os
import unittest
from unittest.mock import patch

from agents.eval.probe_gen import _SynthesizedProbe, generate_probes
from agents.optimize import planner
from agents.optimize.adapters.chunk_prescreener import run as run_chunk_prescreener
from core.schema import Chunk, DiagnosticReport, Document, Finding, Probe
from core.state import AgentDoctorState


class ChunkGroundingIntegrationTest(unittest.TestCase):
    def setUp(self):
        # 생성 개수를 고정한다. 같은 프로세스의 다른 테스트가 graph.py 를 import 하면
        # load_dotenv(override=True) 로 .env 의 EVAL_TESTSET_SIZE 가 프로세스에 들어오고,
        # Probe 개수가 달라지면서 gold_spans 없는 no_answer Probe 가 섞여 아래 단언이
        # 실행 순서에 따라 깨진다. 이 테스트의 관심사는 개수가 아니라 span 그라운딩이다.
        self._env = {k: os.environ.get(k) for k in ("EVAL_TESTSET_SIZE", "EVAL_PROBE_SOURCE")}
        os.environ["EVAL_TESTSET_SIZE"] = "3"
        os.environ.pop("EVAL_PROBE_SOURCE", None)

    def tearDown(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @patch("agents.eval.probe_gen._llm_synthesize_query")
    def test_generated_gold_spans_drive_chunk_prescreener(self, synthesize):
        evidence = "정답" * 100
        content = (
            ("머리" * 50)
            + ". 도입 문장입니다. "
            + evidence
            + "입니다. "
            + ("꼬리" * 100)
            + ". "
            + ("후속" * 500)
        )
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
        grounded_probe_ids = [
            probe.probe_id for probe in state.probes if probe.gold_spans
        ]
        finding = Finding(
            finding_id="f1",
            type="retrieval_failure",
            severity="warning",
            description="검색 context가 너무 깁니다.",
            label="too_long_context",
            affected_probes=grounded_probe_ids,
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

        self.assertTrue(grounded_probe_ids)
        self.assertEqual(request.optimizer, "internal")
        self.assertGreater(len(request.search_space["chunker.chunk_size"]), 1)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.best_config["chunker.chunk_size"], 800)
        self.assertTrue(result.metadata["best_is_baseline"])

    def test_structural_spans_drive_multiple_chunk_candidates(self):
        documents = [
            Document(
                f"d{i}",
                "memory",
                "txt",
                ("가" * 239)
                + ".\n\n"
                + str(i)
                + ("나" * 556),
            )
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
        state.probes = [
            Probe(
                probe_id=f"probe_{i}",
                question=f"문서 {i}의 첫 문단은 무엇인가요?",
                source="taxonomy",
                answer_exists=True,
                ground_truth=document.content[:240],
                gold_chunk_ids=[chunks[i].chunk_id],
                gold_spans=[{"doc_id": document.doc_id, "start": 0, "end": 240}],
                metadata={
                    "span_grounding": {
                        "status": "exact",
                        "span_qualities": ["exact"],
                    }
                },
            )
            for i, document in enumerate(documents)
        ]
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
        candidates = request.search_space["chunker.chunk_size"]
        self.assertGreater(len(candidates), 1)
        self.assertTrue(all(600 <= value < 800 for value in candidates))
        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "grounded",
        )
        self.assertEqual(
            request.metadata["candidate_grounding"]["source"],
            "structural_evidence_windows",
        )


if __name__ == "__main__":
    unittest.main()
