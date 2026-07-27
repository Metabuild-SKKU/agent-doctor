import os
import unittest
from unittest.mock import patch

from agents.eval.probe_gen import _SynthesizedProbe, generate_probes
from agents.optimize import planner
from agents.optimize.adapters.chunk_prescreener import run as run_chunk_prescreener
from core.schema import Chunk, DiagnosticReport, Document, Finding
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

    # 예전엔 휴리스틱 폴백(_llm_generate_single_hop=None)으로 Probe 를 만들었지만,
    # 그 경로는 원문 조각을 질문·정답에 그대로 써서 '질문=정답' Probe 가 되고 품질
    # 게이트가 전부 폐기한다(Probe 0개 → 테스트 주제인 프리스크리너까지 못 감).
    # 관심사는 "문서 여러 개의 gold_span 이 청크 후보를 이끄는지" 이므로, 게이트를
    # 통과하면서 evidence 로 정확한 span 을 남기는 RAGAS 합성 경로를 쓴다.
    @patch("agents.eval.probe_gen._llm_synthesize_query")
    def test_generated_spans_drive_multiple_chunk_candidates(self, synthesize):
        evidence = "정답" * 100          # span 길이 200 → chunk_size 800 보다 작아야 축소 후보가 나온다
        documents = [
            Document(f"d{i}", "memory", "txt", ("머리" * 50) + evidence + ("꼬리" * 600))
            for i in range(3)
        ]
        chunks = [
            Chunk(
                f"d{i}_chunk_000",
                f"d{i}",
                document.content[:800],
                char_span=(0, 800),
                metadata={"chunk_index": 0},
            )
            for i, document in enumerate(documents)
        ]
        # 멀티홉 Probe 는 source 를 2개 받으므로 양쪽에 인용을 준다 — 한쪽만 주면 나머지
        # source 의 span 이 청크 전체(800)로 폴백해 p85 를 끌어올리고, 축소 후보가 사라진다.
        synthesize.return_value = _SynthesizedProbe(
            question="정답 근거는 무엇인가요?",
            ground_truth="문서에 제시된 근거입니다.",
            evidence=[
                {"source_index": 0, "quote": evidence},
                {"source_index": 1, "quote": evidence},
            ],
        )
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
                    # 단일 문서 통합 fixture이므로 작은 표본을 명시적으로 허용한다
                    # (멀티홉 Probe 는 두 source 의 span 을 하나로 묶어 세므로 표본이 준다).
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

        # evidence 인용이 원문에 정확히 앵커돼 span 길이가 인용 길이(200)로 잡힌다.
        # generate_probes 는 ragas/datamorgana 를 섞어 만들므로 개수를 고정하지 않고,
        # "인용 길이 span 이 다수이고 그게 후보 산정을 이끈다" 는 성질을 본다.
        span_lengths = [
            span["end"] - span["start"]
            for probe in state.probes
            for span in probe.gold_spans
        ]
        self.assertGreaterEqual(len(span_lengths), 3)
        self.assertEqual(min(span_lengths), len(evidence))
        self.assertTrue(all(probe.gold_spans for probe in state.probes))
        self.assertEqual(request.optimizer, "internal")
        # span 200 → target 250(=200×1.2 를 50 단위 올림), 현재 800 에서 그 사이를
        # path_fractions 로 끊은 값. span 길이가 바뀌면 이 값도 같이 움직인다.
        self.assertEqual(
            request.search_space["chunker.chunk_size"],
            [600, 450, 250],
        )
        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "grounded",
        )


if __name__ == "__main__":
    unittest.main()
