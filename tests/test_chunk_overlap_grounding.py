import unittest

from agents.eval import diagnose, metrics_common
from agents.eval.types import EvalRecord, Mode
from agents.optimize import optimizer, planner
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
        metrics_common.set_context()
        metrics_common.set_mode(Mode.FAST)

    def test_split_span_does_not_override_confirmed_retrieval_cause(self):
        chunks = _fixed_chunks("d1", 1000, 400, 50)
        metrics_common.set_context(chunks=chunks)
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

        self.assertEqual(findings[0].label, "retrieval_incomplete_enumeration")
        self.assertTrue(findings[0].confirmed)
        self.assertEqual(findings[0].metadata["group"], "A")

    def test_chunk_fallback_span_is_not_used_for_boundary_diagnosis(self):
        chunks = _fixed_chunks("d1", 1000, 400, 50)
        metrics_common.set_context(chunks=chunks)
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

    def test_recall_success_but_split_context_is_preliminary_in_fast_mode(self):
        chunks = _fixed_chunks("d1", 1000, 400, 0)
        metrics_common.set_context(chunks=chunks)
        probe = Probe(
            probe_id="p1",
            question="정답은?",
            source="taxonomy",
            answer_exists=True,
            ground_truth="정답",
            gold_chunk_ids=[chunks[0].chunk_id, chunks[1].chunk_id],
            gold_spans=[{"doc_id": "d1", "start": 350, "end": 450}],
            metadata={"span_grounding": {"status": "exact"}},
        )
        record = EvalRecord(
            probe=probe,
            retrieved_chunk_ids=[chunks[0].chunk_id, chunks[1].chunk_id],
            retrieved_context=[chunks[0].text, chunks[1].text],
            # 오답은 F1_PASS_THRESHOLD 아래로 확실히 떨어지는 문자열을 써야 한다.
            # 예전 fixture 였던 "오답"은 gold "정답"과 '답' 한 글자를 공유해 char-F1 이
            # 정확히 0.5 = 문턱과 동률이 되고, `>=` 비교라 '정답'으로 통과해 버린다
            # (probe 가 성공 판정 → diagnose 가 게이트에서 [] 반환 → 이 테스트가 잡으려는
            #  경계 분할 진단까지 못 감).
            # FAST 에는 이 근접 오답을 걸러낼 수단이 없다 — 의미 판정은 tier3(RAGAS
            # answer_correctness) 몫이다. 문턱을 올려 막지 않는 이유는 _f1_ok 이
            # 'lexical 미달이면 RAGAS 를 보지도 않고 즉시 실패' 구조라, 문턱을 올리면
            # 생기는 오탈락(긴 gold 재진술 등)은 DEEP 에서도 복구가 불가능하기 때문이다.
            # (관대한 문턱의 오통과는 DEEP 강등으로 회복되지만, 그 반대는 회복되지 않는다.)
            generated_answer="전혀 다른 소리",
            oracle_answer="정답",
        )

        findings = diagnose.diagnose(record, Mode.FAST)
        finding = next(
            item for item in findings if item.label == "chunking_context_mismatch"
        )

        self.assertEqual(record.recall_at_k, 1.0)
        self.assertFalse(finding.confirmed)

    # tier4(파이프라인 재실행) 제거로 boundary-merge ablation 확정 경로는 없어졌다.
    # recall 성공 + 경계 분할 케이스는 항상 예비이며, optimize 가 청킹 파라미터를
    # 바꿔 재실행하며 검증한다(위 test_recall_success_..._preliminary 로 커버).


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

    def test_unrecoverable_overlap_moves_to_chunk_size_candidate(self):
        document = Document("d1", "memory", "txt", "가" * 1600)
        chunks = _fixed_chunks("d1", len(document.content), 400, 50)
        probe = Probe(
            probe_id="p1",
            question="질문",
            source="taxonomy",
            answer_exists=True,
            gold_spans=[{"doc_id": "d1", "start": 100, "end": 900}],
            metadata={"span_grounding": {"status": "exact"}},
        )
        finding = Finding(
            finding_id="p1:chunking_context_mismatch",
            type="retrieval_failure",
            severity="warning",
            description="안전한 overlap 범위로 복구할 수 없는 긴 정답",
            label="chunking_context_mismatch",
            affected_probes=["p1"],
        )
        state = AgentDoctorState(
            documents=[document],
            chunks=chunks,
            probes=[probe],
            report=DiagnosticReport(
                report_id="r1",
                findings=[finding],
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

        request, decision = planner.plan(state)
        result = optimizer.run(request)

        self.assertEqual(decision.mode, "apply_optimize")
        self.assertEqual(
            request.metadata["candidate_grounding"]["status"],
            "no_recoverable_crossings",
        )
        self.assertEqual(result.selected_candidate.id, "increase_chunk_size")
        self.assertEqual(result.config_patch.changes, {"chunker.chunk_size": 800})


if __name__ == "__main__":
    unittest.main()
