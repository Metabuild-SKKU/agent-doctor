"""
tests/test_bad_gold_loop.py
D그룹 bad_gold_answer 처리 루프 검증.

Phase 2: 정답셋 오류(bad_gold_answer)로 판정된 probe 는 '거짓 실패'이므로 점수 집계
(composite/overall)에서 제외하되, 진단·리포트(findings)에는 남긴다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch
from core.schema import Probe, Finding, Chunk, Document, EvalSnapshot
from core.state import AgentDoctorState
from agents.eval.types import EvalRecord
from agents.eval.report import build_report, _is_bad_gold_probe
import agents.eval.probe_gen as pg


def _record(pid: str, f1: float, bad_gold: bool = False) -> EvalRecord:
    probe = Probe(probe_id=pid, question="q", source="taxonomy", ground_truth="gt")
    rec = EvalRecord(probe=probe)
    rec.f1_score = f1
    rec.recall_at_k = 1.0
    if bad_gold:
        rec.findings = [Finding(
            finding_id=pid, type="gap", severity="warning", description="bad",
            label="bad_gold_answer", confirmed=True, affected_probes=[pid],
        )]
    return rec


class BadGoldScoreExclusionTest(unittest.TestCase):
    def test_bad_gold_excluded_from_composite(self):
        good = [_record(f"g{i}", 0.9) for i in range(3)]
        bad = [_record("b1", 0.0, bad_gold=True)]

        only_good = build_report(good, 0, mode=1).composite_score["total"]
        with_bad = build_report(good + bad, 0, mode=1).composite_score["total"]
        # 거짓 실패(bad_gold)는 점수에서 빠지므로 두 종합점수가 같아야 한다.
        self.assertEqual(only_good, with_bad)

    def test_bad_gold_still_reported_in_findings(self):
        report = build_report([_record("g1", 0.9), _record("b1", 0.0, bad_gold=True)], 0, mode=1)
        labels = {f.label for f in report.findings}
        self.assertIn("bad_gold_answer", labels)  # 점수엔 빠져도 진단엔 남는다

    def test_only_confirmed_bad_gold_is_excluded(self):
        probe = Probe(probe_id="p", question="q", source="taxonomy", ground_truth="gt")
        rec = EvalRecord(probe=probe)
        rec.findings = [Finding(finding_id="p", type="gap", severity="warning", description="d",
                                label="bad_gold_answer", confirmed=False, affected_probes=["p"])]
        self.assertFalse(_is_bad_gold_probe(rec))  # 예비면 제외 대상 아님


class BadGoldRegenerationTest(unittest.TestCase):
    """Phase 3: bad_gold 로 판정된 우리(llm_generated) probe 를 같은 근거 청크에서 재생성.
    user_log·멀티홉은 제외, 재생성 1회 가드."""

    def _state(self):
        txt = "재택근무는 주 2일까지 가능합니다. 승인 절차는 팀장 결재."
        st = AgentDoctorState()
        st.chunks = [Chunk(chunk_id="c1", doc_id="d1", text=txt)]
        st.documents = [Document(doc_id="d1", source="s", format="md", content=txt)]
        return st

    def test_regenerates_our_probe_and_excludes_user_log(self):
        st = self._state()
        ours = Probe(probe_id="probe_gen_000", question="구질문?", source="llm_generated",
                     ground_truth="틀린", gold_chunk_ids=["c1"], metadata={})
        user = Probe(probe_id="u1", question="q", source="user_log",
                     ground_truth="x", gold_chunk_ids=["c1"])
        with patch.object(pg, "_llm_generate_single_hop", return_value=("재택 며칠?", "주 2일")), \
             patch.object(pg, "probe_quality_issue", return_value=None):
            replaced = pg.regenerate_probes([ours, user], st)
        self.assertEqual(set(replaced), {"probe_gen_000"})  # user_log 제외
        self.assertEqual(replaced["probe_gen_000"].ground_truth, "주 2일")
        self.assertEqual(replaced["probe_gen_000"].metadata["regenerated"], 1)

    def test_regeneration_guard_stops_after_one(self):
        st = self._state()
        already = Probe(probe_id="probe_gen_000", question="q", source="llm_generated",
                        ground_truth="틀린", gold_chunk_ids=["c1"], metadata={"regenerated": 1})
        with patch.object(pg, "_llm_generate_single_hop", return_value=("q?", "a")):
            self.assertEqual(pg.regenerate_probes([already], st), {})

    def test_agent_persists_regenerated_probes(self):
        from agents.eval import agent as eval_agent
        st = self._state()
        probe = Probe(probe_id="probe_gen_000", question="구질문?", source="llm_generated",
                      ground_truth="틀린", gold_chunk_ids=["c1"], metadata={})
        rec = EvalRecord(probe=probe)
        rec.findings = [Finding(finding_id="probe_gen_000", type="gap", severity="warning",
                                description="bad", label="bad_gold_answer", confirmed=True,
                                affected_probes=["probe_gen_000"])]
        with patch.object(pg, "_llm_generate_single_hop", return_value=("재택 며칠?", "주 2일")), \
             patch.object(pg, "probe_quality_issue", return_value=None), \
             patch.object(eval_agent, "save_probes") as save:
            did = eval_agent._maybe_regenerate_bad_gold(st, [rec], [probe], "v1")
        self.assertTrue(did)                 # 재생성·저장 성공 → True(STEP5 가 snapshot 생략)
        save.assert_called_once()
        saved_probes = save.call_args[0][0]
        self.assertEqual(saved_probes[0].ground_truth, "주 2일")  # 교체본 저장

    def _bad_gold_record(self, probe):
        rec = EvalRecord(probe=probe)
        rec.findings = [Finding(finding_id=probe.probe_id, type="gap", severity="warning",
                                description="bad", label="bad_gold_answer", confirmed=True,
                                affected_probes=[probe.probe_id])]
        return rec

    def test_regeneration_skips_snapshot_so_next_eval_cache_misses(self):
        # blocker(루프 미완결): 재생성이 일어난 실행은 stale snapshot 을 남기면 안 된다.
        # 재생성 후 (STEP5 가 저장을 생략하므로) 그 cache key 로 조회하면 None → 다음 Eval 이
        # cache hit 없이 새 probe 를 재평가한다. 여기서는 재생성이 True 를 돌려주고 기존
        # 스냅샷을 무효화해, 이후 조회가 miss 임을 확인한다.
        from agents.eval import agent as eval_agent
        st = self._state()
        probe = Probe(probe_id="probe_gen_000", question="구질문?", source="llm_generated",
                      ground_truth="틀린", gold_chunk_ids=["c1"], metadata={})
        key = "eval-key-1"
        st.eval_cache = [EvalSnapshot(cache_key=key, index_key="i", probes=[probe],
                                      report=None, diagnosis_cache={}, diagnosis_cache_version="v")]
        with patch.object(pg, "_llm_generate_single_hop", return_value=("재택 며칠?", "주 2일")), \
             patch.object(pg, "probe_quality_issue", return_value=None), \
             patch.object(eval_agent, "save_probes"):
            did = eval_agent._maybe_regenerate_bad_gold(st, [self._bad_gold_record(probe)], [probe], "v1")
        self.assertTrue(did)
        # 기존 스냅샷은 무효화됐고, STEP5 는 재생성 실행에서 새 snapshot 을 저장하지 않으므로
        # 다음 Eval 의 조회는 miss 다(옛 성적표 복원 안 됨).
        self.assertIsNone(eval_agent._find_eval_snapshot(st, key))

    def test_regeneration_save_failure_keeps_caches_and_returns_false(self):
        # 저장 실패 시엔 재생성이 반영 안 됐으므로 무효화·snapshot skip 을 하면 안 된다.
        from agents.eval import agent as eval_agent
        st = self._state()
        probe = Probe(probe_id="probe_gen_000", question="구질문?", source="llm_generated",
                      ground_truth="틀린", gold_chunk_ids=["c1"], metadata={})
        st.diagnosis_cache = {"probe_gen_000": {"sig": 1}}
        with patch.object(pg, "_llm_generate_single_hop", return_value=("재택 며칠?", "주 2일")), \
             patch.object(pg, "probe_quality_issue", return_value=None), \
             patch.object(eval_agent, "save_probes", side_effect=OSError("disk full")):
            did = eval_agent._maybe_regenerate_bad_gold(st, [self._bad_gold_record(probe)], [probe], "v1")
        self.assertFalse(did)                                 # 저장 실패 → False(STEP5 는 정상 저장)
        self.assertIn("probe_gen_000", st.diagnosis_cache)    # 무효화 안 함

    def test_regeneration_invalidates_stale_caches(self):
        # blocker#3 회귀: 재생성 probe 는 같은 probe_id·probe_version 을 유지하므로,
        # 그 id 의 진단 신호와 그 probe 를 담은 eval 스냅샷을 무효화해야 다음 평가가
        # 옛 신호/리포트를 재사용(재생성 무효화)하지 않는다.
        from agents.eval import agent as eval_agent
        st = self._state()
        probe = Probe(probe_id="probe_gen_000", question="구질문?", source="llm_generated",
                      ground_truth="틀린", gold_chunk_ids=["c1"], metadata={})
        other = Probe(probe_id="keep_me", question="q2", source="llm_generated",
                      ground_truth="ok", gold_chunk_ids=["c1"])
        st.diagnosis_cache = {"probe_gen_000": {"sig": 1}, "keep_me": {"sig": 2}}
        st.eval_cache = [
            EvalSnapshot(cache_key="k_with", index_key="i", probes=[probe, other],
                         report=None, diagnosis_cache={}, diagnosis_cache_version="v"),
            EvalSnapshot(cache_key="k_without", index_key="i", probes=[other],
                         report=None, diagnosis_cache={}, diagnosis_cache_version="v"),
        ]
        rec = EvalRecord(probe=probe)
        rec.findings = [Finding(finding_id="probe_gen_000", type="gap", severity="warning",
                                description="bad", label="bad_gold_answer", confirmed=True,
                                affected_probes=["probe_gen_000"])]
        with patch.object(pg, "_llm_generate_single_hop", return_value=("재택 며칠?", "주 2일")), \
             patch.object(pg, "probe_quality_issue", return_value=None), \
             patch.object(eval_agent, "save_probes"):
            eval_agent._maybe_regenerate_bad_gold(st, [rec], [probe, other], "v1")
        # 재생성 id 의 진단 신호는 제거, 무관한 id 는 유지
        self.assertNotIn("probe_gen_000", st.diagnosis_cache)
        self.assertIn("keep_me", st.diagnosis_cache)
        # 재생성 probe 를 담은 스냅샷만 제거, 나머지는 유지
        keys = {s.cache_key for s in st.eval_cache}
        self.assertEqual(keys, {"k_without"})


if __name__ == "__main__":
    unittest.main()
