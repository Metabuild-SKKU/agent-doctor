"""
tests/test_fused_refetch.py
fused RAGAS 응답에 필수 키가 빠졌을 때의 1회 재요청 경로.

json_object 모드는 유효한 JSON 만 보장하고 스키마는 강제하지 않아, 블록이 늘면 모델이
뒤쪽 키를 빠뜨린 채 JSON 을 닫는 일이 있다(실행 로그: 30 probe 중 recall_classifications
7회 누락). 개별 지표 보수보다 재요청 1회가 싸므로 그 앞에 둔다.
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import agents.eval.metrics_ragas as R


BLOCKS = ["answer_statements", "faithfulness_verdicts", "relevancy",
          "context_verdicts", "reference_statements", "correctness",
          "recall_classifications"]

# 재요청이 첫 호출과 같은 조건으로 나가는지 보려고 심는 표식.
CACHE_PREFIX = "CACHE-PREFIX"
JSON_SCHEMA = {"type": "object", "marker": "SCHEMA"}


def _full() -> dict:
    return {k: [] for k in R._fused_required_keys(BLOCKS)}


class FusedRequiredKeysTest(unittest.TestCase):
    def test_relevancy_expands_to_two_keys(self):
        # relevancy 블록 하나가 generated_questions·noncommittal 두 키를 만든다 —
        # _fused_prompt 의 required 규칙과 어긋나면 재요청 판정이 헛돈다.
        self.assertEqual(R._fused_required_keys(["relevancy"]),
                         ["generated_questions", "noncommittal"])

    def test_other_blocks_map_to_their_own_name(self):
        self.assertEqual(R._fused_required_keys(["correctness"]), ["correctness"])

    def test_matches_prompt_required_list(self):
        # 프롬프트가 요구하는 목록과 재요청이 검사하는 목록이 같아야 한다.
        prompt = R._fused_prompt(BLOCKS, "q", "a", ["c"], "r")
        for key in R._fused_required_keys(BLOCKS):
            self.assertIn(f'"{key}"', prompt)


class RefetchTest(unittest.TestCase):
    def _run(self, first: dict, retry: dict | None = None):
        calls = []

        def fake_chat(judge, prompt, max_output_tokens=None, label="", **kwargs):
            # 여기 오는 호출은 전부 재요청이다 — 첫 응답(first)은 인자로 이미 넘어온다.
            # **kwargs: _chat 이 provider 별 옵션(cache_prefix/json_schema)을 늘려도
            # 이 스텁이 TypeError 로 죽지 않게. 대신 받은 걸 그대로 모아
            # test_refetch_forwards_provider_options 가 전달 여부를 직접 본다.
            calls.append((prompt, kwargs))
            return {} if retry is None else retry

        buf = io.StringIO()
        with patch.object(R, "_chat", fake_chat), redirect_stdout(buf):
            out = R._refetch_fused_if_incomplete(None, first, "PROMPT", BLOCKS,
                                                 cache_prefix=CACHE_PREFIX,
                                                 json_schema=JSON_SCHEMA)
        return out, calls, buf.getvalue()

    def test_complete_response_does_not_refetch(self):
        out, calls, log = self._run(_full())
        self.assertEqual(len(calls), 0)
        self.assertEqual(log, "")

    def test_missing_key_triggers_one_refetch(self):
        partial = _full()
        del partial["recall_classifications"]
        out, calls, log = self._run(partial, _full())
        self.assertEqual(len(calls), 1)              # 1회만
        self.assertIn("recall_classifications", out)
        self.assertIn("1회 재요청", log)

    def test_refetch_forwards_provider_options(self):
        # 재요청은 첫 호출과 같은 조건으로 나가야 한다 — cache_prefix 가 빠지면 캐시
        # 접두가 갈려 프롬프트 캐시가 안 맞고, json_schema 가 빠지면 스키마 강제가 없는
        # 응답을 받아 애초에 이 경로를 부른 이유(키 누락)가 그대로 재현된다.
        # _chat 이 인자를 늘릴 때 스텁이 **kwargs 로 조용히 삼키므로, 전달 자체는
        # 여기서 못박아 둔다.
        partial = _full()
        del partial["recall_classifications"]
        _out, calls, _log = self._run(partial, _full())
        self.assertEqual(calls[0][1],
                         {"cache_prefix": CACHE_PREFIX, "json_schema": JSON_SCHEMA})

    def test_first_response_wins_on_overlap(self):
        # 재요청분은 빈 자리만 메운다 — 같은 트랙에서 두 응답이 섞이면 판정 근거가
        # 일관되지 않는다.
        partial = _full()
        partial["faithfulness_verdicts"] = ["FIRST"]
        del partial["recall_classifications"]
        retry = _full()
        retry["faithfulness_verdicts"] = ["RETRY"]
        retry["recall_classifications"] = ["FILLED"]
        out, _calls, _log = self._run(partial, retry)
        self.assertEqual(out["faithfulness_verdicts"], ["FIRST"])
        self.assertEqual(out["recall_classifications"], ["FILLED"])

    def test_empty_retry_keeps_original(self):
        partial = _full()
        del partial["correctness"]
        out, calls, _log = self._run(partial, {})
        self.assertEqual(len(calls), 1)
        self.assertNotIn("correctness", out)         # 보수 경로가 이어받는다

    def test_disabled_repair_skips_refetch(self):
        partial = _full()
        del partial["correctness"]
        with patch.object(R, "_fused_repair_enabled", lambda: False):
            out, calls, log = self._run(partial, _full())
        self.assertEqual(len(calls), 0)
        self.assertEqual(log, "")

    def test_no_blocks_is_a_noop(self):
        calls = []
        with patch.object(R, "_chat", lambda *a, **k: calls.append(1)):
            out = R._refetch_fused_if_incomplete(None, {}, "P", [])
        self.assertEqual(out, {})
        self.assertEqual(calls, [])


class PromptRequiredKeyNoticeTest(unittest.TestCase):
    """스키마의 required 는 프롬프트 중간에 묻히므로 끝에서 한 번 더 못박는다."""

    def test_prompt_ends_with_explicit_key_checklist(self):
        prompt = R._fused_prompt(BLOCKS, "q", "a", ["c"], "r")
        tail = prompt[-800:]
        self.assertIn("MUST contain ALL of these top-level keys", tail)
        self.assertIn("recall_classifications", tail)


if __name__ == "__main__":
    unittest.main()
