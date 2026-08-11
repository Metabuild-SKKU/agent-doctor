"""
tests/test_anthropic_batch.py
Message Batches 수집기(core/llm_clients._AnthropicBatchCollector)의 계약 고정.

실제 API 는 부르지 않는다(가짜 client). 못 박는 것:
  1. 동시 다발 submit 이 배치 **하나**로 묶인다 — 쪼개지면 배치당 대기가 곱해진다.
  2. 결과가 custom_id 로 제 주인에게 돌아간다(응답 순서와 무관).
  3. errored 항목은 그 Future 만 예외를 받고 나머지는 산다.
  4. 스위치(EVAL_ANTHROPIC_BATCH)와 fan-out(judge_fanout) 게이트.
"""
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import llm_clients
from agents.eval import llm_provider

# 폴링·수집 창을 테스트용으로 줄인다 — 기본값이면 테스트가 15초씩 잔다.
_FAST_ENV = {
    "EVAL_ANTHROPIC_BATCH_WINDOW": "0.05",
    "EVAL_ANTHROPIC_BATCH_POLL": "0.01",
    "EVAL_ANTHROPIC_BATCH_TIMEOUT": "10",
}


def _message(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=None, stop_reason="end_turn")


class _FakeBatches:
    """messages.batches 흉내. create 호출 목록과 결과 시나리오를 들고 있다."""

    def __init__(self, results_by_id):
        self.created: list[list[dict]] = []
        self._results_by_id = results_by_id

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"batch-{len(self.created)}",
                               processing_status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended",
                               request_counts=SimpleNamespace(succeeded=0))

    def results(self, batch_id):
        # 일부러 역순으로 준다 — custom_id 매칭이 순서에 기대면 여기서 무너진다.
        for cid, result in reversed(list(self._results_by_id.items())):
            yield SimpleNamespace(custom_id=cid, result=result)


def _fake_client(results_by_id):
    batches = _FakeBatches(results_by_id)
    return SimpleNamespace(messages=SimpleNamespace(batches=batches)), batches


class BatchCollectorTest(unittest.TestCase):
    def setUp(self):
        self._patch = patch.dict(os.environ, _FAST_ENV)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def _submit_concurrently(self, collector, client, n):
        futures = [None] * n
        barrier = threading.Barrier(n)

        def worker(i):
            barrier.wait()
            futures[i] = collector.submit(client, {"model": "m", "n": i})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return futures

    def test_concurrent_submits_form_one_batch(self):
        n = 8
        results = {f"req-{i}": SimpleNamespace(type="succeeded", message=_message(f"r{i}"))
                   for i in range(n)}
        client, batches = _fake_client(results)
        collector = llm_clients._AnthropicBatchCollector()
        futures = self._submit_concurrently(collector, client, n)
        for fut in futures:
            fut.result(timeout=10)                                  # 전원 정상 수신
        self.assertEqual(len(batches.created), 1)                   # 배치 하나로 묶임
        self.assertEqual(len(batches.created[0]), n)

    def test_results_return_to_their_owner_by_custom_id(self):
        n = 4
        results = {f"req-{i}": SimpleNamespace(type="succeeded", message=_message(f"r{i}"))
                   for i in range(n)}
        client, _ = _fake_client(results)
        collector = llm_clients._AnthropicBatchCollector()
        futures = self._submit_concurrently(collector, client, n)
        # submit(i) 의 custom_id 는 도착 순서라 i 와 다를 수 있다 — params.n 으로 주인을
        # 역추적하지 않고, 전체 집합이 정확히 한 번씩 돌아왔는지로 계약을 잡는다.
        texts = sorted(f.result(timeout=10).content[0].text for f in futures)
        self.assertEqual(texts, [f"r{i}" for i in range(n)])

    def test_errored_item_fails_alone(self):
        results = {
            "req-0": SimpleNamespace(type="succeeded", message=_message("ok")),
            "req-1": SimpleNamespace(type="errored",
                                     error=SimpleNamespace(message="boom")),
        }
        client, _ = _fake_client(results)
        collector = llm_clients._AnthropicBatchCollector()
        fut0 = collector.submit(client, {"model": "m"})
        fut1 = collector.submit(client, {"model": "m"})
        # 어느 future 가 req-0 인지는 도착 순서에 달렸다 — 성공 1·실패 1 이면 계약 충족.
        outcomes = []
        for fut in (fut0, fut1):
            try:
                outcomes.append(("ok", fut.result(timeout=10)))
            except RuntimeError as exc:
                outcomes.append(("error", str(exc)))
        kinds = sorted(kind for kind, _ in outcomes)
        self.assertEqual(kinds, ["error", "ok"])
        err = next(msg for kind, msg in outcomes if kind == "error")
        self.assertIn("errored", err)

    def test_missing_custom_id_raises(self):
        client, _ = _fake_client({})                                # 결과가 하나도 없음
        collector = llm_clients._AnthropicBatchCollector()
        fut = collector.submit(client, {"model": "m"})
        with self.assertRaises(RuntimeError):
            fut.result(timeout=10)


class BatchGateTest(unittest.TestCase):
    def test_switch_defaults_off(self):
        with patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH": ""}):
            self.assertFalse(llm_clients.anthropic_batch_enabled())
        with patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH": "1"}):
            self.assertTrue(llm_clients.anthropic_batch_enabled())

    def test_judge_fanout_widens_only_for_anthropic_batch(self):
        cases = [
            # (provider, batch, expected)
            ("anthropic", "1", 30),      # 배치 모드 — record 수 전체
            ("anthropic", "0", 10),      # 배치 꺼짐 — 기존 동시성
            ("openrouter", "1", 10),     # 다른 provider — 배치 스위치 무관
        ]
        for provider, batch, expected in cases:
            with self.subTest(provider=provider, batch=batch), \
                    patch.dict(os.environ, {"EVAL_LLM_PROVIDER": provider,
                                            "EVAL_ANTHROPIC_BATCH": batch}):
                self.assertEqual(llm_provider.judge_fanout(30, 10), expected)


if __name__ == "__main__":
    unittest.main()
