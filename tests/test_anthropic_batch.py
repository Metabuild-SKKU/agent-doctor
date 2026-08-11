"""
tests/test_anthropic_batch.py
Message Batches 수집기(core/llm_clients._AnthropicBatchCollector)의 계약 고정.

실제 API 는 부르지 않는다(가짜 client). 못 박는 것:
  1. 동시 다발 submit 이 배치 **하나**로 묶인다 — 쪼개지면 배치당 대기가 곱해진다.
  2. 결과가 custom_id 로 제 주인에게 돌아간다(응답 순서와 무관).
  3. errored 항목은 그 Future 만 예외를 받고 나머지는 산다.
  4. 스위치(EVAL_ANTHROPIC_BATCH)와 fan-out(judge_fanout) 게이트.
  5. 수집기 스레드가 죽어도 호출부가 **hang 하지 않는다** — 무인 실행에서 가장 나쁜
     실패 모드라, 잘못된 env 값·스레드 내부 예외 모두 예외로 깨어나야 한다.
  6. 배치 경로는 호출부가 use_batch 로 켠 호출에만 걸린다 — env 하나가 Eval 의 anthropic
     호출 전체를 비동기로 바꾸지 않는다.
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


class _EchoBatches(_FakeBatches):
    """제출된 custom_id 를 그대로 성공 처리해 돌려준다 — 수집기 내부 seq 에 안 묶인다."""

    def __init__(self):
        super().__init__({})

    def results(self, batch_id):
        for req in self.created[-1]:
            yield SimpleNamespace(custom_id=req["custom_id"],
                                  result=SimpleNamespace(type="succeeded",
                                                         message=_message("batched")))


def _fake_client(results_by_id):
    batches = _FakeBatches(results_by_id)
    return SimpleNamespace(messages=SimpleNamespace(batches=batches)), batches


class _FakeMessages:
    """messages.create(동기 경로) 호출 수를 센다. batches 는 배치 경로용."""

    def __init__(self, batches):
        self.batches = batches
        self.direct_calls = 0

    def create(self, **kwargs):
        self.direct_calls += 1
        return _message("direct")


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

    def test_collector_serves_a_second_wave_after_going_idle(self):
        """수집기가 한 번 놀다 스레드를 접은 뒤에도 다음 요청을 받아야 한다.

        STEP3 는 실제 트랙 팬아웃과 오라클 트랙 팬아웃이 시차를 두고 온다 — 두 번째
        물결이 죽은 스레드에 걸리면 그 배치 전체가 안 깨어난다."""
        client, batches = _fake_client({})
        client.messages.batches = _EchoBatches()
        collector = llm_clients._AnthropicBatchCollector()
        for wave in range(2):
            with self.subTest(wave=wave):
                fut = collector.submit(client, {"model": "m"})
                self.assertEqual(fut.result(timeout=10).content[0].text, "batched")
        self.assertEqual(len(client.messages.batches.created), 2)

    def test_missing_custom_id_raises(self):
        client, _ = _fake_client({})                                # 결과가 하나도 없음
        collector = llm_clients._AnthropicBatchCollector()
        fut = collector.submit(client, {"model": "m"})
        with self.assertRaises(RuntimeError):
            fut.result(timeout=10)


class BatchThreadFailureTest(unittest.TestCase):
    """수집기 스레드가 접히는 모든 경로에서 호출부가 예외로 깨어나야 한다.

    무인 실행에서 hang 은 실패보다 나쁘다 — 로그가 멈춘 채 몇 시간이 흐르고,
    무엇이 잘못됐는지 알려주는 것이 아무것도 남지 않는다."""

    def setUp(self):
        self._patch = patch.dict(os.environ, _FAST_ENV)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_thread_internal_error_wakes_every_future(self):
        class _Exploding(llm_clients._AnthropicBatchCollector):
            def _dispatch(self, items):
                raise BaseException("수집기 내부 폭발")   # noqa: TRY002

        client, _ = _fake_client({})
        collector = _Exploding()
        futures = [collector.submit(client, {"model": "m"}) for _ in range(3)]
        for fut in futures:
            # 예외 종류는 계약이 아니다 — "영원히 잠들지 않는다"가 계약이다.
            with self.assertRaises(BaseException):
                fut.result(timeout=10)

    def test_bad_env_value_falls_back_instead_of_killing_thread(self):
        # 예전엔 float("2s") 가 백그라운드 스레드를 죽였고, 호출부는 Future.result() 에서
        # 영원히 기다렸다. .env.example 에 노출된 값이라 실제로 일어날 수 있는 오타다.
        results = {"req-0": SimpleNamespace(type="succeeded", message=_message("ok"))}
        client, batches = _fake_client(results)
        collector = llm_clients._AnthropicBatchCollector()
        # POLL 은 빠른 값을 유지한다 — 기본값(15s)으로 떨어뜨리면 이 테스트만 15초를 잔다.
        # 파싱 자체의 계약은 아래 _env_float 단위 검사가 잡는다.
        with patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH_WINDOW": "2s"}):
            fut = collector.submit(client, {"model": "m"})
            # WINDOW 가 기본 2.0s 로 떨어져도 끝난다 — 스레드가 죽지 않았다는 뜻.
            self.assertEqual(fut.result(timeout=60).content[0].text, "ok")
        self.assertEqual(len(batches.created), 1)

    def test_env_float_falls_back_on_garbage_and_nonpositive(self):
        for bad in ("빠르게", "15s", "", "0", "-1"):
            with self.subTest(value=bad), \
                    patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH_POLL": bad}):
                self.assertEqual(llm_clients._env_float("EVAL_ANTHROPIC_BATCH_POLL", 15.0), 15.0)
        with patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH_POLL": "3"}):
            self.assertEqual(llm_clients._env_float("EVAL_ANTHROPIC_BATCH_POLL", 15.0), 3.0)

    def test_caller_wait_timeout_exceeds_collector_timeout(self):
        # 호출부 대기 상한이 수집기 상한보다 짧으면, 정상 대기를 오탐으로 끊는다.
        with patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH_TIMEOUT": "100"}):
            self.assertGreater(llm_clients._batch_wait_timeout(), 100.0)


class BatchScopeTest(unittest.TestCase):
    """배치 경로는 호출부가 켠 호출에만 걸린다(EVAL_ANTHROPIC_BATCH 는 허용 스위치일 뿐).

    env 만으로 갈리면 EVAL_LLM_PROVIDER=anthropic 인 실행에서 probe 합성 같은 순차 호출까지
    배치 수집 창과 폴링을 받게 된다 — 비용은 절반이지만 실행 시간이 예상 밖으로 늘어난다."""

    def setUp(self):
        self._patch = patch.dict(os.environ,
                                 {**_FAST_ENV, "EVAL_ANTHROPIC_BATCH": "1"})
        self._patch.start()
        self._collector = llm_clients._AnthropicBatchCollector()
        self._collector_patch = patch.object(llm_clients, "_batch_collector", self._collector)
        self._collector_patch.start()

    def tearDown(self):
        self._collector_patch.stop()
        self._patch.stop()

    def _run_chat(self, **kwargs) -> tuple[str, _FakeMessages, _EchoBatches]:
        batches = _EchoBatches()
        messages = _FakeMessages(batches)
        client = SimpleNamespace(messages=messages)
        fake_sdk = SimpleNamespace(Anthropic=lambda **kw: client)
        with patch.dict(sys.modules, {"anthropic": fake_sdk}):
            text = llm_clients.anthropic_chat(
                "system 지시문", "user 입력", "claude-haiku-4-5", **kwargs)
        return text, messages, batches

    def test_default_call_stays_synchronous_even_with_env_on(self):
        text, messages, batches = self._run_chat()
        self.assertEqual(text, "direct")
        self.assertEqual(messages.direct_calls, 1)
        self.assertEqual(batches.created, [])        # 배치 제출 없음

    def test_opted_in_call_goes_through_batch(self):
        text, messages, batches = self._run_chat(use_batch=True)
        self.assertEqual(text, "batched")
        self.assertEqual(messages.direct_calls, 0)
        self.assertEqual(len(batches.created), 1)

    def test_opt_in_without_env_switch_stays_synchronous(self):
        with patch.dict(os.environ, {"EVAL_ANTHROPIC_BATCH": "0"}):
            text, messages, batches = self._run_chat(use_batch=True)
        self.assertEqual(text, "direct")
        self.assertEqual(messages.direct_calls, 1)
        self.assertEqual(batches.created, [])

    def test_summary_counts_submitted_batches(self):
        self._run_chat(use_batch=True)
        stats = self._collector.stats()
        self.assertEqual(stats["batches"], 1)
        self.assertEqual(stats["items"], 1)


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

    def test_fanout_respects_max_cap(self):
        """fan-out 은 곧 로컬 worker 스레드 수 — 상한 없이 record 수를 그대로 열면
        150 QA 실행에서 스레드가 그만큼 생긴다. 기본값(256)은 그 실행을 한 배치로 담는다."""
        env = {"EVAL_LLM_PROVIDER": "anthropic", "EVAL_ANTHROPIC_BATCH": "1"}
        with patch.dict(os.environ, {**env, "EVAL_ANTHROPIC_BATCH_MAX_FANOUT": "50"}):
            self.assertEqual(llm_provider.judge_fanout(300, 10), 50)
            self.assertEqual(llm_provider.judge_fanout(30, 10), 30)   # 상한 미만은 그대로
        with patch.dict(os.environ, {**env, "EVAL_ANTHROPIC_BATCH_MAX_FANOUT": ""}):
            self.assertEqual(llm_provider.judge_fanout(150, 10), 150)  # 권장 실행은 한 배치
            self.assertEqual(llm_provider.judge_fanout(9999, 10),
                             llm_provider._BATCH_MAX_FANOUT_DEFAULT)
        # 잘못된 값은 기본값으로 — 배치가 1건씩 쪼개지는 사고를 막는다.
        with patch.dict(os.environ, {**env, "EVAL_ANTHROPIC_BATCH_MAX_FANOUT": "많이"}):
            self.assertEqual(llm_provider.judge_fanout(9999, 10),
                             llm_provider._BATCH_MAX_FANOUT_DEFAULT)


if __name__ == "__main__":
    unittest.main()


class BatchSummaryModelTest(unittest.TestCase):
    """요약 줄의 모델 표기(리뷰 제안) — 모델이 갈리면 배치 대기·비용을 실행 간 직접
    비교하면 안 되므로, 요약만 보고 비교하는 사고를 막는 표식이다."""

    def setUp(self):
        self._patch = patch.dict(os.environ, _FAST_ENV)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_stats_collects_models_and_summary_prints_them(self):
        import io
        from contextlib import redirect_stdout

        results = {"req-0": SimpleNamespace(type="succeeded", message=_message("ok"))}
        client, _ = _fake_client(results)
        collector = llm_clients._AnthropicBatchCollector()
        collector.submit(client, {"model": "claude-haiku-4-5"}).result(timeout=10)
        self.assertEqual(collector.stats()["models"], ["claude-haiku-4-5"])

        buf = io.StringIO()
        with patch.object(llm_clients, "_batch_collector", collector), \
                redirect_stdout(buf):
            llm_clients.print_batch_summary()
        self.assertIn("claude-haiku-4-5", buf.getvalue())
