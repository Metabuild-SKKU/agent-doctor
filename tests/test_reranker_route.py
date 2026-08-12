"""리랭크 실행 경로(provider × device) — 로컬 CPU/GPU 와 OpenRouter /rerank.

임베딩 축과 **일부러 다르게** 만든 지점들이 여기서 고정된다.
  - 기본값이 반대다(임베딩 openrouter / 리랭크 local). 과금이 색인 1회가 아니라 질의마다다.
  - 경로를 바꾸면 모델도 바뀐다(로컬 bge-reranker-v2-m3 ↔ voyageai/rerank-2.5-lite). 그래서
    "어디서 계산했나" 가 실행 기록에 남아야 하고, 그 값이 리포트까지 도달해야 한다.
  - 키가 없을 때 임베딩은 예외로 끊지만 리랭크는 원순위를 유지한다(optional 이라서).
"""
from __future__ import annotations

import argparse
import os
import unittest
from unittest.mock import Mock, patch

from agents.eval.report import build_report
from agents.eval.types import EvalRecord
from agents.index import qdrant_store
from agents.rag.retriever import RetrievalSettings, Retriever
from core import embedding_cli, llm_clients
from core.schema import Probe


def _clear_reranker_state() -> None:
    qdrant_store._rerankers.clear()
    qdrant_store._failed_rerankers.clear()
    qdrant_store._reranker_max_lengths.clear()
    qdrant_store._reranker_routes.clear()
    qdrant_store._reranker_device_overrides.clear()
    qdrant_store._reranker_applied_devices.clear()
    qdrant_store._embed_routes_notified.clear()


class RerankerRouteResolutionTest(unittest.TestCase):
    def tearDown(self):
        _clear_reranker_state()

    def test_default_provider_is_local(self):
        """기본값이 local 이어야 한다. 임베딩(openrouter)을 따라가면 Eval 한 번이
        질문 수만큼 유료 검색이 되고, 그 사실이 어디에도 안 드러난다."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INDEX_RERANKER_PROVIDER", None)
            self.assertEqual(qdrant_store.resolve_reranker_provider(), "local")

    def test_provider_aliases_are_normalized(self):
        with patch.dict(os.environ, {"INDEX_RERANKER_PROVIDER": "Open-Router"}):
            self.assertEqual(qdrant_store.resolve_reranker_provider(), "openrouter")

    def test_device_inherits_embedding_axis(self):
        """--embed gpu 로 GPU 를 쓰겠다고 한 실행에서 리랭커만 CPU 에 남으면,
        정작 검색 시간의 대부분을 차지하는 쪽이 그대로 느리다."""
        with patch.dict(os.environ, {"INDEX_EMBED_DEVICE": "cpu"}):
            os.environ.pop("INDEX_RERANKER_DEVICE", None)
            self.assertEqual(qdrant_store.resolve_reranker_device(), "cpu")

    def test_reranker_device_overrides_embedding_axis(self):
        """VRAM 이 둘을 다 못 받칠 때 리랭커만 떼어 내릴 수 있어야 한다."""
        with patch.dict(
            os.environ,
            {"INDEX_EMBED_DEVICE": "cuda", "INDEX_RERANKER_DEVICE": "cpu"},
        ):
            self.assertEqual(qdrant_store.resolve_reranker_device(), "cpu")

    def test_gpu_is_accepted_as_cuda_alias(self):
        """`--rerank gpu` 가 CLI 어휘라 env 에 그대로 옮겨 적기 쉬운데, torch 는 "gpu" 를
        모른다 — 그대로 넘기면 로드가 죽고 리랭커가 300초 쿨다운으로 조용히 빠진다."""
        with patch.dict(os.environ, {"INDEX_RERANKER_DEVICE": "gpu"}):
            self.assertIn(
                qdrant_store._reranker_soft_device("test/alias"), {"cuda", "cpu"}
            )
            self.assertNotEqual(
                qdrant_store._reranker_soft_device("test/alias"), "gpu"
            )
        # 임베딩 축도 같은 어휘를 쓴다(`--embed gpu`).
        with patch.dict(os.environ, {"INDEX_EMBED_DEVICE": "gpu"}):
            self.assertNotEqual(qdrant_store.resolve_embedding_device(), "gpu")

    def test_openrouter_model_is_not_derived_from_local_name(self):
        """로컬 이름 소문자화(임베딩 규칙)를 쓰면 404 다 — 카탈로그에 bge 계열이 없다."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INDEX_RERANKER_MODEL_OPENROUTER", None)
            model = qdrant_store.openrouter_reranker_model(
                qdrant_store.DEFAULT_RERANKER_MODEL
            )
        self.assertEqual(model, "voyageai/rerank-2.5-lite")
        self.assertNotIn("bge", model)


class RerankerCliFlagTest(unittest.TestCase):
    @staticmethod
    def _parse(argv: list[str]):
        parser = argparse.ArgumentParser()
        embedding_cli.add_embedding_args(parser)
        return parser.parse_args(argv)

    def test_rerank_flag_sets_its_own_axis(self):
        with patch.dict(os.environ, {}, clear=False):
            applied = embedding_cli.apply_embedding_args(self._parse(["--rerank", "gpu"]))

        self.assertEqual(applied["INDEX_RERANKER_PROVIDER"], "local")
        self.assertEqual(applied["INDEX_RERANKER_DEVICE"], "cuda")
        # 임베딩 장치는 건드리지 않는다 — `--embed cpu --rerank gpu` 가 임베딩까지
        # GPU 로 끌고 가면, VRAM 이 한쪽만 감당할 때 쓰려던 조합이 무너진다.
        self.assertNotIn("INDEX_EMBED_DEVICE", applied)

    def test_rerank_openrouter_sets_provider_only(self):
        with patch.dict(os.environ, {}, clear=False):
            applied = embedding_cli.apply_embedding_args(
                self._parse(["--rerank", "openrouter"])
            )

        self.assertEqual(applied["INDEX_RERANKER_PROVIDER"], "openrouter")
        self.assertNotIn("INDEX_RERANKER_DEVICE", applied)

    def test_embed_flag_does_not_move_reranker(self):
        """두 축은 기본값이 반대라 --embed 가 리랭커를 끌고 가면 안 된다."""
        with patch.dict(os.environ, {}, clear=False):
            applied = embedding_cli.apply_embedding_args(
                self._parse(["--embed", "openrouter"])
            )

        self.assertEqual(applied["INDEX_EMBED_PROVIDER"], "openrouter")
        self.assertNotIn("INDEX_RERANKER_PROVIDER", applied)


class OpenRouterRerankTransportTest(unittest.TestCase):
    """core.llm_clients.openrouter_rerank — HTTP 규약만."""

    @staticmethod
    def _response(status_code=200, payload=None, text=""):
        resp = Mock()
        resp.status_code = status_code
        resp.text = text
        resp.json.return_value = payload or {}
        return resp

    def test_sends_cohere_style_body_and_returns_indexed_scores(self):
        payload = {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.5},
            ],
            "usage": {"cost": 0.002},
        }
        post = Mock(return_value=self._response(payload=payload))

        with patch("requests.post", post):
            scored = llm_clients.openrouter_rerank(
                "질문", ["a", "b", "c"], "voyageai/rerank-2.5-lite", api_key="key"
            )

        self.assertEqual(scored, [(2, 0.9), (0, 0.1), (1, 0.5)])
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "voyageai/rerank-2.5-lite")
        self.assertEqual(body["query"], "질문")
        self.assertEqual(body["documents"], ["a", "b", "c"])
        # top_n 은 보내지 않는다 — 보내면 나머지 후보의 점수가 응답에서 빠지는데,
        # 채점(과금) 대상 문서 수는 그대로라 줄여도 싸지지 않는다.
        self.assertNotIn("top_n", body)

    def test_reported_cost_is_logged(self):
        payload = {"results": [{"index": 0, "relevance_score": 0.4}], "usage": {"cost": 0.002}}

        with patch("requests.post", Mock(return_value=self._response(payload=payload))):
            with patch("core.llm_clients.log_usage") as log:
                llm_clients.openrouter_rerank("질문", ["a"], "m", api_key="key")

        self.assertEqual(log.call_args.kwargs["cost_usd"], 0.002)

    def test_missing_cost_is_unpriced_not_zero(self):
        """$0 으로 뭉개면 유료 호출이 비용표에서 조용히 사라진다."""
        payload = {"results": [{"index": 0, "relevance_score": 0.4}]}

        with patch("requests.post", Mock(return_value=self._response(payload=payload))):
            with patch("core.llm_clients.log_usage") as log:
                llm_clients.openrouter_rerank("질문", ["a"], "m", api_key="key")

        self.assertIsNone(log.call_args.kwargs["cost_usd"])

    def test_http_error_carries_status_code_for_retry_policy(self):
        """429/5xx 만 재시도해야 한다 — 문자열이 아니라 상태 코드로 갈리게 한다."""
        from core.llm_retry import is_rate_limit, is_transient

        with patch("requests.post", Mock(return_value=self._response(429, text="slow down"))):
            with self.assertRaises(RuntimeError) as caught:
                llm_clients.openrouter_rerank("질문", ["a"], "m", api_key="key")

        self.assertEqual(caught.exception.status_code, 429)
        self.assertTrue(is_rate_limit(caught.exception))
        self.assertTrue(is_transient(caught.exception))

    def test_auth_error_is_not_retried(self):
        from core.llm_retry import is_transient

        with patch("requests.post", Mock(return_value=self._response(401, text="no key"))):
            with self.assertRaises(RuntimeError) as caught:
                llm_clients.openrouter_rerank("질문", ["a"], "m", api_key="bad")

        self.assertFalse(is_transient(caught.exception))


class OpenRouterRerankerModelTest(unittest.TestCase):
    """qdrant_store 쪽 — CrossEncoder 자리에 끼워 넣은 API 리랭커."""

    def setUp(self):
        _clear_reranker_state()
        self.env = patch.dict(
            os.environ,
            {
                "INDEX_RERANKER_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        _clear_reranker_state()

    def test_scores_are_placed_back_by_input_index(self):
        """응답은 점수 내림차순이라 입력 순서가 아니다. 그대로 zip 하면 점수가
        엉뚱한 청크에 붙어 검색이 조용히 망가진다."""
        transport = Mock(return_value=[(1, 0.9), (0, 0.2)])
        results = [
            {"chunk_id": "c1", "text": "첫째", "score": 0.5},
            {"chunk_id": "c2", "text": "둘째", "score": 0.4},
        ]

        with patch.object(qdrant_store, "openrouter_rerank", transport):
            reranked, status, _seconds = qdrant_store.rerank_with_status(
                "질문", results, top_k=2
            )

        self.assertEqual(status, "applied")
        self.assertEqual([item["chunk_id"] for item in reranked], ["c2", "c1"])
        self.assertEqual(reranked[0]["score"], 0.9)
        # 원 검색 점수도 보존된다(기존 계약).
        self.assertEqual(reranked[0]["retrieval_score"], 0.4)

    def test_route_is_recorded_for_the_report(self):
        transport = Mock(return_value=[(0, 0.3)])
        results = [{"chunk_id": "c1", "text": "본문", "score": 0.5}]

        with patch.object(qdrant_store, "openrouter_rerank", transport):
            qdrant_store.rerank_with_status("질문", results, top_k=1)

        self.assertEqual(
            qdrant_store.reranker_route(qdrant_store.DEFAULT_RERANKER_MODEL),
            "openrouter:voyageai/rerank-2.5-lite",
        )
        # 입력 길이 상한은 로컬 전용 장치다 — API 경로는 uncapped 로 남아야 한다.
        self.assertIsNone(
            qdrant_store.reranker_max_length(qdrant_store.DEFAULT_RERANKER_MODEL)
        )

    def test_short_response_fails_instead_of_scoring_zero(self):
        """점수가 빠진 후보를 0점으로 깔면 조용히 강등된다. 원순위 유지가 낫다."""
        transport = Mock(return_value=[(0, 0.9)])          # 문서 2건인데 점수 1건
        results = [
            {"chunk_id": "c1", "text": "첫째", "score": 0.5},
            {"chunk_id": "c2", "text": "둘째", "score": 0.4},
        ]

        with patch.object(qdrant_store, "openrouter_rerank", transport):
            reranked, status, seconds = qdrant_store.rerank_with_status(
                "질문", results, top_k=2
            )

        self.assertEqual(status, "inference_failed")
        self.assertEqual([item["chunk_id"] for item in reranked], ["c1", "c2"])
        self.assertEqual(seconds, 0.0)

    def test_empty_document_keeps_its_slot(self):
        """빈 문서는 API 가 400 으로 거절한다. 빼 버리면 인덱스가 밀린다."""
        transport = Mock(return_value=[(0, 0.1), (1, 0.2)])
        results = [
            {"chunk_id": "c1", "text": "", "score": 0.5},
            {"chunk_id": "c2", "text": "본문", "score": 0.4},
        ]

        with patch.object(qdrant_store, "openrouter_rerank", transport):
            qdrant_store.rerank_with_status("질문", results, top_k=2)

        sent_documents = transport.call_args.args[1]
        self.assertEqual(len(sent_documents), 2)
        self.assertTrue(sent_documents[0].strip() == "")
        self.assertNotEqual(sent_documents[0], "")

    def test_missing_api_key_keeps_original_order(self):
        """임베딩은 예외로 끊지만 리랭커는 optional 이다 — 여기서 던지면 키 오타
        하나로 모든 검색이 죽는다."""
        del os.environ["OPENROUTER_API_KEY"]
        results = [{"chunk_id": "c1", "text": "본문", "score": 0.5}]

        reranked, status, _seconds = qdrant_store.rerank_with_status(
            "질문", results, top_k=1
        )

        self.assertEqual(status, "api_key_missing")
        self.assertEqual([item["chunk_id"] for item in reranked], ["c1"])

    def test_missing_api_key_capability_is_not_retryable(self):
        """설정을 고쳐야 풀리는 실패다. retryable 로 두면 Optimize 가 매 회차
        같은 처방을 다시 꺼내 든다."""
        del os.environ["OPENROUTER_API_KEY"]

        capability = qdrant_store.probe_reranker_capability()

        self.assertEqual(capability["status"], "unavailable")
        self.assertEqual(capability["reason"], "api_key_missing")
        self.assertFalse(capability["retryable"])

    def test_missing_api_key_judgment_survives_repeat_probe(self):
        """키 없음을 쿨다운에 넣으면 300초 안의 두 번째 조회가 "cooldown"(retryable=True)
        으로 바뀌어 첫 판정(설정을 고쳐야 풀림, retryable=False)을 스스로 뒤집는다 —
        상태를 비우지 않고 연달아 조회해도 같은 판정이 나와야 한다."""
        del os.environ["OPENROUTER_API_KEY"]

        first = qdrant_store.probe_reranker_capability()
        second = qdrant_store.probe_reranker_capability()

        self.assertEqual(first["reason"], "api_key_missing")
        self.assertEqual(second["reason"], "api_key_missing")
        self.assertFalse(second["retryable"])

    def test_capability_model_names_the_actual_scoring_model(self):
        """요청받은 로컬 이름을 그대로 돌려주면 기록이 거짓말한다 — Optimize 의 baseline
        정체성이 이 필드로 두 경로를 가르므로, 로컬과 API 측정이 한 baseline 으로 섞인다."""
        transport = Mock(return_value=[(0, 0.5)])

        with patch.object(qdrant_store, "openrouter_rerank", transport):
            capability = qdrant_store.probe_reranker_capability(
                qdrant_store.DEFAULT_RERANKER_MODEL
            )

        self.assertEqual(capability["model"], "voyageai/rerank-2.5-lite")
        self.assertNotEqual(capability["model"], qdrant_store.DEFAULT_RERANKER_MODEL)
        # API 경로에는 토크나이저가 없어 입력 상한이라는 개념 자체가 없다.
        self.assertIsNone(capability["max_length"])

    def test_capability_carries_the_route(self):
        transport = Mock(return_value=[(0, 0.5)])

        with patch.object(qdrant_store, "openrouter_rerank", transport):
            capability = qdrant_store.probe_reranker_capability()

        self.assertEqual(capability["status"], "verified")
        self.assertEqual(capability["route"], "openrouter:voyageai/rerank-2.5-lite")


class RerankRouteReachesDecisionKeysTest(unittest.TestCase):
    """route 가 "보이는 기록"에만 있으면, 정작 판단 계층에서는 두 경로가 한 실행으로 섞인다."""

    @staticmethod
    def _cache_key(state):
        from agents.eval.agent import _eval_cache_key

        return _eval_cache_key(state, 1, "pipeline-v1", "probe-v1")

    def _state(self):
        from core.state import AgentDoctorState

        return AgentDoctorState(
            index_config={"use_reranker": True, "top_k": 5},
            active_index_key="idx-1",
        )

    def test_reranker_axes_change_the_eval_cache_key(self):
        """config 가 같아도 리랭커 경로가 바뀌면 검색 결과가 달라진다 — 캐시가 히트하면
        이전 경로의 리포트가 복원돼, 경로 차이가 report 에 기록되기도 전에 묻힌다."""
        state = self._state()
        axes = {
            "INDEX_RERANKER_PROVIDER": "openrouter",
            "INDEX_RERANKER_MODEL_OPENROUTER": "cohere/rerank-v3.5",
            "INDEX_RERANKER_DEVICE": "cpu",
            "INDEX_RERANKER_MAX_LENGTH": "512",
        }

        with patch.dict(os.environ, {"INDEX_RERANKER_PROVIDER": "local"}):
            baseline = self._cache_key(state)

        for name, value in axes.items():
            with self.subTest(env=name):
                with patch.dict(os.environ, {**{"INDEX_RERANKER_PROVIDER": "local"},
                                             name: value}):
                    self.assertNotEqual(self._cache_key(state), baseline)


class RerankTimingTest(unittest.TestCase):
    """rerank_seconds 는 추론(호출) 시간만 재야 한다 — 이 계측이 리랭커 존폐의 심판이다."""

    def setUp(self):
        _clear_reranker_state()

    def tearDown(self):
        _clear_reranker_state()

    def test_rerank_seconds_prefers_model_reported_pure_time(self):
        """모델이 순수 호출 시간을 보고하면 벽시계 대신 그 값을 쓴다. API 경로의
        429 재시도 대기(기본 5~10초)가 벽시계에 섞이면 쌍당 ms 가 수 배로 부푼다."""
        model_name = "test/pure-time"
        model = Mock()
        model.predict.return_value = [0.5]
        model.last_predict_seconds = 0.2
        qdrant_store._rerankers[model_name] = model
        ticks = iter([100.0, 160.0])          # 벽시계로는 60초가 걸린 것처럼 보인다

        with patch(
            "agents.index.qdrant_store.time.monotonic",
            side_effect=lambda: next(ticks),
        ):
            _out, status, seconds = qdrant_store.rerank_with_status(
                "질문",
                [{"chunk_id": "c1", "text": "본문", "score": 0.5}],
                model_name=model_name,
                top_k=1,
            )

        self.assertEqual(status, "applied")
        self.assertEqual(seconds, 0.2)

    def test_openrouter_predict_excludes_retry_wait(self):
        """429 재시도의 대기·실패한 시도 시간은 last_predict_seconds 에서 빠진다 —
        성공한 시도의 HTTP 호출 시간만 남는다."""
        calls = {"n": 0}

        def flaky(_query, _documents, _model, *, api_key, tag):
            calls["n"] += 1
            if calls["n"] == 1:
                error = RuntimeError("429 too many requests")
                error.status_code = 429
                raise error
            return [(0, 0.9)]

        reranker = qdrant_store._OpenRouterReranker("test/model", "key")
        # 시도1 시작 100(실패, 끝 시각은 안 읽음) / 시도2 시작 110 → 성공 112 = 2초
        ticks = iter([100.0, 110.0, 112.0])

        with patch.object(qdrant_store, "openrouter_rerank", side_effect=flaky):
            with patch(
                "agents.index.qdrant_store.time.monotonic",
                side_effect=lambda: next(ticks),
            ):
                with patch("core.llm_retry.time.sleep"):
                    scores = reranker.predict([("질문", "본문")])

        self.assertEqual(scores, [0.9])
        self.assertEqual(calls["n"], 2)
        self.assertEqual(reranker.last_predict_seconds, 2.0)


class RerankerPreflightCostTest(unittest.TestCase):
    """리랭커를 쓰지도 않는 실행이 Index 마다 유료 호출을 내면 안 된다.

    이 preflight 는 optimize → index 루프를 도는 횟수만큼 반복된다 — 파이프라인당
    1건이 아니다."""

    def setUp(self):
        _clear_reranker_state()

    def tearDown(self):
        _clear_reranker_state()

    @staticmethod
    def _smoke(use_reranker, provider):
        from agents.index.agent import _should_smoke_test_reranker

        with patch.dict(os.environ, {"INDEX_RERANKER_PROVIDER": provider}):
            return _should_smoke_test_reranker(
                "eager", {"use_reranker": use_reranker}
            )

    def test_disabled_reranker_skips_paid_smoke_call(self):
        self.assertFalse(self._smoke(False, "openrouter"))

    def test_enabled_reranker_still_verifies_the_api_path(self):
        """켠 실행은 키·모델명 오타를 Eval 수십 건을 태우기 전에 잡아야 한다."""
        self.assertTrue(self._smoke(True, "openrouter"))

    def test_local_path_always_smokes(self):
        """로컬 smoke 는 이미 올린 모델을 한 번 돌리는 것이라 공짜다."""
        self.assertTrue(self._smoke(False, "local"))


class RerankerRouteSwitchTest(unittest.TestCase):
    def setUp(self):
        _clear_reranker_state()

    def tearDown(self):
        _clear_reranker_state()

    def test_cached_model_is_replaced_when_route_changes(self):
        """provider 를 바꾼 실행이 이전 경로의 객체를 계속 쓰면, 리포트에는 새 경로가
        찍히는데 실제로는 옛 모델이 채점한다 — 비교 실험의 라벨이 통째로 거짓말이 된다."""
        model_name = qdrant_store.DEFAULT_RERANKER_MODEL
        stale = Mock()
        stale.predict.return_value = [0.1]
        qdrant_store._rerankers[model_name] = stale
        qdrant_store._reranker_routes[model_name] = "local:cpu"

        with patch.dict(
            os.environ,
            {"INDEX_RERANKER_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k"},
        ):
            model, status = qdrant_store._load_reranker(model_name)

        self.assertEqual(status, "ready")
        self.assertIsNot(model, stale)
        self.assertEqual(
            qdrant_store._reranker_routes[model_name],
            "openrouter:voyageai/rerank-2.5-lite",
        )

    def test_injected_double_without_route_is_kept(self):
        """경로 기록이 없는 객체는 테스트가 직접 꽂은 더블이다. 갈아끼우면 기존
        리랭커 테스트 전부가 실모델을 내려받는다."""
        model_name = "test/injected-double"
        double = Mock()
        qdrant_store._rerankers[model_name] = double

        model, status = qdrant_store._load_reranker(model_name)

        self.assertIs(model, double)
        self.assertEqual(status, "ready")

    def test_unsupported_provider_value_does_not_reload_every_query(self):
        """미지원 값(오타·"gpu" 등)은 dispatch 가 local 로 처리한다. 캐시 유지 판정이
        "local" 리터럴만 보면 그 값이 항상 경로 불일치가 돼, 질의마다 2GB 대 모델을
        새로 올린다 — 리뷰에서 3회 호출에 3회 로드로 실측된 문제."""
        import sys
        import types

        model_name = "test/typo-provider"
        loads = []
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = lambda _name, **kw: (
            loads.append(_name) or Mock(predict=Mock(return_value=[0.1]))
        )

        with patch.dict(os.environ, {"INDEX_RERANKER_PROVIDER": "gpu"}):
            with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
                first, status1 = qdrant_store._load_reranker(model_name)
                second, status2 = qdrant_store._load_reranker(model_name)

        self.assertEqual((status1, status2), ("ready", "ready"))
        self.assertIs(first, second)
        self.assertEqual(len(loads), 1)


class RerankerCudaFallbackTest(unittest.TestCase):
    def setUp(self):
        _clear_reranker_state()

    def tearDown(self):
        _clear_reranker_state()

    def test_inference_oom_demotes_to_cpu_without_cooldown(self):
        """쿨다운을 걸면 300초 뒤 같은 GPU 로 재시도해 같은 자리에서 또 죽는다.
        장치를 CPU 로 내렸으면 다음 질의는 곧바로 다시 시도해야 한다."""
        model_name = qdrant_store.DEFAULT_RERANKER_MODEL
        oom = Mock()
        oom.predict.side_effect = RuntimeError(
            "CUDA out of memory. Tried to allocate 2.00 GiB"
        )
        qdrant_store._rerankers[model_name] = oom
        qdrant_store._reranker_routes[model_name] = "local:cuda"
        results = [{"chunk_id": "c1", "text": "본문", "score": 0.5}]

        reranked, status, _seconds = qdrant_store.rerank_with_status(
            "질문", results, top_k=1
        )

        self.assertEqual(status, "inference_failed")
        self.assertEqual([item["chunk_id"] for item in reranked], ["c1"])
        self.assertEqual(qdrant_store._reranker_device_overrides[model_name], "cpu")
        self.assertNotIn(model_name, qdrant_store._failed_rerankers)
        self.assertNotIn(model_name, qdrant_store._rerankers)

    def test_non_oom_failure_still_cools_down(self):
        model_name = qdrant_store.DEFAULT_RERANKER_MODEL
        broken = Mock()
        broken.predict.side_effect = ValueError("weights corrupted")
        qdrant_store._rerankers[model_name] = broken
        qdrant_store._reranker_routes[model_name] = "local:cuda"

        qdrant_store.rerank_with_status(
            "질문", [{"chunk_id": "c1", "text": "본문", "score": 0.5}], top_k=1
        )

        self.assertIn(model_name, qdrant_store._failed_rerankers)
        self.assertNotIn(model_name, qdrant_store._reranker_device_overrides)


class RerankRouteReachesReportTest(unittest.TestCase):
    def setUp(self):
        _clear_reranker_state()

    def tearDown(self):
        _clear_reranker_state()

    def test_search_details_carry_the_route(self):
        model_name = "test/route-reranker"
        model = Mock()
        model.predict.return_value = [0.3, 0.9]
        qdrant_store._rerankers[model_name] = model
        # 경로 기록은 남기되 provider(local 기본)와 어긋나게 두지 않는다 — provider 가
        # 다르면 _load_reranker 가 이 더블을 갈아끼우려 실모델 로드에 나선다.
        qdrant_store._reranker_routes[model_name] = "local:cuda"
        chunks = [
            {"chunk_id": f"c{i}", "doc_id": "d1", "text": f"alpha {i}", "metadata": {}}
            for i in range(2)
        ]
        retriever = Retriever(
            chunks,
            RetrievalSettings(
                use_reranker=True, reranker_model=model_name, rerank_candidates=2
            ),
            client=None,
        )

        result = retriever.search_with_details("alpha", top_k=2)

        self.assertEqual(result["rerank_route"], "local:cuda")

    def test_failed_attempt_keeps_the_route_it_tried(self):
        """실패야말로 "OpenRouter 를 켰는데 왜 리랭크가 안 됐지" 를 봐야 하는 순간이다 —
        여기서 경로가 사라지면 리포트만 보고는 원인을 못 따라간다."""
        model_name = "test/failing-route"
        broken = Mock()
        broken.predict.side_effect = ValueError("boom")
        qdrant_store._rerankers[model_name] = broken
        chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "alpha", "metadata": {}}]
        retriever = Retriever(
            chunks,
            RetrievalSettings(
                use_reranker=True, reranker_model=model_name, rerank_candidates=1
            ),
            client=None,
        )

        with patch.dict(
            os.environ,
            {"INDEX_RERANKER_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "k"},
        ):
            result = retriever.search_with_details("alpha", top_k=1)

        self.assertEqual(result["reranker_status"], "inference_failed")
        self.assertEqual(
            result["rerank_route"], "openrouter:voyageai/rerank-2.5-lite"
        )

    def test_disabled_reranker_reports_no_route(self):
        """시도조차 안 한 실행에 경로가 찍히면 "리랭크가 돌았다" 로 읽힌다."""
        chunks = [{"chunk_id": "c1", "doc_id": "d1", "text": "alpha", "metadata": {}}]
        retriever = Retriever(chunks, RetrievalSettings(use_reranker=False), client=None)

        result = retriever.search_with_details("alpha", top_k=1)

        self.assertIsNone(result["rerank_route"])

    def test_report_separates_applied_and_attempted_routes(self):
        records = [
            EvalRecord(
                probe=Probe("p1", "질문", "test"),
                retrieval_details={
                    "reranker_attempted": True, "reranker_status": "applied",
                    "reranked": True, "rerank_pairs": 3, "rerank_seconds": 0.1,
                    "rerank_route": "local:cuda",
                },
            ),
            EvalRecord(
                probe=Probe("p2", "질문", "test"),
                retrieval_details={
                    "reranker_attempted": True, "reranker_status": "api_key_missing",
                    "reranked": False, "rerank_pairs": 0,
                    "rerank_route": "openrouter:voyageai/rerank-2.5-lite",
                },
            ),
        ]

        rerank = build_report(records, iteration=1).runtime_summary["reranker"]

        self.assertEqual(rerank["routes"], ["local:cuda"])
        self.assertEqual(
            rerank["attempted_routes"],
            ["local:cuda", "openrouter:voyageai/rerank-2.5-lite"],
        )
        # 실제로 채점한 모델은 하나뿐이다 — 실패한 시도는 점수를 만들지 않았다.
        self.assertFalse(rerank["mixed_models"])

    def test_report_flags_runs_scored_by_different_models(self):
        """local + openrouter 가 섞이면 점수 분포가 섞여 처방 전후 비교가 성립하지 않는다."""
        def _record(probe_id, route):
            return EvalRecord(
                probe=Probe(probe_id, "질문", "test"),
                retrieval_details={
                    "reranker_attempted": True, "reranker_status": "applied",
                    "reranked": True, "rerank_pairs": 3, "rerank_seconds": 0.1,
                    "rerank_route": route,
                },
            )

        mixed = build_report(
            [
                _record("p1", "local:cuda"),
                _record("p2", "openrouter:voyageai/rerank-2.5-lite"),
            ],
            iteration=1,
        ).runtime_summary["reranker"]
        # CUDA OOM 강등은 경로만 둘이고 모델은 같다 — 점수가 같으므로 경고 대상이 아니다.
        demoted = build_report(
            [_record("p1", "local:cuda"), _record("p2", "local:cpu")],
            iteration=1,
        ).runtime_summary["reranker"]

        self.assertTrue(mixed["mixed_models"])
        self.assertFalse(demoted["mixed_models"])
        self.assertEqual(len(demoted["routes"]), 2)

    def test_report_lists_every_route_that_actually_ran(self):
        """경로가 섞인 실행은 리포트에서 드러나야 한다 — 두 경로는 서로 다른 모델이라
        점수 차이를 처방 효과로 읽으면 안 된다."""
        records = [
            EvalRecord(
                probe=Probe("p1", "질문", "test"),
                retrieval_details={
                    "reranker_status": "applied", "reranked": True,
                    "rerank_pairs": 3, "rerank_seconds": 0.1,
                    "rerank_route": "local:cuda",
                },
            ),
            EvalRecord(
                probe=Probe("p2", "질문", "test"),
                retrieval_details={
                    "reranker_status": "applied", "reranked": True,
                    "rerank_pairs": 3, "rerank_seconds": 0.2,
                    "rerank_route": "openrouter:voyageai/rerank-2.5-lite",
                },
            ),
            # 돌지 않은 실행의 경로는 세지 않는다(쌍 0).
            EvalRecord(
                probe=Probe("p3", "질문", "test"),
                retrieval_details={
                    "reranker_status": "load_failed", "reranked": False,
                    "rerank_pairs": 0, "rerank_route": "local:cpu",
                },
            ),
        ]

        report = build_report(records, iteration=1)

        self.assertEqual(
            report.runtime_summary["reranker"]["routes"],
            ["local:cuda", "openrouter:voyageai/rerank-2.5-lite"],
        )


if __name__ == "__main__":
    unittest.main()
