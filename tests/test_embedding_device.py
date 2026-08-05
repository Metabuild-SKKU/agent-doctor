# 임베딩 장치(cuda/cpu) 선택 계층의 계약을 고정한다. 실제 모델도 GPU 도 필요 없다 —
# sentence_transformers·torch 를 가짜로 바꿔 분기만 확인한다.
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from agents.index import qdrant_store as store


def _fake_torch(cuda_available: bool):
    """torch.cuda.is_available() 만 흉내내는 최소 모듈."""
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        empty_cache=lambda: None,
        OutOfMemoryError=RuntimeError,
    )
    return module


class _FakeEncoder:
    """SentenceTransformer 대역. encode 호출의 batch_size 를 기록한다."""

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.batch_sizes: list[int] = []
        self.tokenizer = types.SimpleNamespace(encode=lambda text: list(text))

    def encode(self, texts, batch_size=32, normalize_embeddings=True,
               show_progress_bar=False):
        self.batch_sizes.append(batch_size)
        return [_Vec([float(len(text))] * 4) for text in texts]


class _Vec(list):
    def tolist(self):
        return list(self)


class ResolveDeviceTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(True)}):
            self.assertEqual(store.resolve_embedding_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu_without_cuda(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(False)}):
            self.assertEqual(store.resolve_embedding_device("auto"), "cpu")

    def test_auto_falls_back_to_cpu_without_torch(self):
        # torch 가 아예 없는 환경(설치 안 됨)에서도 죽지 않아야 한다.
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(store.resolve_embedding_device("auto"), "cpu")

    def test_explicit_cuda_downgrades_loudly(self):
        # 조용히 내리면 "GPU 로 돌리는 중" 이라 믿은 실행이 CPU 속도로 기어간다.
        with patch.dict(sys.modules, {"torch": _fake_torch(False)}), \
             patch("builtins.print") as printed:
            self.assertEqual(store.resolve_embedding_device("cuda"), "cpu")
        self.assertTrue(printed.called)

    def test_env_supplies_default(self):
        with patch.dict(sys.modules, {"torch": _fake_torch(True)}), \
             patch.dict("os.environ", {"INDEX_EMBED_DEVICE": "cpu"}):
            self.assertEqual(store.resolve_embedding_device(), "cpu")


class _CacheIsolated(unittest.TestCase):
    """모델 캐시는 전역이라 테스트마다 비운다.

    provider 도 local 로 고정한다 — 이 파일은 로컬 경로(장치 선택·OOM)를 다루고,
    기본값 provider 로 두면 실행 환경의 env 에 따라 API 경로로 새어 나간다."""

    def setUp(self):
        patcher = patch.dict("os.environ", {"INDEX_EMBED_PROVIDER": "local"})
        patcher.start()
        self.addCleanup(patcher.stop)
        store._models.clear()
        store._failed_models.clear()
        store._tokenizers.clear()
        self.addCleanup(store._models.clear)
        self.addCleanup(store._failed_models.clear)
        self.addCleanup(store._tokenizers.clear)


class DeviceKeyedCacheTests(_CacheIsolated):
    def test_same_model_cached_per_device(self):
        loaded = []

        def _factory(model_name, device="cpu"):
            loaded.append(device)
            return _FakeEncoder(model_name, device)

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = _factory
        with patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                      "torch": _fake_torch(True)}):
            store._load_embedding_model("m", "cuda")
            store._load_embedding_model("m", "cpu")
            store._load_embedding_model("m", "cuda")   # 캐시 적중

        self.assertEqual(loaded, ["cuda", "cpu"])
        self.assertIn(("m", "cuda"), store._models)
        self.assertIn(("m", "cpu"), store._models)

    def test_gpu_load_failure_falls_back_to_cpu(self):
        def _factory(model_name, device="cpu"):
            if device != "cpu":
                raise RuntimeError("no gpu here")
            return _FakeEncoder(model_name, device)

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = _factory
        with patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                      "torch": _fake_torch(True)}), \
             patch("builtins.print"):
            model, device = store._load_embedding_model("m", "cuda")

        # 해시 fallback 이 아니라 CPU 실모델로 내려가야 한다.
        self.assertIsNotNone(model)
        self.assertEqual(device, "cpu")
        self.assertIn(("m", "cuda"), store._failed_models)

    def test_is_fallback_false_when_cpu_usable(self):
        # 기준은 "해시 벡터를 쓰는가" 이지 "어느 장치인가" 가 아니다.
        # Eval 의 로컬 임베딩 가용성 판정이 이 함수를 본다.
        def _factory(model_name, device="cpu"):
            if device != "cpu":
                raise RuntimeError("no gpu here")
            return _FakeEncoder(model_name, device)

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = _factory
        with patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                      "torch": _fake_torch(True)}), \
             patch("builtins.print"):
            self.assertFalse(store.embedding_is_fallback("m", device="cuda"))


class CountTokensCacheTests(_CacheIsolated):
    def test_finds_tokenizer_regardless_of_device_key(self):
        """캐시 키가 (이름, 장치) 인데 이름만으로 조회하면 항상 미스가 난다.

        미스는 예외가 아니라 어림짐작 토큰 수로 조용히 떨어지는 형태로 나타나고,
        그 값은 optimize 의 chunk_size 처방에 쓰인다."""
        store._models[("m", "cuda")] = _FakeEncoder("m", "cuda")
        # tokenizer.encode 는 문자 단위라 길이가 그대로 나온다.
        self.assertEqual(store.count_tokens("hello", model_name="m"), 5)

    def test_falls_back_to_heuristic_when_tokenizer_unavailable(self):
        # tokenizer 마저 못 받으면 어림짐작으로 떨어지되, 전체 모델을 지연 로드하지는
        # 않는다 — 그러면 토큰 세기 한 번이 2.2GB 다운로드를 유발한다(리뷰 #36).
        with patch.dict(sys.modules, {"transformers": None}), patch("builtins.print"):
            self.assertGreaterEqual(store.count_tokens("a b c", model_name="m"), 1)
        self.assertEqual(store._models, {})

    def test_tokenizer_loaded_separately_for_api_provider(self):
        """API provider 는 전체 모델을 안 올리므로 tokenizer 만 따로 받아야 한다."""
        fake_tf = types.ModuleType("transformers")
        calls = []

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(name):
                calls.append(name)
                return types.SimpleNamespace(encode=lambda text: list(text))

        fake_tf.AutoTokenizer = _AutoTokenizer
        with patch.dict(sys.modules, {"transformers": fake_tf}):
            self.assertEqual(store.count_tokens("hello", model_name="m"), 5)
            self.assertEqual(store.count_tokens("hi", model_name="m"), 2)

        # 전체 모델은 올라오지 않고, tokenizer 는 한 번만 받는다.
        self.assertEqual(store._models, {})
        self.assertEqual(calls, ["m"])

    def test_tokenizer_failure_is_cached(self):
        # 실패를 캐시하지 않으면 청크마다 네트워크를 두드려 색인이 기어간다.
        fake_tf = types.ModuleType("transformers")
        calls = []

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(name):
                calls.append(name)
                raise OSError("offline")

        fake_tf.AutoTokenizer = _AutoTokenizer
        with patch.dict(sys.modules, {"transformers": fake_tf}), patch("builtins.print"):
            store.count_tokens("a b", model_name="m")
            store.count_tokens("c d", model_name="m")
        self.assertEqual(calls, ["m"])


class BatchOomTests(_CacheIsolated):
    def test_cuda_oom_halves_batch_size(self):
        encoder = _FakeEncoder("m", "cuda")
        calls = {"n": 0}

        def _encode(texts, batch_size=32, **kwargs):
            calls["n"] += 1
            encoder.batch_sizes.append(batch_size)
            if batch_size > 8:
                raise RuntimeError("CUDA out of memory")
            return [_Vec([1.0]) for _ in texts]

        encoder.encode = _encode
        with patch.dict(sys.modules, {"torch": _fake_torch(True)}), \
             patch("builtins.print"):
            vectors = store._encode_batch(encoder, ["a", "b"], 32, "cuda")

        self.assertEqual(len(vectors), 2)
        self.assertEqual(encoder.batch_sizes, [32, 16, 8])

    def test_non_oom_error_propagates(self):
        # OOM 이 아닌 실패까지 배치를 줄여 재시도하면 진짜 버그를 삼킨다.
        encoder = _FakeEncoder("m", "cuda")

        def _encode(texts, batch_size=32, **kwargs):
            raise ValueError("schema mismatch")

        encoder.encode = _encode
        with patch.dict(sys.modules, {"torch": _fake_torch(True)}):
            with self.assertRaises(ValueError):
                store._encode_batch(encoder, ["a"], 32, "cuda")

    def test_cpu_oom_string_does_not_loop(self):
        # CPU 경로에서는 OOM 문구가 보여도 배치 축소 대상이 아니다.
        encoder = _FakeEncoder("m", "cpu")

        def _encode(texts, batch_size=32, **kwargs):
            raise RuntimeError("CUDA out of memory")

        encoder.encode = _encode
        with patch.dict(sys.modules, {"torch": _fake_torch(False)}):
            with self.assertRaises(RuntimeError):
                store._encode_batch(encoder, ["a"], 32, "cpu")


class EmbedPathTests(_CacheIsolated):
    def test_embed_matches_embed_batch(self):
        """단건 embed() 와 embed_batch() 가 같은 경로를 타는지 고정한다."""
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = _FakeEncoder
        with patch.dict(sys.modules, {"sentence_transformers": fake_st,
                                      "torch": _fake_torch(False)}):
            single = store.embed("hello", model_name="m")
            batched = store.embed_batch(["hello"], model_name="m")
        self.assertEqual(single, batched[0])

    def test_empty_input_short_circuits(self):
        self.assertEqual(store.embed_batch([], model_name="m"), [])
        self.assertEqual(store._models, {})


if __name__ == "__main__":
    unittest.main()
