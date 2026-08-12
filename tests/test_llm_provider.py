"""
tests/test_llm_provider.py
llm_provider.chat_json 단위 테스트.

chat_json 은 그동안 직접 테스트가 없었다(합성 경로 테스트는 chat_json 을 목킹해 우회).
여기서는 (1) Gemini 가 dict 를 [ {…} ] 로 감싸 반환한 경우의 언랩, (2) 빈 응답/파싱
실패/타입 불일치의 사유별 로그를 검증한다. 실제 API 대신 _run_with_retry 를 patch 해
raw 응답 문자열만 주입한다(transport·재시도는 core 쪽 관심사).
"""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.eval import llm_provider


def _chat_json_with_raw(raw: str):
    """_run_with_retry 를 patch 해 raw 를 주입하고 (반환값, stdout) 을 돌려준다."""
    buf = io.StringIO()
    with patch.object(llm_provider, "_run_with_retry", return_value=raw), \
            redirect_stdout(buf):
        result = llm_provider.chat_json("sys", "user")
    return result, buf.getvalue()


class ChatJsonUnwrapTest(unittest.TestCase):
    def test_dict_passthrough(self):
        result, _ = _chat_json_with_raw('{"question": "q", "ground_truth": "a"}')
        self.assertEqual(result, {"question": "q", "ground_truth": "a"})

    def test_single_element_list_is_unwrapped(self):
        # Gemini 가 dict 를 한 겹 감싸 반환한 경우.
        result, _ = _chat_json_with_raw('[{"question": "q", "ground_truth": "a"}]')
        self.assertEqual(result, {"question": "q", "ground_truth": "a"})

    def test_multi_element_list_is_not_unwrapped(self):
        # 길이 2+ 는 스키마 위반 — 억지로 풀지 않는다.
        result, log = _chat_json_with_raw('[{"a": 1}, {"b": 2}]')
        self.assertEqual(result, {})
        self.assertIn("타입 불일치", log)

    def test_list_of_non_dict_is_not_unwrapped(self):
        result, log = _chat_json_with_raw('[1]')
        self.assertEqual(result, {})
        self.assertIn("타입 불일치", log)


class ChatJsonFailureReasonTest(unittest.TestCase):
    def test_empty_response_logs_reason(self):
        result, log = _chat_json_with_raw("")
        self.assertEqual(result, {})
        self.assertIn("빈 응답", log)

    def test_whitespace_only_response_is_treated_as_empty(self):
        result, log = _chat_json_with_raw("   \n  ")
        self.assertEqual(result, {})
        self.assertIn("빈 응답", log)

    def test_parse_failure_logs_reason(self):
        result, log = _chat_json_with_raw("이건 JSON 이 아니다")
        self.assertEqual(result, {})
        self.assertIn("파싱 실패", log)


if __name__ == "__main__":
    unittest.main()


class EmbedProviderOverrideTest(unittest.TestCase):
    """EVAL_EMBED_PROVIDER — 임베딩 provider 를 심판(EVAL_LLM_PROVIDER)에서 분리하는 스위치.

    실제 사고(2026-08-11)의 회귀 방지다: 심판만 anthropic 으로 바꿨는데 임베딩이 로컬
    BGE-M3 로 끌려가 (1) fan-out 동시 로드 race 로 OS 프리즈, (2) 8GB GPU 에서 리랭커와
    VRAM 경합(검색 192초→31분)을 만들었다. openrouter 명시가 로컬 경로를 피하는지,
    오타·키 부재가 조용히 새 경로를 만들지 않는지를 고정한다."""

    def setUp(self):
        llm_provider._embed_notified = False

    def _run(self, env: dict, local_ok: bool = True):
        """embed_texts 1회. 어느 embed 가 불렸는지와 stdout 을 돌려준다."""
        called = {}
        buf = io.StringIO()

        def fake_openrouter(texts, model):
            called["openrouter"] = model
            return [[1.0]] * len(texts)

        def fake_openai(texts, model):
            called["openai"] = model
            return [[1.0]] * len(texts)

        def fake_local(texts, provider=None):
            called["local"] = provider
            return [[0.0]] * len(texts)

        base_env = {"EVAL_LLM_PROVIDER": "anthropic", "EVAL_EMBED_PROVIDER": "",
                    "OPENROUTER_API_KEY": "", "OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict(os.environ, {**base_env, **env}, clear=False), \
                patch.object(llm_provider, "_openrouter_embed", fake_openrouter), \
                patch.object(llm_provider, "_openai_embed", fake_openai), \
                patch.object(llm_provider, "_local_embeddings_available",
                             return_value=local_ok), \
                patch("agents.index.qdrant_store.embed_batch", fake_local), \
                redirect_stdout(buf):
            llm_provider.embed_texts(["텍스트"])
        return called, buf.getvalue()

    def test_override_routes_to_openrouter_even_with_anthropic_judge(self):
        called, _ = self._run({"EVAL_EMBED_PROVIDER": "openrouter",
                               "OPENROUTER_API_KEY": "k"})
        self.assertIn("openrouter", called)
        self.assertNotIn("local", called)          # 사고 경로(로컬)를 타지 않는다

    def test_override_without_key_skips_instead_of_falling_back(self):
        """명시 override 는 강제값 — 못 쓰면 결측이지, 다른 provider 로 새지 않는다.

        사고의 원인이 '로컬로 조용히 새서 GPU 경합'이었으므로, override 를 적은 사용자의
        의도(그 경로만)를 지킨다(리뷰 ③ 정책 결정). 폴백을 원하면 값을 지우면 된다."""
        called, log = self._run({"EVAL_EMBED_PROVIDER": "openrouter"})
        self.assertNotIn("openrouter", called)     # 키 없이 호출하지 않는다
        self.assertNotIn("local", called)          # 사고 경로(로컬)로도 새지 않는다
        self.assertIn("폴백하지 않습니다", log)      # 조용히 결측되지 않는다

    def test_invalid_override_warns_and_uses_chain(self):
        called, log = self._run({"EVAL_EMBED_PROVIDER": "openroutre",   # 오타
                                 "EVAL_LLM_PROVIDER": "openrouter",
                                 "OPENROUTER_API_KEY": "k"})
        self.assertIn("지원하지 않는 값", log)
        self.assertIn("openrouter", called)        # 기본 사슬(심판 provider API)로 동작

    def test_override_local_skips_api_even_with_keys(self):
        called, _ = self._run({"EVAL_EMBED_PROVIDER": "local",
                               "OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "k"})
        self.assertEqual(called.get("local"), "local")
        self.assertNotIn("openrouter", called)
        self.assertNotIn("openai", called)

    def test_route_report_and_availability_agree(self):
        """실제 경로 · 리포트 메타데이터 · embeddings_available 이 한 판정을 공유한다.

        갈리면 벤치마크 리포트의 embedding_source 가 거짓이 되어 실행 간 코사인 비교의
        근거가 무너진다(리뷰 ① — 예전엔 리포트가 심판 provider 기준으로 따로 판정했다)."""
        from agents.eval import report
        env = {"EVAL_LLM_PROVIDER": "anthropic", "EVAL_EMBED_PROVIDER": "openrouter",
               "OPENROUTER_API_KEY": "k", "OPENAI_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False), \
                patch.object(llm_provider, "_local_embeddings_available",
                             return_value=True):    # 로컬이 가능해도 route 를 따라야 한다
            self.assertEqual(llm_provider.embedding_route(), "openrouter")
            self.assertEqual(report._embedding_source(), "openrouter")
            self.assertTrue(llm_provider.embeddings_available())
        # override 경로를 못 쓰면 셋 다 '없음'으로 일치해야 한다 — 로컬로 새면 안 된다.
        with patch.dict(os.environ, {**env, "OPENROUTER_API_KEY": ""}, clear=False), \
                patch.object(llm_provider, "_local_embeddings_available",
                             return_value=True):
            self.assertEqual(llm_provider.embedding_route(), "none")
            self.assertEqual(report._embedding_source(), "none")
            self.assertFalse(llm_provider.embeddings_available())
