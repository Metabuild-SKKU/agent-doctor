# --embed / --query-embed 플래그의 계약을 고정한다.
# 파이프라인은 돌리지 않는다 — 인자 → os.environ 변환만 본다.
from __future__ import annotations

import argparse
import os
import unittest
from unittest.mock import patch

from core.embedding_cli import (
    EMBED_CHOICES,
    add_embedding_args,
    apply_embedding_args,
)

_KEYS = ("INDEX_EMBED_PROVIDER", "INDEX_EMBED_DEVICE", "INDEX_QUERY_EMBED_PROVIDER")


def _apply(argv, env=None):
    """깨끗한 env 에서 argv 를 적용하고 결과 env 를 돌려준다."""
    base = dict.fromkeys(_KEYS, "")
    base.update(env or {})
    with patch.dict(os.environ, base):
        for key, value in base.items():
            if not value:
                os.environ.pop(key, None)
        parser = argparse.ArgumentParser()
        add_embedding_args(parser)
        apply_embedding_args(parser.parse_args(argv))
        return {k: os.environ.get(k) for k in _KEYS}


class ChoiceMappingTests(unittest.TestCase):
    """사용자가 고르는 3지선다를 코드의 2축(provider, device)으로 옮긴다."""

    def test_openrouter(self):
        env = _apply(["--embed", "openrouter"])
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "openrouter")
        self.assertEqual(env["INDEX_QUERY_EMBED_PROVIDER"], "openrouter")
        # device 는 local 일 때만 의미가 있어 건드리지 않는다.
        self.assertIsNone(env["INDEX_EMBED_DEVICE"])

    def test_gpu(self):
        env = _apply(["--embed", "gpu"])
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "local")
        self.assertEqual(env["INDEX_EMBED_DEVICE"], "cuda")

    def test_cpu(self):
        env = _apply(["--embed", "cpu"])
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "local")
        self.assertEqual(env["INDEX_EMBED_DEVICE"], "cpu")

    def test_choices_are_the_three_the_user_thinks_in(self):
        self.assertEqual(set(EMBED_CHOICES), {"openrouter", "gpu", "cpu"})


class PrecedenceTests(unittest.TestCase):
    def test_flag_beats_dotenv(self):
        """이 모듈이 존재하는 이유다.

        엔트리포인트들은 load_dotenv(override=True) 라 셸 값보다 .env 가 이긴다.
        그래서 INDEX_EMBED_PROVIDER=local python graph.py 가 조용히 무시된다.
        플래그는 .env 로드 뒤에 적용되므로 항상 이겨야 한다."""
        env = _apply(["--embed", "cpu"],
                     env={"INDEX_EMBED_PROVIDER": "openrouter",
                          "INDEX_QUERY_EMBED_PROVIDER": "openrouter"})
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "local")
        self.assertEqual(env["INDEX_QUERY_EMBED_PROVIDER"], "local")

    def test_no_flag_leaves_env_alone(self):
        """미지정 인자는 건드리지 않는다 — .env·셸 설정이 그대로 살아 있어야 한다."""
        env = _apply([], env={"INDEX_EMBED_PROVIDER": "openrouter",
                              "INDEX_EMBED_DEVICE": "cuda"})
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "openrouter")
        self.assertEqual(env["INDEX_EMBED_DEVICE"], "cuda")
        self.assertIsNone(env["INDEX_QUERY_EMBED_PROVIDER"])


class QueryAxisTests(unittest.TestCase):
    def test_embed_applies_to_both_axes(self):
        # --embed cpu 로 골랐는데 질의만 API 를 타면 혼란스럽다.
        env = _apply(["--embed", "cpu"])
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "local")
        self.assertEqual(env["INDEX_QUERY_EMBED_PROVIDER"], "local")

    def test_query_embed_overrides_only_query(self):
        """대표 조합: 색인은 API 로 두고 질의만 로컬로 내린다.

        429 재시도가 단건 질의 지연으로 그대로 나타날 때 쓴다."""
        env = _apply(["--embed", "openrouter", "--query-embed", "cpu"])
        self.assertEqual(env["INDEX_EMBED_PROVIDER"], "openrouter")
        self.assertEqual(env["INDEX_QUERY_EMBED_PROVIDER"], "local")
        # 질의가 로컬이므로 장치가 필요하다.
        self.assertEqual(env["INDEX_EMBED_DEVICE"], "cpu")

    def test_query_embed_alone_does_not_touch_index(self):
        env = _apply(["--query-embed", "cpu"])
        self.assertIsNone(env["INDEX_EMBED_PROVIDER"])
        self.assertEqual(env["INDEX_QUERY_EMBED_PROVIDER"], "local")

    def test_index_device_wins_when_both_are_local(self):
        # --embed gpu 로 장치를 정했으면 질의도 같은 장치를 쓴다(모델을 두 벌 올리지 않게).
        env = _apply(["--embed", "gpu", "--query-embed", "cpu"])
        self.assertEqual(env["INDEX_EMBED_DEVICE"], "cuda")


if __name__ == "__main__":
    unittest.main()
