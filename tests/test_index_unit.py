# 외부 모델이나 Qdrant 서버 없이 Index 계약만 확인하는 테스트.
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from qdrant_client.models import Distance, VectorParams

from agents.index.agent import CHUNK_STRATEGIES, IndexTools, _chunk_document, run
from agents.index.graph_index import build_graph_artifacts
from agents.index import qdrant_store
from agents.index.qdrant_store import (
    COLLECTION,
    build_sparse_vector,
    build_client,
    delete_document_chunks,
    ensure_collection,
    hybrid_search,
    search,
    upsert_chunks,
)
from core.schema import Chunk, Document
from core.state import AgentDoctorState


def _document(doc_id: str, content: str) -> Document:
    return Document(
        doc_id=doc_id,
        source=f"https://example.com/{doc_id}",
        format="md",
        content=content,
        metadata={"title": doc_id},
    )


def _index_tools() -> IndexTools:
    return IndexTools(
        get_retriever=lambda *_args, **_kwargs: Mock(),
        embed=lambda _text, **_kwargs: [1.0, 0.0, 0.0, 0.0],
        count_tokens=lambda _text, **_kwargs: 3,
        build_sparse_vector=lambda _text: {"indices": [], "values": []},
        build_graph_artifacts=lambda _chunks, _config: {},
    )


class ChunkingTests(unittest.TestCase):
    def test_all_chunk_strategies_are_registered(self):
        self.assertEqual(
            set(CHUNK_STRATEGIES),
            {"fixed", "markdown", "recursive", "markdown_recursive",
             "recursive_sentence"},
        )

    def test_recursive_sentence_breaks_on_sentence_boundaries(self):
        # 문장 종결부 우선. 분할이 일어난 청크의 끝은 문장 종결부여야 한다
        # (마지막 청크와 강제 분할 폴백은 예외).
        source = ("첫째 문장입니다. 둘째 문장은 조금 더 깁니다. 셋째 문장도 있습니다. "
                  "넷째 문장으로 마무리합니다. 다섯째 문장 추가. 여섯째 문장 추가입니다.")
        document = _document("sent-doc", source)
        drafts = _chunk_document(
            document, chunk_size=40, chunk_overlap=0, strategy="recursive_sentence"
        )
        self.assertGreater(len(drafts), 1)
        for draft in drafts[:-1]:
            self.assertTrue(
                draft.text.rstrip().endswith((".", "。", "?", "!", "…")),
                f"문장 경계로 안 끊김: {draft.text!r}",
            )

    def test_overlap_and_max_size_are_respected(self):
        source = "가나다라마바사 " * 30
        document = _document("fixed-doc", source)
        drafts = _chunk_document(document, chunk_size=40, chunk_overlap=8, strategy="fixed")

        self.assertGreater(len(drafts), 1)
        self.assertTrue(all(0 < len(d.text) <= 40 for d in drafts))
        self.assertTrue(all(document.content[d.start:d.end] == d.text for d in drafts))

    def test_markdown_section_is_preserved(self):
        document = _document(
            "guide",
            "# 설치\n설치 방법입니다.\n\n## Windows\nPowerShell을 사용합니다.",
        )

        drafts = _chunk_document(
            document,
            chunk_size=100,
            chunk_overlap=10,
            strategy="markdown_recursive",
        )

        self.assertEqual(drafts[0].section, "설치")
        self.assertEqual(drafts[1].section, "설치 > Windows")
        self.assertEqual(document.content[drafts[1].start : drafts[1].end], drafts[1].text)

    def test_strategies_can_be_swapped_with_one_config_value(self):
        document = _document(
            "guide",
            "# 설치\n" + ("설치 설명 문장입니다. " * 8)
            + "\n## Windows\n" + ("PowerShell 설명입니다. " * 8),
        )

        fixed = _chunk_document(document, 40, 8, strategy="fixed")
        markdown = _chunk_document(document, 40, 8, strategy="markdown")
        recursive = _chunk_document(document, 40, 8, strategy="recursive")
        combined = _chunk_document(
            document,
            40,
            8,
            strategy="markdown_recursive",
        )

        self.assertTrue(all(chunk.section is None for chunk in fixed))
        self.assertEqual([chunk.section for chunk in markdown], ["설치", "설치 > Windows"])
        self.assertTrue(any(len(chunk.text) > 40 for chunk in markdown))
        self.assertTrue(all(chunk.section is None for chunk in recursive))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in recursive))
        self.assertTrue(all(chunk.section is not None for chunk in combined))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in combined))

    def test_numbered_chunk_stages_are_supported(self):
        document = _document(
            "guide",
            "# 설치\n" + ("설치 설명 문장입니다. " * 8)
            + "\n## Windows\n" + ("PowerShell 설명입니다. " * 8),
        )

        stage_1 = _chunk_document(document, 40, 8, strategy=1)
        stage_2 = _chunk_document(document, 40, 8, strategy=2)
        stage_3 = _chunk_document(document, 40, 8, strategy=3)

        self.assertTrue(all(chunk.section is None for chunk in stage_1))
        self.assertTrue(all(chunk.section is None for chunk in stage_2))
        self.assertTrue(all(chunk.section is not None for chunk in stage_3))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in stage_3))

    def test_math_document_keeps_problem_boundaries(self):
        document = _document(
            "math",
            "\n".join(
                [
                    "SET 17",
                    "171번 x=2cos^3(t), y=3sin^3(t)의 접선 기울기를 구하라.",
                    "해설: dy/dx를 계산한다.",
                    "172번 적분법 문제의 정답은 58이다.",
                    "해설: 표의 정답란을 확인한다.",
                    "173번 수열 Sn의 첫째항과 공비를 이용해 급수의 합을 구하라.",
                    "해설: 무한등비급수 공식을 사용한다.",
                ]
            ),
        )
        document.metadata["document_type"] = "math"

        drafts = _chunk_document(
            document,
            chunk_size=120,
            chunk_overlap=20,
            strategy="markdown_recursive",
        )

        sections = [draft.section for draft in drafts]
        self.assertIn("문제 171", sections)
        self.assertIn("문제 172", sections)
        self.assertIn("문제 173", sections)
        for draft in drafts:
            if draft.section == "문제 172":
                self.assertIn("정답은 58", draft.text)
                self.assertNotIn("173번", draft.text)
            self.assertEqual(document.content[draft.start : draft.end], draft.text)

    def test_math_problem_sections_preserve_preamble_before_first_problem(self):
        document = _document(
            "math",
            "\n".join(
                [
                    "# 미적분 개요",
                    "이 장에서 사용할 핵심 정의와 공식입니다.",
                    "1. 첫 번째 문제와 풀이입니다.",
                    "2. 두 번째 문제와 풀이입니다.",
                ]
            ),
        )
        document.metadata["document_type"] = "math"

        drafts = _chunk_document(
            document,
            chunk_size=120,
            chunk_overlap=20,
            strategy="markdown_recursive",
        )

        self.assertEqual(drafts[0].section, "preamble")
        self.assertIn("미적분 개요", drafts[0].text)
        self.assertIn("핵심 정의와 공식", drafts[0].text)
        problem_sections = [draft.section for draft in drafts[1:]]
        self.assertTrue(any(str(section).endswith("1") for section in problem_sections))
        self.assertTrue(any(str(section).endswith("2") for section in problem_sections))
        covered = "".join(document.content[draft.start : draft.end] for draft in drafts)
        compact_original = "".join(document.content.split())
        compact_covered = "".join(covered.split())
        self.assertEqual(compact_covered, compact_original)

    def test_general_numbered_document_does_not_use_math_problem_sections(self):
        document = _document(
            "policy",
            "1. API rate limit 정책을 설명한다.\n2. user_id별 요청 제한을 설명한다.",
        )

        drafts = _chunk_document(
            document,
            chunk_size=200,
            chunk_overlap=20,
            strategy="markdown_recursive",
        )

        self.assertEqual(len(drafts), 1)
        self.assertIsNone(drafts[0].section)

    # ── issue #100: 트림이 만든 좌표 틈 ────────────────────────────
    #
    # char_span 은 앞뒤 공백을 뗀 좌표라 섹션 경계마다 틈이 남는다. raw_start/raw_end
    # 는 트림 전 경계라 그 틈이 닫혀야 하고, 동시에 각 청크가 제 char_span 을 덮어야
    # 한다(안 그러면 그 청크만 검색됐을 때 제 본문을 못 덮는다).

    def _raw_gaps(self, drafts):
        return [(a.raw_end, b.raw_start) for a, b in zip(drafts, drafts[1:])
                if b.raw_start > a.raw_end]

    def _char_gaps(self, drafts):
        return [(a.end, b.start) for a, b in zip(drafts, drafts[1:]) if b.start > a.end]

    def _assert_raw_span_contract(self, document, drafts):
        self.assertTrue(drafts)
        self.assertEqual(self._raw_gaps(drafts), [], "raw 좌표에 틈이 남았다")
        for draft in drafts:
            self.assertLessEqual(draft.raw_start, draft.start)
            self.assertGreaterEqual(draft.raw_end, draft.end)
            # char_span 쪽 불변식은 그대로여야 한다 — probe_gen 이 이걸 검증한다.
            self.assertEqual(document.content[draft.start : draft.end], draft.text)

    def test_section_strategies_leave_char_span_gaps_but_no_raw_gaps(self):
        document = _document(
            "policy",
            "# 1장 총칙\n이 규정은 인사 원칙을 정한다.\n\n"
            "## 2장 연차\n연차는 15일이다.\n\n"
            "## 3장 평가\n평가는 연 2회 실시한다.",
        )

        for strategy in ("markdown", "markdown_recursive"):
            with self.subTest(strategy=strategy):
                drafts = _chunk_document(document, 512, 50, strategy=strategy)
                # 이 전략들은 겹침이 없어 트림 틈이 실제로 생긴다(재현 조건 고정).
                self.assertTrue(self._char_gaps(drafts), "틈이 안 생기면 회귀가 무의미")
                self._assert_raw_span_contract(document, drafts)

    def test_raw_spans_survive_subsection_recursive_split(self):
        """섹션이 chunk_size 를 넘어 하위 재귀분할될 때도 섹션 경계가 이어져야 한다."""
        document = _document(
            "big",
            "\n\n".join(f"## {i}장\n" + ("이 절의 설명 문장입니다. " * 40) for i in range(1, 5)),
        )

        drafts = _chunk_document(document, 200, 20, strategy="markdown_recursive")

        self.assertGreater(len(drafts), 4, "하위분할이 일어나야 의미 있는 케이스")
        self._assert_raw_span_contract(document, drafts)

    def test_gapless_strategies_keep_raw_spans_gapless(self):
        document = _document(
            "guide",
            "# 설치\n" + ("설치 설명 문장입니다. " * 20) + "\n\n## Windows\n"
            + ("PowerShell 설명입니다. " * 20),
        )

        for strategy in ("fixed", "recursive", "recursive_sentence"):
            with self.subTest(strategy=strategy):
                drafts = _chunk_document(document, 200, 20, strategy=strategy)
                self.assertEqual(self._char_gaps(drafts), [])
                self._assert_raw_span_contract(document, drafts)

    def test_math_sections_keep_raw_spans_contiguous(self):
        document = _document(
            "math",
            "\n".join(["# 미적분", "핵심 정의입니다.", "1. 첫 문제입니다.", "2. 둘째 문제입니다."]),
        )
        document.metadata["document_type"] = "math"

        drafts = _chunk_document(document, 120, 20, strategy="markdown_recursive")

        self._assert_raw_span_contract(document, drafts)

    def test_default_chunk_strategy_is_fixed(self):
        document = _document(
            "guide",
            "# 설치\n" + ("설치 설명 문장입니다. " * 8),
        )

        drafts = _chunk_document(document, 40, 8)

        self.assertTrue(drafts)
        self.assertTrue(all(chunk.section is None for chunk in drafts))
        self.assertTrue(all(len(chunk.text) <= 40 for chunk in drafts))


class IndexRunTests(unittest.TestCase):
    def _state(self) -> AgentDoctorState:
        state = AgentDoctorState()
        state.index_config.update(
            {
                "chunk_size": 60,
                "chunk_overlap": 10,
                "embedding_model": "test-model",
                "embedding_dimension": 4,
                "graph_enabled": False,
            }
        )
        return state

    def test_graph_usage_summary_runs_when_graph_builder_fails(self):
        state = self._state()
        state.documents = [_document("doc-1", "그래프 생성 실패 테스트 문서입니다.")]
        state.index_config.update(
            {
                "graph_enabled": True,
                "corpus_visualization_enabled": False,
            }
        )
        tools = IndexTools(
            get_retriever=lambda *_args, **_kwargs: Mock(),
            embed=lambda _text, **_kwargs: [1.0, 0.0, 0.0, 0.0],
            count_tokens=lambda _text, **_kwargs: 3,
            build_sparse_vector=lambda _text: {"indices": [], "values": []},
            build_graph_artifacts=Mock(side_effect=RuntimeError("graph boom")),
        )
        baseline = {
            "calls": 0,
            "prompt": 0,
            "output": 0,
            "cost": 0.0,
            "unpriced_calls": 0,
        }

        with (
            patch("agents.index.agent.snapshot_usage", return_value=baseline),
            patch("agents.index.agent.print_summary") as summary,
        ):
            result = run(state, tools=tools)

        self.assertEqual(result.status, "error")
        self.assertIn("graph boom", result.error)
        summary.assert_called_once_with(
            tag="Index",
            stage="그래프 생성",
            since=baseline,
        )

    @patch("agents.index.agent.get_retriever")
    @patch("agents.index.agent.embed_batch", None)  # 단건 embed 폴백 경로로 강제
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_run_validates_deduplicates_and_writes_metadata(
        self, mock_embed, mock_get_retriever
    ):
        state = self._state()
        content = "# 규정\n재택근무는 주 2일까지 가능합니다."
        state.documents = [_document("doc-1", content), _document("doc-2", content)]
        state.index_config["use_hybrid"] = True
        state.index_config["chunk_strategy"] = "markdown_recursive"

        result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertIsNone(result.error)
        self.assertEqual(result.index_artifacts["documents"], 1)
        self.assertTrue(result.chunks)
        self.assertEqual(result.chunks[0].section, "규정")
        self.assertEqual(
            content[result.chunks[0].char_span[0] : result.chunks[0].char_span[1]],
            result.chunks[0].text,
        )
        self.assertGreater(result.chunks[0].token_count, 0)
        self.assertEqual(len(result.chunks[0].hash), 16)
        self.assertEqual(result.chunks[0].metadata["embedding_model"], "test-model")
        self.assertIsNotNone(result.chunks[0].sparse_vector)
        self.assertEqual(mock_embed.call_count, len(result.chunks))
        mock_get_retriever.assert_called_once()

    @patch("agents.index.agent.get_retriever")
    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_same_signature_reuses_embeddings(
        self, first_embed, _mock_get_retriever
    ):
        state = self._state()
        state.documents = [_document("doc-1", "동일한 문서 본문입니다.")]
        first = run(state)
        self.assertEqual(first_embed.call_count, 1)

        with patch("agents.index.agent.embed") as second_embed:
            second = run(first)

        self.assertEqual(second.status, "indexed")
        self.assertEqual(second.index_artifacts["reused_embeddings"], 1)
        second_embed.assert_not_called()

    def test_reused_chunks_refresh_retrieval_metadata(self):
        state = self._state()
        state.documents = [_document("doc-1", "metadata refresh target")]
        first = run(state, tools=_index_tools())

        first.index_config["top_k"] = 9
        first.index_config["use_reranker"] = True
        first.index_config["rerank_candidates"] = 40
        second = run(first, tools=_index_tools())

        self.assertEqual(second.status, "indexed")
        self.assertEqual(second.index_artifacts["reused_embeddings"], 1)
        self.assertEqual(second.chunks[0].metadata["top_k"], 9)
        self.assertTrue(second.chunks[0].metadata["use_reranker"])
        self.assertEqual(
            second.chunks[0].metadata["reranker_model"],
            "BAAI/bge-reranker-v2-m3",
        )
        self.assertEqual(second.chunks[0].metadata["rerank_candidates"], 40)

    def test_model_recovery_reembeds_fallback_chunks(self):
        # 리뷰 회귀: 최초 색인이 fallback(해시 벡터)으로 이뤄진 뒤 모델이 복구되면,
        # 같은 문서·설정으로 재색인해도 fallback 청크를 그대로 재사용하면 안 되고
        # 실제 모델 벡터로 강제 재임베딩해야 한다(문서·질의 벡터 공간 불일치 방지).
        FALLBACK_VEC = [0.5, 0.5, 0.5, 0.5]
        REAL_VEC = [1.0, 0.0, 0.0, 0.0]

        def _tools(is_fallback: bool) -> IndexTools:
            vec = FALLBACK_VEC if is_fallback else REAL_VEC
            return IndexTools(
                get_retriever=lambda *_a, **_k: Mock(),
                embed=lambda _t, **_k: list(vec),
                count_tokens=lambda _t, **_k: 3,
                build_sparse_vector=lambda _t: {"indices": [], "values": []},
                build_graph_artifacts=lambda _c, _cfg: {},
                embed_batch=lambda texts, **_k: [list(vec) for _ in texts],
                embedding_is_fallback=lambda *_a, **_k: is_fallback,
            )

        state = self._state()
        state.documents = [_document("doc-1", "복구 대상 문서 본문입니다.")]

        # 1) 모델 로드 실패 상태로 색인 → fallback 벡터 + provenance 기록
        first = run(state, tools=_tools(is_fallback=True))
        self.assertEqual(first.status, "indexed")
        self.assertTrue(first.chunks[0].metadata["embedding_fallback"])
        self.assertEqual(first.chunks[0].embedding, FALLBACK_VEC)

        # 2) 모델 복구 후 재색인 → fallback 청크는 실제 벡터로 재임베딩, 캐시 reset
        with patch("agents.index.agent.reset_retriever_cache") as mock_reset:
            second = run(first, tools=_tools(is_fallback=False))

        self.assertEqual(second.status, "indexed")
        # fallback 이었으므로 재사용이 아니라 재임베딩돼야 한다.
        self.assertEqual(second.index_artifacts["reused_embeddings"], 0)
        self.assertEqual(second.index_artifacts["reembedded_fallback"], len(second.chunks))
        self.assertEqual(second.chunks[0].embedding, REAL_VEC)
        self.assertFalse(second.chunks[0].metadata["embedding_fallback"])
        mock_reset.assert_called_once()

    def test_fallback_flag_survives_repeated_failures_then_recovery(self):
        # 리뷰 회귀(@SeonUI): 재사용 경로가 embedding_fallback 을 이어주지 않으면,
        # 모델이 두 번 연속 실패하는 사이에 플래그가 기본값 False 로 덮여
        # 이후 모델이 복구돼도 재임베딩 대상으로 잡히지 않는다(해시 벡터 영구 고착).
        # 위 2회 테스트는 이 경로를 지나지 않으므로 3회(실패→실패→복구)로 검증한다.
        FALLBACK_VEC = [0.5, 0.5, 0.5, 0.5]
        REAL_VEC = [1.0, 0.0, 0.0, 0.0]

        def _tools(is_fallback: bool) -> IndexTools:
            vec = FALLBACK_VEC if is_fallback else REAL_VEC
            return IndexTools(
                get_retriever=lambda *_a, **_k: Mock(),
                embed=lambda _t, **_k: list(vec),
                count_tokens=lambda _t, **_k: 3,
                build_sparse_vector=lambda _t: {"indices": [], "values": []},
                build_graph_artifacts=lambda _c, _cfg: {},
                embed_batch=lambda texts, **_k: [list(vec) for _ in texts],
                embedding_is_fallback=lambda *_a, **_k: is_fallback,
            )

        state = self._state()
        state.documents = [_document("doc-1", "두 번 실패 후 복구되는 문서 본문입니다.")]

        # 1) 모델 실패 → fallback 벡터로 색인되고 provenance 기록
        first = run(state, tools=_tools(is_fallback=True))
        self.assertTrue(first.chunks[0].metadata["embedding_fallback"])

        # 2) 모델 여전히 실패 → 임베딩 재사용 경로. 여기서 플래그가 유실되면 안 된다.
        second = run(first, tools=_tools(is_fallback=True))
        self.assertEqual(second.index_artifacts["reused_embeddings"], len(second.chunks))
        self.assertTrue(
            second.chunks[0].metadata["embedding_fallback"],
            "재사용 경로가 embedding_fallback 을 유실하면 복구 후 재임베딩이 동작하지 않는다",
        )
        self.assertEqual(second.chunks[0].embedding, FALLBACK_VEC)

        # 3) 모델 복구 → 2회차를 거쳤어도 여전히 재임베딩 대상이어야 한다
        with patch("agents.index.agent.reset_retriever_cache") as mock_reset:
            third = run(second, tools=_tools(is_fallback=False))

        self.assertEqual(third.status, "indexed")
        self.assertEqual(third.index_artifacts["reused_embeddings"], 0)
        self.assertEqual(third.index_artifacts["reembedded_fallback"], len(third.chunks))
        self.assertEqual(third.chunks[0].embedding, REAL_VEC)
        self.assertFalse(third.chunks[0].metadata["embedding_fallback"])
        mock_reset.assert_called_once()

    def test_recovered_model_still_reuses_non_fallback_chunks(self):
        # 정상(fallback 아님) 벡터로 색인된 청크는 모델이 로드 가능해도 그대로 재사용한다
        # (재임베딩은 fallback provenance 가 있는 청크에만 적용).
        def _tools() -> IndexTools:
            return IndexTools(
                get_retriever=lambda *_a, **_k: Mock(),
                embed=lambda _t, **_k: [1.0, 0.0, 0.0, 0.0],
                count_tokens=lambda _t, **_k: 3,
                build_sparse_vector=lambda _t: {"indices": [], "values": []},
                build_graph_artifacts=lambda _c, _cfg: {},
                embed_batch=lambda texts, **_k: [[1.0, 0.0, 0.0, 0.0] for _ in texts],
                embedding_is_fallback=lambda *_a, **_k: False,
            )

        state = self._state()
        state.documents = [_document("doc-1", "정상 색인 문서 본문입니다.")]
        first = run(state, tools=_tools())
        self.assertFalse(first.chunks[0].metadata["embedding_fallback"])

        with patch("agents.index.agent.reset_retriever_cache") as mock_reset:
            second = run(first, tools=_tools())

        self.assertEqual(second.index_artifacts["reused_embeddings"], len(second.chunks))
        self.assertEqual(second.index_artifacts["reembedded_fallback"], 0)
        mock_reset.assert_not_called()

    def test_runtime_only_config_change_skips_reindex_work(self):
        state = self._state()
        state.documents = [_document("doc-1", "기존 문서")]
        state.chunks = [
            Chunk("c1", "doc-1", "기존 문서", embedding=[1.0, 0.0, 0.0, 0.0])
        ]
        state.reindex_required = False
        tools = IndexTools(
            get_retriever=Mock(),
            embed=Mock(),
            count_tokens=Mock(),
            build_sparse_vector=Mock(),
            build_graph_artifacts=Mock(),
        )

        result = run(state, tools=tools)

        self.assertEqual(result.status, "indexed")
        self.assertTrue(result.index_artifacts["reindex_skipped"])
        self.assertTrue(result.reindex_required)
        tools.get_retriever.assert_not_called()
        tools.embed.assert_not_called()

    def test_reused_chunks_still_seed_chunk_deduplication(self):
        state = self._state()
        state.index_config.update(
            {
                "chunk_size": 6,
                "chunk_overlap": 0,
                "chunk_stage": 1,
            }
        )
        state.documents = [_document("doc-a", "shared")]
        first = run(state, tools=_index_tools())

        first.documents = [
            _document("doc-a", "shared"),
            _document("doc-b", "sharedzz"),
        ]
        second = run(first, tools=_index_tools())

        self.assertEqual(second.status, "indexed")
        self.assertEqual(second.index_artifacts["reused_embeddings"], 1)
        self.assertEqual(
            [(chunk.doc_id, chunk.text) for chunk in second.chunks],
            [("doc-a", "shared"), ("doc-b", "zz")],
        )

    def test_invalid_overlap_returns_error_state(self):
        state = self._state()
        state.documents = [_document("doc-1", "본문")]
        state.index_config.update({"chunk_size": 10, "chunk_overlap": 10})

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("chunk_overlap", result.error)

    def test_unknown_chunk_strategy_returns_error_state(self):
        state = self._state()
        state.documents = [_document("doc-1", "본문")]
        state.index_config["chunk_strategy"] = "unknown"

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("chunk_strategy", result.error)

    @patch("agents.index.agent.get_retriever")
    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_chunk_stage_config_overrides_default_strategy(
        self, _mock_embed, _mock_get_retriever
    ):
        state = self._state()
        state.documents = [_document("doc-1", "# 제목\n" + ("본문입니다. " * 8))]
        state.index_config["chunk_stage"] = 1

        result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertEqual(result.index_artifacts["chunk_strategy"], "fixed")
        self.assertTrue(all(chunk.section is None for chunk in result.chunks))

    def test_run_accepts_swapped_index_tools(self):
        state = self._state()
        state.documents = [_document("doc-1", "도구 교체 테스트 본문입니다.")]
        upserted: list[list[Chunk]] = []
        tools = IndexTools(
            get_retriever=lambda chunks, *_args, **_kwargs: upserted.append(chunks),
            embed=lambda _text, **_kwargs: [0.0, 1.0, 0.0, 0.0],
            count_tokens=lambda _text, **_kwargs: 7,
            build_sparse_vector=lambda _text: {"indices": [], "values": []},
            build_graph_artifacts=lambda _chunks, _config: {},
        )

        result = run(state, tools=tools)

        self.assertEqual(result.status, "indexed")
        self.assertEqual(result.chunks[0].embedding, [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(result.chunks[0].token_count, 7)
        self.assertEqual(len(upserted[0]), len(result.chunks))

    def test_blank_document_fails_pydantic_validation(self):
        state = self._state()
        state.documents = [_document("doc-1", "   \n")]

        result = run(state)

        self.assertEqual(result.status, "error")
        self.assertIn("문서 검증 실패", result.error)

    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_same_doc_id_with_different_content_skips_conflicting_document(self, _mock_embed):
        # 충돌 문서만 건너뛰고 먼저 들어온 문서는 정상 인덱싱되어야 한다.
        state = self._state()
        state.documents = [
            _document("same-id", "첫 번째 본문"),
            _document("same-id", "두 번째 본문"),
        ]

        result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "첫 번째 본문")
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["doc_id"], "same-id")
        self.assertIn("같은 doc_id", failed[0]["error"])

    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_partial_failure_preserves_valid_documents(self, _mock_embed):
        # 불량 문서 1개가 나머지 정상 문서들의 작업을 버리게 만들면 안 된다.
        state = self._state()
        state.documents = [
            _document("doc-ok-1", "정상 문서 본문입니다."),
            _document("doc-bad", "   \n"),  # 공백뿐 → pydantic 검증 실패
            _document("doc-ok-2", "또 다른 정상 문서 본문입니다."),
        ]

        result = run(state)

        self.assertEqual(result.status, "indexed")
        indexed_doc_ids = {chunk.doc_id for chunk in result.chunks}
        self.assertEqual(indexed_doc_ids, {"doc-ok-1", "doc-ok-2"})
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual([f["doc_id"] for f in failed], ["doc-bad"])

    def test_failed_document_does_not_pollute_chunk_dedup(self):
        # 임베딩 도중 실패한 문서의 청크 해시가 dedup 집합에 남으면
        # 뒤에 오는 동일 텍스트 청크가 중복으로 오인되어 누락된다.
        state = self._state()
        shared_text = "실패 후에도 인덱싱되어야 하는 본문입니다."
        state.documents = [
            _document("doc-fails", shared_text),
            _document("doc-succeeds", shared_text),
        ]

        calls = {"n": 0}

        def flaky_embed(text, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("임베딩 일시 실패")
            return [1.0, 0.0, 0.0, 0.0]

        with patch("agents.index.agent.embed_batch", None), \
                patch("agents.index.agent.embed", side_effect=flaky_embed):
            result = run(state)

        self.assertEqual(result.status, "indexed")
        self.assertEqual({c.doc_id for c in result.chunks}, {"doc-succeeds"})
        self.assertEqual(result.chunks[0].text, shared_text)
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual([f["doc_id"] for f in failed], ["doc-fails"])

    # ── dedup 이 버린 자리를 생존 청크가 대표한다(duplicate_spans) ──────
    #
    # dedup 은 본문 완전일치만 버리므로 버려진 글자는 생존 청크에 그대로 살아 있다.
    # 그런데 좌표 지도에서는 그 자리가 빈 채로 남아, 근거를 실제로 다 본 답이
    # span_recall 0 으로 떨어졌다. Index 가 버릴 때 좌표를 적어 두면 Eval 이 메꾼다.

    _DUP_SECTION = "## 안내\n\n재택근무는 주 2일까지 허용하며 팀장과 협의해 요일을 정한다."

    def _duplicated_document(self, doc_id: str = "doc-dup"):
        content = (
            f"{self._DUP_SECTION}\n\n"
            "## 서론\n\n이 문서는 사내 규정을 설명한다.\n\n"
            f"{self._DUP_SECTION}\n\n"
            "## 결론\n\n규정은 매년 갱신한다."
        )
        return _document(doc_id, content), content

    def test_dropped_duplicate_chunk_records_its_span_on_the_survivor(self):
        document, content = self._duplicated_document()
        state = self._state()
        state.documents = [document]
        state.index_config["chunk_strategy"] = "markdown"

        result = run(state, tools=_index_tools())

        self.assertEqual(result.status, "indexed")
        survivor = next(c for c in result.chunks if c.text.startswith("## 안내"))
        # 중복이라 빠진 두 번째 '안내' 절의 자리가 생존 청크에 적혀야 한다.
        self.assertTrue(survivor.duplicate_spans, "dedup 이 버린 자리가 기록되지 않았다")
        alias_doc, alias_start, alias_end = survivor.duplicate_spans[0]
        self.assertEqual(alias_doc, "doc-dup")
        # 좌표는 트림 전 기준이고, 그 구간의 본문이 곧 생존 청크의 본문이다.
        self.assertEqual(content[alias_start:alias_end].strip(), survivor.text)
        self.assertGreaterEqual(alias_end - alias_start, len(survivor.text))
        # metadata 로도 나가야 payload 왕복(qdrant·chunks.json)에서 안 사라진다.
        self.assertEqual(
            survivor.metadata["duplicate_spans"],
            [list(span) for span in survivor.duplicate_spans],
        )

    def test_unique_chunks_carry_no_duplicate_spans(self):
        # 대개의 문서는 중복이 없다 — 그때는 필드도 payload 키도 안 생겨야 한다.
        state = self._state()
        state.documents = [_document("doc-1", "중복이 없는 평범한 문서 본문입니다.")]

        result = run(state, tools=_index_tools())

        for chunk in result.chunks:
            self.assertEqual(chunk.duplicate_spans, [])
            self.assertNotIn("duplicate_spans", chunk.metadata)

    def test_alias_survives_embedding_reuse(self):
        # 재색인 때 별칭을 안 물려주면 dedup 구멍이 되살아난다.
        document, _content = self._duplicated_document()
        state = self._state()
        state.documents = [document]
        state.index_config["chunk_strategy"] = "markdown"
        first = run(state, tools=_index_tools())
        before = next(c for c in first.chunks if c.text.startswith("## 안내"))

        second = run(first, tools=_index_tools())

        self.assertEqual(second.index_artifacts["reused_embeddings"], len(first.chunks))
        after = next(c for c in second.chunks if c.text.startswith("## 안내"))
        self.assertEqual(after.duplicate_spans, before.duplicate_spans)

    def test_cross_document_duplicate_records_the_other_documents_span(self):
        # 문서 간 dedup — 별칭의 doc_id 는 청크 제 doc_id 가 아니라 버려진 쪽이다.
        # 두 문서 전체가 같으면 문서 단위 dedup 이 먼저 걸리므로, 절 하나만 겹치게 둔다.
        shared_section = "## 안내\n\n재택근무는 주 2일까지 허용한다."
        second_content = f"## 서론\n\n이 문서는 규정을 설명한다.\n\n{shared_section}"
        state = self._state()
        state.index_config["chunk_strategy"] = "markdown"
        state.documents = [
            _document("doc-first", shared_section),
            _document("doc-second", second_content),
        ]

        result = run(state, tools=_index_tools())

        survivor = next(c for c in result.chunks if c.doc_id == "doc-first")
        self.assertEqual([span[0] for span in survivor.duplicate_spans], ["doc-second"])
        _alias_doc, alias_start, alias_end = survivor.duplicate_spans[0]
        self.assertEqual(second_content[alias_start:alias_end].strip(), survivor.text)

    def test_failed_document_does_not_leave_alias_spans(self):
        # 실패한 문서의 좌표를 남기면, 색인되지도 않은 구간을 Eval 이 '덮였다'고 센다.
        # 앞 테스트와 같은 구성(절 하나가 겹침)이라 성공했다면 별칭이 남았을 상황이다.
        shared_section = "## 안내\n\n재택근무는 주 2일까지 허용한다."
        state = self._state()
        state.index_config["chunk_strategy"] = "markdown"
        state.documents = [
            _document("doc-ok", shared_section),
            _document("doc-fails", f"## 서론\n\n이 문서는 규정을 설명한다.\n\n{shared_section}"),
        ]

        calls = {"n": 0}

        def flaky_embed(_text, **_kwargs):
            calls["n"] += 1
            if calls["n"] > 1:          # 첫 문서만 통과시킨다
                raise RuntimeError("임베딩 일시 실패")
            return [1.0, 0.0, 0.0, 0.0]

        tools = IndexTools(
            get_retriever=lambda *_args, **_kwargs: Mock(),
            embed=flaky_embed,
            count_tokens=lambda _text, **_kwargs: 3,
            build_sparse_vector=lambda _text: {"indices": [], "values": []},
            build_graph_artifacts=lambda _chunks, _config: {},
        )

        result = run(state, tools=tools)

        self.assertEqual(result.status, "indexed")
        self.assertEqual({c.doc_id for c in result.chunks}, {"doc-ok"})
        self.assertEqual(
            [f["doc_id"] for f in result.index_artifacts["failed_documents"]],
            ["doc-fails"],
        )
        for chunk in result.chunks:
            self.assertEqual(chunk.duplicate_spans, [])


class SearchAndGraphTests(unittest.TestCase):
    def test_qdrant_dense_search_round_trip(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        chunks = [
            Chunk(
                chunk_id="remote",
                doc_id="policy",
                text="재택근무 규정",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                chunk_id="vacation",
                doc_id="policy",
                text="연차 규정",
                embedding=[0.0, 1.0],
            ),
        ]
        upsert_chunks(client, chunks)

        results = search(client, [1.0, 0.0], top_k=1)

        self.assertEqual(results[0]["chunk_id"], "remote")

        delete_document_chunks(client, ["policy"])
        self.assertEqual(search(client, [1.0, 0.0], top_k=5), [])

    @patch("agents.index.qdrant_store.search")
    def test_hybrid_search_recovers_exact_keyword(self, dense_search):
        dense_search.return_value = [
            {
                "chunk_id": "dense",
                "doc_id": "d1",
                "text": "근무 제도 안내",
                "metadata": {},
                "score": 0.9,
                "section": None,
            }
        ]
        chunks = [
            {
                "chunk_id": "keyword",
                "doc_id": "d2",
                "text": "RAGAS Oracle Test 설정",
                "metadata": {},
            }
        ]

        results = hybrid_search(
            Mock(),
            query_vector=[1.0],
            query="Oracle Test",
            chunks=chunks,
            top_k=2,
            dense_weight=0.5,
        )

        self.assertEqual({item["chunk_id"] for item in results}, {"dense", "keyword"})

    def test_native_hybrid_search_uses_qdrant_sparse_vector(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        chunks = [
            Chunk(
                chunk_id="dense",
                doc_id="d1",
                text="semantic policy guide",
                embedding=[1.0, 0.0],
                sparse_vector=build_sparse_vector("semantic policy guide"),
            ),
            Chunk(
                chunk_id="keyword",
                doc_id="d2",
                text="RAGAS Oracle Test setting",
                embedding=[0.0, 1.0],
                sparse_vector=build_sparse_vector("RAGAS Oracle Test setting"),
            ),
        ]
        upsert_chunks(client, chunks)

        with patch(
            "agents.index.qdrant_store.search",
            side_effect=AssertionError("local fusion used"),
        ):
            results = hybrid_search(
                client,
                query_vector=[1.0, 0.0],
                query="Oracle Test",
                chunks=[],
                top_k=2,
                dense_weight=0.5,
            )

        self.assertEqual({item["chunk_id"] for item in results}, {"dense", "keyword"})

    def test_native_hybrid_search_is_limited_to_retrieval_scope(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        chunks = [
            Chunk(
                chunk_id="shared",
                doc_id="doc-a",
                text="Oracle Test from corpus A",
                embedding=[0.0, 1.0],
                sparse_vector=build_sparse_vector("Oracle Test from corpus A"),
                metadata={"retrieval_scope_id": "scope-a"},
            ),
            Chunk(
                chunk_id="shared",
                doc_id="doc-b",
                text="Oracle Test from corpus B",
                embedding=[1.0, 0.0],
                sparse_vector=build_sparse_vector("Oracle Test from corpus B"),
                metadata={"retrieval_scope_id": "scope-b"},
            ),
        ]
        upsert_chunks(client, chunks)

        results = hybrid_search(
            client,
            query_vector=[1.0, 0.0],
            query="Oracle Test",
            chunks=[],
            top_k=2,
            dense_weight=0.5,
            retrieval_scope_id="scope-b",
        )

        self.assertEqual([item["doc_id"] for item in results], ["doc-b"])

    def test_point_id_includes_scope_and_delete_respects_scope(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        chunks = [
            Chunk(
                chunk_id="same-chunk",
                doc_id="same-doc",
                text="first corpus",
                embedding=[1.0, 0.0],
                metadata={"retrieval_scope_id": "scope-a"},
            ),
            Chunk(
                chunk_id="same-chunk",
                doc_id="same-doc",
                text="second corpus",
                embedding=[0.0, 1.0],
                metadata={"retrieval_scope_id": "scope-b"},
            ),
        ]
        upsert_chunks(client, chunks)

        self.assertEqual(len(search(client, [1.0, 0.0], top_k=5)), 2)

        delete_document_chunks(client, ["same-doc"], retrieval_scope_id="scope-a")

        remaining = search(client, [0.0, 1.0], top_k=5)
        self.assertEqual([item["retrieval_scope_id"] for item in remaining], ["scope-b"])

    def test_legacy_dense_only_collection_uses_dense_upsert_and_search(self):
        client = build_client(":memory:")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        ensure_collection(client, vector_dim=2)
        upsert_chunks(
            client,
            [
                Chunk(
                    chunk_id="legacy",
                    doc_id="doc",
                    text="legacy dense collection",
                    embedding=[1.0, 0.0],
                    sparse_vector=build_sparse_vector("legacy dense collection"),
                )
            ],
        )

        results = search(client, [1.0, 0.0], top_k=1)

        self.assertEqual(results[0]["chunk_id"], "legacy")

    def test_legacy_dense_search_uses_query_points_without_search_api(self):
        client = build_client(":memory:")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )
        ensure_collection(client, vector_dim=2)
        upsert_chunks(
            client,
            [
                Chunk(
                    chunk_id="legacy",
                    doc_id="doc",
                    text="legacy dense collection",
                    embedding=[1.0, 0.0],
                )
            ],
        )

        real_query_points = client.query_points
        calls = []

        def spy_query_points(**kwargs):
            calls.append(kwargs)
            return real_query_points(**kwargs)

        with patch.object(client, "query_points", side_effect=spy_query_points):
            results = search(client, [1.0, 0.0], top_k=1)

        self.assertEqual(results[0]["chunk_id"], "legacy")
        self.assertTrue(calls)
        self.assertNotIn("using", calls[0])

    def test_dense_search_rechecks_shape_cache_after_collection_migration(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        upsert_chunks(
            client,
            [
                Chunk(
                    chunk_id="native",
                    doc_id="doc",
                    text="native hybrid collection",
                    embedding=[1.0, 0.0],
                    sparse_vector=build_sparse_vector("native hybrid collection"),
                )
            ],
        )
        qdrant_store._collection_native_hybrid_cache[client] = False

        results = search(client, [1.0, 0.0], top_k=1)

        self.assertEqual(results[0]["chunk_id"], "native")
        self.assertTrue(qdrant_store._collection_has_native_hybrid(client))

    def test_upsert_rechecks_shape_after_transient_probe_failure(self):
        client = build_client(":memory:")
        ensure_collection(client, vector_dim=2)
        chunk = Chunk(
            chunk_id="native",
            doc_id="doc",
            text="native hybrid collection",
            embedding=[1.0, 0.0],
            sparse_vector=build_sparse_vector("native hybrid collection"),
        )
        qdrant_store._clear_collection_shape_cache(client)
        real_get_collection = client.get_collection
        calls = {"n": 0}

        def flaky_get_collection(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("temporary qdrant probe failure")
            return real_get_collection(*args, **kwargs)

        with patch.object(client, "get_collection", side_effect=flaky_get_collection):
            upsert_chunks(client, [chunk])
            upsert_chunks(client, [chunk])

        results = search(client, [1.0, 0.0], top_k=1)
        self.assertEqual(results[0]["chunk_id"], "native")
        self.assertTrue(qdrant_store._collection_has_native_hybrid(client))

    def test_recreate_legacy_collection_logs_shape_migration(self):
        client = build_client(":memory:")
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=2, distance=Distance.COSINE),
        )

        with patch("builtins.print") as print_mock:
            ensure_collection(client, vector_dim=2, recreate_on_mismatch=True)

        self.assertTrue(
            any("legacy dense-only 컬렉션 재생성" in str(call) for call in print_mock.call_args_list)
        )

    def test_graph_artifacts_are_written(self):
        chunk = Chunk(
            chunk_id="doc_chunk_000",
            doc_id="doc",
            text="Qdrant는 벡터 검색과 metadata filter를 지원한다.",
            embedding=[1.0, 0.0],
            metadata={"title": "설계"},
        )
        with tempfile.TemporaryDirectory() as directory:
            artifacts = build_graph_artifacts(
                [chunk],
                {
                    "graph_extraction": "keyword",
                    "graph_output_dir": directory,
                    "graph_similarity_threshold": 0.9,
                },
            )

            self.assertTrue(Path(artifacts["graphml"]).exists())
            self.assertTrue(Path(artifacts["mermaid"]).exists())
            self.assertGreater(artifacts["graph_nodes"], 2)


class ModelLoadCooldownTests(unittest.TestCase):
    """로드 실패를 영구 캐시하지 않고 쿨다운 후 재시도하는지 확인(embedding·reranker).

    sentence_transformers import 를 None 으로 막아 로드를 실패시키고, time.monotonic 을
    가짜로 진행시켜 '쿨다운 중에는 재시도 안 함 / 지나면 재시도함'을 검증한다.
    """

    def setUp(self):
        import agents.index.qdrant_store as store
        self.store = store
        # 각 테스트가 깨끗한 실패 캐시에서 시작하도록 초기화한다.
        store._failed_models.clear()
        store._failed_rerankers.clear()
        # 임베딩 모델 캐시 키는 (model_name, device) 다 — 장치가 달라도 같은 모델을
        # 동시에 들고 있어야 하기 때문. 리랭커는 아직 이름만으로 캐시한다.
        store._models.pop(("m", "cpu"), None)
        store._rerankers.pop("m", None)
        self.addCleanup(store._failed_models.clear)
        self.addCleanup(store._failed_rerankers.clear)

    def test_embedding_model_retries_after_cooldown(self):
        store = self.store
        # import 를 막아 로드를 실패시킨다.
        with patch.dict(sys.modules, {"sentence_transformers": None}), \
             patch.object(store, "_FAILED_MODEL_RETRY_SEC", 300.0), \
             patch.object(store.time, "monotonic") as clock:
            # device 를 고정한다 — 생략하면 실행 머신의 CUDA 유무로 캐시 키가
            # 달라져 테스트가 환경 의존이 된다.
            key = ("m", "cpu")
            clock.return_value = 1000.0
            self.assertIsNone(store._get_embedding_model("m", device="cpu"))
            self.assertIn(key, store._failed_models)

            # 쿨다운 중(=재시도 안 함): 실패 시각이 그대로여야 한다.
            clock.return_value = 1100.0   # +100s < 300s
            first_failed_at = store._failed_models[key]
            self.assertIsNone(store._get_embedding_model("m", device="cpu"))
            self.assertEqual(store._failed_models[key], first_failed_at)

            # 쿨다운 경과 후: 재시도하여 실패 시각이 갱신된다.
            clock.return_value = 1400.0   # +400s > 300s
            self.assertIsNone(store._get_embedding_model("m", device="cpu"))
            self.assertEqual(store._failed_models[key], 1400.0)

    def test_reranker_retries_after_cooldown(self):
        store = self.store
        results = [{"chunk_id": "c1", "text": "t1", "score": 0.5}]
        with patch.dict(sys.modules, {"sentence_transformers": None}), \
             patch.object(store, "_FAILED_RERANKER_RETRY_SEC", 300.0), \
             patch.object(store.time, "monotonic") as clock:
            clock.return_value = 1000.0
            store.rerank("q", results, model_name="m", top_k=5)
            self.assertIn("m", store._failed_rerankers)
            first_failed_at = store._failed_rerankers["m"]

            # 쿨다운 중: 실패 시각 유지(재시도 안 함).
            clock.return_value = 1100.0
            store.rerank("q", results, model_name="m", top_k=5)
            self.assertEqual(store._failed_rerankers["m"], first_failed_at)

            # 쿨다운 경과 후: 재시도로 실패 시각 갱신.
            clock.return_value = 1400.0
            store.rerank("q", results, model_name="m", top_k=5)
            self.assertEqual(store._failed_rerankers["m"], 1400.0)


class TransientFailureStatusTests(unittest.TestCase):
    """일시적 실패로 문서가 빠진 색인은 "완료" 가 아니다.

    영구 실패(빈 문서·doc_id 충돌)와 구분한다 — 그런 파일이 코퍼스에 하나 섞여 있으면
    매 실행이 영원히 partial 이 되어 신호가 죽는다."""

    def _state(self):
        from core.state import AgentDoctorState

        return AgentDoctorState(
            documents=[], index_config={"embedding_model": "test-model",
                                        "embedding_dimension": 4},
        )

    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_transient_failure_marks_partial(self, _mock_embed):
        from agents.index import agent as index_agent

        state = self._state()
        state.documents = [
            _document("doc-ok", "정상 문서 본문입니다."),
            _document("doc-flaky", "임베딩 중 5xx 가 나는 문서입니다."),
        ]
        real = index_agent._process_document

        def _flaky(document, **kwargs):
            if document.doc_id == "doc-flaky":
                raise RuntimeError("503 Service Unavailable")
            return real(document, **kwargs)

        with patch.object(index_agent, "_process_document", _flaky):
            result = index_agent.run(state)

        self.assertEqual(result.status, "partial")
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual([f["doc_id"] for f in failed], ["doc-flaky"])
        self.assertTrue(failed[0]["transient"])

    @patch("agents.index.agent.embed_batch", None)
    @patch("agents.index.agent.embed", return_value=[1.0, 0.0, 0.0, 0.0])
    def test_permanent_failure_stays_indexed(self, _mock_embed):
        # 빈 문서는 다시 돌려도 같다. 매번 partial 로 올리면 알람이 무뎌진다.
        state = self._state()
        state.documents = [
            _document("doc-ok", "정상 문서 본문입니다."),
            _document("doc-bad", "   \n"),
        ]

        result = run(state)

        self.assertEqual(result.status, "indexed")
        failed = result.index_artifacts["failed_documents"]
        self.assertEqual([f["doc_id"] for f in failed], ["doc-bad"])
        self.assertFalse(failed[0]["transient"])


if __name__ == "__main__":
    unittest.main()
