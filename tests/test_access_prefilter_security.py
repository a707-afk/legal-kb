"""Security tests for access_prefilter — verify allowed_ids enforcement."""
from __future__ import annotations

from unittest.mock import MagicMock

from llama_index.core.schema import NodeWithScore, TextNode


def _make_result(node_id: str, text: str = "test", score: float = 0.8) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text, id_=node_id), score=score)


class TestAccessPrefilterSecurity:
    """C1: access_prefilter must filter results by allowed_ids."""

    def test_filters_by_allowed_ids(self):
        """Results NOT in allowed_ids must be excluded."""
        from src.access_prefilter import vector_retrieve_access_filtered

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            _make_result("node-1"),
            _make_result("node-2"),
            _make_result("node-3"),
        ]

        # Only allow node-1 and node-3
        allowed = frozenset({"node-1", "node-3"})
        settings = MagicMock()
        results = vector_retrieve_access_filtered(
            mock_index, "query", 5, settings, allowed_ids=allowed
        )

        assert len(results) == 2
        ids = {r.node.node_id for r in results}
        assert ids == {"node-1", "node-3"}
        assert "node-2" not in ids

    def test_empty_allowed_ids_blocks_all(self):
        """Empty allowed_ids set should return no results."""
        from src.access_prefilter import vector_retrieve_access_filtered

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            _make_result("node-1"),
        ]

        settings = MagicMock()
        results = vector_retrieve_access_filtered(
            mock_index, "query", 5, settings, allowed_ids=frozenset()
        )
        assert results == []

    def test_none_allowed_ids_passes_all(self):
        """When allowed_ids is None (no access control), all results pass."""
        from src.access_prefilter import vector_retrieve_access_filtered

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            _make_result("node-1"),
            _make_result("node-2"),
        ]

        settings = MagicMock()
        results = vector_retrieve_access_filtered(
            mock_index, "query", 5, settings, allowed_ids=None
        )
        assert len(results) == 2

    def test_respects_top_k_after_filter(self):
        """After filtering, only top_k results are returned."""
        from src.access_prefilter import vector_retrieve_access_filtered

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        # 10 results, all allowed
        mock_retriever.retrieve.return_value = [
            _make_result(f"node-{i}") for i in range(10)
        ]

        settings = MagicMock()
        all_ids = frozenset(f"node-{i}" for i in range(10))
        results = vector_retrieve_access_filtered(
            mock_index, "query", 3, settings, allowed_ids=all_ids
        )
        assert len(results) == 3
