"""访问控制 Pre-filter 单元测试（不依赖 Chroma / Embedding）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.access_control import can_access_chunk_metadata
from src.access_prefilter import resolve_allowed_node_ids
from src.bm25_store import bm25_search, clear_bm25_memory_cache, persist_bm25_corpus
from src.config_LEGACY_REFERENCE import Settings
from llama_index.core.schema import TextNode


@pytest.fixture
def tiny_bm25_settings(tmp_path: Path) -> Settings:
    corpus = tmp_path / "bm25_tiny.jsonl"
    nodes = [
        TextNode(
            text="public overview",
            id_="n-public",
            metadata={
                "tenant_id": "corp-default",
                "security_level": "public",
                "audience": "all",
            },
        ),
        TextNode(
            text="restricted case refund",
            id_="n-restricted",
            metadata={
                "tenant_id": "corp-default",
                "security_level": "restricted",
                "audience": "support",
            },
        ),
        TextNode(
            text="qa only matrix",
            id_="n-qa",
            metadata={
                "tenant_id": "corp-default",
                "security_level": "internal",
                "audience": "qa",
            },
        ),
    ]
    s = Settings(bm25_corpus_path=str(corpus), hybrid_bm25_enabled=True)
    persist_bm25_corpus(nodes, s)
    clear_bm25_memory_cache()
    return s


def test_support_role_matches_support_agent_audience():
    doc = {
        "tenant_id": "corp-default",
        "security_level": "internal",
        "audience": "support_agent,team_lead",
    }
    assert can_access_chunk_metadata(
        doc, roles=["support"], tenant_id="corp-default", security_clearance=1
    )


def test_can_access_clearance_and_audience():
    pub = {"tenant_id": "corp-default", "security_level": "public", "audience": "all"}
    assert can_access_chunk_metadata(pub, roles=["support"], tenant_id="corp-default", security_clearance=0)
    restr = {"tenant_id": "corp-default", "security_level": "restricted", "audience": "support"}
    assert not can_access_chunk_metadata(restr, roles=["support"], tenant_id="corp-default", security_clearance=0)
    qa_doc = {"tenant_id": "corp-default", "security_level": "internal", "audience": "qa"}
    assert not can_access_chunk_metadata(qa_doc, roles=[], tenant_id="corp-default", security_clearance=3)
    assert can_access_chunk_metadata(qa_doc, roles=["qa"], tenant_id="corp-default", security_clearance=2)


def test_resolve_allowed_node_ids(tiny_bm25_settings: Settings):
    low = resolve_allowed_node_ids(
        tiny_bm25_settings,
        roles=["support"],
        tenant_id="corp-default",
        security_clearance=0,
    )
    assert low == frozenset({"n-public"})

    high = resolve_allowed_node_ids(
        tiny_bm25_settings,
        roles=["support"],
        tenant_id="corp-default",
        security_clearance=5,
    )
    assert "n-restricted" in high
    assert "n-qa" not in high


def test_bm25_search_respects_allowed_ids(tiny_bm25_settings: Settings):
    allowed = frozenset({"n-public"})
    hits = bm25_search(
        tiny_bm25_settings, "overview public", 5, allowed_ids=allowed
    )
    assert hits
    assert all(nid == "n-public" for nid, _ in hits)
