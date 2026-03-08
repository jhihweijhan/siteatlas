import app.main  # noqa: F401

from app.routers.rag import (
    build_rag_meta,
    collect_relation_scope_info,
    extract_result_relation_group,
    merge_results_with_image_captions,
)
from app.services.filter_builder import (
    build_combined_where,
    build_image_caption_where,
    build_relation_where,
    build_source_where,
)


def test_extract_result_relation_group_falls_back_domain():
    assert extract_result_relation_group({"relation_group": "group-a"}) == "group-a"
    assert extract_result_relation_group({"domain": "group-b"}) == "group-b"
    assert extract_result_relation_group({}) == "general"


def test_collect_relation_scope_info_handles_legacy_domain_records():
    docs = [
        {"url": "https://a", "relation_group": "edition.cnn.com"},
        {"url": "https://b", "domain": "ball-in.club"},
    ]
    info = collect_relation_scope_info(docs, [])
    assert info["mode"] == "all"
    assert info["known_groups"] == ["ball-in.club", "edition.cnn.com"]
    assert info["in_scope_groups"] == ["ball-in.club", "edition.cnn.com"]


def test_build_relation_where_matches_relation_or_domain():
    assert build_relation_where([]) is None
    assert build_relation_where(["edition.cnn.com"]) == {
        "$or": [{"relation_group": "edition.cnn.com"}, {"domain": "edition.cnn.com"}]
    }


def test_build_source_where_and_combined_where():
    assert build_source_where([]) is None
    assert build_source_where(["https://a"]) == {"source_url": "https://a"}
    assert build_image_caption_where(["https://a"]) == {
        "$and": [{"source_url": "https://a"}, {"type": "image_caption"}]
    }
    assert build_combined_where(["edition.cnn.com"], ["https://a"]) == {
        "$and": [
            {"$or": [{"relation_group": "edition.cnn.com"}, {"domain": "edition.cnn.com"}]},
            {"source_url": "https://a"},
        ]
    }


def test_build_rag_meta_counts_hit_rate():
    results = [
        {
            "metadata": {
                "source_url": "https://a",
                "relation_group": "edition.cnn.com",
            }
        },
        {
            "metadata": {
                "source_url": "https://b",
                "domain": "ball-in.club",
            }
        },
    ]
    scope_info = {
        "mode": "all",
        "in_scope_groups": ["ball-in.club", "edition.cnn.com", "skillsmp.com"],
        "selected_groups": [],
    }
    meta = build_rag_meta(
        results=results,
        top_k=5,
        model="qwen2.5-coder:3b",
        embedding_model="bge-m3:latest",
        scope_info=scope_info,
        selected_source_urls=["https://a"],
    )
    assert meta["retrieved_chunks"] == 2
    assert meta["retrieval_hit_rate"] == 0.4
    assert meta["scope_group_count"] == 3
    assert meta["hit_group_count"] == 2
    assert meta["relation_group_hit_rate"] == 0.6667
    assert meta["selected_source_count"] == 1


def test_merge_results_with_image_captions_deduplicates_and_appends():
    base_results = [
        {
            "id": "doc1_chunk_0",
            "document": "text chunk",
            "metadata": {"type": "text", "source_url": "https://a"},
        },
        {
            "id": "doc1_img_0",
            "document": "[圖片描述] hotel front",
            "metadata": {"type": "image_caption", "source_url": "https://a", "image_index": 0},
        },
    ]
    caption_rows = [
        {
            "id": "doc1_img_0",
            "document": "[圖片描述] hotel front",
            "metadata": {"type": "image_caption", "source_url": "https://a", "image_index": 0},
        },
        {
            "id": "doc1_img_1",
            "document": "[圖片描述] studio scene",
            "metadata": {"type": "image_caption", "source_url": "https://a", "image_index": 1},
        },
    ]

    merged = merge_results_with_image_captions(base_results, caption_rows)
    assert len(merged) == 3
    assert [row["id"] for row in merged] == ["doc1_chunk_0", "doc1_img_0", "doc1_img_1"]


def test_build_rag_meta_hit_rate_is_capped_to_1():
    results = [
        {"metadata": {"source_url": "https://a", "relation_group": "edition.cnn.com"}},
        {"metadata": {"source_url": "https://a", "relation_group": "edition.cnn.com"}},
    ]
    scope_info = {
        "mode": "all",
        "in_scope_groups": ["edition.cnn.com"],
        "selected_groups": [],
    }
    meta = build_rag_meta(
        results=results,
        top_k=1,
        model="qwen2.5-coder:3b",
        embedding_model="bge-m3:latest",
        scope_info=scope_info,
        selected_source_urls=["https://a"],
    )
    assert meta["retrieval_hit_rate"] == 1.0
