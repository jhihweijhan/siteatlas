from app.services.filter_builder import (
    build_combined_where,
    build_relation_where,
    build_source_where,
    build_tags_where,
)


# --- build_relation_where ---


def test_build_relation_where_single_group():
    result = build_relation_where(["edition.cnn.com"])
    assert result == {
        "$or": [{"relation_group": "edition.cnn.com"}, {"domain": "edition.cnn.com"}]
    }


def test_build_relation_where_multiple_groups():
    result = build_relation_where(["group-b", "group-a"])
    assert result == {
        "$or": [
            {"relation_group": {"$in": ["group-a", "group-b"]}},
            {"domain": {"$in": ["group-a", "group-b"]}},
        ]
    }


def test_build_relation_where_empty_returns_none():
    assert build_relation_where([]) is None


def test_build_relation_where_whitespace_only_returns_none():
    assert build_relation_where(["", "  "]) is None


def test_build_relation_where_deduplicates():
    result = build_relation_where(["group-a", "group-a", " group-a "])
    assert result == {
        "$or": [{"relation_group": "group-a"}, {"domain": "group-a"}]
    }


# --- build_tags_where ---


def test_build_tags_where_single_tag():
    result = build_tags_where(["Python"])
    assert result == {
        "$or": [
            {"tag_1": "Python"},
            {"tag_2": "Python"},
            {"tag_3": "Python"},
        ]
    }


def test_build_tags_where_empty_returns_none():
    assert build_tags_where([]) is None


# --- build_combined_where ---


def test_build_combined_where_all_none():
    assert build_combined_where([], []) is None
    assert build_combined_where([], [], tags=[]) is None


def test_build_combined_where_only_relation():
    result = build_combined_where(["group-a"], [])
    assert result == {
        "$or": [{"relation_group": "group-a"}, {"domain": "group-a"}]
    }


def test_build_combined_where_relation_and_source():
    result = build_combined_where(["group-a"], ["https://example.com"])
    assert result == {
        "$and": [
            {"$or": [{"relation_group": "group-a"}, {"domain": "group-a"}]},
            {"source_url": "https://example.com"},
        ]
    }


def test_build_combined_where_all_three():
    result = build_combined_where(["group-a"], ["https://a"], tags=["Python"])
    assert "$and" in result
    assert len(result["$and"]) == 3


# --- build_source_where ---


def test_build_source_where_single():
    assert build_source_where(["https://a"]) == {"source_url": "https://a"}


def test_build_source_where_multiple():
    result = build_source_where(["https://b", "https://a"])
    assert result == {"source_url": {"$in": ["https://a", "https://b"]}}


def test_build_source_where_empty_returns_none():
    assert build_source_where([]) is None
