import app.main  # noqa: F401

from app.services.filter_builder import build_combined_where, build_tags_where


def test_build_tags_where_single():
    result = build_tags_where(["Python 開發"])
    assert result == {"$or": [
        {"tag_1": "Python 開發"},
        {"tag_2": "Python 開發"},
        {"tag_3": "Python 開發"},
    ]}


def test_build_tags_where_multiple():
    result = build_tags_where(["Python 開發", "API 設計"])
    assert "$or" in result
    assert len(result["$or"]) == 6


def test_build_tags_where_empty():
    result = build_tags_where([])
    assert result is None


def test_build_combined_where_with_tags_only():
    result = build_combined_where([], [], tags=["Python 開發"])
    assert result is not None
    assert "$or" in result


def test_build_combined_where_with_tags_and_groups():
    result = build_combined_where(["group1"], [], tags=["Python 開發"])
    assert "$and" in result
    assert len(result["$and"]) == 2


def test_build_combined_where_with_tags_and_source_urls():
    result = build_combined_where([], ["https://a"], tags=["Python 開發"])
    assert "$and" in result
    assert len(result["$and"]) == 2


def test_build_combined_where_with_all_three():
    result = build_combined_where(["group1"], ["https://a"], tags=["Python 開發"])
    assert "$and" in result
    assert len(result["$and"]) == 3
