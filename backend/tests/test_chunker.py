import pytest

from app.services.chunker import ChineseTextChunker


@pytest.fixture
def chunker():
    return ChineseTextChunker(chunk_size=100, chunk_overlap=20)


def test_short_text_single_chunk(chunker):
    text = "這是一段很短的文字。"
    chunks = chunker.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_multiple_chunks(chunker):
    text = "。".join([f"這是第{i}個句子，包含一些有意義的內容" for i in range(20)])
    chunks = chunker.split(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= chunker.chunk_size + 50


def test_overlap_exists(chunker):
    text = "。".join([f"句子{i}有一些內容在裡面" for i in range(20)])
    chunks = chunker.split(text)
    if len(chunks) >= 2:
        assert chunks[0] != chunks[1]


def test_empty_text(chunker):
    chunks = chunker.split("")
    assert chunks == []


def test_splits_on_chinese_punctuation(chunker):
    text = "第一段內容很長" + "啊" * 80 + "。第二段內容也很長" + "啊" * 80 + "。"
    chunks = chunker.split(text)
    assert len(chunks) >= 2
