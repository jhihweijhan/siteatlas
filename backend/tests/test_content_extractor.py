import pytest

from app.services.content_extractor import ContentExtractor, ExtractedContent


@pytest.fixture
def extractor():
    return ContentExtractor()


class TestExtractedContent:
    def test_has_text_and_image_urls(self):
        content = ExtractedContent(text="hello", image_urls=["http://img.jpg"])
        assert content.text == "hello"
        assert content.image_urls == ["http://img.jpg"]

    def test_defaults(self):
        content = ExtractedContent(text="hello")
        assert content.image_urls == []


class TestContentExtractor:
    def test_extract_basic_html(self, extractor):
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <article>
                <h1>Test Article</h1>
                <p>This is the main content of the article. It contains important information
                that should be extracted by trafilatura. The content needs to be long enough
                for trafilatura to consider it as main content rather than boilerplate.</p>
                <p>Here is another paragraph with more details about the topic. Trafilatura
                uses heuristics to determine what constitutes the main content versus
                navigation, ads, and other noise elements on a web page.</p>
            </article>
            <nav><a href="/home">Home</a><a href="/about">About</a></nav>
            <footer>Copyright 2026</footer>
        </body>
        </html>
        """
        result = extractor.extract(html, url="http://example.com/article")
        assert isinstance(result, ExtractedContent)
        assert len(result.text) > 0
        assert "main content" in result.text.lower()

    def test_extract_empty_html_returns_empty(self, extractor):
        result = extractor.extract("", url="http://example.com")
        assert result.text == ""
        assert result.image_urls == []

    def test_extract_no_content_html(self, extractor):
        html = "<html><body><nav>Just navigation</nav></body></html>"
        result = extractor.extract(html, url="http://example.com")
        assert isinstance(result, ExtractedContent)

    def test_extract_filters_data_uri_images(self, extractor):
        html = """
        <html><body>
            <article>
                <h1>Article</h1>
                <p>Content paragraph with enough text for extraction to work properly
                and pass the minimum content threshold of trafilatura.</p>
                <img src="data:image/svg+xml;base64,abc" alt="SVG icon" />
                <img src="http://example.com/real-photo.jpg" alt="Real photo" />
            </article>
        </body></html>
        """
        result = extractor.extract(html, url="http://example.com")
        for url in result.image_urls:
            assert not url.startswith("data:")
