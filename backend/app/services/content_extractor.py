"""Content extraction service using trafilatura."""

import re
from dataclasses import dataclass, field

from trafilatura import bare_extraction


@dataclass
class ExtractedContent:
    text: str
    image_urls: list[str] = field(default_factory=list)


_IGNORE_IMAGE_RE = re.compile(
    r"^data:|\.gif(\?|$)|/pixel\.|/beacon\.|/track(er|ing)",
    re.IGNORECASE,
)

_MARKDOWN_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class ContentExtractor:
    def extract(self, html: str, url: str) -> ExtractedContent:
        if not html or not html.strip():
            return ExtractedContent(text="")

        result = bare_extraction(
            html,
            url=url,
            include_images=True,
            include_tables=True,
            include_links=False,
            deduplicate=True,
            favor_recall=True,
        )

        if result is None:
            return ExtractedContent(text="")

        text = getattr(result, "text", "") or ""

        # Extract image URLs from markdown syntax in text
        image_urls: list[str] = []
        for match in _MARKDOWN_IMG_RE.finditer(text):
            src = match.group(2)
            if src and not _IGNORE_IMAGE_RE.search(src):
                image_urls.append(src)

        # Clean markdown image syntax from text
        text = _MARKDOWN_IMG_RE.sub("", text).strip()

        return ExtractedContent(text=text, image_urls=image_urls)
