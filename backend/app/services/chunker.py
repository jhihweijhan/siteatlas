"""Chinese-optimized recursive text chunker."""


class ChineseTextChunker:
    SEPARATORS = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，",
        "、",
        " ",
        "",
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return self._recursive_split(text.strip(), self.SEPARATORS)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[-1]
        for sep in separators:
            if sep and sep in text:
                separator = sep
                break

        if separator:
            parts = text.split(separator)
        else:
            parts = list(text)

        chunks = []
        current = ""

        if separator in separators:
            idx = separators.index(separator)
            remaining_separators = separators[idx + 1 :]
        else:
            remaining_separators = separators[-1:]

        for part in parts:
            candidate = current + separator + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(part) > self.chunk_size and remaining_separators:
                sub_chunks = self._recursive_split(part, remaining_separators)
                chunks.extend(sub_chunks[:-1])
                current = sub_chunks[-1] if sub_chunks else ""
            else:
                current = part

        if current:
            chunks.append(current)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            for sep in ["。", "！", "？", "；", "，", " "]:
                idx = overlap_text.find(sep)
                if idx >= 0:
                    overlap_text = overlap_text[idx + 1 :]
                    break
            result.append(overlap_text + chunks[i])
        return result
