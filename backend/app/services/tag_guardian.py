import logging
import math

from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class TagGuardian:
    """新標籤守門員：用 embedding 相似度防止標籤發散"""

    def __init__(self, embedding_svc: EmbeddingService, threshold: float = 0.3):
        self._embedding_svc = embedding_svc
        self._threshold = threshold
        self._existing_embeddings: dict[str, list[float]] = {}

    async def resolve_tags(
        self,
        suggested_tags: list[str],
        existing_tags: set[str],
    ) -> list[str]:
        """
        For each suggested tag:
        - If it matches an existing tag (case-insensitive) -> use the existing one
        - If it's new -> embed and compare with existing tag embeddings
          - If similar (distance < threshold) -> merge to closest existing tag
          - If not similar -> keep as new tag
        - If existing_tags is empty -> return all suggested as-is
        - If embedding fails -> return original suggested tags (graceful degradation)
        """
        if not existing_tags:
            return list(suggested_tags)

        try:
            result: list[str] = []
            new_tags: list[str] = []

            for tag in suggested_tags:
                match = self._find_exact_match(tag, existing_tags)
                if match is not None:
                    result.append(match)
                else:
                    new_tags.append(tag)

            if not new_tags:
                return result

            await self._ensure_existing_embeddings(existing_tags)

            for tag in new_tags:
                embedding = await self._embedding_svc.embed_query(tag)
                closest_tag, distance = self._find_closest(embedding)
                if closest_tag is not None and distance < self._threshold:
                    logger.info(
                        "Tag '%s' merged to '%s' (distance=%.4f)",
                        tag,
                        closest_tag,
                        distance,
                    )
                    result.append(closest_tag)
                else:
                    result.append(tag)

            return result

        except Exception:
            logger.warning(
                "Embedding failed during tag resolution, keeping original tags",
                exc_info=True,
            )
            return list(suggested_tags)

    def _matches_existing(self, tag: str, existing_tags: set[str]) -> bool:
        """Case-insensitive match check."""
        return self._find_exact_match(tag, existing_tags) is not None

    def _find_exact_match(self, tag: str, existing_tags: set[str]) -> str | None:
        """Returns the existing tag with original casing, or None."""
        tag_lower = tag.lower()
        for existing in existing_tags:
            if existing.lower() == tag_lower:
                return existing
        return None

    async def _ensure_existing_embeddings(self, existing_tags: set[str]) -> None:
        """Batch embed existing tags that are not yet cached."""
        tags_to_embed = [
            t for t in existing_tags if t not in self._existing_embeddings
        ]
        if not tags_to_embed:
            return
        embeddings = await self._embedding_svc.embed_texts(tags_to_embed)
        for tag, emb in zip(tags_to_embed, embeddings):
            self._existing_embeddings[tag] = emb

    def _find_closest(
        self, embedding: list[float]
    ) -> tuple[str | None, float]:
        """Find closest existing tag by cosine distance."""
        if not self._existing_embeddings:
            return None, float("inf")

        best_tag: str | None = None
        best_distance = float("inf")

        for tag, existing_emb in self._existing_embeddings.items():
            dist = self._cosine_distance(embedding, existing_emb)
            if dist < best_distance:
                best_distance = dist
                best_tag = tag

        return best_tag, best_distance

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """Returns 1.0 - cosine_similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - dot / (norm_a * norm_b)
