"""ChromaDB vector store service."""

import chromadb


COLLECTION_NAME = "web_knowledge"


class VectorStoreService:
    def __init__(self, host: str = "localhost", port: int = 8100):
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def upsert_with_ids(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        """Upsert documents with explicit IDs (supports mixed text + image chunks)."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items = []
        for i in range(len(ids)):
            items.append(
                {
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return items

    def search_by_embedding(
        self,
        query_embedding: list[float],
        exclude_doc_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search by embedding vector, excluding chunks from a specific doc_id.

        Returns deduplicated results by source_url with aggregated tags.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"doc_id": {"$ne": exclude_doc_id}},
            include=["metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        seen: dict[str, dict] = {}
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0
            url = meta.get("source_url", "")
            if not url:
                continue
            if url not in seen or dist < seen[url]["distance"]:
                tags = [meta.get(k, "") for k in ("tag_1", "tag_2", "tag_3")]
                tags = [t for t in tags if t]
                seen[url] = {
                    "doc_id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "url": url,
                    "relation_group": meta.get("relation_group", "general"),
                    "tags": tags,
                    "distance": dist,
                }

        return list(seen.values())

    def list_documents(self) -> list[dict]:
        all_data = self._collection.get(include=["metadatas"])
        seen: dict[str, dict] = {}
        schema_key_map: dict[str, set[str]] = {}

        for meta in (all_data.get("metadatas") or []):
            url = meta.get("source_url", "")
            if not url:
                continue
            relation_group = meta.get("relation_group") or meta.get("domain") or "general"
            if url not in seen:
                seen[url] = {
                    "id": meta.get("doc_id", ""),
                    "title": meta.get("title", ""),
                    "url": url,
                    "created_at": meta.get("created_at", ""),
                    "chunks_count": 0,
                    "relation_group": relation_group,
                    "text_chunks_count": 0,
                    "image_chunks_count": 0,
                    "schema_keys": [],
                    "tags": set(),
                }
                schema_key_map[url] = set()
            seen[url]["chunks_count"] += 1
            for key in ("tag_1", "tag_2", "tag_3"):
                val = meta.get(key, "")
                if val:
                    seen[url]["tags"].add(val)
            chunk_type = meta.get("type", "text")
            if chunk_type == "image_caption":
                seen[url]["image_chunks_count"] += 1
            else:
                seen[url]["text_chunks_count"] += 1
            schema_key_map[url].update(meta.keys())

        for url, item in seen.items():
            item["schema_keys"] = sorted(schema_key_map.get(url, set()))
            item["tags"] = sorted(item.get("tags", set()))

        return list(seen.values())

    def get_chunks(
        self,
        where: dict | None = None,
        include_documents: bool = True,
    ) -> list[dict]:
        include = ["metadatas"]
        if include_documents:
            include.append("documents")

        kwargs = {"include": include}
        if where:
            kwargs["where"] = where

        results = self._collection.get(**kwargs)
        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        rows: list[dict] = []
        for i in range(len(ids)):
            rows.append(
                {
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        return rows

    def get_document_detail(self, doc_id: str) -> dict | None:
        rows = self.get_chunks(where={"doc_id": doc_id}, include_documents=True)
        if not rows:
            return None

        def _sort_key(row: dict) -> tuple:
            meta = row.get("metadata") or {}
            chunk_type = meta.get("type", "text")
            if chunk_type == "image_caption":
                return (1, int(meta.get("image_index", meta.get("chunk_index", 0)) or 0))
            return (0, int(meta.get("chunk_index", 0) or 0))

        rows = sorted(rows, key=_sort_key)
        first_meta = rows[0].get("metadata") or {}
        relation_group = first_meta.get("relation_group") or first_meta.get("domain") or "general"

        type_counts: dict[str, int] = {}
        schema_types: dict[str, set[str]] = {}
        text_samples: list[dict] = []
        image_items: list[dict] = []
        text_chunks_count = 0
        image_chunks_count = 0

        for row in rows:
            meta = row.get("metadata") or {}
            chunk_type = str(meta.get("type", "text") or "text")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

            for key, value in meta.items():
                schema_types.setdefault(key, set()).add(type(value).__name__)

            if chunk_type == "image_caption":
                image_chunks_count += 1
                image_items.append(
                    {
                        "id": row.get("id", ""),
                        "image_index": meta.get("image_index", meta.get("chunk_index", 0)),
                        "image_url": meta.get("image_url", ""),
                        "image_alt": meta.get("image_alt", ""),
                        "caption": row.get("document", ""),
                        "metadata": meta,
                    }
                )
            else:
                text_chunks_count += 1
                if len(text_samples) < 5:
                    text = str(row.get("document", "") or "")
                    text_samples.append(
                        {
                            "id": row.get("id", ""),
                            "chunk_index": meta.get("chunk_index", 0),
                            "preview": text[:220],
                            "metadata": meta,
                        }
                    )

        metadata_schema = {key: sorted(list(type_names)) for key, type_names in schema_types.items()}
        detail_tags: set[str] = set()
        for row in rows:
            m = row.get("metadata", {})
            for key in ("tag_1", "tag_2", "tag_3"):
                val = m.get(key, "")
                if val:
                    detail_tags.add(val)

        item = {
            "id": doc_id,
            "title": first_meta.get("title", ""),
            "url": first_meta.get("source_url", ""),
            "created_at": first_meta.get("created_at", ""),
            "chunks_count": len(rows),
            "relation_group": relation_group,
            "text_chunks_count": text_chunks_count,
            "image_chunks_count": image_chunks_count,
            "schema_keys": sorted(metadata_schema.keys()),
            "tags": sorted(detail_tags),
        }

        return {
            "item": item,
            "type_counts": type_counts,
            "metadata_schema": metadata_schema,
            "text_samples": text_samples,
            "image_items": image_items,
        }

    def find_document_by_source_url(self, source_url: str) -> dict | None:
        rows = self.get_chunks(where={"source_url": source_url}, include_documents=False)
        if not rows:
            return None

        meta = rows[0].get("metadata") or {}
        return {
            "doc_id": meta.get("doc_id", ""),
            "source_url": meta.get("source_url", ""),
            "title": meta.get("title", ""),
            "content_hash": meta.get("content_hash"),
            "file_name": meta.get("file_name"),
        }

    def update_document_relation(self, doc_id: str, relation_group: str) -> bool:
        normalized = relation_group.strip()
        if not normalized:
            return False

        results = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        if not ids:
            return False

        updated_metadatas = []
        for meta in metadatas:
            next_meta = dict(meta or {})
            next_meta["relation_group"] = normalized
            updated_metadatas.append(next_meta)

        self._collection.update(ids=ids, metadatas=updated_metadatas)
        return True

    def collect_existing_tags(self) -> set[str]:
        """蒐集知識庫中所有既有的唯一標籤"""
        result = self._collection.get(include=["metadatas"])
        tags: set[str] = set()
        for meta in result.get("metadatas", []):
            for key in ("tag_1", "tag_2", "tag_3"):
                val = meta.get(key, "")
                if val:
                    tags.add(val)
        return tags

    def update_document_tags(self, doc_id: str, tags: list[str]) -> bool:
        """更新指定文件所有 chunk 的 tag_1/2/3"""
        result = self._collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            return False

        tag_values = (tags + ["", "", ""])[:3]
        updated_metadatas = []
        for meta in result["metadatas"]:
            next_meta = dict(meta or {})
            next_meta["tag_1"] = tag_values[0]
            next_meta["tag_2"] = tag_values[1]
            next_meta["tag_3"] = tag_values[2]
            updated_metadatas.append(next_meta)

        self._collection.update(ids=ids, metadatas=updated_metadatas)
        return True

    def delete_document(self, doc_id: str) -> bool:
        results = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = results.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
            return True
        return False

    def check_health(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:  # noqa: BLE001
            return False
