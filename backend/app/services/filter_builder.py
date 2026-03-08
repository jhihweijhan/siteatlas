def _normalize_relation_group(value: str | None) -> str:
    return (value or "").strip()


def build_relation_where(relation_groups: list[str]) -> dict | None:
    groups = sorted({_normalize_relation_group(g) for g in relation_groups if _normalize_relation_group(g)})
    if not groups:
        return None
    if len(groups) == 1:
        g = groups[0]
        return {"$or": [{"relation_group": g}, {"domain": g}]}
    return {
        "$or": [
            {"relation_group": {"$in": groups}},
            {"domain": {"$in": groups}},
        ]
    }


def build_source_where(source_urls: list[str]) -> dict | None:
    urls = sorted({(u or "").strip() for u in source_urls if (u or "").strip()})
    if not urls:
        return None
    if len(urls) == 1:
        return {"source_url": urls[0]}
    return {"source_url": {"$in": urls}}


def build_image_caption_where(source_urls: list[str]) -> dict | None:
    source_where = build_source_where(source_urls)
    if not source_where:
        return None
    return {"$and": [source_where, {"type": "image_caption"}]}


def build_tags_where(tags: list[str]) -> dict | None:
    if not tags:
        return None
    conditions = []
    for tag in tags:
        conditions.extend([
            {"tag_1": tag},
            {"tag_2": tag},
            {"tag_3": tag},
        ])
    return {"$or": conditions}


def build_combined_where(
    relation_groups: list[str],
    source_urls: list[str],
    tags: list[str] | None = None,
) -> dict | None:
    parts = []
    relation_where = build_relation_where(relation_groups)
    if relation_where:
        parts.append(relation_where)
    source_where = build_source_where(source_urls)
    if source_where:
        parts.append(source_where)
    tags_where = build_tags_where(tags or [])
    if tags_where:
        parts.append(tags_where)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return {"$and": parts}
