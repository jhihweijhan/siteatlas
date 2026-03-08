from app.models.agentic_schemas import AgenticState


async def execute(state: AgenticState) -> tuple[AgenticState, str]:
    query = state.query_plan.semantic_query

    # 1. duplicate query detection
    if query in state.previous_queries:
        state.evidence_verdict = "exhausted"
        state.current_node = "REFINER"
        return state, "NO_ANSWER"

    # 2. strip soft tag filters (keep only UI hard constraints)
    hard_tags = set(state.ui_hard_constraints.get("tags", []))
    state.query_plan.tag_filters = [
        t for t in state.query_plan.tag_filters if t in hard_tags
    ]

    # 3. strip soft relation filters
    hard_relations = set(state.ui_hard_constraints.get("relation_groups", []))
    state.query_plan.relation_filters = [
        r for r in state.query_plan.relation_filters if r in hard_relations
    ]

    # 4. track query
    state.previous_queries.append(query)

    # 5. bump round
    state.refine_round += 1
    state.current_node = "REFINER"

    return state, "RETRIEVE"
