import time

import pytest
from pydantic import ValidationError

from app.core.agentic_settings import AgenticConfig
from app.models.agentic_schemas import (
    AgenticChatRequest,
    AgenticState,
    EvidenceDetail,
    QueryPlan,
    RouteAction,
)


class TestRouteAction:
    def test_enum_values(self):
        assert RouteAction.SEARCH_KNOWLEDGE == "search_knowledge"
        assert RouteAction.DIRECT_CHAT == "direct_chat"
        assert RouteAction.NEED_CLARIFICATION == "need_clarification"


class TestQueryPlan:
    def test_valid(self):
        qp = QueryPlan(semantic_query="test query")
        assert qp.semantic_query == "test query"
        assert qp.tag_filters == []
        assert qp.relation_filters == []

    def test_max_length_exceeded(self):
        with pytest.raises(ValidationError):
            QueryPlan(semantic_query="x" * 513)

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            QueryPlan(semantic_query="")


class TestEvidenceDetail:
    def test_optional_fields(self):
        ed = EvidenceDetail(best_distance=0.3, caption_ratio=0.1, heuristic_zone="pass")
        assert ed.llm_verdict is None
        assert ed.reject_reason is None

    def test_with_optional_fields(self):
        ed = EvidenceDetail(
            best_distance=0.3,
            caption_ratio=0.1,
            heuristic_zone="pass",
            llm_verdict="accept",
            reject_reason="too far",
        )
        assert ed.llm_verdict == "accept"
        assert ed.reject_reason == "too far"


class TestAgenticState:
    def test_initial_values(self):
        state = AgenticState(
            user_message="hello",
            conversation_history=[],
            ui_hard_constraints={},
            existing_system_tags=[],
            config=AgenticConfig(),
            chat_model="qwen2.5:14b",
            embedding_model="bge-m3",
            start_time=time.time(),
        )
        assert state.current_node == "START"
        assert state.route_action is None
        assert state.query_plan is None
        assert state.refine_round == 0
        assert state.final_query_params is None
        assert state.retrieved_chunks == []
        assert state.previous_queries == []
        assert state.evidence_verdict is None
        assert state.evidence_detail is None
        assert state.final_answer == ""
        assert state.transition_log == []


class TestAgenticChatRequest:
    def test_minimal(self):
        req = AgenticChatRequest(query="what is this?")
        assert req.query == "what is this?"
        assert req.model is None
        assert req.embedding_model is None
        assert req.relation_groups is None
        assert req.source_urls is None
        assert req.tags is None
        assert req.top_k == 5
        assert req.conversation_history == []

    def test_full(self):
        req = AgenticChatRequest(
            query="summarize",
            model="qwen2.5:14b",
            embedding_model="bge-m3",
            relation_groups=["group1"],
            source_urls=["https://example.com"],
            tags=["tag1", "tag2"],
            top_k=10,
            conversation_history=[{"role": "user", "content": "hi"}],
        )
        assert req.model == "qwen2.5:14b"
        assert req.top_k == 10
        assert req.tags == ["tag1", "tag2"]
        assert len(req.conversation_history) == 1

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            AgenticChatRequest(query="")

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            AgenticChatRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            AgenticChatRequest(query="test", top_k=21)
