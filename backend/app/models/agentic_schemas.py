from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.core.agentic_settings import AgenticConfig


class RouteAction(str, Enum):
    SEARCH_KNOWLEDGE = "search_knowledge"
    DIRECT_CHAT = "direct_chat"
    NEED_CLARIFICATION = "need_clarification"


class QueryPlan(BaseModel):
    semantic_query: str = Field(min_length=1, max_length=512)
    tag_filters: list[str] = []
    relation_filters: list[str] = []


class EvidenceDetail(BaseModel):
    best_distance: float
    caption_ratio: float
    heuristic_zone: str
    llm_verdict: Optional[str] = None
    reject_reason: Optional[str] = None


class AgenticState(BaseModel):
    user_message: str
    conversation_history: list[dict]
    ui_hard_constraints: dict
    existing_system_tags: list[str]
    config: AgenticConfig
    chat_model: str
    embedding_model: str
    start_time: float
    current_node: str = "START"
    route_action: Optional[RouteAction] = None
    query_plan: Optional[QueryPlan] = None
    refine_round: int = 0
    final_query_params: Optional[dict] = None
    retrieved_chunks: list[dict] = []
    previous_queries: list[str] = []
    evidence_verdict: Optional[str] = None
    evidence_detail: Optional[EvidenceDetail] = None
    final_answer: str = ""
    transition_log: list[dict] = []


class AgenticChatRequest(BaseModel):
    query: str = Field(min_length=1)
    model: Optional[str] = None
    embedding_model: Optional[str] = None
    relation_groups: Optional[list[str]] = None
    source_urls: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    conversation_history: list[dict] = []
