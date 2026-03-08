from pydantic import BaseModel


class AgenticConfig(BaseModel):
    evidence_auto_pass_threshold: float = 0.40
    evidence_reject_threshold: float = 0.65
    caption_dominance_threshold: float = 0.80
    max_refine_rounds: int = 1
    global_timeout_seconds: float = 30.0
    router_num_predict: int = 200
    evaluator_num_predict: int = 100
    query_similarity_threshold: float = 0.95
