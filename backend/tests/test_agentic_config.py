from app.core.agentic_settings import AgenticConfig


class TestAgenticConfigDefaults:
    def test_default_values(self):
        cfg = AgenticConfig()
        assert cfg.evidence_auto_pass_threshold == 0.40
        assert cfg.evidence_reject_threshold == 0.65
        assert cfg.caption_dominance_threshold == 0.80
        assert cfg.max_refine_rounds == 1
        assert cfg.global_timeout_seconds == 30.0
        assert cfg.router_num_predict == 200
        assert cfg.evaluator_num_predict == 100
        assert cfg.query_similarity_threshold == 0.95

    def test_threshold_ordering(self):
        cfg = AgenticConfig()
        assert cfg.evidence_auto_pass_threshold < cfg.evidence_reject_threshold

    def test_custom_values_from_kwargs(self):
        cfg = AgenticConfig(
            evidence_auto_pass_threshold=0.30,
            evidence_reject_threshold=0.70,
            max_refine_rounds=3,
            global_timeout_seconds=20.0,
        )
        assert cfg.evidence_auto_pass_threshold == 0.30
        assert cfg.evidence_reject_threshold == 0.70
        assert cfg.max_refine_rounds == 3
        assert cfg.global_timeout_seconds == 20.0
