"""Integration test for generate_report_themes() quote verification.

No real API is called: a fake labeler exposing domain_hint/call_raw stands
in for LLMLabeler, following the pattern used in test_llm_granularity.py.
Uses a freshly-fit model (not the shared session-scoped fitted_model
fixture) since generate_report_themes mutates model state in place.
"""

import json

import pytest

from graphweave import GraphWeave, GraphWeaveConfig


class FakeThemeLabeler:
    """Dispatches canned responses to the proposer vs. narrative-writer calls
    based on which system prompt _propose_meta_themes / _write_meta_theme_narrative
    use (see graphweave/core/model.py)."""

    def __init__(self, proposer_response: str, narrative_response: str):
        self.domain_hint = None
        self._proposer_response = proposer_response
        self._narrative_response = narrative_response
        self.calls: list[str] = []

    def call_raw(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append(system_prompt)
        if "consolidate fine-grained" in system_prompt:
            return self._proposer_response
        return self._narrative_response


@pytest.fixture
def small_model(fake_documents, _fake_embeddings):
    cfg = GraphWeaveConfig(
        use_dim_reduction=False,
        use_iterative_refinement=False,
        use_lexical_view=True,
        n_consensus_runs=3,
        min_cluster_size=5,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )
    model = GraphWeave(config=cfg)
    model.fit(fake_documents, embeddings=_fake_embeddings)
    for topic in model.topics_:
        if topic.topic_id != -1:
            topic.label = f"Topic {topic.topic_id} Label"
            topic.description = f"A description for topic {topic.topic_id}."
    return model


def _one_theme_proposal(model) -> str:
    topic_ids = [t.topic_id for t in model.topics_ if t.topic_id != -1]
    return json.dumps({"themes": [{"title": "One Consolidated Theme Here", "topic_ids": topic_ids}]})


class TestGenerateReportThemesQuoteVerification:
    def test_fabricated_quote_is_flagged_on_theme(self, small_model):
        proposer_resp = _one_theme_proposal(small_model)
        narrative_resp = json.dumps({
            "narrative": (
                "Respondents repeatedly said the process was "
                "'a uniquely bewildering bureaucratic nightmare no one warned them about'. "
                "This pattern recurred across accounts and reflects a deeper concern."
            )
        })
        labeler = FakeThemeLabeler(proposer_resp, narrative_resp)

        themes = small_model.generate_report_themes(labeler)

        assert len(themes) == 1
        assert themes[0].unverified_quotes == [
            "a uniquely bewildering bureaucratic nightmare no one warned them about"
        ]

    def test_fabricated_quote_past_truncation_is_flagged(self, small_model):
        """A quote appearing only *after* the 1200-char truncation boundary
        must still be flagged. The LLM is shown doc[:1200]; verifying against
        the full untruncated documents would wrongly 'confirm' a fabricated
        quote that happens to match text the LLM never saw. Regression guard
        for the _write_meta_theme_narrative / verify_quotes mismatch."""
        buried = "the coffee machine on the third floor was permanently broken"
        filler = "routine intake notes and standard paperwork details. " * 30
        long_doc = f"{filler} {buried} and closing remarks follow here."
        assert long_doc.index(buried) > 1200  # phrase sits past what the LLM sees
        # Every representative doc is now this long document, so the buried
        # phrase is present in the full text but absent from the shown prefix.
        small_model.documents_ = [long_doc] * len(small_model.documents_)

        proposer_resp = _one_theme_proposal(small_model)
        narrative_resp = json.dumps({
            "narrative": (
                "Respondents kept returning to small daily frustrations. One participant "
                f"noted that '{buried}', a detail that captured a broader sense of neglect. "
                "This pattern recurred across accounts and points to a deeper erosion of trust."
            )
        })
        labeler = FakeThemeLabeler(proposer_resp, narrative_resp)

        themes = small_model.generate_report_themes(labeler)

        assert themes[0].unverified_quotes == [buried]

    def test_no_quote_means_no_flag(self, small_model):
        proposer_resp = _one_theme_proposal(small_model)
        narrative_resp = json.dumps({
            "narrative": (
                "Respondents consistently raised the same underlying concern across "
                "documents. This pattern reflects a deeper, shared experience that "
                "matters for the report."
            )
        })
        labeler = FakeThemeLabeler(proposer_resp, narrative_resp)

        themes = small_model.generate_report_themes(labeler)

        assert themes[0].unverified_quotes == []

    def test_export_report_flags_unverified_quotes(self, small_model, tmp_path):
        proposer_resp = _one_theme_proposal(small_model)
        narrative_resp = json.dumps({
            "narrative": (
                "Respondents said the office was "
                "'a place that fabricated nothing anyone actually said out loud'. "
                "This pattern reflects a deeper concern worth naming."
            )
        })
        labeler = FakeThemeLabeler(proposer_resp, narrative_resp)
        small_model.generate_report_themes(labeler)

        out_path = tmp_path / "report.md"
        small_model.export_report(str(out_path))
        content = out_path.read_text()

        assert "Review needed" in content
        assert "⚠" in content  # warning glyph next to the flagged theme heading
        assert "Unverified quote" in content
