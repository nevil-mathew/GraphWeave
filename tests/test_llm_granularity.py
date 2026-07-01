"""Tests for LLM-guided granularity calibration (tritopic.labeling.llm_granularity).

No real API is ever called: a fake labeler exposing call_raw/call_structured
stands in for LLMLabeler. Unit tests exercise the module's helper functions
directly; the integration test drives a freshly-fit TriTopic model (NOT the
shared session-scoped ``fitted_model`` fixture, since tune_resolution_with_llm
mutates model state in place).
"""

import json

import numpy as np
import pytest

from tritopic import TriTopic, TriTopicConfig
from tritopic.labeling.llm_granularity import (
    _sample_triplets,
    _candidate_resolutions,
    _triplet_agreement,
    _parse_triplet_response,
    _default_n_triplets,
    llm_select_resolution,
)


class FakeLabeler:
    """Returns a canned string from both call_raw and call_structured."""

    def __init__(self, response):
        self._resp = response
        self.calls: list[tuple[str, str]] = []

    def call_raw(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        return self.call_raw(system_prompt, user_prompt, max_tokens=max_tokens)


class AllBLabeler(FakeLabeler):
    """Always answers 'B' for every item in every batch it receives."""

    def __init__(self):
        super().__init__(None)

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        n_items = user_prompt.count("Item ")
        resp = json.dumps({"answers": ["B"] * n_items})
        self.calls.append((system_prompt, user_prompt))
        return resp


# --------------------------------------------------------------------------- #
# Triplet sampling
# --------------------------------------------------------------------------- #
class TestSampleTriplets:
    def _make_labels_embeddings(self, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(3, 8))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        labels = np.repeat([0, 1, 2], 20)
        embs = np.vstack([
            centers[t] + 0.05 * rng.normal(size=(20, 8)) for t in range(3)
        ])
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return labels, embs.astype(np.float32)

    def test_determinism_same_seed(self):
        labels, embs = self._make_labels_embeddings()
        t1 = _sample_triplets(labels, embs, n_triplets=10, random_state=42)
        t2 = _sample_triplets(labels, embs, n_triplets=10, random_state=42)
        assert t1 == t2

    def test_different_seed_can_differ(self):
        labels, embs = self._make_labels_embeddings()
        t1 = _sample_triplets(labels, embs, n_triplets=10, random_state=42)
        t2 = _sample_triplets(labels, embs, n_triplets=10, random_state=7)
        assert t1 != t2

    def test_b_same_cluster_c_different_cluster(self):
        labels, embs = self._make_labels_embeddings()
        triplets = _sample_triplets(labels, embs, n_triplets=15, random_state=0)
        assert len(triplets) > 0
        for a, b, c in triplets:
            assert labels[a] == labels[b]
            assert labels[a] != labels[c]
            assert a != b

    def test_outliers_excluded(self):
        labels, embs = self._make_labels_embeddings()
        labels = labels.copy()
        labels[:5] = -1
        triplets = _sample_triplets(labels, embs, n_triplets=15, random_state=0)
        for a, b, c in triplets:
            assert a >= 5 and b >= 5 and c >= 5

    def test_single_cluster_returns_empty(self):
        labels = np.zeros(10, dtype=int)
        embs = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        assert _sample_triplets(labels, embs, n_triplets=5, random_state=0) == []


# --------------------------------------------------------------------------- #
# Candidate resolutions
# --------------------------------------------------------------------------- #
class TestCandidateResolutions:
    def test_count_and_bounds(self):
        cands = _candidate_resolutions((0.1, 2.0), 6)
        assert len(cands) == 6
        assert abs(cands[0] - 0.1) < 1e-9
        assert abs(cands[-1] - 2.0) < 1e-9

    def test_monotonic_increasing(self):
        cands = _candidate_resolutions((0.1, 2.0), 6)
        assert all(cands[i] < cands[i + 1] for i in range(len(cands) - 1))


# --------------------------------------------------------------------------- #
# Triplet agreement scoring
# --------------------------------------------------------------------------- #
class TestTripletAgreement:
    def test_perfect_agreement(self):
        labels = np.array([0, 0, 1])  # a=0,b=1 same cluster; c=2 different
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["B"]) == 1.0

    def test_perfect_disagreement(self):
        labels = np.array([0, 0, 1])
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["C"]) == 0.0

    def test_uninformative_triplet_skipped_both_merged(self):
        labels = np.array([0, 0, 0])  # a, b, c all in same cluster: uninformative
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["B"]) == 0.0

    def test_uninformative_triplet_skipped_both_separated(self):
        labels = np.array([0, 1, 2])  # a,b,c all different clusters: uninformative
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["B"]) == 0.0

    def test_mixed_batch_only_informative_counted(self):
        labels = np.array([0, 0, 1, 2, 2])
        # triplet1: a=0(lbl0), b=1(lbl0, same), c=2(lbl1, diff) -> informative, implied "B"
        # triplet2: a=2(lbl1), b=0(lbl0, diff), c=1(lbl0, diff) -> both diff, uninformative
        triplets = [(0, 1, 2), (2, 0, 1)]
        answers = ["B", "C"]
        assert _triplet_agreement(labels, triplets, answers) == 1.0


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
class TestParseTripletResponse:
    def test_valid_json(self):
        raw = '{"answers": ["B", "C", "B"]}'
        assert _parse_triplet_response(raw, 3) == ["B", "C", "B"]

    def test_valid_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"answers": ["C", "B"]}\nThanks!'
        assert _parse_triplet_response(raw, 2) == ["C", "B"]

    def test_truncated_json_falls_back_to_regex_tier(self):
        raw = '{"answers": ["B", "C", "B"'  # missing closing brackets
        result = _parse_triplet_response(raw, 3)
        assert result == ["B", "C", "B"]

    def test_garbage_input_falls_back_to_all_b(self):
        with pytest.warns(UserWarning, match="could not parse"):
            result = _parse_triplet_response("not json at all, sorry!", 4)
        assert result == ["B", "B", "B", "B"]

    def test_short_answer_list_padded(self):
        raw = '{"answers": ["C"]}'
        with pytest.warns(UserWarning, match="LLM returned 1 answers"):
            result = _parse_triplet_response(raw, 3)
        assert result == ["C", "B", "B"]

    def test_long_answer_list_truncated(self):
        raw = '{"answers": ["B", "C", "B", "C", "B"]}'
        result = _parse_triplet_response(raw, 3)
        assert result == ["B", "C", "B"]

    def test_invalid_enum_values_default_to_b(self):
        raw = '{"answers": ["B", "X", "C"]}'
        result = _parse_triplet_response(raw, 3)
        assert result == ["B", "B", "C"]


# --------------------------------------------------------------------------- #
# Default n_triplets scaling
# --------------------------------------------------------------------------- #
class TestDefaultNTriplets:
    def test_small_corpus_floor(self):
        assert _default_n_triplets(50) == 24
        assert _default_n_triplets(100) == 24

    def test_scales_up_for_large_corpus(self):
        assert _default_n_triplets(1000) > 24
        assert _default_n_triplets(50_000) <= 120
        assert _default_n_triplets(50_000) >= 80

    def test_ceiling(self):
        assert _default_n_triplets(10_000_000) == 120


# --------------------------------------------------------------------------- #
# Integration: TriTopic.tune_resolution_with_llm
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_fitted_model(fake_documents, _fake_embeddings):
    """A freshly-fit, non-shared TriTopic model (safe to mutate)."""
    cfg = TriTopicConfig(
        use_dim_reduction=False,
        use_iterative_refinement=False,
        use_lexical_view=True,
        n_consensus_runs=3,
        min_cluster_size=5,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )
    model = TriTopic(config=cfg)
    model.fit(fake_documents, embeddings=_fake_embeddings)
    return model


class TestTuneResolutionWithLLM:
    def test_raises_if_not_fitted(self):
        model = TriTopic()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.tune_resolution_with_llm(FakeLabeler('{"answers": ["B"]}'))

    def test_returns_self(self, fresh_fitted_model):
        labeler = AllBLabeler()
        result = fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert result is fresh_fitted_model

    def test_topics_refreshed(self, fresh_fitted_model):
        labeler = AllBLabeler()
        topics_before = fresh_fitted_model.topics_
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        # _extract_topic_info always rebuilds self.topics_ as a brand-new list
        assert fresh_fitted_model.topics_ is not topics_before
        assert len(fresh_fitted_model.topics_) > 0

    def test_labels_length_preserved(self, fresh_fitted_model, fake_documents):
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert len(fresh_fitted_model.labels_) == len(fake_documents)

    def test_probabilities_refreshed(self, fresh_fitted_model):
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        n_topics = len([t for t in fresh_fitted_model.topics_ if t.topic_id != -1])
        assert fresh_fitted_model.probabilities_.shape == (
            len(fresh_fitted_model.documents_), n_topics
        )

    def test_llm_was_called(self, fresh_fitted_model):
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert len(labeler.calls) > 0

    def test_chosen_resolution_is_from_candidate_sweep(self, fresh_fitted_model):
        labeler = AllBLabeler()
        best_res = llm_select_resolution(
            labeler,
            fresh_fitted_model.documents_,
            fresh_fitted_model.graph_,
            fresh_fitted_model.embeddings_,
            resolution_range=(0.1, 2.0),
            n_candidates=4,
            n_triplets=12,
            batch_size=4,
        )
        candidates = _candidate_resolutions((0.1, 2.0), 4)
        assert any(abs(best_res - c) < 1e-9 for c in candidates)

    def test_does_not_rebuild_graph(self, fresh_fitted_model, monkeypatch):
        """tune_resolution_with_llm must reuse self.graph_, never call
        self._graph_builder.build_multiview_graph."""
        called = {"count": 0}
        original = fresh_fitted_model._graph_builder.build_multiview_graph

        def _spy(*args, **kwargs):
            called["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            fresh_fitted_model._graph_builder, "build_multiview_graph", _spy
        )
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert called["count"] == 0
