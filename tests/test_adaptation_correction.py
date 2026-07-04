"""Tests for tritopic.adaptation.correction."""

import json

import numpy as np

from tritopic import TriTopic, TriTopicConfig
from tritopic.adaptation.correction import reassign_low_confidence


def _fit_small_model():
    rng = np.random.default_rng(0)
    docs, labels = [], []
    for t in range(3):
        for i in range(20):
            docs.append(f"topic {t} document {i} words about area {t}")
            labels.append(t)
    centers = rng.normal(size=(3, 12))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embs = np.vstack([centers[t] + 0.4 * rng.normal(size=(20, 12)) for t in range(3)])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    cfg = TriTopicConfig(
        use_dim_reduction=False, use_iterative_refinement=False,
        n_consensus_runs=3, min_cluster_size=5, n_neighbors=10,
        random_state=42, verbose=False,
    )
    return TriTopic(config=cfg).fit(docs, embeddings=embs.astype(np.float32))


class _AlwaysNoneLabeler:
    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        return json.dumps({"choice": "none"})


class _AlwaysFirstChoiceLabeler:
    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        return json.dumps({"choice": "0"})


def test_dry_run_leaves_labels_unchanged():
    model = _fit_small_model()
    original_labels = model.labels_.copy()
    df = reassign_low_confidence(
        model, _AlwaysFirstChoiceLabeler(), margin_threshold=1.0, max_docs=10, dry_run=True
    )
    np.testing.assert_array_equal(model.labels_, original_labels)
    assert not df.empty


def test_zero_threshold_selects_no_docs():
    model = _fit_small_model()
    df = reassign_low_confidence(model, _AlwaysNoneLabeler(), margin_threshold=0.0, max_docs=10)
    assert df.empty


def test_max_docs_respected():
    model = _fit_small_model()
    df = reassign_low_confidence(model, _AlwaysNoneLabeler(), margin_threshold=1.0, max_docs=5)
    assert len(df) <= 5


class _AlwaysSecondChoiceLabeler:
    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        return json.dumps({"choice": "1"})


def test_topic_emptied_entirely_does_not_crash():
    """Force every document in one topic to be reassigned to another topic,
    fully draining it. Used to crash _compute_topic_centroids() (np.average
    / .mean over an empty slice -> NaN centroid, then NaN probabilities)."""
    model = _fit_small_model()
    topics = [t for t in model.topics_ if t.topic_id != -1]
    topic_order = [t.topic_id for t in topics]  # column order matches probabilities_
    n_topics = len(topic_order)
    target_topic = topic_order[0]
    target_mask = model.labels_ == target_topic

    # Hand-craft probabilities_ so every target_topic doc has a tiny margin
    # between its own topic (col 0, current top-1) and topic_order[1]
    # (col 1, top-2) — with choice="1" that reassigns 0 -> topic_order[1] for
    # every one of them. Every other doc gets a huge margin so it sorts last
    # and stays outside max_docs, leaving the rest of the model untouched.
    proba = np.full((len(model.labels_), n_topics), 0.01)
    proba[~target_mask, 0] = 0.9
    proba[target_mask, 0] = 0.40
    proba[target_mask, 1] = 0.35
    model.probabilities_ = proba

    df = reassign_low_confidence(
        model,
        _AlwaysSecondChoiceLabeler(),
        margin_threshold=1.0,  # select every document regardless of confidence
        top_k=2,
        max_docs=int(target_mask.sum()),
    )

    assert not df.empty
    assert (df["new_topic"] == topic_order[1]).all()
    remaining_topic_ids = {t.topic_id for t in model.topics_ if t.topic_id != -1}
    assert target_topic not in remaining_topic_ids
    for topic_id in remaining_topic_ids:
        assert np.any(model.labels_ == topic_id)  # every surviving topic still has members
    assert not np.any(np.isnan(model.topic_embeddings_))
    assert not np.any(np.isnan(model.probabilities_))
