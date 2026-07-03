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
