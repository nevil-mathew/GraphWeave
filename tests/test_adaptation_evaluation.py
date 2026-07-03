"""Tests for tritopic.adaptation.evaluation."""

import math

import numpy as np

from tritopic.adaptation.evaluation import (
    compare_embedders,
    forgetting_check,
    hungarian_accuracy,
    triplet_accuracy,
)
from tritopic.adaptation.triplets import TripletJudgment


def test_triplet_accuracy_hand_built_geometry():
    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
    ])
    correct = [TripletJudgment(anchor=0, positive=1, negative=2, swapped=False, raw_answer="B", split="holdout")]
    assert triplet_accuracy(embs, correct) == 1.0

    wrong = [TripletJudgment(anchor=0, positive=2, negative=1, swapped=False, raw_answer="B", split="holdout")]
    assert triplet_accuracy(embs, wrong) == 0.0


def test_triplet_accuracy_empty_is_nan():
    assert math.isnan(triplet_accuracy(np.zeros((2, 2)), []))


def test_hungarian_accuracy_permuted_labels_is_one():
    labels_true = np.array([0, 0, 1, 1, 2, 2])
    labels_pred = np.array([5, 5, 9, 9, 1, 1])
    assert hungarian_accuracy(labels_pred, labels_true) == 1.0


def test_hungarian_accuracy_excludes_outliers():
    labels_true = np.array([0, 0, 1, 1])
    labels_pred = np.array([-1, 0, 1, 1])
    acc = hungarian_accuracy(labels_pred, labels_true)
    assert 0.0 <= acc <= 1.0


def test_forgetting_check_identity_encoder_zero_delta():
    def fake_encode(texts):
        return np.array([[len(t), t.count(" ")] for t in texts], dtype=float)

    result = forgetting_check(fake_encode, fake_encode)
    assert set(result) == {"spearman_before", "spearman_after", "delta"}
    assert abs(result["delta"]) < 1e-9


def test_compare_embedders_prefers_clean_over_noisy():
    from tritopic.core.model import TriTopicConfig
    from tritopic.cumulative.datasets import make_streaming_corpus

    corpus = make_streaming_corpus(n_topics=3, docs_per_batch=40, n_batches=1, random_state=0)
    docs = corpus.all_documents
    clean = corpus.all_embeddings
    rng = np.random.default_rng(0)
    noisy = clean + rng.normal(scale=2.0, size=clean.shape)
    noisy = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)

    base_config = TriTopicConfig(
        use_dim_reduction=False, use_iterative_refinement=False,
        n_consensus_runs=3, min_cluster_size=5, n_neighbors=10, verbose=False,
    )

    df = compare_embedders(
        docs, {"clean": clean, "noisy": noisy}, labels_true=corpus.all_labels,
        base_config=base_config, n_seeds=1,
    )
    assert set(df["variant"]) == {"clean", "noisy"}
    clean_ari = df.loc[df.variant == "clean", "ari"].iloc[0]
    noisy_ari = df.loc[df.variant == "noisy", "ari"].iloc[0]
    assert clean_ari >= noisy_ari
    assert "keyword_overlap_vs_baseline" in df.columns
