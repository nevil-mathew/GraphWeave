"""Tests for graphweave.adaptation.triplets (sampling + TripletBank).

No real API is ever called: fake labelers stand in for LLMLabeler, mirroring
the pattern in tests/test_llm_granularity.py.
"""

import json

import numpy as np
import pytest

from graphweave.adaptation.triplets import (
    TripletBank,
    sample_triplets_entropy,
    sample_triplets_hard_margin,
)


def _make_labels_embeddings(seed=0, n_per_cluster=20, dim=8):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat([0, 1, 2], n_per_cluster)
    embs = np.vstack([
        centers[t] + 0.05 * rng.normal(size=(n_per_cluster, dim)) for t in range(3)
    ])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return labels, embs.astype(np.float32)


def _make_probabilities(labels, n_classes=3, low_conf_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    proba = np.zeros((n, n_classes))
    for i, lbl in enumerate(labels):
        proba[i, lbl] = 0.9
        others = [c for c in range(n_classes) if c != lbl]
        for c in others:
            proba[i, c] = 0.1 / len(others)
    n_low_conf = int(low_conf_frac * n)
    low_conf_idx = rng.choice(n, size=n_low_conf, replace=False)
    proba[low_conf_idx] = 1.0 / n_classes
    return proba, low_conf_idx


class TestSampleTripletsEntropy:
    def test_determinism(self):
        labels, embs = _make_labels_embeddings()
        proba, _ = _make_probabilities(labels)
        t1 = sample_triplets_entropy(labels, embs, proba, n_triplets=15, random_state=42)
        t2 = sample_triplets_entropy(labels, embs, proba, n_triplets=15, random_state=42)
        assert t1 == t2

    def test_anchors_favor_high_entropy(self):
        # top_frac's candidate pool (25% of 120 = 30) must be <= the number
        # of true maximum-entropy (uniform-probability) docs (30% = 36), or
        # ties with the next-highest-entropy docs make the subset check
        # flaky by construction.
        labels, embs = _make_labels_embeddings(n_per_cluster=40)
        proba, low_conf_idx = _make_probabilities(labels, low_conf_frac=0.3)
        triplets = sample_triplets_entropy(
            labels, embs, proba, n_triplets=10, random_state=0, top_frac=0.25
        )
        anchors = {a for a, _, _ in triplets}
        assert anchors.issubset(set(low_conf_idx.tolist()))

    def test_outliers_excluded(self):
        labels, embs = _make_labels_embeddings()
        proba, _ = _make_probabilities(labels)
        labels = labels.copy()
        labels[:5] = -1
        triplets = sample_triplets_entropy(labels, embs, proba, n_triplets=15, random_state=0)
        anchors = {a for a, _, _ in triplets}
        assert not anchors & set(range(5))

    def test_fallback_to_fast_without_probabilities(self):
        labels, embs = _make_labels_embeddings()
        triplets = sample_triplets_entropy(labels, embs, None, n_triplets=10, random_state=0)
        assert len(triplets) > 0
        for a, b, c in triplets:
            assert labels[a] == labels[b]
            assert labels[a] != labels[c]


class TestSampleTripletsHardMargin:
    def test_returns_valid_same_diff_triplets(self):
        labels, embs = _make_labels_embeddings()
        triplets = sample_triplets_hard_margin(labels, embs, n_triplets=10, random_state=0)
        assert len(triplets) > 0
        for a, b, c in triplets:
            assert labels[a] == labels[b]
            assert labels[a] != labels[c]


class _CannedLabeler:
    """Cycles through a fixed list of answer-lists, one per call."""

    def __init__(self, answers_per_call):
        self._answers = answers_per_call
        self._i = 0
        self.calls = []

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        ans = self._answers[self._i % len(self._answers)]
        self._i += 1
        return json.dumps({"answers": ans})


class _AllBLabeler:
    def __init__(self):
        self.calls = []

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        n_items = user_prompt.count("Item ")
        self.calls.append((system_prompt, user_prompt))
        return json.dumps({"answers": ["B"] * n_items})


class _RecordingLabeler:
    """Records the max_tokens each call was made with, answers "B" always."""

    def __init__(self):
        self.max_tokens_seen: list[int] = []

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        self.max_tokens_seen.append(max_tokens)
        n_items = user_prompt.count("Item ")
        return json.dumps({"answers": ["B"] * n_items})


class TestTripletBank:
    def test_all_b_labeler_canonical_positive_matches_presented_slot(self):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        bank = TripletBank(random_state=42)
        bank.collect(
            _AllBLabeler(), docs, embs, labels,
            n_triplets=20, sampling="fast", batch_size=5, holdout_frac=0.0,
        )
        assert len(bank.judgments) > 0
        for j in bank.judgments:
            if j.swapped:
                assert labels[j.anchor] != labels[j.positive]
            else:
                assert labels[j.anchor] == labels[j.positive]

    def test_unparseable_answers_excluded_from_train(self):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        bank = TripletBank(random_state=42)
        garbage_labeler = _CannedLabeler([["X"]])  # "X" is not in {B, C} -> None
        bank.collect(
            garbage_labeler, docs, embs, labels,
            n_triplets=10, sampling="fast", batch_size=3, holdout_frac=0.0,
        )
        assert len(bank.judgments) > 0
        assert all(j.raw_answer is None for j in bank.judgments)
        assert bank.train == []
        assert bank.n_unparsed == len(bank.judgments)

    def test_cache_round_trip_avoids_second_llm_call(self, tmp_path):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        cache_path = str(tmp_path / "cache.jsonl")

        bank1 = TripletBank(cache_path=cache_path, random_state=42)
        bank1.collect(_AllBLabeler(), docs, embs, labels, n_triplets=10, sampling="fast", batch_size=4)
        assert bank1.n_llm_calls > 0
        assert bank1.n_cache_hits == 0

        labeler2 = _AllBLabeler()
        bank2 = TripletBank(cache_path=cache_path, random_state=42)
        bank2.collect(labeler2, docs, embs, labels, n_triplets=10, sampling="fast", batch_size=4)
        assert bank2.n_llm_calls == 0
        assert bank2.n_cache_hits == len(bank2.judgments)
        assert len(labeler2.calls) == 0

    def test_holdout_split_disjoint_and_roughly_sized(self):
        labels, embs = _make_labels_embeddings(n_per_cluster=40)
        docs = [f"doc{i}" for i in range(len(labels))]
        bank = TripletBank(random_state=42)
        bank.collect(
            _AllBLabeler(), docs, embs, labels, n_triplets=60, sampling="fast",
            batch_size=8, holdout_frac=0.2,
        )
        train_keys = {(j.anchor, j.positive, j.negative) for j in bank.train}
        holdout_keys = {(j.anchor, j.positive, j.negative) for j in bank.holdout}
        assert train_keys.isdisjoint(holdout_keys)
        frac = len(bank.holdout) / max(len(bank.judgments), 1)
        assert 0.05 < frac < 0.4

    def test_to_training_texts_dedupes_anchors(self):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        bank = TripletBank(random_state=42)
        bank.collect(
            _AllBLabeler(), docs, embs, labels, n_triplets=30, sampling="fast",
            batch_size=5, holdout_frac=0.0,
        )
        texts = bank.to_training_texts(docs)
        assert len(texts["anchor"]) == len(set(texts["anchor"]))
        assert len(texts["anchor"]) == len(texts["positive"]) == len(texts["negative"])

    def test_save_load_round_trip(self, tmp_path):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        bank = TripletBank(random_state=42)
        bank.collect(_AllBLabeler(), docs, embs, labels, n_triplets=15, sampling="fast", batch_size=4)

        path = str(tmp_path / "bank.json")
        bank.save(path)
        loaded = TripletBank.load(path)
        assert loaded.n_llm_calls == bank.n_llm_calls
        assert len(loaded.judgments) == len(bank.judgments)
        assert [j.anchor for j in loaded.judgments] == [j.anchor for j in bank.judgments]

    def test_max_tokens_override_reaches_labeler(self):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        labeler = _RecordingLabeler()
        bank = TripletBank(random_state=42)
        bank.collect(
            labeler, docs, embs, labels, n_triplets=15, sampling="fast",
            batch_size=4, max_tokens=2000,
        )
        assert len(labeler.max_tokens_seen) > 0
        assert all(mt == 2000 for mt in labeler.max_tokens_seen)

    def test_max_tokens_default_is_auto_sized(self):
        labels, embs = _make_labels_embeddings()
        docs = [f"doc{i}" for i in range(len(labels))]
        labeler = _RecordingLabeler()
        bank = TripletBank(random_state=42)
        bank.collect(
            labeler, docs, embs, labels, n_triplets=15, sampling="fast", batch_size=4,
        )
        assert len(labeler.max_tokens_seen) > 0
        # auto-sized (max(128, batch_size * 20 + 64)) — not the override value from the
        # previous test, and small, since this task's answers are compact "B"/"C" tokens.
        assert all(mt < 500 for mt in labeler.max_tokens_seen)
