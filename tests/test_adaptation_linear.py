"""Tests for tritopic.adaptation.adapter.LinearAdapter — pure numpy, no
network, no fine-tuning dependencies required."""

import numpy as np

from tritopic.adaptation.adapter import LinearAdapter
from tritopic.adaptation.evaluation import triplet_accuracy
from tritopic.adaptation.triplets import TripletJudgment


def _make_clusters(seed=0, n_per_cluster=30, dim=16, noise=0.5):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.repeat([0, 1, 2], n_per_cluster)
    embs = np.vstack([
        centers[t] + noise * rng.normal(size=(n_per_cluster, dim)) for t in range(3)
    ])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return labels.astype(int), embs.astype(np.float64)


def _oracle_judgments(labels, n=400, seed=0):
    rng = np.random.default_rng(seed)
    n_docs = len(labels)
    judgments = []
    for i in range(n):
        a = int(rng.integers(n_docs))
        same_idx = np.where(labels == labels[a])[0]
        same_idx = same_idx[same_idx != a]
        diff_idx = np.where(labels != labels[a])[0]
        if len(same_idx) == 0 or len(diff_idx) == 0:
            continue
        b = int(rng.choice(same_idx))
        c = int(rng.choice(diff_idx))
        split = "holdout" if i % 5 == 0 else "train"
        judgments.append(
            TripletJudgment(anchor=a, positive=b, negative=c, swapped=False, raw_answer="B", split=split)
        )
    return judgments


class TestLinearAdapter:
    def test_identity_init_reproduces_input(self):
        _, embs = _make_clusters()
        adapter = LinearAdapter(dim=embs.shape[1])
        out = adapter.transform(embs)
        expected = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        np.testing.assert_allclose(out, expected, atol=1e-8)

    def test_training_improves_or_maintains_holdout_triplet_accuracy(self):
        labels, embs = _make_clusters(noise=0.5)
        judgments = _oracle_judgments(labels, n=400)
        train = [j for j in judgments if j.split == "train"]
        holdout = [j for j in judgments if j.split == "holdout"]
        assert len(train) > 20 and len(holdout) > 5

        acc_before = triplet_accuracy(embs, holdout)

        adapter = LinearAdapter(dim=embs.shape[1], random_state=42)
        adapter.fit(embs, train, epochs=60, lr=0.1, margin=0.2, l2=1e-4, batch_size=64)
        acc_after = triplet_accuracy(adapter.transform(embs), holdout)

        assert acc_after >= acc_before
        assert adapter.history_[-1] <= adapter.history_[0]

    def test_l2_shrinkage_bounds_deviation_from_identity(self):
        labels, embs = _make_clusters()
        train = [j for j in _oracle_judgments(labels, n=300) if j.split == "train"]

        weak = LinearAdapter(dim=embs.shape[1], random_state=0).fit(embs, train, epochs=30, l2=1e-6)
        strong = LinearAdapter(dim=embs.shape[1], random_state=0).fit(embs, train, epochs=30, l2=1.0)

        identity = np.eye(embs.shape[1])
        dev_weak = np.linalg.norm(weak.W - identity)
        dev_strong = np.linalg.norm(strong.W - identity)
        assert dev_strong < dev_weak

    def test_determinism_same_seed(self):
        labels, embs = _make_clusters()
        train = [j for j in _oracle_judgments(labels, n=100) if j.split == "train"]

        a1 = LinearAdapter(dim=embs.shape[1], random_state=7).fit(embs, train, epochs=10)
        a2 = LinearAdapter(dim=embs.shape[1], random_state=7).fit(embs, train, epochs=10)
        np.testing.assert_allclose(a1.W, a2.W)

    def test_save_load_round_trip(self, tmp_path):
        labels, embs = _make_clusters()
        train = [j for j in _oracle_judgments(labels, n=100) if j.split == "train"]

        adapter = LinearAdapter(dim=embs.shape[1], random_state=1).fit(embs, train, epochs=10)
        path = str(tmp_path / "adapter")
        adapter.save(path)
        loaded = LinearAdapter.load(path)
        np.testing.assert_allclose(adapter.W, loaded.W)

    def test_empty_judgments_is_noop(self):
        _, embs = _make_clusters()
        adapter = LinearAdapter(dim=embs.shape[1])
        adapter.fit(embs, [])
        np.testing.assert_allclose(adapter.W, np.eye(embs.shape[1]))
