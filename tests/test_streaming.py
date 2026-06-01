"""Tests for the streaming / incremental batch pipeline."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from tritopic import StreamingTriTopic, TriTopic, TriTopicConfig

D = 64
N_BASE_TOPICS = 5
RNG_SEED = 42


def _topic_center(topic_id: int) -> np.ndarray:
    g = np.random.default_rng(1000 + topic_id)
    c = g.normal(size=D)
    return c / np.linalg.norm(c) * 5.0


CENTERS = {tid: _topic_center(tid) for tid in range(N_BASE_TOPICS + 1)}
VOCAB = {
    0: ["finance", "markets", "stocks", "bonds", "trading"],
    1: ["sports", "football", "league", "player", "match"],
    2: ["politics", "election", "policy", "government", "senate"],
    3: ["tech", "software", "startup", "cloud", "ai"],
    4: ["health", "doctor", "hospital", "patient", "study"],
    5: ["climate", "carbon", "emissions", "renewable", "solar"],
}


def gen_batch(topic_distribution, noise=0.4, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    docs, embs = [], []
    for topic_id, n in topic_distribution.items():
        words = VOCAB[topic_id]
        center = CENTERS[topic_id]
        for _ in range(n):
            length = rng.integers(8, 16)
            chosen = rng.choice(words, size=length, replace=True)
            docs.append(" ".join(chosen))
            embs.append(center + rng.normal(scale=noise, size=D))
    embs_arr = np.stack(embs).astype(np.float32)
    perm = rng.permutation(len(docs))
    return [docs[i] for i in perm], embs_arr[perm]


def _streaming_config(**overrides):
    base = dict(
        mode="streaming",
        use_dim_reduction=False,
        use_iterative_refinement=False,
        n_consensus_runs=3,
        min_cluster_size=10,
        reseed_pool_size=80,
        promote_min_batches=2,
        promote_min_docs=120,
        promote_min_coherence=0.55,
        refit_every_n_batches=0,
        keyword_refresh_every_n_batches=0,
        verbose=False,
    )
    base.update(overrides)
    return TriTopicConfig(**base)


@pytest.fixture(scope="module")
def seeded_model():
    cfg = _streaming_config()
    model = TriTopic(config=cfg)
    docs, embs = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120}, seed=1)
    model.fit(docs, embeddings=embs)
    return model


class TestSeeding:
    def test_themes_created(self, seeded_model):
        backend = seeded_model._streaming_backend
        assert len(backend.themes) >= 2  # depending on clustering, may merge some

    def test_per_theme_thresholds_calibrated(self, seeded_model):
        backend = seeded_model._streaming_backend
        for theme in backend.themes.values():
            assert 0.0 < theme.assign_threshold < 1.0
            assert theme.review_threshold <= theme.assign_threshold

    def test_facade_topics_synced(self, seeded_model):
        backend = seeded_model._streaming_backend
        assert len(seeded_model.topics_) == len(backend.themes)
        assert seeded_model.topic_embeddings_.shape[0] == len(backend.themes)
        assert seeded_model._is_fitted is True


class TestRouting:
    def test_assigns_known_distribution(self):
        cfg = _streaming_config()
        model = TriTopic(config=cfg)
        d0, e0 = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120}, seed=1)
        model.fit(d0, embeddings=e0)

        d1, e1 = gen_batch({0: 40, 1: 40, 2: 40, 3: 40, 4: 40}, seed=2)
        result = model.add_batch(d1, embeddings=e1)

        statuses = [a["status"] for a in result["assignments"]]
        n_assigned = statuses.count("assigned")
        # On synthetic well-separated blobs, the majority must land in the
        # assigned bucket — review + outlier are the long tail.
        assert n_assigned / len(d1) >= 0.55

    def test_unassigned_pool_fills_on_new_distribution(self):
        cfg = _streaming_config(reseed_pool_size=10_000)  # disable reseed for this test
        model = TriTopic(config=cfg)
        d0, e0 = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120}, seed=1)
        model.fit(d0, embeddings=e0)

        d_new, e_new = gen_batch({5: 100}, seed=3)
        model.add_batch(d_new, embeddings=e_new)
        backend = model._streaming_backend
        # Most brand-new-distribution docs should hit the pool.
        assert len(backend.unassigned_pool) >= 40


class TestPublicCountMonotonic:
    def test_historical_max_never_decreases(self):
        cfg = _streaming_config()
        model = TriTopic(config=cfg)
        d0, e0 = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120}, seed=1)
        model.fit(d0, embeddings=e0)

        backend = model._streaming_backend
        prev_max = {tid: t.historical_max for tid, t in backend.themes.items()}
        for seed in (2, 3, 4):
            d, e = gen_batch({0: 30, 1: 30, 2: 30, 3: 30, 4: 30}, seed=seed)
            model.add_batch(d, embeddings=e)
            for tid, t in backend.themes.items():
                if tid in prev_max:
                    assert t.historical_max >= prev_max[tid], (
                        f"theme {tid} historical_max regressed"
                    )
                prev_max[tid] = t.historical_max


class TestEmergingPath:
    def test_pool_reseed_pushes_emerging(self):
        cfg = _streaming_config(reseed_pool_size=60, merge_threshold=0.99)
        # merge_threshold=0.99 forces the reseed to create emerging clusters
        # rather than merging into existing themes.
        model = TriTopic(config=cfg)
        d0, e0 = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120}, seed=1)
        model.fit(d0, embeddings=e0)

        d_new, e_new = gen_batch({5: 80}, seed=3)
        model.add_batch(d_new, embeddings=e_new)
        backend = model._streaming_backend
        assert len(backend.emerging) >= 1

    def test_emerging_promotes_after_enough_evidence(self):
        cfg = _streaming_config(
            reseed_pool_size=60,
            merge_threshold=0.99,
            promote_min_batches=2,
            promote_min_docs=80,
            promote_min_coherence=0.5,
        )
        model = TriTopic(config=cfg)
        d0, e0 = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120}, seed=1)
        model.fit(d0, embeddings=e0)

        starting_themes = len(model._streaming_backend.themes)

        for seed in (3, 4, 5):
            d, e = gen_batch({5: 70}, seed=seed)
            model.add_batch(d, embeddings=e)

        backend = model._streaming_backend
        # Either the emerging cluster promoted (most likely) or it's grown
        # large enough to be on the verge.
        promoted = len(backend.themes) > starting_themes
        emerging_large = any(len(ec.docs) >= 80 for ec in backend.emerging)
        assert promoted or emerging_large


class TestModeFlag:
    def test_single_mode_still_works(self):
        cfg = TriTopicConfig(
            mode="single",
            use_dim_reduction=False,
            use_iterative_refinement=False,
            n_consensus_runs=3,
            min_cluster_size=10,
            verbose=False,
        )
        model = TriTopic(config=cfg)
        d, e = gen_batch({0: 80, 1: 80, 2: 80}, seed=1)
        model.fit(d, embeddings=e)
        assert model._is_fitted
        assert model._streaming_backend is None
        assert model.labels_ is not None

    def test_add_batch_rejected_in_single_mode(self):
        cfg = TriTopicConfig(
            mode="single",
            use_dim_reduction=False,
            use_iterative_refinement=False,
            n_consensus_runs=3,
            min_cluster_size=10,
            verbose=False,
        )
        model = TriTopic(config=cfg)
        d, e = gen_batch({0: 80, 1: 80, 2: 80}, seed=1)
        model.fit(d, embeddings=e)
        d2, e2 = gen_batch({0: 30}, seed=2)
        with pytest.raises(RuntimeError, match="streaming"):
            model.add_batch(d2, embeddings=e2)


class TestTransform:
    def test_transform_uses_per_theme_thresholds(self, seeded_model):
        d, e = gen_batch({0: 30, 1: 30}, seed=2)
        labels = seeded_model.transform(d, embeddings=e)
        assert labels.shape == (len(d),)
        # The bulk should not be outliers since these are in-distribution.
        assigned = (labels != -1).sum()
        assert assigned / len(d) >= 0.5


class TestSaveLoad:
    def test_streaming_save_load_roundtrip(self):
        cfg = _streaming_config()
        model = TriTopic(config=cfg)
        d0, e0 = gen_batch({0: 100, 1: 100, 2: 100}, seed=1)
        model.fit(d0, embeddings=e0)
        d1, e1 = gen_batch({0: 30, 1: 30, 2: 30}, seed=2)
        model.add_batch(d1, embeddings=e1)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = TriTopic.load(path)
            assert loaded._streaming_backend is not None
            assert len(loaded._streaming_backend.themes) == len(model._streaming_backend.themes)
            for tid in model._streaming_backend.themes:
                assert tid in loaded._streaming_backend.themes
                np.testing.assert_array_almost_equal(
                    loaded._streaming_backend.themes[tid].centroid,
                    model._streaming_backend.themes[tid].centroid,
                )
        finally:
            os.remove(path)
