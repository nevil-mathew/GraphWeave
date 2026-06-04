"""Tests for cumulative / batch-wise clustering (tritopic.cumulative).

Self-contained: synthetic blob embeddings are passed in precomputed, so these
tests never download an embedding model. A lightweight TriTopicConfig keeps the
reused full-batch fit() fast (no UMAP, no iterative refinement, no lexical view).
"""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from tritopic import TriTopic, TriTopicConfig
from tritopic.core.hierarchy import TopicHierarchy
from tritopic.cumulative import CumulativeConfig, CumulativeTriTopic
from tritopic.cumulative.strategies import STRATEGY_NAMES
from tritopic.cumulative.evaluation import benchmark_strategies, compare_to_full_batch


# --------------------------------------------------------------------------- #
# Synthetic data: fixed topic centers so "same topic" means the same blob.
# --------------------------------------------------------------------------- #
DIM = 32
N_TOPICS = 4
_RNG0 = np.random.default_rng(12345)
CENTERS = _RNG0.normal(size=(N_TOPICS, DIM))
CENTERS /= np.linalg.norm(CENTERS, axis=1, keepdims=True)
TOPIC_WORDS = {t: [f"t{t}_term{j}" for j in range(6)] for t in range(N_TOPICS)}


def make_dataset(topic_ids, per_topic=20, noise=0.05, seed=0):
    """Return (documents, embeddings, truth) for the given topic ids."""
    rng = np.random.default_rng(seed)
    docs, embs, truth = [], [], []
    for t in topic_ids:
        for _ in range(per_topic):
            v = CENTERS[t] + noise * rng.normal(size=DIM)
            v /= np.linalg.norm(v)
            embs.append(v)
            words = rng.choice(TOPIC_WORDS[t], size=8)
            docs.append(" ".join(words))
            truth.append(t)
    return docs, np.asarray(embs, dtype=np.float32), np.asarray(truth)


@pytest.fixture
def light_config():
    """Lightweight full-batch config reused by the cumulative model."""
    return TriTopicConfig(
        use_dim_reduction=False,
        use_iterative_refinement=False,
        use_lexical_view=False,
        n_consensus_runs=3,
        min_cluster_size=3,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )


# --------------------------------------------------------------------------- #
class TestAddBatch:
    def test_first_batch_reclusters(self, light_config):
        docs, embs, _ = make_dataset([0, 1], seed=1)
        cum = CumulativeTriTopic(CumulativeConfig(base_config=light_config))
        res = cum.add_batch(docs, embeddings=embs)

        assert res.reclustered is True
        assert res.epoch == 1
        assert res.novelty is None  # nothing to compare against on the first batch
        assert cum.labels_ is not None
        assert len(cum.labels_) == len(docs)
        assert res.assignments is not None
        assert len(res.assignments) == len(docs)

    def test_second_batch_assigned_without_recluster(self, light_config):
        cfg = CumulativeConfig(base_config=light_config, recluster_trigger="manual")
        cum = CumulativeTriTopic(cfg)
        d1, e1, _ = make_dataset([0, 1], seed=1)
        cum.add_batch(d1, embeddings=e1)

        d2, e2, _ = make_dataset([0, 1], seed=2)
        res = cum.add_batch(d2, embeddings=e2)

        assert res.reclustered is False           # manual trigger never auto-fires
        assert res.novelty is not None
        assert len(res.assignments) == len(d2)
        assert len(cum.labels_) == len(d1) + len(d2)  # but the doc is still accumulated


class TestRecluster:
    @pytest.mark.parametrize("strategy", STRATEGY_NAMES)
    def test_strategy_produces_clusters(self, light_config, strategy):
        cfg = CumulativeConfig(
            base_config=light_config, strategy=strategy, recluster_trigger="manual"
        )
        cum = CumulativeTriTopic(cfg)
        d1, e1, _ = make_dataset([0, 1], seed=1)
        d2, e2, _ = make_dataset([2, 3], seed=2)
        cum.add_batch(d1, embeddings=e1)   # epoch 1 (first always reclusters)
        cum.add_batch(d2, embeddings=e2)   # accumulated, no auto-recluster
        cum.recluster()                    # epoch 2 over everything

        assert cum.model_ is not None
        assert len(cum.labels_) == len(d1) + len(d2)
        assert cum.n_global_topics >= 2
        assert cum.history_[-1].strategy == strategy


class TestTopicAlignment:
    def test_stable_ids_on_identical_data(self, light_config):
        cfg = CumulativeConfig(
            base_config=light_config,
            strategy="global_refit",
            recluster_trigger="manual",
            align_topics=True,
        )
        cum = CumulativeTriTopic(cfg)
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=7)
        cum.add_batch(docs, embeddings=embs)      # epoch 1
        ids1 = set(int(x) for x in cum.labels_)
        cum.recluster()                            # epoch 2, identical accumulator
        ids2 = set(int(x) for x in cum.labels_)

        # Identical data + Hungarian alignment => identical stable global IDs.
        assert ids1 == ids2

    def test_new_topic_gets_fresh_global_id(self, light_config):
        cfg = CumulativeConfig(
            base_config=light_config, strategy="global_refit", recluster_trigger="manual"
        )
        cum = CumulativeTriTopic(cfg)
        d1, e1, _ = make_dataset([0, 1], seed=1)
        cum.add_batch(d1, embeddings=e1)
        n_after_two = cum.n_global_topics

        d2, e2, _ = make_dataset([0, 1, 2, 3], seed=3)
        cum.add_batch(d2, embeddings=e2)
        cum.recluster()

        # Introducing two new blobs must grow the global topic count.
        assert cum.n_global_topics > n_after_two


class TestDriftTrigger:
    def test_drift_fires_on_novel_batch(self, light_config):
        cfg = CumulativeConfig(
            base_config=light_config, recluster_trigger="drift", novelty_threshold=0.3
        )
        cum = CumulativeTriTopic(cfg)
        d1, e1, _ = make_dataset([0, 1], seed=1)
        cum.add_batch(d1, embeddings=e1)

        d2, e2, _ = make_dataset([2, 3], seed=3)   # entirely new themes
        res = cum.add_batch(d2, embeddings=e2)
        assert res.novelty > 0.3
        assert res.reclustered is True

    def test_drift_holds_on_familiar_batch(self, light_config):
        cfg = CumulativeConfig(
            base_config=light_config, recluster_trigger="drift", novelty_threshold=0.3
        )
        cum = CumulativeTriTopic(cfg)
        d1, e1, _ = make_dataset([0, 1], seed=1)
        cum.add_batch(d1, embeddings=e1)

        d2, e2, _ = make_dataset([0, 1], seed=9)   # same themes
        res = cum.add_batch(d2, embeddings=e2)
        assert res.novelty < 0.3
        assert res.reclustered is False

    def test_manual_never_auto_fires(self, light_config):
        cfg = CumulativeConfig(base_config=light_config, recluster_trigger="manual")
        cum = CumulativeTriTopic(cfg)
        d1, e1, _ = make_dataset([0, 1], seed=1)
        cum.add_batch(d1, embeddings=e1)
        d2, e2, _ = make_dataset([2, 3], seed=3)
        res = cum.add_batch(d2, embeddings=e2)
        assert res.reclustered is False


class TestRegimeSwitch:
    def test_coreset_path_above_budget(self, light_config):
        # Force Regime B: cap below the corpus size.
        cfg = CumulativeConfig(
            base_config=light_config,
            strategy="global_refit",
            max_inmemory_docs=30,
        )
        cum = CumulativeTriTopic(cfg)
        docs, embs, _ = make_dataset([0, 1, 2, 3], per_topic=20, seed=1)  # 80 docs
        cum.add_batch(docs, embeddings=embs)

        assert cum.history_[-1].regime == "B"
        assert cum.history_[-1].n_docs_clustered <= 30
        assert len(cum.labels_) == len(docs)   # all docs still get a global label


class TestBiggerPicture:
    def test_hierarchy_built(self, light_config):
        cfg = CumulativeConfig(base_config=light_config, strategy="global_refit")
        cum = CumulativeTriTopic(cfg)
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=1)
        cum.add_batch(docs, embeddings=embs)

        view = cum.bigger_picture(n_levels=2)
        assert isinstance(view["hierarchy"], TopicHierarchy)
        assert view["hierarchy"].n_levels == 2
        assert view["themes"] is None  # no labeler supplied

    def test_get_topic_info_has_global_column(self, light_config):
        cfg = CumulativeConfig(base_config=light_config, strategy="global_refit")
        cum = CumulativeTriTopic(cfg)
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=1)
        cum.add_batch(docs, embeddings=embs)

        df = cum.get_topic_info()
        assert isinstance(df, pd.DataFrame)
        assert "GlobalTopic" in df.columns


class TestReuseAndRegression:
    def test_global_refit_matches_full_batch(self, light_config):
        """global_refit (Regime A) must reproduce a standalone full-batch fit() —
        proving fit() is reused unchanged and the orchestration is faithful."""
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=7)

        full = TriTopic(config=copy.deepcopy(light_config))
        full.fit(docs, embeddings=embs)

        cum = CumulativeTriTopic(
            CumulativeConfig(base_config=light_config, strategy="global_refit")
        )
        cum.add_batch(docs, embeddings=embs)

        # Same data, same seed, same pipeline => identical partition (up to IDs).
        assert adjusted_rand_score(full.labels_, cum.labels_) >= 0.99

    def test_importing_cumulative_does_not_break_fit(self, light_config):
        import tritopic.cumulative  # noqa: F401

        docs, embs, _ = make_dataset([0, 1], seed=1)
        model = TriTopic(config=copy.deepcopy(light_config))
        labels = model.fit_transform(docs, embeddings=embs)
        assert len(labels) == len(docs)


class TestBenchmarkHarness:
    def test_benchmark_strategies_runs(self, light_config):
        d1, e1, t1 = make_dataset([0, 1], seed=1)
        d2, e2, t2 = make_dataset([2, 3], seed=2)
        batches = [d1, d2]
        batch_embs = [e1, e2]
        truth = np.concatenate([t1, t2])

        df = benchmark_strategies(
            batches,
            base_config=light_config,
            labels_true=truth,
            precomputed_batch_embeddings=batch_embs,
        )

        assert isinstance(df, pd.DataFrame)
        assert set(df["strategy"]) == set(STRATEGY_NAMES)
        for col in ("ari_vs_full", "nmi_vs_full", "topic_count_drift", "keyword_overlap"):
            assert col in df.columns
