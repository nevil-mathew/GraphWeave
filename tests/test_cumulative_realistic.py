"""Realistic, model-free tests for cumulative clustering.

These exercise the *full* pipeline (real text -> TF-IDF + LSA embeddings ->
multi-view graph -> Leiden -> c-TF-IDF keywords) on a corpus with the messiness
that breaks naive clustering: overlapping vocabulary, imbalanced topic sizes,
noise documents, and an emerging topic that appears partway through the stream.
No embedding model is downloaded — embeddings come from LSA (TruncatedSVD on
TF-IDF), the classic pre-neural document embedder.
"""

import numpy as np
import pytest

from tritopic import TriTopicConfig
from tritopic.cumulative import (
    CumulativeConfig,
    CumulativeTriTopic,
    STRATEGY_NAMES,
    lsa_embed,
    make_streaming_corpus,
)
from tritopic.cumulative.evaluation import benchmark_strategies


@pytest.fixture(scope="module")
def realistic_config():
    """Full multi-view pipeline on real-ish text, kept fast (no UMAP / no iteration)."""
    return TriTopicConfig(
        use_dim_reduction=False,        # LSA already produced dense, low-dim vectors
        use_lexical_view=True,          # real text -> TF-IDF lexical graph (more realistic)
        use_iterative_refinement=False,
        n_consensus_runs=4,
        min_cluster_size=5,
        n_neighbors=15,
        random_state=42,
        verbose=False,
    )


@pytest.fixture(scope="module")
def stationary_corpus():
    return make_streaming_corpus(
        n_topics=6, docs_per_batch=120, n_batches=5,
        overlap=0.18, noise_frac=0.05, random_state=1,
    )


@pytest.fixture(scope="module")
def emerging_corpus():
    return make_streaming_corpus(
        n_topics=6, docs_per_batch=120, n_batches=5,
        overlap=0.18, noise_frac=0.05,
        emerging_topic_at=3, emerging_frac=0.35, random_state=2,
    )


# --------------------------------------------------------------------------- #
class TestLsaEmbedder:
    def test_shape_and_normalized(self):
        docs = [f"alpha beta gamma word{i % 7}" for i in range(50)]
        emb = lsa_embed(docs, dim=16)
        assert emb.shape[0] == 50
        assert emb.shape[1] <= 16
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_deterministic(self):
        docs = [f"alpha beta word{i % 5} term{i % 3}" for i in range(40)]
        assert np.allclose(lsa_embed(docs, dim=12), lsa_embed(docs, dim=12))


class TestRealisticStationary:
    """Stable themes: cheap transform-assignment between reclusters should stay
    close to a full-batch fit, and the structure should be recovered well."""

    def test_recovers_structure(self, realistic_config, stationary_corpus):
        corp = stationary_corpus
        cum = CumulativeTriTopic(
            CumulativeConfig(base_config=realistic_config, strategy="global_refit",
                             recluster_trigger="drift", novelty_threshold=0.25)
        )
        for docs, emb in zip(corp.batches, corp.batch_embeddings):
            cum.add_batch(docs, embeddings=emb)

        from tritopic.utils.metrics import compute_ari
        ari_truth = compute_ari(cum.labels_, corp.all_labels)

        assert len(cum.labels_) == corp.n_docs
        assert ari_truth >= 0.70                      # recovers the real topics well
        assert corp.n_topics - 2 <= cum.n_global_topics <= corp.n_topics + 2

    def test_close_to_full_batch(self, realistic_config, stationary_corpus):
        corp = stationary_corpus
        full_cfg = realistic_config
        from tritopic import TriTopic
        import copy
        full = TriTopic(config=copy.deepcopy(full_cfg))
        full.fit(corp.all_documents, embeddings=corp.all_embeddings)

        cum = CumulativeTriTopic(
            CumulativeConfig(base_config=full_cfg, strategy="global_refit",
                             recluster_trigger="drift", novelty_threshold=0.25)
        )
        for docs, emb in zip(corp.batches, corp.batch_embeddings):
            cum.add_batch(docs, embeddings=emb)

        from tritopic.utils.metrics import compute_ari
        # Even when most of the stream is assigned cheaply (no refit), the
        # cumulative labelling stays very close to a full-batch fit.
        assert compute_ari(cum.labels_, full.labels_) >= 0.85


class TestEmergingTopicDrift:
    """A new theme appears mid-stream; drift detection should catch it and a
    recluster should discover the new topic."""

    def test_drift_detects_new_theme(self, realistic_config, emerging_corpus):
        corp = emerging_corpus
        cum = CumulativeTriTopic(
            CumulativeConfig(base_config=realistic_config, strategy="global_refit",
                             recluster_trigger="drift", novelty_threshold=0.15)
        )
        # Capture state *during* streaming (not after).
        results = []
        topic_counts = []
        for d, e in zip(corp.batches, corp.batch_embeddings):
            results.append(cum.add_batch(d, embeddings=e))
            topic_counts.append(cum.n_global_topics)

        # Batches 1 and 2 are stationary -> no recluster.
        assert results[1].reclustered is False
        assert results[2].reclustered is False

        # Batch 3 introduces the new theme -> novelty spikes -> recluster fires.
        assert results[3].novelty is not None and results[3].novelty > 0.15
        assert results[3].reclustered is True

        # The recluster at batch 3 discovered an additional global topic.
        assert topic_counts[3] > topic_counts[2]
        assert cum.n_global_topics >= corp.n_topics - 1


class TestStrategyComparisonRealistic:
    """Head-to-head on the same stream (recluster every batch) so the
    quality/cost tradeoff between engines is visible and ordered as expected."""

    def test_global_refit_is_best(self, realistic_config, emerging_corpus):
        corp = emerging_corpus
        df = benchmark_strategies(
            corp.batches,
            base_config=realistic_config,
            labels_true=corp.all_labels,
            precomputed_batch_embeddings=corp.batch_embeddings,
            cumulative_kwargs=dict(recluster_trigger="schedule",
                                   schedule_every_n_docs=120),
        )

        assert set(df["strategy"]) == set(STRATEGY_NAMES)
        row = {r["strategy"]: r for _, r in df.iterrows()}

        # global_refit reproduces the full-batch baseline very closely.
        assert row["global_refit"]["ari_vs_full"] >= 0.95
        # It is at least as good as the cheap batch-local merge.
        assert row["global_refit"]["ari_vs_full"] >= row["batch_merge"]["ari_vs_full"] - 1e-9
        # Every strategy still recovers the real structure reasonably.
        for strat in STRATEGY_NAMES:
            assert row[strat]["ari_vs_truth_cumulative"] >= 0.6
            assert np.isfinite(row[strat]["ari_vs_full"])
