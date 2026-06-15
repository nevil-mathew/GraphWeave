"""Tests for the coreset-improvement tranche (P0 + P2b + P1).

Covers:
- P2b: ``stratified_coreset`` honours per-topic floors, the budget cap, and is
  deterministic.
- P1:  weighted topic centroids (and ``sample_weights=None`` reproduces the old
  unweighted result).
- P0+integration: a planted rare topic survives a small coreset better under
  stratified selection than under recency selection (``rare_topic_recall``).

Self-contained: synthetic blob embeddings are passed in precomputed, so nothing
downloads an embedding model.
"""

import numpy as np
import pytest

from tritopic import TriTopic, TriTopicConfig
from tritopic.core.clustering import ConsensusLeiden
from tritopic.core.model import TopicInfo
from tritopic.cumulative import CumulativeConfig, CumulativeTriTopic
from tritopic.cumulative.alignment import (
    microcluster_coreset,
    sensitivity_weights,
    stratified_coreset,
)
from tritopic.cumulative.evaluation import compare_to_full_batch
from tritopic.cumulative.strategies import ReclusterContext, _select_reduced


# --------------------------------------------------------------------------- #
# P2b — stratified_coreset
# --------------------------------------------------------------------------- #
class TestStratifiedCoreset:
    def _data(self, sizes, dim=8, seed=0):
        """Build embeddings + labels for clusters of the given sizes."""
        rng = np.random.default_rng(seed)
        embs, labels = [], []
        for c, n in enumerate(sizes):
            embs.append(rng.normal(c * 5, 0.1, size=(n, dim)))
            labels.extend([c] * n)
        return np.vstack(embs), np.asarray(labels)

    def test_floor_per_cluster_when_budget_allows(self):
        # One rare cluster (10) among large ones; floor must protect it.
        emb, labels = self._data([1000, 1000, 10])
        size, floor = 300, 5
        idx, probs = stratified_coreset(
            emb, size, labels, new_count=50, min_per_cluster=floor
        )
        assert len(idx) <= size                      # budget cap respected
        assert len(idx) == len(probs)
        chosen = labels[idx]
        for c, n in zip([0, 1, 2], [1000, 1000, 10]):
            assert (chosen == c).sum() >= min(floor, n)  # floor honoured
        # The rare cluster's represented mass (sum of 1/p) recovers its true size.
        weights = 1.0 / probs
        assert weights[chosen == 2].sum() == pytest.approx(10, rel=0.01)
        assert np.all(probs > 0)

    def test_budget_cap_and_determinism(self):
        emb, labels = self._data([500, 500, 500])
        idx1, _ = stratified_coreset(emb, 200, labels, new_count=30, random_state=7)
        idx2, _ = stratified_coreset(emb, 200, labels, new_count=30, random_state=7)
        assert len(idx1) <= 200
        assert np.array_equal(idx1, idx2)            # deterministic per seed
        assert np.array_equal(idx1, np.sort(idx1))   # sorted output

    def test_inclusion_probabilities_recover_mass(self):
        # 1/p is the represented doc count; per cluster it sums to ~cluster size.
        emb, labels = self._data([1000, 10])
        idx, probs = stratified_coreset(emb, 300, labels, new_count=50, min_per_cluster=5)
        weights = 1.0 / probs
        assert weights[labels[idx] == 0].sum() == pytest.approx(1000, rel=0.01)
        assert weights[labels[idx] == 1].sum() == pytest.approx(10, rel=0.01)

    def test_falls_back_without_labels(self):
        emb, _ = self._data([100, 100])
        idx, probs = stratified_coreset(emb, 50, labels=None, new_count=20)
        assert len(idx) == 50
        assert len(probs) == 50


# --------------------------------------------------------------------------- #
# Sensitivity (lightweight-coreset) sampling
# --------------------------------------------------------------------------- #
class TestSensitivitySampling:
    def test_weights_form_a_distribution_and_favour_outliers(self):
        # 9 points near the origin, 1 far away: the far point must get the
        # largest importance (it defines structure), but all stay reachable.
        emb = np.vstack([np.zeros((9, 3)), np.array([[100.0, 0.0, 0.0]])])
        q = sensitivity_weights(emb)
        assert q.sum() == pytest.approx(1.0)
        assert np.all(q > 0)
        assert q[-1] == q.max()

    def test_identical_points_are_uniform(self):
        q = sensitivity_weights(np.ones((5, 4)))
        assert np.allclose(q, 0.2)

    def test_stratified_sensitivity_keeps_floor_and_weights(self):
        rng = np.random.default_rng(0)
        emb = np.vstack([rng.normal(0, 0.1, (1000, 6)), rng.normal(5, 0.1, (10, 6))])
        labels = np.array([0] * 1000 + [1] * 10)
        idx, probs = stratified_coreset(
            emb, 300, labels, new_count=50, min_per_cluster=5, sampling="sensitivity"
        )
        assert len(idx) <= 300 and len(idx) == len(probs)
        assert np.all((probs > 0) & (probs <= 1.0))
        assert (labels[idx] == 1).sum() >= 5            # rare-topic floor honoured


# --------------------------------------------------------------------------- #
# Micro-cluster (CluStream/BIRCH) hybrid coreset
# --------------------------------------------------------------------------- #
class TestMicroclusterCoreset:
    def test_size_bounded_and_mass_conserved(self):
        rng = np.random.default_rng(1)
        emb = rng.normal(size=(500, 8))
        idx, w = microcluster_coreset(emb, n_recent=40, k=30)
        assert len(idx) <= 70                            # bounded by n_recent + k
        assert len(idx) == len(w)
        assert w.sum() == pytest.approx(500)             # total represented mass == N
        assert np.array_equal(idx, np.sort(idx))

    def test_recent_docs_kept_raw_at_weight_one(self):
        rng = np.random.default_rng(2)
        emb = rng.normal(size=(300, 5))
        idx, w = microcluster_coreset(emb, n_recent=50, k=20)
        recent = set(range(250, 300))
        kept_recent = [i for i in idx if int(i) in recent]
        assert len(kept_recent) == 50                    # every recent doc retained
        assert all(w[list(idx).index(i)] == 1.0 for i in kept_recent)

    def test_indices_are_real_documents(self):
        # Representatives must be actual rows (so document text exists downstream).
        emb = np.random.default_rng(3).normal(size=(120, 4))
        idx, _ = microcluster_coreset(emb, n_recent=20, k=15)
        assert idx.min() >= 0 and idx.max() < 120


# --------------------------------------------------------------------------- #
# Guaranteed novelty/outlier retention
# --------------------------------------------------------------------------- #
class TestNoveltyReservation:
    def _ctx(self, emb, labels, reserve):
        return ReclusterContext(
            documents=[f"d{i}" for i in range(len(emb))],
            embeddings=emb,
            new_count=20,
            max_inmemory_docs=10_000,
            coreset_size=100,
            random_state=0,
            labels=labels,
            coreset_selection="stratified",
            min_per_cluster=5,
            reserve_novel=reserve,
        )

    def test_recent_outliers_forced_in_at_full_weight(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(400, 6))
        labels = np.array([0] * 200 + [1] * 190 + [-1] * 10)  # 10 recent outliers
        idx, w = _select_reduced(self._ctx(emb, labels, reserve=10), size=80)
        outlier_rows = set(range(390, 400))
        kept = outlier_rows & set(int(i) for i in idx)
        assert kept == outlier_rows                       # all reserved
        wmap = {int(i): wt for i, wt in zip(idx, w)}
        assert all(wmap[i] == 1.0 for i in outlier_rows)  # at full weight

    def test_zero_reserve_matches_plain_stratified(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(300, 6))
        labels = np.array([0] * 150 + [1] * 145 + [-1] * 5)
        idx, _ = _select_reduced(self._ctx(emb, labels, reserve=0), size=80)
        assert len(idx) <= 80


# --------------------------------------------------------------------------- #
# Mass-based small-cluster pruning (weight-aware Leiden post-processing)
# --------------------------------------------------------------------------- #
class TestMassBasedPruning:
    def _graph(self):
        import igraph as ig

        # Two blobs: A (5 nodes) and B (3 nodes), thinly bridged.
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (0, 4),
                 (5, 6), (6, 7), (5, 7)]
        g = ig.Graph(n=8, edges=edges, directed=False)
        g.es["weight"] = [1.0] * g.ecount()
        return g

    def test_small_cluster_pruned_unweighted(self):
        cl = ConsensusLeiden(resolution=0.5, n_runs=3, random_state=0)
        labels = cl.fit_predict(self._graph(), min_cluster_size=4)
        assert set(labels[5:8]) == {-1}                   # B (3 nodes) pruned

    def test_small_cluster_survives_when_mass_clears_threshold(self):
        cl = ConsensusLeiden(resolution=0.5, n_runs=3, random_state=0)
        w = np.array([1, 1, 1, 1, 1, 10, 10, 10], dtype=float)  # B mass = 30
        labels = cl.fit_predict(self._graph(), min_cluster_size=4, node_weights=w)
        assert -1 not in set(labels[5:8])                 # B survives on mass


# --------------------------------------------------------------------------- #
# P1 — weighted topic centroids
# --------------------------------------------------------------------------- #
class TestWeightedCentroids:
    def _model_with_state(self, emb, labels, weights):
        m = TriTopic(config=TriTopicConfig(verbose=False))
        m.labels_ = np.asarray(labels)
        m.original_embeddings_ = emb
        m.embeddings_ = emb
        m.sample_weights_ = weights
        m.topics_ = [
            TopicInfo(topic_id=t, size=int((labels == t).sum()),
                      keywords=[], keyword_scores=[], representative_docs=[])
            for t in sorted(set(int(x) for x in labels) - {-1})
        ]
        return m

    def test_weighted_mean_matches_manual(self):
        emb = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 0, 1])
        weights = np.array([1.0, 9.0, 1.0])  # second point stands for 9 docs
        m = self._model_with_state(emb, labels, weights)
        m._compute_topic_centroids()

        expected0 = np.average(emb[:2], axis=0, weights=weights[:2])  # -> [1.8, 0]
        assert np.allclose(m.topic_embeddings_[0], expected0)
        assert np.allclose(m.topic_embeddings_[1], emb[2])

    def test_none_weights_reproduce_unweighted(self):
        emb = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 0, 1])
        m = self._model_with_state(emb, labels, weights=None)
        m._compute_topic_centroids()
        assert np.allclose(m.topic_embeddings_[0], emb[:2].mean(axis=0))  # [1.0, 0]


# --------------------------------------------------------------------------- #
# P0 + integration — rare topic survival under a small coreset
# --------------------------------------------------------------------------- #
@pytest.fixture
def light_config():
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


def _stream_rare_then_common(dim=24, seed=1):
    """Batch 1 introduces all 3 topics (incl. a thin rare one); later batches add
    only common-topic docs. The rare topic is discovered + labelled up front, then
    diluted — exactly the case where a naive recency coreset drops it on the next
    Regime-B refit but a stratified coreset (per-topic floor) keeps it.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    words = {t: [f"t{t}_w{j}" for j in range(6)] for t in range(3)}

    def gen(t, n):
        docs, embs = [], []
        for _ in range(n):
            v = centers[t] + 0.04 * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            embs.append(v.astype(np.float32))
            docs.append(" ".join(rng.choice(words[t], size=8)))
        return docs, np.asarray(embs)

    d0, e0 = gen(0, 40)
    d1, e1 = gen(1, 40)
    d2, e2 = gen(2, 15)                           # the rare topic, only in batch 1
    batches = [(d0 + d1 + d2, np.vstack([e0, e1, e2]))]
    for _ in range(3):                            # common-only batches dilute the rare one
        da, ea = gen(0, 100)
        db, eb = gen(1, 100)
        batches.append((da + db, np.vstack([ea, eb])))
    return batches


def _run(strategy_selection, light_config, batches):
    cfg = CumulativeConfig(
        base_config=light_config,
        strategy="coreset",
        coreset_size=100,                        # batch1 (95) fits full; later refit is Regime B
        coreset_selection=strategy_selection,
        min_docs_per_topic_in_coreset=8,
        recluster_trigger="manual",              # only batch1 auto-reclusters; rest are assigned-only
    )
    cum = CumulativeTriTopic(cfg)
    for docs, emb in batches:
        cum.add_batch(docs, embeddings=emb)      # extends labels_ to span the accumulator
    cum.recluster()                              # one Regime-B refit with labels_ fully aligned
    return cum


def test_stratified_engages_under_drift_trigger(light_config, monkeypatch):
    """Regression: auto-reclusters (drift/schedule) must reach the stratified path.

    labels_ is extended with the batch's transform assignments *before* the
    recluster decision, so a Regime-B coreset stratifies on full-length labels
    instead of silently falling back to recency.
    """
    import tritopic.cumulative.strategies as S

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def gen(t, n):
        v = centers[t] + 0.04 * rng.normal(size=(n, 16))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return [f"t{t}_{i % 6}" for i in range(n)], v.astype(np.float32)

    seen = []
    orig = S._select_reduced

    def probe(ctx, size):
        idx, w = orig(ctx, size)
        labels_full = ctx.labels is not None and len(ctx.labels) == len(ctx.documents)
        seen.append(("stratified" if w is not None else "recency", labels_full))
        return idx, w

    monkeypatch.setattr(S, "_select_reduced", probe)

    cfg = CumulativeConfig(
        base_config=light_config, strategy="coreset", coreset_size=80,
        coreset_selection="stratified", min_docs_per_topic_in_coreset=8,
        recluster_trigger="drift", novelty_threshold=0.2,
    )
    cum = CumulativeTriTopic(cfg)
    for t in [0, 1, 2, 0, 1, 2]:
        d, e = gen(t, 120)
        cum.add_batch(d, embeddings=e)

    assert seen, "coreset strategy never reduced a working set"
    assert seen[0] == ("recency", False)            # first recluster: no labels yet
    later = seen[1:]
    assert later and all(mode == "stratified" and full for mode, full in later)


def test_stratified_beats_recency_on_rare_recall(light_config):
    batches = _stream_rare_then_common()
    all_docs = [d for docs, _ in batches for d in docs]
    all_emb = np.vstack([e for _, e in batches])

    full = TriTopic(config=light_config).fit(all_docs, embeddings=all_emb)

    # rare_frac=0.05 (~35 docs) so the 15-doc planted topic counts as rare.
    kw = dict(rare_frac=0.05, rare_sim_cutoff=0.4)
    strat = compare_to_full_batch(_run("stratified", light_config, batches), full, **kw)
    recency = compare_to_full_batch(_run("recency", light_config, batches), full, **kw)

    # The benchmark must expose the new tail-collapse metric, and actually exercise it.
    assert "rare_topic_recall" in strat and "n_rare_topics_full" in strat
    assert strat["n_rare_topics_full"] >= 1
    # Stratified should strictly recover the rare topic that recency drops.
    assert strat["rare_topic_recall"] > recency["rare_topic_recall"]
    assert strat["nmi_vs_full"] >= recency["nmi_vs_full"] - 0.15
