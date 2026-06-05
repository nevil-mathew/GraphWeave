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
from tritopic.core.model import TopicInfo
from tritopic.cumulative import CumulativeConfig, CumulativeTriTopic
from tritopic.cumulative.alignment import stratified_coreset
from tritopic.cumulative.evaluation import compare_to_full_batch


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
