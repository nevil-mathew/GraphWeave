"""
Integration tests on real 20 Newsgroups text — no embedding model required.

The lightweight (non-slow) tests use 2k docs with LSA embeddings so they
run in CI without torch. The @pytest.mark.slow tests use 6k docs and
sentence-transformers when available (otherwise LSA), exercising the HNSW
backend path (triggered above 5k docs).

Run fast tests only:   pytest tests/test_integration_20ng.py -v -m "not slow"
Run slow tests too:    pytest tests/test_integration_20ng.py -v
"""

import copy

import numpy as np
import pytest
from sklearn.datasets import fetch_20newsgroups

from tritopic import TriTopic, TriTopicConfig
from tritopic.cumulative import CumulativeConfig, CumulativeTriTopic
from tritopic.cumulative.datasets import lsa_embed
from tritopic.cumulative.evaluation import compare_to_full_batch
from tritopic.utils.metrics import compute_ari, compute_nmi


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ng20(n_docs: int, cats: list[int], seed: int = 42) -> tuple[list[str], np.ndarray]:
    """Load a balanced sample of 20NG and return (documents, labels)."""
    data = fetch_20newsgroups(
        subset="all",
        categories=[fetch_20newsgroups(subset="all").target_names[c] for c in cats],
        remove=("headers", "footers", "quotes"),
    )
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(data.data), size=min(n_docs, len(data.data)), replace=False)
    docs  = [data.data[i].strip() or "empty" for i in idx]
    labels = data.target[idx]
    return docs, labels


def _light_cfg() -> TriTopicConfig:
    return TriTopicConfig(
        use_dim_reduction=False,
        use_lexical_view=True,
        use_iterative_refinement=False,
        n_consensus_runs=3,
        min_cluster_size=5,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )


def _try_sentence_transformers(docs: list[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return m.encode(docs, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=128).astype(np.float32)
    except ImportError:
        return lsa_embed(docs, dim=64)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ng20_small():
    """2k docs from 10 categories with LSA embeddings — CI-safe, no torch."""
    cats  = list(range(10))
    docs, labels = _ng20(2000, cats, seed=1)
    embs  = lsa_embed(docs, dim=64)
    return docs, embs, labels


@pytest.fixture(scope="module")
def ng20_drift():
    """
    Drift scenario: 8 categories for the first 2 batches,
    then 4 new categories appear in batch 3.

    Established: categories 1-5  (comp.* topics)
    Emerging:    categories 11-14 (sci.crypt, sci.electronics, sci.med, sci.space)
    These groups are vocabulary-distinct enough for LSA to detect drift cleanly.
    Returns (batch_docs, batch_embs, batch_labels).
    """
    # comp.* is vocabulary-technical; rec.sport.* is sports — very different in LSA space,
    # giving ~22 % novelty and a clean drift signal even at 64-dim.
    cats_old = [1, 2, 3, 4, 5]   # comp.graphics, comp.os.*, comp.sys.*, comp.windows.*
    cats_new = [9, 10]            # rec.sport.baseball, rec.sport.hockey
    old_docs, old_lbl = _ng20(1200, cats_old, seed=2)
    new_docs, new_lbl = _ng20(600,  cats_new, seed=3)

    all_docs = old_docs + new_docs
    all_embs = lsa_embed(all_docs, dim=64)
    all_lbl  = np.concatenate([old_lbl, new_lbl])

    old_e, new_e = all_embs[:len(old_docs)], all_embs[len(old_docs):]

    batches_docs = [old_docs[:600], old_docs[600:], new_docs]
    batches_embs = [old_e[:600],    old_e[600:],    new_e]
    batches_lbl  = [old_lbl[:600],  old_lbl[600:],  new_lbl]
    return batches_docs, batches_embs, batches_lbl


# ── TestRealDataPipeline ─────────────────────────────────────────────────────

class TestRealDataPipeline:
    """End-to-end pipeline on real 20NG text with LSA embeddings."""

    def test_labels_cover_all_docs(self, ng20_small):
        docs, embs, _ = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        assert len(cum.labels_) == len(docs)

    def test_recovers_reasonable_structure(self, ng20_small):
        docs, embs, labels = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        ari = compute_ari(cum.labels_, labels)
        # LSA on 10 real, heavily-overlapping newsgroups categories is hard;
        # even a modest ARI proves the pipeline finds non-trivial structure.
        assert ari >= 0.10, f"ARI {ari:.3f} unexpectedly low"

    def test_second_batch_accumulates(self, ng20_small):
        docs, embs, _ = ng20_small
        half = len(docs) // 2
        cum = CumulativeTriTopic(
            CumulativeConfig(base_config=_light_cfg(), recluster_trigger="manual")
        )
        cum.add_batch(docs[:half], embeddings=embs[:half])
        cum.add_batch(docs[half:], embeddings=embs[half:])
        assert len(cum.labels_) == len(docs)

    def test_close_to_full_batch(self, ng20_small):
        docs, embs, labels = ng20_small
        full = TriTopic(config=copy.deepcopy(_light_cfg()))
        full.fit(docs, embeddings=embs)

        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)

        m = compare_to_full_batch(cum, full, labels_true=labels)
        # When the whole dataset is one batch, global_refit IS the full-batch fit
        assert m["ari_vs_full"] >= 0.98


# ── TestRealDriftDetection ───────────────────────────────────────────────────

class TestRealDriftDetection:
    """Emerging categories in batch 3 must spike novelty and trigger a recluster."""

    def test_drift_fires_on_new_categories(self, ng20_drift):
        batches_docs, batches_embs, batches_lbl = ng20_drift
        cum = CumulativeTriTopic(CumulativeConfig(
            base_config=_light_cfg(),
            recluster_trigger="drift",
            novelty_threshold=0.15,
        ))
        results = []
        topic_counts = []
        for d, e in zip(batches_docs, batches_embs):
            results.append(cum.add_batch(d, embeddings=e))
            topic_counts.append(cum.n_global_topics)

        # Batches 1-2: established categories — stable, no recluster
        assert results[1].reclustered is False
        # Batch 3: new categories arrive → novelty spikes → recluster fires
        assert results[2].novelty is not None
        assert results[2].novelty > 0.15
        assert results[2].reclustered is True
        # New topics must have been discovered
        assert topic_counts[2] > topic_counts[1]

    def test_novelty_stays_low_on_familiar(self, ng20_drift):
        batches_docs, batches_embs, _ = ng20_drift
        cum = CumulativeTriTopic(CumulativeConfig(
            base_config=_light_cfg(),
            recluster_trigger="drift",
            novelty_threshold=0.15,
        ))
        cum.add_batch(batches_docs[0], embeddings=batches_embs[0])
        r = cum.add_batch(batches_docs[1], embeddings=batches_embs[1])
        # Same 8 categories → novelty must stay well below threshold
        assert r.novelty < 0.15


# ── TestRealBiggerPicture ────────────────────────────────────────────────────

class TestRealBiggerPicture:
    """bigger_picture() builds a hierarchy on real topics."""

    def test_hierarchy_has_requested_levels(self, ng20_small):
        docs, embs, _ = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        view = cum.bigger_picture(n_levels=2)
        h = view["hierarchy"]
        assert h.n_levels == 2

    def test_coarse_level_has_fewer_nodes(self, ng20_small):
        docs, embs, _ = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        view = cum.bigger_picture(n_levels=2)
        h = view["hierarchy"]
        assert len(h.cut(0)) <= len(h.cut(1))

    def test_themes_none_without_labeler(self, ng20_small):
        docs, embs, _ = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        assert cum.bigger_picture()["themes"] is None


# ── TestRealEvaluate ─────────────────────────────────────────────────────────

class TestRealEvaluate:
    """evaluate() must return finite, in-range metrics on real text."""

    def test_metrics_are_finite(self, ng20_small):
        docs, embs, _ = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        m = cum.evaluate()
        for key in ("coherence_mean", "diversity", "n_total_docs", "n_global_topics"):
            assert key in m
            assert np.isfinite(m[key]), f"{key} = {m[key]}"

    def test_diversity_in_range(self, ng20_small):
        docs, embs, _ = ng20_small
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs, embeddings=embs)
        m = cum.evaluate()
        assert 0.0 <= m["diversity"] <= 1.0


# ── TestRealTransform ────────────────────────────────────────────────────────

class TestRealTransform:
    """transform() on held-out docs returns valid global topic IDs."""

    def test_transform_returns_valid_ids(self, ng20_small):
        docs, embs, _ = ng20_small
        half = len(docs) // 2
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs[:half], embeddings=embs[:half])
        assigned = cum.transform(docs[half:], embeddings=embs[half:])
        valid_ids = set(cum.labels_[cum.labels_ != -1]) | {-1}
        assert all(a in valid_ids for a in assigned)

    def test_transform_shape(self, ng20_small):
        docs, embs, _ = ng20_small
        half = len(docs) // 2
        cum = CumulativeTriTopic(CumulativeConfig(base_config=_light_cfg()))
        cum.add_batch(docs[:half], embeddings=embs[:half])
        assigned = cum.transform(docs[half:], embeddings=embs[half:])
        assert len(assigned) == len(docs) - half


# ── TestHnswPath (slow) ──────────────────────────────────────────────────────

@pytest.mark.slow
class TestHnswPath:
    """
    6k docs triggers the HNSW backend (threshold=5k). Uses sentence-transformers
    when available; falls back to LSA otherwise. Both paths are valid — the test
    verifies correct results, not which backend ran.
    """

    @pytest.fixture(scope="class")
    def ng20_6k(self):
        cats  = list(range(12))
        docs, labels = _ng20(6000, cats, seed=10)
        embs  = _try_sentence_transformers(docs)
        return docs, embs, labels

    def test_pipeline_handles_6k_docs(self, ng20_6k):
        docs, embs, labels = ng20_6k
        cum = CumulativeTriTopic(CumulativeConfig(
            base_config=TriTopicConfig(
                use_dim_reduction=False,
                use_lexical_view=True,
                use_iterative_refinement=False,
                n_consensus_runs=4,
                min_cluster_size=10,
                low_memory=True,  # validates the memory-safe path
                n_neighbors=15,
                random_state=42,
                verbose=False,
            )
        ))
        cum.add_batch(docs, embeddings=embs)
        assert len(cum.labels_) == len(docs)
        ari = compute_ari(cum.labels_, labels)
        assert ari >= 0.20   # real overlapping newsgroups — realistic floor

    def test_low_memory_path_used(self, ng20_6k):
        """Verify low_memory=True doesn't OOM and produces a valid model."""
        docs, embs, _ = ng20_6k
        cum = CumulativeTriTopic(CumulativeConfig(
            base_config=TriTopicConfig(
                use_dim_reduction=False, use_lexical_view=False,
                use_iterative_refinement=False,
                n_consensus_runs=3, min_cluster_size=10,
                low_memory=True, n_neighbors=15, random_state=0, verbose=False,
            )
        ))
        cum.add_batch(docs, embeddings=embs)
        assert cum.model_ is not None
        assert cum.n_global_topics >= 2
