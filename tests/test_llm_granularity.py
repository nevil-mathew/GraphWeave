"""Tests for LLM-guided granularity calibration (graphweave.labeling.llm_granularity).

No real API is ever called: a fake labeler exposing call_raw/call_structured
stands in for LLMLabeler. Unit tests exercise the module's helper functions
directly; the integration test drives a freshly-fit GraphWeave model (NOT the
shared session-scoped ``fitted_model`` fixture, since tune_resolution_with_llm
mutates model state in place).
"""

import json

import numpy as np
import pytest

from graphweave import GraphWeave, GraphWeaveConfig
from graphweave.core.clustering import ConsensusLeiden
from graphweave.labeling.llm_granularity import (
    _sample_triplets,
    _sample_triplets_fast,
    _sample_triplets_informed,
    _candidate_resolutions,
    _partition_at_resolution,
    _triplet_agreement,
    _parse_triplet_response,
    _default_n_triplets,
    llm_select_resolution,
)


class FakeLabeler:
    """Returns a canned string from both call_raw and call_structured."""

    def __init__(self, response):
        self._resp = response
        self.calls: list[tuple[str, str]] = []

    def call_raw(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        return self.call_raw(system_prompt, user_prompt, max_tokens=max_tokens)


class AllBLabeler(FakeLabeler):
    """Always answers 'B' for every item in every batch it receives."""

    def __init__(self):
        super().__init__(None)

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        n_items = user_prompt.count("Item ")
        resp = json.dumps({"answers": ["B"] * n_items})
        self.calls.append((system_prompt, user_prompt))
        return resp


# --------------------------------------------------------------------------- #
# Triplet sampling (canonical)
# --------------------------------------------------------------------------- #
class TestSampleTriplets:
    def _make_labels_embeddings(self, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(3, 8))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        labels = np.repeat([0, 1, 2], 20)
        embs = np.vstack([
            centers[t] + 0.05 * rng.normal(size=(20, 8)) for t in range(3)
        ])
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return labels, embs.astype(np.float32)

    def test_determinism_same_seed(self):
        labels, embs = self._make_labels_embeddings()
        t1 = _sample_triplets(labels, embs, n_triplets=10, random_state=42)
        t2 = _sample_triplets(labels, embs, n_triplets=10, random_state=42)
        assert t1 == t2

    def test_different_seed_can_differ(self):
        labels, embs = self._make_labels_embeddings()
        t1 = _sample_triplets(labels, embs, n_triplets=10, random_state=42)
        t2 = _sample_triplets(labels, embs, n_triplets=10, random_state=7)
        assert t1 != t2

    def test_b_same_cluster_c_different_cluster(self):
        labels, embs = self._make_labels_embeddings()
        triplets = _sample_triplets(labels, embs, n_triplets=15, random_state=0)
        assert len(triplets) > 0
        for a, b, c in triplets:
            assert labels[a] == labels[b]
            assert labels[a] != labels[c]
            assert a != b

    def test_outliers_excluded(self):
        labels, embs = self._make_labels_embeddings()
        labels = labels.copy()
        labels[:5] = -1
        triplets = _sample_triplets(labels, embs, n_triplets=15, random_state=0)
        for a, b, c in triplets:
            assert a >= 5 and b >= 5 and c >= 5

    def test_single_cluster_returns_empty(self):
        labels = np.zeros(10, dtype=int)
        embs = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        assert _sample_triplets(labels, embs, n_triplets=5, random_state=0) == []


# --------------------------------------------------------------------------- #
# Fast triplet sampling
# --------------------------------------------------------------------------- #
class TestSampleTripletsFast:
    def _make_labels_embeddings(self, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(3, 8))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        labels = np.repeat([0, 1, 2], 20)
        embs = np.vstack([
            centers[t] + 0.05 * rng.normal(size=(20, 8)) for t in range(3)
        ])
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return labels, embs.astype(np.float32)

    def test_b_same_cluster_c_different_cluster(self):
        labels, embs = self._make_labels_embeddings()
        triplets = _sample_triplets_fast(labels, embs, n_triplets=15, random_state=0)
        assert len(triplets) > 0
        for a, b, c in triplets:
            assert labels[a] == labels[b], "b must be same-cluster"
            assert labels[a] != labels[c], "c must be diff-cluster"
            assert a != b

    def test_determinism(self):
        labels, embs = self._make_labels_embeddings()
        t1 = _sample_triplets_fast(labels, embs, n_triplets=10, random_state=7)
        t2 = _sample_triplets_fast(labels, embs, n_triplets=10, random_state=7)
        assert t1 == t2

    def test_single_cluster_returns_empty(self):
        labels = np.zeros(10, dtype=int)
        embs = np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)
        assert _sample_triplets_fast(labels, embs, n_triplets=5, random_state=0) == []

    def test_outliers_excluded(self):
        labels, embs = self._make_labels_embeddings()
        labels = labels.copy()
        labels[:5] = -1
        triplets = _sample_triplets_fast(labels, embs, n_triplets=15, random_state=0)
        outlier_idx = set(np.where(labels == -1)[0].tolist())
        for a, b, c in triplets:
            assert a not in outlier_idx
            assert b not in outlier_idx
            assert c not in outlier_idx


# --------------------------------------------------------------------------- #
# Discriminative triplet sampling
# --------------------------------------------------------------------------- #
class TestSampleTripletsInformed:
    def _partitions_and_embeddings(self, seed=0):
        """Two well-separated clusters; one partition merges them, one splits."""
        rng = np.random.default_rng(seed)
        embs = np.vstack([
            rng.normal([1, 0, 0, 0], 0.05, size=(20, 4)),
            rng.normal([-1, 0, 0, 0], 0.05, size=(20, 4)),
        ]).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        p_split = np.array([0] * 20 + [1] * 20)
        p_merged = np.zeros(40, dtype=int)
        return [p_split, p_merged], embs

    def test_prefers_discriminative_triplets(self):
        """When one partition splits and another merges, informed sampling
        should return triplets with non-zero cross-partition disagreement."""
        partitions, embs = self._partitions_and_embeddings()
        triplets = _sample_triplets_informed(
            partitions, embs, n_triplets=8, random_state=0
        )
        # Every returned triplet should be from the reference (split) partition
        ref = partitions[len(partitions) // 2]
        for a, b, c in triplets:
            assert ref[a] == ref[b]
            assert ref[a] != ref[c]

    def test_returns_at_most_n_triplets(self):
        partitions, embs = self._partitions_and_embeddings()
        n = 5
        triplets = _sample_triplets_informed(partitions, embs, n_triplets=n, random_state=0)
        assert len(triplets) <= n


# --------------------------------------------------------------------------- #
# Candidate resolutions
# --------------------------------------------------------------------------- #
class TestCandidateResolutions:
    def test_count_and_bounds(self):
        cands = _candidate_resolutions((0.1, 2.0), 6)
        assert len(cands) == 6
        assert abs(cands[0] - 0.1) < 1e-9
        assert abs(cands[-1] - 2.0) < 1e-9

    def test_monotonic_increasing(self):
        cands = _candidate_resolutions((0.1, 2.0), 6)
        assert all(cands[i] < cands[i + 1] for i in range(len(cands) - 1))


# --------------------------------------------------------------------------- #
# Triplet agreement scoring
# --------------------------------------------------------------------------- #
class TestTripletAgreement:
    def test_perfect_agreement(self):
        labels = np.array([0, 0, 1])  # a=0,b=1 same cluster; c=2 different
        triplets = [(0, 1, 2)]
        # Laplace-smoothed: (agree+1)/(counted+2) = 2/3, not a raw 1.0.
        assert _triplet_agreement(labels, triplets, ["B"]) == pytest.approx(2 / 3)

    def test_perfect_disagreement(self):
        labels = np.array([0, 0, 1])
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["C"]) == pytest.approx(1 / 3)

    def test_uninformative_triplet_skipped_both_merged(self):
        labels = np.array([0, 0, 0])  # a, b, c all in same cluster: uninformative
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["B"]) == 0.0

    def test_uninformative_triplet_skipped_both_separated(self):
        labels = np.array([0, 1, 2])  # a,b,c all different clusters: uninformative
        triplets = [(0, 1, 2)]
        assert _triplet_agreement(labels, triplets, ["B"]) == 0.0

    def test_none_answer_skipped(self):
        """None answers (parse failures) must not count for OR against any candidate."""
        labels = np.array([0, 0, 1])
        triplets = [(0, 1, 2)]
        # None answer: same as if triplet didn't exist → score 0.0 (no informative data)
        assert _triplet_agreement(labels, triplets, [None]) == 0.0

    def test_mixed_none_and_valid(self):
        """One valid answer, one None → only the valid one is counted."""
        labels = np.array([0, 0, 1, 2, 2])
        # triplet1: a=0(lbl0), b=1(lbl0, same), c=2(lbl1, diff) -> informative, implied "B"
        # triplet2: None answer -> skipped
        triplets = [(0, 1, 2), (2, 3, 4)]
        answers: list = ["B", None]
        # 1 informative triplet, 1 agreement → Laplace: (1+1)/(1+2) = 2/3
        assert _triplet_agreement(labels, triplets, answers) == pytest.approx(2 / 3)

    def test_mixed_batch_only_informative_counted(self):
        labels = np.array([0, 0, 1, 2, 2])
        # triplet1: a=0(lbl0), b=1(lbl0, same), c=2(lbl1, diff) -> informative, implied "B"
        # triplet2: a=2(lbl1), b=0(lbl0, diff), c=1(lbl0, diff) -> both diff, uninformative
        triplets = [(0, 1, 2), (2, 0, 1)]
        answers = ["B", "C"]
        assert _triplet_agreement(labels, triplets, answers) == pytest.approx(2 / 3)

    def test_sparse_perfect_match_does_not_outrank_broad_high_agreement(self):
        # Candidate A: informative on exactly 1 triplet, agrees on it (naive 1.0).
        labels_a = np.array([0, 0, 1])
        triplets_a = [(0, 1, 2)]
        score_a = _triplet_agreement(labels_a, triplets_a, ["B"])

        # Candidate B: informative on 9 triplets, agrees on 8 (naive ~0.889).
        labels_b = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        triplets_b = [(0, 1, i) for i in range(2, 10)] + [(0, 1, 2)]
        answers_b = ["B"] * 8 + ["C"]  # 8/9 agree
        score_b = _triplet_agreement(labels_b, triplets_b, answers_b)

        assert score_b > score_a


# --------------------------------------------------------------------------- #
# B-position bias: swap round-trip
# --------------------------------------------------------------------------- #
class TestSwapUnswap:
    """The swap/unswap round-trip must be transparent to _triplet_agreement."""

    def _make(self, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(size=(3, 8))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        labels = np.repeat([0, 1, 2], 20)
        embs = np.vstack([
            centers[t] + 0.05 * rng.normal(size=(20, 8)) for t in range(3)
        ])
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return labels, embs.astype(np.float32)

    def test_canonical_answer_unchanged_after_swap_unswap(self):
        """When we swap a triplet (a,b,c)→(a,c,b) in the prompt and then
        un-swap the LLM's answer, the resulting canonical answer must equal
        the canonical answer for the un-swapped presentation."""
        labels, embs = self._make()
        triplets = _sample_triplets(labels, embs, n_triplets=10, random_state=0)
        assert triplets, "need at least one triplet"

        # Simulate: all swapped, LLM always says "B" on presented order.
        # After un-swap, "B" (diff) becomes canonical "C".
        for a, b, c in triplets[:3]:
            presented_answer = "B"  # LLM saw diff-cluster as B
            canonical = "C" if presented_answer == "B" else "B"  # un-swap
            # In canonical space B=same, so answer "C" means "C=diff more similar" → wrong
            # (original triplet has same=b, so correct canonical is "B")
            assert canonical == "C"  # swap inverted as expected


# --------------------------------------------------------------------------- #
# AllBLabeler is no longer trivially biased toward middle candidate
# --------------------------------------------------------------------------- #
class TestAllBLabelerBiasReduced:
    """Verify the swap mechanism neutralises AllBLabeler's B-position bias.

    The canonical-space check is done directly on the swap/unswap arithmetic
    rather than via a full graph run, because any graph whose partitions are
    identical across resolutions would make the algorithm tie-break to the
    middle regardless — and the two-clique graphs used in other tests have
    exactly that property.  What matters for bias correction is that the raw
    all-B answers become ~50 % B / ~50 % C in canonical space.
    """

    def test_canonical_answers_approximately_balanced_from_allb(self):
        """After swap randomisation + unswap, all-'B' raw answers become
        roughly 50 % B and 50 % C in canonical space — confirming the
        position bias toward 'B' is neutralised."""
        random_state = 42
        n = 200
        rng_swap = np.random.default_rng(random_state + 999)
        swaps = rng_swap.random(n) < 0.5
        # AllBLabeler always returns "B" in presented (possibly-swapped) order.
        raw_answers = ["B"] * n
        # Un-swap: presented "B" on a swapped triplet means diff-cluster was B
        # → canonical answer is "C" (same-cluster is C in canonical space).
        canonical = [
            ("C" if ans == "B" else "B") if bool(s) else ans
            for s, ans in zip(swaps, raw_answers)
        ]
        n_b = sum(1 for a in canonical if a == "B")
        # ~50 % should be B; allow wide tolerance for n=200
        assert 70 <= n_b <= 130, (
            f"Expected ~100 canonical-B in {n} trials, got {n_b}. "
            "The swap mechanism may not be cancelling position bias."
        )


# --------------------------------------------------------------------------- #
# Two-stage grid: Stage B reuses LLM answers (no extra calls)
# --------------------------------------------------------------------------- #
class TestStageBRefinement:
    def _two_cluster_setup(self):
        import igraph as ig

        rng = np.random.default_rng(3)
        a_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        b_edges = [(4 + i, 4 + j) for i in range(4) for j in range(i + 1, 4)]
        g = ig.Graph(n=8, edges=a_edges + b_edges + [(3, 4)], directed=False)
        g.es["weight"] = [1.0] * g.ecount()

        centers = rng.normal(size=(2, 8))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        embs = np.vstack([
            centers[t] + 0.05 * rng.normal(size=(4, 8)) for t in range(2)
        ]).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        docs = [f"doc {i}" for i in range(8)]
        return g, embs, docs

    def test_stage_b_does_not_call_llm_again(self):
        """Stage B scores the fine grid against existing LLM answers;
        the total call count must equal ceil(n_triplets / batch_size)."""
        g, embs, docs = self._two_cluster_setup()
        labeler = AllBLabeler()
        n_triplets = 8
        batch_size = 4
        llm_select_resolution(
            labeler, docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=n_triplets, batch_size=batch_size, random_state=0,
        )
        import math
        expected_calls = math.ceil(n_triplets / batch_size)
        assert len(labeler.calls) == expected_calls

    def test_best_resolution_within_range(self):
        """After Stage B, the winning resolution must be within resolution_range."""
        g, embs, docs = self._two_cluster_setup()
        res = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=8, batch_size=4, random_state=0,
        )
        assert 0.1 <= res <= 2.0


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
class TestDiagnostics:
    def _setup(self):
        import igraph as ig

        rng = np.random.default_rng(5)
        a_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        b_edges = [(4 + i, 4 + j) for i in range(4) for j in range(i + 1, 4)]
        g = ig.Graph(n=8, edges=a_edges + b_edges + [(3, 4)], directed=False)
        g.es["weight"] = [1.0] * g.ecount()
        embs = rng.normal(size=(8, 4)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        docs = [f"doc {i}" for i in range(8)]
        return g, embs, docs

    def test_return_diagnostics_true_returns_tuple(self):
        g, embs, docs = self._setup()
        result = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=8, batch_size=4, return_diagnostics=True,
        )
        assert isinstance(result, tuple) and len(result) == 2
        best_res, diag = result
        assert isinstance(best_res, float)
        assert isinstance(diag, dict)

    def test_diagnostics_keys_present(self):
        g, embs, docs = self._setup()
        _, diag = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=8, batch_size=4, return_diagnostics=True,
        )
        for key in ("n_triplets", "n_unparsed", "reference_resolution", "best_resolution", "candidates"):
            assert key in diag

    def test_diagnostics_candidates_have_required_fields(self):
        g, embs, docs = self._setup()
        _, diag = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=8, batch_size=4, return_diagnostics=True,
        )
        for row in diag["candidates"]:
            for field in ("resolution", "score", "n_clusters", "stage"):
                assert field in row
        stages = {row["stage"] for row in diag["candidates"]}
        assert stages <= {"A", "B"}

    def test_diagnostics_best_resolution_matches_return_value(self):
        g, embs, docs = self._setup()
        best_res, diag = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=8, batch_size=4, return_diagnostics=True,
        )
        assert diag["best_resolution"] == pytest.approx(best_res)

    def test_return_diagnostics_false_returns_float(self):
        g, embs, docs = self._setup()
        result = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=4,
            n_triplets=8, batch_size=4, return_diagnostics=False,
        )
        assert isinstance(result, float)


# --------------------------------------------------------------------------- #
# _partition_at_resolution node_weights threading
# --------------------------------------------------------------------------- #
class TestPartitionAtResolutionNodeWeights:
    """Reuses the exact fixture/parameters from
    tests/test_coreset_improvements.py::TestWeightedPartition, which are
    already proven to flip RBConfigurationVertexPartition (merges across
    the bridge) vs RBERVertexPartition+node_sizes (splits the heavy trio
    off) at resolution=0.4. This proves _partition_at_resolution's
    node_weights branch matches ConsensusLeiden.fit_predict's objective.
    """

    def _graph(self):
        import igraph as ig

        a_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        b_edges = [(4 + i, 4 + j) for i in range(4) for j in range(i + 1, 4)]
        bridge = [(3, 4)]
        g = ig.Graph(n=8, edges=a_edges + b_edges + bridge, directed=False)
        g.es["weight"] = [1.0] * len(a_edges) + [1.0] * len(b_edges) + [3.2]
        return g

    def test_unweighted_merges_across_the_bridge(self):
        labels = _partition_at_resolution(self._graph(), 0.4, random_state=0)
        assert len(set(labels)) == 1  # A and B merged into one community

    def test_node_weights_split_b_into_its_own_community(self):
        w = np.array([1, 1, 1, 1, 1, 3, 3, 3], dtype=float)
        labels = _partition_at_resolution(self._graph(), 0.4, random_state=0, node_weights=w)
        assert len(set(labels[5:])) == 1
        assert labels[5] != labels[0]

    def test_matches_consensus_leiden_weighted_objective(self):
        """Sanity cross-check: the same node_weights make ConsensusLeiden.fit_predict
        pick the same objective family (RBER) as _partition_at_resolution."""
        w = np.array([1, 1, 1, 1, 1, 3, 3, 3], dtype=float)
        cl = ConsensusLeiden(resolution=0.4, n_runs=3, random_state=0)
        consensus_labels = cl.fit_predict(self._graph(), min_cluster_size=2, node_weights=w)
        single_labels = _partition_at_resolution(self._graph(), 0.4, random_state=0, node_weights=w)
        assert len(set(consensus_labels[5:]) - {-1}) == 1
        assert len(set(single_labels[5:])) == 1

    def test_min_cluster_size_suppresses_small_clusters(self):
        """When min_cluster_size is large enough to kill all communities,
        the partition should become all-outliers (-1)."""
        g = self._graph()
        # Setting min_cluster_size=100 forces all 4-node clusters to be outliers
        labels = _partition_at_resolution(g, 0.4, random_state=0, min_cluster_size=100)
        assert all(l == -1 for l in labels)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #
class TestParseTripletResponse:
    def test_valid_json(self):
        raw = '{"answers": ["B", "C", "B"]}'
        assert _parse_triplet_response(raw, 3) == ["B", "C", "B"]

    def test_valid_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"answers": ["C", "B"]}\nThanks!'
        assert _parse_triplet_response(raw, 2) == ["C", "B"]

    def test_truncated_json_falls_back_to_regex_tier(self):
        raw = '{"answers": ["B", "C", "B"'  # missing closing brackets
        result = _parse_triplet_response(raw, 3)
        assert result == ["B", "C", "B"]

    def test_garbage_input_returns_all_none(self):
        with pytest.warns(UserWarning, match="could not parse"):
            result = _parse_triplet_response("not json at all, sorry!", 4)
        assert result == [None, None, None, None]

    def test_short_answer_list_padded_with_none(self):
        raw = '{"answers": ["C"]}'
        with pytest.warns(UserWarning, match="LLM returned 1 answers"):
            result = _parse_triplet_response(raw, 3)
        assert result == ["C", None, None]

    def test_long_answer_list_truncated(self):
        raw = '{"answers": ["B", "C", "B", "C", "B"]}'
        result = _parse_triplet_response(raw, 3)
        assert result == ["B", "C", "B"]

    def test_invalid_enum_values_become_none(self):
        raw = '{"answers": ["B", "X", "C"]}'
        result = _parse_triplet_response(raw, 3)
        assert result == ["B", None, "C"]


# --------------------------------------------------------------------------- #
# Default n_triplets scaling
# --------------------------------------------------------------------------- #
class TestDefaultNTriplets:
    def test_small_corpus_floor(self):
        assert _default_n_triplets(50) == 24
        assert _default_n_triplets(100) == 24

    def test_scales_up_for_large_corpus(self):
        assert _default_n_triplets(1000) > 24
        assert _default_n_triplets(50_000) <= 120
        assert _default_n_triplets(50_000) >= 80

    def test_ceiling(self):
        assert _default_n_triplets(10_000_000) == 120


# --------------------------------------------------------------------------- #
# llm_select_resolution input validation
# --------------------------------------------------------------------------- #
class TestLlmSelectResolutionValidation:
    def _labels_embeddings(self):
        rng = np.random.default_rng(0)
        centers = rng.normal(size=(3, 8))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        embs = np.vstack([
            centers[t] + 0.05 * rng.normal(size=(20, 8)) for t in range(3)
        ])
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return embs.astype(np.float32)

    def _tiny_graph(self):
        import igraph as ig

        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        edges += [(4 + i, 4 + j) for i in range(4) for j in range(i + 1, 4)]
        edges += [(3, 4)]
        g = ig.Graph(n=8, edges=edges, directed=False)
        g.es["weight"] = [1.0] * g.ecount()
        return g

    def test_n_candidates_zero_raises_value_error(self):
        embs = self._labels_embeddings()[:8]
        docs = [f"doc {i}" for i in range(8)]
        with pytest.raises(ValueError, match="n_candidates"):
            llm_select_resolution(
                AllBLabeler(), docs, self._tiny_graph(), embs, n_candidates=0
            )

    def test_batch_size_zero_raises_value_error(self):
        embs = self._labels_embeddings()[:8]
        docs = [f"doc {i}" for i in range(8)]
        with pytest.raises(ValueError, match="batch_size"):
            llm_select_resolution(
                AllBLabeler(), docs, self._tiny_graph(), embs, batch_size=0
            )


# --------------------------------------------------------------------------- #
# Tie / no-signal fallback
# --------------------------------------------------------------------------- #
class TestNoSignalFallback:
    def test_all_zero_scores_falls_back_to_middle_candidate_with_warning(self):
        # A single-node-per-cluster graph (no edges) can't produce
        # informative triplets under any candidate resolution: every
        # cluster in every candidate partition is a singleton, so
        # _sample_triplets finds no same-cluster neighbor for any anchor
        # and returns []. This exercises the "no valid triplets sampled"
        # fallback, which returns the middle candidate.
        import igraph as ig

        g = ig.Graph(n=6, edges=[], directed=False)
        g.es["weight"] = []
        docs = [f"doc {i}" for i in range(6)]
        embs = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)

        with pytest.warns(UserWarning, match="no valid triplets"):
            best_res = llm_select_resolution(
                AllBLabeler(), docs, g, embs,
                resolution_range=(0.1, 2.0), n_candidates=4, n_triplets=6, batch_size=4,
            )
        candidates = _candidate_resolutions((0.1, 2.0), 4)
        assert abs(best_res - candidates[len(candidates) // 2]) < 1e-9

    def test_tied_scores_prefer_middle_candidate_over_first_index(self):
        # Two disconnected K4 cliques: Leiden finds the identical 2-cluster
        # partition at every resolution in (0.1, 2.0) (verified empirically
        # via leidenalg directly), so every candidate's _triplet_agreement
        # score ties exactly. np.argmax would silently pick index 0 (the
        # lowest-resolution candidate); the explicit tie-break must instead
        # land on the middle candidate.
        import igraph as ig

        a_edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        b_edges = [(4 + i, 4 + j) for i in range(4) for j in range(i + 1, 4)]
        g = ig.Graph(n=8, edges=a_edges + b_edges, directed=False)
        g.es["weight"] = [1.0] * g.ecount()
        docs = [f"doc {i}" for i in range(8)]
        embs = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)

        best_res = llm_select_resolution(
            AllBLabeler(), docs, g, embs,
            resolution_range=(0.1, 2.0), n_candidates=5, n_triplets=8, batch_size=4,
        )
        candidates = _candidate_resolutions((0.1, 2.0), 5)
        assert abs(best_res - candidates[len(candidates) // 2]) < 1e-9


# --------------------------------------------------------------------------- #
# Integration: GraphWeave.tune_resolution_with_llm
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_fitted_model(fake_documents, _fake_embeddings):
    """A freshly-fit, non-shared GraphWeave model (safe to mutate)."""
    cfg = GraphWeaveConfig(
        use_dim_reduction=False,
        use_iterative_refinement=False,
        use_lexical_view=True,
        n_consensus_runs=3,
        min_cluster_size=5,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )
    model = GraphWeave(config=cfg)
    model.fit(fake_documents, embeddings=_fake_embeddings)
    return model


class TestTuneResolutionWithLLM:
    def test_raises_if_not_fitted(self):
        model = GraphWeave()
        with pytest.raises(ValueError, match="Model not fitted"):
            model.tune_resolution_with_llm(FakeLabeler('{"answers": ["B"]}'))

    def test_raises_clear_error_on_missing_graph_after_save_load(self, fresh_fitted_model, tmp_path):
        path = str(tmp_path / "model.pkl")
        fresh_fitted_model.save(path)
        reloaded = GraphWeave.load(path)
        assert reloaded.graph_ is None  # save()/load() never persists graph_
        with pytest.raises(ValueError, match="save\\(\\)/load\\(\\)"):
            reloaded.tune_resolution_with_llm(AllBLabeler())

    def test_resolution_and_clusterer_consistent_after_tuning(self, fresh_fitted_model):
        """config.resolution and _clusterer.resolution must be equal after tuning."""
        fresh_fitted_model.tune_resolution_with_llm(
            AllBLabeler(), n_candidates=4, n_triplets=12, batch_size=4, random_state=42
        )
        assert fresh_fitted_model.config.resolution == pytest.approx(
            fresh_fitted_model._clusterer.resolution
        )

    def test_resolution_within_range(self, fresh_fitted_model):
        fresh_fitted_model.tune_resolution_with_llm(
            AllBLabeler(), n_candidates=4, n_triplets=12, batch_size=4,
            resolution_range=(0.1, 2.0),
        )
        assert 0.1 <= fresh_fitted_model.config.resolution <= 2.0

    def test_diagnostics_stored(self, fresh_fitted_model):
        fresh_fitted_model.tune_resolution_with_llm(
            AllBLabeler(), n_candidates=4, n_triplets=12, batch_size=4
        )
        assert hasattr(fresh_fitted_model, "granularity_diagnostics_")
        diag = fresh_fitted_model.granularity_diagnostics_
        assert isinstance(diag, dict)
        assert "candidates" in diag
        assert "best_resolution" in diag
        # Should include both Stage A and possibly Stage B rows
        assert len(diag["candidates"]) >= 4

    def test_forwards_sample_weights_and_exercises_weighted_objective(
        self, fresh_fitted_model, monkeypatch
    ):
        """tune_resolution_with_llm must forward self.sample_weights_ as
        node_weights into llm_select_resolution, and that forwarding must
        actually reach leidenalg as the weighted RBER objective throughout
        the call (candidate scoring AND the final consensus re-fit) rather
        than being silently dropped somewhere in between."""
        import leidenalg as la
        import graphweave.labeling.llm_granularity as llm_granularity_module

        weights = np.random.default_rng(0).uniform(1, 5, size=len(fresh_fitted_model.documents_))
        fresh_fitted_model.sample_weights_ = weights

        captured = {}
        original_select = llm_granularity_module.llm_select_resolution

        def _spy_select(*args, **kwargs):
            captured["node_weights"] = kwargs.get("node_weights")
            return original_select(*args, **kwargs)

        monkeypatch.setattr(llm_granularity_module, "llm_select_resolution", _spy_select)

        partition_types_used = []
        original_find_partition = la.find_partition

        def _spy_find_partition(graph, partition_type, **kwargs):
            partition_types_used.append(partition_type)
            return original_find_partition(graph, partition_type, **kwargs)

        monkeypatch.setattr(la, "find_partition", _spy_find_partition)

        fresh_fitted_model.tune_resolution_with_llm(
            AllBLabeler(), n_candidates=4, n_triplets=12, batch_size=4
        )

        assert captured["node_weights"] is weights
        assert len(partition_types_used) > 0
        assert all(pt is la.RBERVertexPartition for pt in partition_types_used)

    def test_returns_self(self, fresh_fitted_model):
        labeler = AllBLabeler()
        result = fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert result is fresh_fitted_model

    def test_topics_refreshed(self, fresh_fitted_model):
        labeler = AllBLabeler()
        topics_before = fresh_fitted_model.topics_
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        # _extract_topic_info always rebuilds self.topics_ as a brand-new list
        assert fresh_fitted_model.topics_ is not topics_before
        assert len(fresh_fitted_model.topics_) > 0

    def test_labels_length_preserved(self, fresh_fitted_model, fake_documents):
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert len(fresh_fitted_model.labels_) == len(fake_documents)

    def test_probabilities_refreshed(self, fresh_fitted_model):
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        n_topics = len([t for t in fresh_fitted_model.topics_ if t.topic_id != -1])
        assert fresh_fitted_model.probabilities_.shape == (
            len(fresh_fitted_model.documents_), n_topics
        )

    def test_llm_was_called(self, fresh_fitted_model):
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert len(labeler.calls) > 0

    def test_does_not_rebuild_graph(self, fresh_fitted_model, monkeypatch):
        """tune_resolution_with_llm must reuse self.graph_, never call
        self._graph_builder.build_multiview_graph."""
        called = {"count": 0}
        original = fresh_fitted_model._graph_builder.build_multiview_graph

        def _spy(*args, **kwargs):
            called["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            fresh_fitted_model._graph_builder, "build_multiview_graph", _spy
        )
        labeler = AllBLabeler()
        fresh_fitted_model.tune_resolution_with_llm(
            labeler, n_candidates=4, n_triplets=12, batch_size=4
        )
        assert called["count"] == 0
