"""
Consensus Leiden Clustering
============================

Robust community detection with:
- Leiden algorithm (better than Louvain)
- Consensus clustering for stability
- Resolution parameter tuning
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from collections import Counter


class ConsensusLeiden:
    """
    Leiden clustering with consensus for stability.
    
    Runs multiple Leiden clusterings with different seeds and combines
    results using consensus clustering. This dramatically improves
    reproducibility and reduces sensitivity to random initialization.
    
    Parameters
    ----------
    resolution : float
        Resolution parameter for Leiden. Higher = more clusters. Default: 1.0
    n_runs : int
        Number of consensus runs. Default: 10
    random_state : int
        Random seed for reproducibility. Default: 42
    consensus_threshold : float
        Minimum agreement ratio for consensus. Default: 0.5
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        n_runs: int = 10,
        random_state: int = 42,
        consensus_threshold: float = 0.5,
        low_memory: bool = False,
        consensus_method: str = "graph",
        consensus_threshold_tau: float = 0.5,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.resolution = resolution
        self.n_runs = n_runs
        self.random_state = random_state
        self.consensus_threshold = consensus_threshold
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        if consensus_method not in ("graph", "hierarchical"):
            raise ValueError(
                f"consensus_method must be 'graph' or 'hierarchical', got {consensus_method!r}"
            )
        self.consensus_method = consensus_method
        self.consensus_threshold_tau = consensus_threshold_tau
        
        self.labels_: np.ndarray | None = None
        self.stability_score_: float | None = None
        self._all_partitions: list[np.ndarray] = []
    
    def fit_predict(
        self,
        graph: "igraph.Graph",
        min_cluster_size: int = 5,
        resolution: float | None = None,
        compute_stability: bool = True,
        node_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit Leiden clustering with consensus.

        Parameters
        ----------
        graph : igraph.Graph
            Input graph with edge weights.
        min_cluster_size : int
            Minimum cluster size. Smaller clusters become outliers. When
            *node_weights* is given this is compared against the summed
            represented mass of a cluster, not its raw node count.
        resolution : float, optional
            Override default resolution.
        node_weights : np.ndarray, optional
            Per-node representation weight, aligned row-wise with the graph's
            vertices (e.g. how many real documents a coreset point stands for).
            When given, small-cluster pruning judges a cluster by its summed
            represented mass rather than its raw node count, so a large theme
            that is sparsely sampled in a coreset is not deleted as "small".
            ``None`` (default) reproduces the unweighted behaviour exactly.
            
        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        from graphweave.utils.timing import step_timer
        import leidenalg as la

        res = resolution or self.resolution
        n_nodes = graph.vcount()

        # Represented mass per node: used for mass-based small-cluster pruning
        # (see _handle_small_clusters) AND, below, for the partition objective
        # itself. leidenalg's RBConfigurationVertexPartition (this codebase's
        # default, degree-based configuration null model) has no node_sizes
        # parameter, so it can't see node mass directly. RBERVertexPartition
        # (Erdős–Rényi null model) does accept node_sizes, but its resolution
        # semantics differ from RBConfiguration's — switching every fit over
        # would silently change behaviour for every existing unweighted config.
        # We avoid that by switching objective *only* when node_weights is
        # given (i.e. only on the already-distinct weighted-coreset branch);
        # every unweighted fit keeps using RBConfigurationVertexPartition
        # exactly as before, so resolution semantics there are untouched.
        #
        # An earlier attempt scaled edge weights by w_i * w_j instead (a
        # graph-contraction approximation) to avoid touching the objective at
        # all, but that inflates a heavy node's *every* edge, including the
        # spurious cross-topic edges a real kNN graph always has near sparse
        # (exactly the under-sampled, heavily-weighted) regions — which made
        # coreset fidelity measurably worse on realistic data. node_sizes only
        # enters RBER's null-model density term, not the raw edges, so it
        # doesn't amplify that noise; benchmarked clearly better.
        self._node_weights = (
            np.asarray(node_weights, dtype=float) if node_weights is not None else None
        )
        use_node_sizes = self._node_weights is not None and len(self._node_weights) == n_nodes

        from joblib import Parallel, delayed

        def _run_one(seed: int) -> np.ndarray:
            import leidenalg as _la
            if use_node_sizes:
                part = _la.find_partition(
                    graph,
                    _la.RBERVertexPartition,
                    weights="weight",
                    node_sizes=self._node_weights.tolist(),
                    resolution_parameter=res,
                    seed=seed,
                )
            else:
                part = _la.find_partition(
                    graph,
                    _la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=res,
                    seed=seed,
                )
            return np.array(part.membership)

        # prefer="threads" so the GIL doesn't block leidenalg's C backend, but
        # cap at n_jobs=4 — running all n_runs in parallel multiplies C-level
        # partition memory by n_jobs concurrent allocations.
        parallel_jobs = min(self.n_jobs if self.n_jobs > 0 else 4, 4)
        seeds = [self.random_state + run for run in range(self.n_runs)]
        with step_timer(f"leiden-runs ×{self.n_runs} ({parallel_jobs} concurrent)", verbose=self.verbose, indent=12):
            self._all_partitions = Parallel(n_jobs=parallel_jobs, prefer="threads")(
                delayed(_run_one)(seed) for seed in seeds
            )

        # Compute consensus
        with step_timer("leiden-cooccur", verbose=self.verbose, indent=12):
            self.labels_ = self._compute_consensus(self._all_partitions)

        # Handle small clusters as outliers
        self.labels_ = self._handle_small_clusters(self.labels_, min_cluster_size)

        # Stability is expensive (45 ARI calls); skip during iterative refinement
        # and compute once at the end via the compute_stability flag.
        self.stability_score_ = self._compute_stability() if compute_stability else None

        return self.labels_
    
    def _compute_consensus(self, partitions: list[np.ndarray]) -> np.ndarray:
        """
        Compute consensus partition from multiple runs.

        Two strategies are supported, chosen by ``self.consensus_method``:

        - ``"graph"`` (default, memory-efficient): threshold the sparse
          co-occurrence and run Leiden once on the resulting weighted graph.
          Peak memory ~O(E).  Reference: Lancichinetti & Fortunato,
          *Consensus clustering in complex networks*, Sci. Rep. 2:336 (2012).
        - ``"hierarchical"`` (legacy): average-linkage on the full
          co-occurrence distance.  Peak memory ~O(N²); use only for small N.
        """
        import math
        from scipy.sparse import coo_matrix as sp_coo

        n_nodes = len(partitions[0])
        n_runs = len(partitions)
        tau = float(self.consensus_threshold_tau)
        # Integer threshold: e.g. tau=0.5, n_runs=10 → 5 runs must agree.
        # math.ceil makes the intent explicit for non-integer products (e.g. 0.6×7=4.2→5).
        threshold_count = math.ceil(tau * n_runs)

        co_occur = None
        for r_idx, partition in enumerate(partitions):
            # Group node indices by cluster for this run.
            cluster_to_nodes: dict[int, list[int]] = {}
            for node_idx, cluster_id in enumerate(partition):
                cluster_to_nodes.setdefault(int(cluster_id), []).append(node_idx)

            # Enumerate upper-triangle pairs (i < j, no diagonal) within each cluster.
            # Upper triangle only: the downstream consumer at line 307 reads only
            # coo.row < coo.col, so lower-triangle and diagonal entries are always
            # discarded — no point computing or storing them.
            # This replaces the M @ M.T approach: same counts, no M matrix, no
            # intermediate co_run dense step, half the pairs to store.
            run_rows: list[np.ndarray] = []
            run_cols: list[np.ndarray] = []
            for members in cluster_to_nodes.values():
                if len(members) < 2:
                    continue
                m = np.sort(np.asarray(members, dtype=np.int32))
                ii, jj = np.triu_indices(len(m), k=1)  # k=1 skips diagonal
                run_rows.append(m[ii])
                run_cols.append(m[jj])

            if not run_rows:
                continue

            r = np.concatenate(run_rows)
            c = np.concatenate(run_cols)
            del run_rows, run_cols
            # int16: counts ∈ [0, n_runs ≤ 32k], 2 bytes vs float32's 4 bytes.
            # Each node is in exactly one cluster per run, so no duplicate (r,c)
            # pairs exist within a single run — .tocsr() handles conversion cleanly.
            co_run = sp_coo(
                (np.ones(len(r), dtype=np.int16), (r, c)),
                shape=(n_nodes, n_nodes),
            ).tocsr()
            del r, c

            co_occur = co_run if co_occur is None else co_occur + co_run

            # Early pruning: after run r_idx, the max a pair can still reach is
            # current_count + runs_remaining.  Drop pairs whose ceiling falls
            # below threshold_count — they can never survive the final cut.
            runs_remaining = n_runs - r_idx - 1
            min_reachable = threshold_count - runs_remaining
            if min_reachable > 1:
                co_occur.data[co_occur.data < min_reachable] = 0
                co_occur.eliminate_zeros()

        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

        if self.consensus_method == "graph":
            coo = co_occur.tocoo()
            rows, cols = coo.row, coo.col
            freq = coo.data.astype(np.float64) / float(n_runs)
            del coo, co_occur
            return self._consensus_via_leiden_on_graph(
                rows, cols, freq, n_nodes, n_runs, partitions
            )

        # ------------------------------------------------------------------
        # Legacy hierarchical path (consensus_method == "hierarchical")
        # ------------------------------------------------------------------
        from scipy.spatial.distance import squareform

        if self.low_memory:
            # Build condensed distance directly from sparse co-occurrence.
            # Symmetrize in sparse form (guards against FP asymmetry, same
            # as the dense (distance + distance.T)/2 step below).
            co_occur = (co_occur + co_occur.T) * 0.5
            coo = co_occur.tocoo()
            coo.sum_duplicates()

            # Default 1.0 = max distance for pairs that never co-clustered
            # in any of n_runs partitions.
            # float64 (not float32): scipy.linkage calls _convert_to_double on
            # non-float64 input, creating a hidden copy that doubles peak RAM.
            # float64 from the start lets scipy reuse the array in-place.
            n_pairs = n_nodes * (n_nodes - 1) // 2
            condensed = np.ones(n_pairs, dtype=np.float64)

            # Upper triangle only.  Condensed index for (i,j) with i<j is
            # i*n - i*(i+1)/2 + (j-i-1), matching scipy.squareform's layout.
            mask = coo.row < coo.col
            i = coo.row[mask].astype(np.int64)
            j = coo.col[mask].astype(np.int64)
            v = coo.data[mask].astype(np.float64) / float(n_runs)
            idx = n_nodes * i - i * (i + 1) // 2 + (j - i - 1)
            condensed[idx] = 1.0 - v
            np.clip(condensed, 0.0, 1.0, out=condensed)

            # Free sparse workspace before linkage allocates its own.
            del co_occur, coo
        else:
            # Original dense path.
            # For very large datasets (>50k) this remains the bottleneck;
            # at that scale set ``low_memory=True`` on GraphWeaveConfig.
            co_occur_dense = co_occur.toarray() / n_runs
            np.fill_diagonal(co_occur_dense, 1.0)
            distance = 1.0 - co_occur_dense

            # Ensure perfect symmetry and no negative values (floating-point)
            distance = np.clip((distance + distance.T) / 2, 0.0, 1.0)

            condensed = squareform(distance, checks=False)

        # Average linkage tends to work well for consensus.
        # Prefer fastcluster (C++, Θ(N²) time, no hidden float64 copy) when
        # available; fall back to scipy for environments without it.
        try:
            import fastcluster

            Z = fastcluster.linkage(condensed, method="average")
        except ImportError:
            Z = linkage(condensed, method="average")

        # Cut at threshold that matches approximate number of clusters
        # from the most frequent partition
        n_clusters_list = [len(np.unique(p)) for p in partitions]
        median_n_clusters = int(np.median(n_clusters_list))

        # Find optimal cut
        best_labels = None
        best_score = -1

        for n_clusters in range(max(2, median_n_clusters - 2), median_n_clusters + 3):
            try:
                labels = fcluster(Z, n_clusters, criterion="maxclust")
                labels = labels - 1  # 0-indexed

                # Score by average ARI with original partitions
                ari_scores = [adjusted_rand_score(labels, p) for p in partitions]
                avg_ari = np.mean(ari_scores)

                if avg_ari > best_score:
                    best_score = avg_ari
                    best_labels = labels
            except Exception as e:
                warnings.warn(f"Consensus partition failed for n_clusters={n_clusters}: {e}")
                continue

        if best_labels is None:
            # Fallback: pick the partition with the highest average ARI
            # against all others
            best_fallback_score = -1
            for p in partitions:
                avg = np.mean([adjusted_rand_score(p, q) for q in partitions])
                if avg > best_fallback_score:
                    best_fallback_score = avg
                    best_labels = p

        return best_labels

    def _consensus_via_leiden_on_graph(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        freq: np.ndarray,
        n_nodes: int,
        n_runs: int,
        partitions: list[np.ndarray],
    ) -> np.ndarray:
        """
        Graph-based consensus (Lancichinetti & Fortunato 2012).

        Threshold the sparse co-occurrence at ``self.consensus_threshold_tau``
        (fraction of runs that must agree on a pair), build an igraph weighted
        graph from the surviving edges, and run Leiden once to obtain the
        consensus partition.  Peak memory is O(E) for E surviving edges,
        avoiding the N×N dense matrix and the scipy.linkage workspace.
        """
        from graphweave.utils.timing import step_timer
        import igraph as ig
        import leidenalg as la

        # rows/cols/freq: upper-triangle pairs extracted by caller so co_occur
        # can be freed before igraph allocates its own graph structure.
        tau = float(self.consensus_threshold_tau)

        def _build_and_cluster(threshold: float) -> np.ndarray | None:
            keep = freq >= threshold
            if not np.any(keep):
                return None
            keep_rows = rows[keep]
            keep_cols = cols[keep]
            keep_weights = freq[keep]
            del keep
            edge_array = np.column_stack([keep_rows, keep_cols])
            del keep_rows, keep_cols
            n_edges = edge_array.shape[0]
            g = ig.Graph(n=n_nodes, edges=edge_array, directed=False)
            del edge_array
            g.es["weight"] = keep_weights
            del keep_weights
            # Same node_sizes-on-RBER swap as the per-run partition above:
            # keeps a heavily-weighted coreset point's community from being
            # washed out in the consensus step too.
            node_weights = getattr(self, "_node_weights", None)
            with step_timer(f"leiden-consensus-run ({n_edges:,} edges)", verbose=self.verbose, indent=15):
                if node_weights is not None and len(node_weights) == n_nodes:
                    part = la.find_partition(
                        g,
                        la.RBERVertexPartition,
                        weights="weight",
                        node_sizes=node_weights.tolist(),
                        resolution_parameter=self.resolution,
                        seed=self.random_state,
                    )
                else:
                    part = la.find_partition(
                        g,
                        la.RBConfigurationVertexPartition,
                        weights="weight",
                        resolution_parameter=self.resolution,
                        seed=self.random_state,
                    )
            labels = np.asarray(part.membership)
            del part, g
            return labels

        labels = _build_and_cluster(tau)

        # If thresholding wiped out the graph (or left everything isolated),
        # back off once.  Isolated nodes form singleton clusters in Leiden,
        # so "degenerate" here means *every* node became a singleton.
        if labels is None or len(np.unique(labels)) >= n_nodes:
            relaxed = max(tau * 0.7, 1.0 / n_runs)
            if relaxed < tau:
                labels = _build_and_cluster(relaxed)

        if labels is None or len(np.unique(labels)) >= n_nodes:
            # Final fallback: pick the input partition with highest mean ARI.
            best_fallback_score = -1.0
            best = partitions[0]
            for p in partitions:
                avg = float(np.mean([adjusted_rand_score(p, q) for q in partitions]))
                if avg > best_fallback_score:
                    best_fallback_score = avg
                    best = p
            labels = best

        return labels

    def _handle_small_clusters(
        self,
        labels: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """Mark small clusters as outliers (-1).

        When per-node weights are present (a weighted coreset), "small" is
        judged by the cluster's summed represented mass, not its raw node count
        — otherwise a large real theme that happens to be sparsely sampled gets
        deleted as small (tail-collapse), the very failure the coreset weighting
        exists to prevent.
        """
        result = labels.copy()
        node_weights = getattr(self, "_node_weights", None)

        if node_weights is not None and len(node_weights) == len(result):
            for cid in np.unique(result):
                if cid != -1 and node_weights[result == cid].sum() < min_size:
                    result[result == cid] = -1
        else:
            unique, counts = np.unique(result, return_counts=True)
            for cid, cnt in zip(unique, counts):
                if cid != -1 and cnt < min_size:
                    result[result == cid] = -1

        # Relabel to consecutive integers (vectorized)
        non_outlier = np.sort(np.unique(result[result != -1]))
        if len(non_outlier) == 0:
            return result
        max_id = int(non_outlier[-1])
        remap = np.full(max_id + 1, -1, dtype=np.int64)
        remap[non_outlier] = np.arange(len(non_outlier), dtype=np.int64)
        out = np.where(result == -1, np.int64(-1), remap[np.clip(result, 0, max_id)])
        return out
    
    def _compute_stability(self) -> float:
        """Compute stability score as average pairwise ARI."""
        if len(self._all_partitions) < 2:
            return 1.0

        from joblib import Parallel, delayed

        pairs = [
            (i, j)
            for i in range(len(self._all_partitions))
            for j in range(i + 1, len(self._all_partitions))
        ]
        ari_scores = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(adjusted_rand_score)(
                self._all_partitions[i], self._all_partitions[j]
            )
            for i, j in pairs
        )
        return float(np.mean(ari_scores))
    
    def find_optimal_resolution(
        self,
        graph: "igraph.Graph",
        resolution_range: tuple[float, float] = (0.1, 2.0),
        n_steps: int = 10,
        target_n_topics: int | None = None,
    ) -> float:
        """
        Find optimal resolution parameter.

        When *target_n_topics* is given, uses binary search for much higher
        precision (O(log n) instead of O(n)).  Falls back to a linear sweep
        only when no target is specified.

        Parameters
        ----------
        graph : igraph.Graph
            Input graph.
        resolution_range : tuple
            Range of resolutions to search.
        n_steps : int
            Number of search steps (binary-search iterations when
            *target_n_topics* is given, linear sweep points otherwise).
        target_n_topics : int, optional
            If provided, find resolution closest to this number of topics.

        Returns
        -------
        optimal_resolution : float
            Best resolution parameter.
        """
        import leidenalg as la

        def _n_clusters_at(res: float) -> int:
            partition = la.find_partition(
                graph,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=res,
                seed=self.random_state,
            )
            return len(set(partition.membership))

        if target_n_topics is not None:
            # Binary search: higher resolution → more clusters
            lo, hi = resolution_range
            best_res, best_diff = lo, abs(_n_clusters_at(lo) - target_n_topics)

            for _ in range(n_steps):
                mid = (lo + hi) / 2
                n_clust = _n_clusters_at(mid)
                diff = abs(n_clust - target_n_topics)

                if diff < best_diff:
                    best_diff = diff
                    best_res = mid

                if n_clust == target_n_topics:
                    return mid
                elif n_clust < target_n_topics:
                    lo = mid
                else:
                    hi = mid

            return best_res
        else:
            # Linear sweep for maximum modularity
            resolutions = np.linspace(resolution_range[0], resolution_range[1], n_steps)
            best_res = resolutions[0]
            best_mod = -float("inf")

            for res in resolutions:
                partition = la.find_partition(
                    graph,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=res,
                    seed=self.random_state,
                )
                if partition.modularity > best_mod:
                    best_mod = partition.modularity
                    best_res = res

            return best_res


class HDBSCANClusterer:
    """
    Alternative clustering using HDBSCAN.
    
    Useful for datasets with varying density or many outliers.
    """
    
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels_: np.ndarray | None = None
        self.probabilities_: np.ndarray | None = None
    
    def fit_predict(
        self,
        embeddings: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit HDBSCAN clustering.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Document embeddings (optionally reduced with UMAP first).
            
        Returns
        -------
        labels : np.ndarray
            Cluster assignments. -1 for outliers.
        """
        import hdbscan
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            **kwargs,
        )
        
        self.labels_ = clusterer.fit_predict(embeddings)
        self.probabilities_ = clusterer.probabilities_
        
        return self.labels_
