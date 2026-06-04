"""
Topic alignment & data-reduction helpers for cumulative clustering
===================================================================

Two concerns live here, both shared by the orchestrator and the recluster
strategies (see :mod:`tritopic.cumulative.strategies`):

1. **Cross-epoch topic alignment** — when the corpus is re-clustered, the new
   topic IDs are arbitrary. :func:`align_topics` matches new topic centroids to
   the persistent *global registry* of previous topics via Hungarian assignment
   on cosine similarity, so a theme keeps a *stable global ID* across epochs.
   This is the mechanism that realises the "global topic merge" of approach #2.

2. **Data reduction for unbounded growth (Regime B)** — :func:`select_coreset`
   draws a bounded (optionally recency-weighted) representative sample, and
   :func:`summarize_embeddings` folds history into micro-cluster centroids
   (CluStream/BIRCH-style sufficient statistics) so memory/compute stay bounded
   regardless of total accumulated N.

None of this touches the full-batch :class:`~tritopic.TriTopic` path.
"""

from __future__ import annotations

import numpy as np


def align_topics(
    new_centroids: np.ndarray,
    registry_centroids: np.ndarray | None,
    registry_ids: list[int],
    threshold: float,
    next_id: int,
) -> tuple[dict[int, int], int, list[tuple[int, int, float]]]:
    """Match new topic centroids to the global registry via Hungarian assignment.

    Parameters
    ----------
    new_centroids : np.ndarray
        ``(k_new, d)`` centroids of the freshly clustered topics, in the same
        order as the model's non-outlier topics.
    registry_centroids : np.ndarray | None
        ``(k_reg, d)`` centroids of the persistent global topics, or ``None`` /
        empty for the first epoch.
    registry_ids : list[int]
        Global IDs aligned row-wise with *registry_centroids*.
    threshold : float
        Minimum cosine similarity for a match. Below this, a new topic is
        treated as a genuinely new global topic and is minted a fresh ID.
    next_id : int
        Next free global ID counter.

    Returns
    -------
    mapping : dict[int, int]
        ``new_row_index -> global_id`` for every new topic.
    next_id : int
        Updated free-ID counter.
    matches : list[(new_idx, registry_idx, similarity)]
        The accepted matches (useful for registry centroid updates / debugging).
    """
    k_new = len(new_centroids)
    mapping: dict[int, int] = {}

    # First epoch (or no registry): assign sequential global IDs.
    if registry_centroids is None or len(registry_centroids) == 0:
        for i in range(k_new):
            mapping[i] = next_id
            next_id += 1
        return mapping, next_id, []

    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(new_centroids, registry_centroids)  # (k_new, k_reg)
    # linear_sum_assignment minimises cost -> negate to maximise similarity.
    row_ind, col_ind = linear_sum_assignment(-sim)

    matches: list[tuple[int, int, float]] = []
    for r, c in zip(row_ind, col_ind):
        if sim[r, c] >= threshold:
            mapping[int(r)] = registry_ids[int(c)]
            matches.append((int(r), int(c), float(sim[r, c])))

    # Unmatched new topics -> fresh global IDs.
    for i in range(k_new):
        if i not in mapping:
            mapping[i] = next_id
            next_id += 1

    return mapping, next_id, matches


def identity_mapping(k_new: int, next_id: int) -> tuple[dict[int, int], int]:
    """Sequential ``new_row_index -> global_id`` map (alignment disabled)."""
    mapping = {i: next_id + i for i in range(k_new)}
    return mapping, next_id + k_new


def assign_to_registry(
    embeddings: np.ndarray,
    registry_centroids: np.ndarray,
    registry_ids: list[int],
    outlier_threshold: float,
) -> np.ndarray:
    """Assign each embedding to the nearest global topic by cosine similarity.

    Mirrors :meth:`TriTopic.transform` but against the persistent global
    registry rather than a single model's centroids. Docs whose best match is
    below *outlier_threshold* are labelled ``-1``.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if registry_centroids is None or len(registry_centroids) == 0:
        return np.full(len(embeddings), -1, dtype=int)

    ids = np.asarray(registry_ids)
    sim = cosine_similarity(embeddings, registry_centroids)
    nearest = np.argmax(sim, axis=1)
    max_sim = sim[np.arange(len(embeddings)), nearest]
    labels = ids[nearest]
    labels[max_sim < outlier_threshold] = -1
    return labels


def recency_weights(n: int, new_count: int, floor: float = 0.25) -> np.ndarray:
    """Importance weights that favour the newest docs for coreset sampling.

    The newest *new_count* docs get weight 1.0; older docs ramp linearly down to
    *floor*. Keeps recent signal dense while still sampling history (so emerging
    themes are not starved — mitigates coreset tail-collapse).
    """
    w = np.full(n, floor, dtype=float)
    if new_count >= n:
        return np.ones(n, dtype=float)
    old_n = n - new_count
    # linear ramp floor -> 1.0 across the older region, then 1.0 for the new tail
    w[:old_n] = np.linspace(floor, 1.0, old_n, endpoint=False)
    w[old_n:] = 1.0
    return w


def select_coreset(
    embeddings: np.ndarray,
    size: int,
    random_state: int = 42,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Pick a bounded representative subset of row indices.

    Uniform by default; pass *weights* for importance sampling (e.g.
    :func:`recency_weights`). Returns sorted indices so downstream order is
    stable. If ``size >= len(embeddings)`` returns all indices.
    """
    n = len(embeddings)
    if size >= n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    if weights is None:
        idx = rng.choice(n, size=size, replace=False)
    else:
        p = np.asarray(weights, dtype=float)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        idx = rng.choice(n, size=size, replace=False, p=p)
    return np.sort(idx)


def summarize_embeddings(
    embeddings: np.ndarray,
    k: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold embeddings into *k* micro-cluster centroids + member counts.

    CluStream/BIRCH-style summary: the centroids are weighted representative
    points that stand in for the raw history in Regime B, so old themes survive
    as compact summaries instead of being windowed away. Uses MiniBatchKMeans
    for scalability.

    Returns
    -------
    centroids : np.ndarray   # (k_eff, d)
    counts : np.ndarray      # (k_eff,) members per micro-cluster
    """
    from sklearn.cluster import MiniBatchKMeans

    n = len(embeddings)
    k_eff = int(min(k, n))
    if k_eff <= 1:
        return embeddings.mean(axis=0, keepdims=True), np.array([n], dtype=float)

    km = MiniBatchKMeans(n_clusters=k_eff, random_state=random_state, n_init=3)
    labels = km.fit_predict(embeddings)
    counts = np.bincount(labels, minlength=k_eff).astype(float)
    # Drop any empty micro-clusters.
    keep = counts > 0
    return km.cluster_centers_[keep], counts[keep]
