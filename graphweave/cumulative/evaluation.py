"""
Metrics harness for cumulative clustering
==========================================

Compares cumulative, batch-wise clustering against the **full-batch baseline**
(a single :meth:`GraphWeave.fit` over all accumulated documents — the quality
ceiling). Implements the metric set requested for the POC:

ARI, NMI, topic-count drift, keyword overlap, silhouette, stability, plus
coherence / diversity and operational stats. :func:`benchmark_strategies` runs
every pluggable engine through the *same* batch stream so the best path is
visible side-by-side rather than argued in the abstract.

All comparisons reuse existing helpers in :mod:`graphweave.utils.metrics`.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from graphweave.core.embeddings import EmbeddingEngine
from graphweave.core.model import GraphWeave, GraphWeaveConfig
from graphweave.cumulative.cumulative import CumulativeConfig, CumulativeGraphWeave
from graphweave.cumulative.strategies import STRATEGY_NAMES
from graphweave.utils.metrics import (
    compute_ari,
    compute_nmi,
    compute_silhouette,
    keyword_jaccard,
)


def _match_topics_by_centroid(
    centroids_a: np.ndarray, centroids_b: np.ndarray
) -> list[tuple[int, int]]:
    """Hungarian match between two centroid sets on cosine similarity."""
    if centroids_a is None or centroids_b is None or len(centroids_a) == 0 or len(centroids_b) == 0:
        return []
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(centroids_a, centroids_b)
    row_ind, col_ind = linear_sum_assignment(-sim)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def _mean_keyword_overlap(cum_model: GraphWeave, full_model: GraphWeave) -> float:
    """Mean Jaccard keyword overlap over centroid-matched topic pairs."""
    cum_topics = [t for t in cum_model.topics_ if t.topic_id != -1]
    full_topics = [t for t in full_model.topics_ if t.topic_id != -1]
    pairs = _match_topics_by_centroid(
        cum_model.topic_embeddings_, full_model.topic_embeddings_
    )
    if not pairs:
        return 0.0
    overlaps = [
        keyword_jaccard(cum_topics[i].keywords, full_topics[j].keywords)
        for i, j in pairs
        if i < len(cum_topics) and j < len(full_topics)
    ]
    return float(np.mean(overlaps)) if overlaps else 0.0


def _rare_topic_recall(
    cum_model: GraphWeave,
    full_model: GraphWeave,
    n_docs: int,
    rare_frac: float,
    sim_cutoff: float,
) -> tuple[float, int]:
    """Fraction of *small* full-batch topics that survive in the cumulative model.

    A full-batch topic is "rare" if its size is below ``rare_frac * n_docs``; it is
    "recovered" if some cumulative topic centroid is within ``sim_cutoff`` cosine of
    it. This is the headline tail-collapse metric: random/recency coresets drop rare
    topics, stratified coresets keep them.

    Returns ``(recall, n_rare)``; recall is ``nan`` when there are no rare topics.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    full_topics = [t for t in full_model.topics_ if t.topic_id != -1]
    full_centroids = full_model.topic_embeddings_
    cum_centroids = cum_model.topic_embeddings_ if cum_model is not None else None

    rare = [j for j, t in enumerate(full_topics) if t.size < rare_frac * max(n_docs, 1)]
    n_rare = len(rare)
    if n_rare == 0 or full_centroids is None or len(full_centroids) == 0:
        return float("nan"), n_rare
    if cum_centroids is None or len(cum_centroids) == 0:
        return 0.0, n_rare

    sim = cosine_similarity(full_centroids, cum_centroids)  # (k_full, k_cum)
    recovered = sum(1 for j in rare if j < sim.shape[0] and sim[j].max() >= sim_cutoff)
    return recovered / n_rare, n_rare


def coreset_cost_ratio(
    working_emb: np.ndarray,
    working_weights: np.ndarray | None,
    full_emb: np.ndarray,
    k: int,
    random_state: int = 42,
) -> float:
    """Theory-aligned coreset quality: weighted k-means distortion on the working
    set ÷ distortion on the full data, under reference centers fit on the full data.

    A value near 1.0 means the coreset preserves the k-means cost (a good coreset);
    large values mean the working set misrepresents the data's geometry. Call this
    when a single strategy's working set + weights are on hand (it is intentionally
    not part of the multi-strategy table, which does not retain working sets).
    """
    try:
        from cuml.cluster import MiniBatchKMeans
    except ImportError:
        from sklearn.cluster import MiniBatchKMeans

    k_eff = int(min(k, len(full_emb)))
    if k_eff <= 1:
        return float("nan")
    km = MiniBatchKMeans(n_clusters=k_eff, random_state=random_state, n_init=3).fit(full_emb)
    centers = km.cluster_centers_

    def _distortion(emb: np.ndarray, w: np.ndarray | None) -> float:
        # min squared distance of each point to any center, optionally weighted.
        d2 = ((emb[:, None, :] - centers[None, :, :]) ** 2).sum(-1).min(axis=1)
        if w is None:
            return float(d2.mean())
        w = np.asarray(w, dtype=float)
        return float((d2 * w).sum() / max(w.sum(), 1e-12))

    full_cost = _distortion(full_emb, None)
    work_cost = _distortion(working_emb, working_weights)
    return work_cost / max(full_cost, 1e-12)


def compare_to_full_batch(
    cumulative: CumulativeGraphWeave,
    full_model: GraphWeave,
    labels_true: np.ndarray | None = None,
    rare_frac: float = 0.01,
    rare_sim_cutoff: float = 0.5,
) -> dict:
    """Compare a cumulative model to the full-batch baseline on the same corpus.

    Both must cover the *same documents in the same order* (feed the cumulative
    model the batches that concatenate to ``full_model``'s training docs).

    Parameters
    ----------
    cumulative : CumulativeGraphWeave
        Fitted cumulative model (its ``labels_`` span all accumulated docs).
    full_model : GraphWeave
        Full-batch model fit once on the same accumulated docs.
    labels_true : np.ndarray, optional
        Ground-truth labels (if the benchmark dataset has them).

    Returns
    -------
    dict of metrics.
    """
    cum_labels = cumulative.labels_
    full_labels = full_model.labels_
    if cum_labels is None or full_labels is None:
        raise ValueError("Both models must be fitted before comparison.")
    if len(cum_labels) != len(full_labels):
        raise ValueError(
            f"Label lengths differ ({len(cum_labels)} vs {len(full_labels)}); "
            "the cumulative and full-batch corpora must match."
        )

    emb = cumulative.embeddings_
    cum_n_topics = cumulative.n_global_topics
    full_n_topics = len([t for t in full_model.topics_ if t.topic_id != -1])

    metrics = {
        # Agreement with the full-batch baseline (permutation-invariant).
        "ari_vs_full": compute_ari(cum_labels, full_labels),
        "nmi_vs_full": compute_nmi(cum_labels, full_labels),
        # Topic structure.
        "n_topics_cumulative": cum_n_topics,
        "n_topics_full": full_n_topics,
        "topic_count_drift": abs(cum_n_topics - full_n_topics),
        # Keyword stability over matched topics.
        "keyword_overlap": _mean_keyword_overlap(cumulative.model_, full_model),
        # Intrinsic cluster quality (delta = cumulative - full; >= 0 is good).
        "silhouette_cumulative": compute_silhouette(emb, cum_labels),
        "silhouette_full": compute_silhouette(emb, full_labels),
        # Outliers.
        "outlier_ratio_cumulative": float(np.mean(cum_labels == -1)),
        "outlier_ratio_full": float(np.mean(full_labels == -1)),
        # Bookkeeping.
        "n_epochs": cumulative.epoch,
        "n_docs": len(cum_labels),
    }
    metrics["silhouette_delta"] = metrics["silhouette_cumulative"] - metrics["silhouette_full"]

    # Tail-collapse: do the full-batch's rare topics survive in the cumulative model?
    recall, n_rare = _rare_topic_recall(
        cumulative.model_, full_model, len(cum_labels), rare_frac, rare_sim_cutoff
    )
    metrics["rare_topic_recall"] = recall
    metrics["n_rare_topics_full"] = n_rare

    if labels_true is not None:
        metrics["ari_vs_truth_cumulative"] = compute_ari(cum_labels, labels_true)
        metrics["ari_vs_truth_full"] = compute_ari(full_labels, labels_true)
        metrics["nmi_vs_truth_cumulative"] = compute_nmi(cum_labels, labels_true)
        metrics["nmi_vs_truth_full"] = compute_nmi(full_labels, labels_true)

    return metrics


def benchmark_strategies(
    batches: list[list[str]],
    base_config: GraphWeaveConfig | None = None,
    strategies: tuple[str, ...] = STRATEGY_NAMES,
    labels_true: np.ndarray | None = None,
    cumulative_kwargs: dict | None = None,
    precomputed_batch_embeddings: list[np.ndarray] | None = None,
) -> pd.DataFrame:
    """Stream the same batches through every strategy and tabulate metrics.

    Embeds each batch **once** (shared across strategies and the baseline), fits
    the full-batch reference, then runs each cumulative strategy and compares it
    to that reference.

    Parameters
    ----------
    batches : list[list[str]]
        Ordered batches of documents.
    base_config : GraphWeaveConfig, optional
        Full-batch config reused everywhere (defaults to quiet defaults).
    strategies : tuple[str, ...]
        Which engines to benchmark (default: all three).
    labels_true : np.ndarray, optional
        Ground-truth labels over the concatenated corpus (same order as batches).
    cumulative_kwargs : dict, optional
        Extra :class:`CumulativeConfig` fields (e.g. ``recluster_trigger``).
    precomputed_batch_embeddings : list[np.ndarray], optional
        Per-batch embeddings to skip encoding entirely.

    Returns
    -------
    pandas.DataFrame
        One row per strategy, columns = metrics + wall-clock.
    """
    base_config = base_config or GraphWeaveConfig(verbose=False)
    cumulative_kwargs = cumulative_kwargs or {}

    # Embed every batch once, reused by the baseline and all strategies.
    if precomputed_batch_embeddings is not None:
        batch_embs = [np.asarray(e) for e in precomputed_batch_embeddings]
    else:
        engine = EmbeddingEngine(
            model_name=base_config.embedding_model,
            batch_size=base_config.embedding_batch_size,
            provider=base_config.embedding_provider,
            api_key=base_config.embedding_api_key,
            verbose=False,
        )
        batch_embs = [engine.encode(b) for b in batches]

    all_docs = [d for b in batches for d in b]
    all_emb = np.vstack(batch_embs)

    # Full-batch baseline (the quality ceiling).
    full_model = GraphWeave(config=copy.deepcopy(base_config))
    full_model.fit(all_docs, embeddings=all_emb)

    rows = []
    for strat in strategies:
        cfg = CumulativeConfig(base_config=base_config, strategy=strat, **cumulative_kwargs)
        cum = CumulativeGraphWeave(cfg)

        t0 = time.perf_counter()
        for docs, emb in zip(batches, batch_embs):
            cum.add_batch(docs, embeddings=emb)
        elapsed = time.perf_counter() - t0

        metrics = compare_to_full_batch(cum, full_model, labels_true=labels_true)
        metrics["strategy"] = strat
        metrics["wall_clock_s"] = round(elapsed, 3)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    cols = ["strategy"] + [c for c in df.columns if c != "strategy"]
    return df[cols]
