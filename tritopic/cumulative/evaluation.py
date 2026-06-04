"""
Metrics harness for cumulative clustering
==========================================

Compares cumulative, batch-wise clustering against the **full-batch baseline**
(a single :meth:`TriTopic.fit` over all accumulated documents — the quality
ceiling). Implements the metric set requested for the POC:

ARI, NMI, topic-count drift, keyword overlap, silhouette, stability, plus
coherence / diversity and operational stats. :func:`benchmark_strategies` runs
every pluggable engine through the *same* batch stream so the best path is
visible side-by-side rather than argued in the abstract.

All comparisons reuse existing helpers in :mod:`tritopic.utils.metrics`.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from tritopic.core.embeddings import EmbeddingEngine
from tritopic.core.model import TriTopic, TriTopicConfig
from tritopic.cumulative.cumulative import CumulativeConfig, CumulativeTriTopic
from tritopic.cumulative.strategies import STRATEGY_NAMES
from tritopic.utils.metrics import (
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


def _mean_keyword_overlap(cum_model: TriTopic, full_model: TriTopic) -> float:
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


def compare_to_full_batch(
    cumulative: CumulativeTriTopic,
    full_model: TriTopic,
    labels_true: np.ndarray | None = None,
) -> dict:
    """Compare a cumulative model to the full-batch baseline on the same corpus.

    Both must cover the *same documents in the same order* (feed the cumulative
    model the batches that concatenate to ``full_model``'s training docs).

    Parameters
    ----------
    cumulative : CumulativeTriTopic
        Fitted cumulative model (its ``labels_`` span all accumulated docs).
    full_model : TriTopic
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

    if labels_true is not None:
        metrics["ari_vs_truth_cumulative"] = compute_ari(cum_labels, labels_true)
        metrics["ari_vs_truth_full"] = compute_ari(full_labels, labels_true)
        metrics["nmi_vs_truth_cumulative"] = compute_nmi(cum_labels, labels_true)
        metrics["nmi_vs_truth_full"] = compute_nmi(full_labels, labels_true)

    return metrics


def benchmark_strategies(
    batches: list[list[str]],
    base_config: TriTopicConfig | None = None,
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
    base_config : TriTopicConfig, optional
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
    base_config = base_config or TriTopicConfig(verbose=False)
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
    full_model = TriTopic(config=copy.deepcopy(base_config))
    full_model.fit(all_docs, embeddings=all_emb)

    rows = []
    for strat in strategies:
        cfg = CumulativeConfig(base_config=base_config, strategy=strat, **cumulative_kwargs)
        cum = CumulativeTriTopic(cfg)

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
