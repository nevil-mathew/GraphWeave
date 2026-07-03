"""Evaluation harness: did the adapted embeddings actually get better?

Three layers, cheapest/most-general first:

1. **Held-out LLM triplet accuracy** (:func:`triplet_accuracy`) — label-free,
   corpus-specific. Works on any corpus, adapted or not.
2. **Intrinsic + ground-truth clustering metrics** (:func:`compare_embedders`)
   — same TriTopic config for every variant, averaged over seeds.
3. **Forgetting check** (:func:`forgetting_check`) — a small shipped-in-file
   generic-English sanity set to catch over-specialization.

``compare_embedders`` is the one function here that drives TriTopic itself
(the actual clustering comparison needs it); it and its helper import
``tritopic.core.model`` / ``tritopic.utils.metrics`` lazily inside the
function body — the one documented exception to this package's import
boundary (see ``tritopic/adaptation/__init__.py``).
"""

from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np
import pandas as pd

from .triplets import TripletBank


# ---------------------------------------------------------------------------
# Label-free triplet accuracy
# ---------------------------------------------------------------------------

def triplet_accuracy(embeddings: np.ndarray, judgments) -> float:
    """Fraction of judgments where ``cos(anchor, positive) >
    cos(anchor, negative)`` in *embeddings*'s space.

    Accepts a list of :class:`~tritopic.adaptation.triplets.TripletJudgment`
    or plain ``(anchor, positive, negative)`` index tuples. This is
    ClusterLLM's own validation signal: it needs no ground-truth labels and
    directly measures whether the embedding geometry moved toward the LLM's
    judgments on *this* corpus.
    """
    judgments = list(judgments)
    if not judgments:
        return float("nan")

    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    correct = 0
    for j in judgments:
        a, p, n = (j.anchor, j.positive, j.negative) if hasattr(j, "anchor") else j
        if float(np.dot(normed[a], normed[p])) > float(np.dot(normed[a], normed[n])):
            correct += 1
    return correct / len(judgments)


# ---------------------------------------------------------------------------
# Hungarian cluster accuracy
# ---------------------------------------------------------------------------

def hungarian_accuracy(labels_pred: np.ndarray, labels_true: np.ndarray) -> float:
    """Best-case cluster accuracy: match predicted clusters to true classes
    via the Hungarian algorithm on the contingency table, then score the
    fraction of (non-outlier) documents in a correctly matched pair."""
    from scipy.optimize import linear_sum_assignment

    labels_pred = np.asarray(labels_pred)
    labels_true = np.asarray(labels_true)
    mask = labels_pred != -1
    if mask.sum() == 0:
        return 0.0

    yp, yt = labels_pred[mask], labels_true[mask]
    pred_ids, true_ids = np.unique(yp), np.unique(yt)
    pred_index = {c: i for i, c in enumerate(pred_ids)}
    true_index = {c: i for i, c in enumerate(true_ids)}

    contingency = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for p, t in zip(yp, yt):
        contingency[pred_index[p], true_index[t]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    correct = contingency[row_ind, col_ind].sum()
    return float(correct / mask.sum())


# ---------------------------------------------------------------------------
# Forgetting check (regression guard)
# ---------------------------------------------------------------------------

# ~40 hand-written generic-English pairs spanning paraphrase / related /
# unrelated, gold similarity on a 0-5 scale (STS-B convention). No downloads.
_GENERIC_STS_PAIRS: list[tuple[str, str, float]] = [
    ("A man is playing a guitar.", "A man is playing an acoustic guitar.", 4.6),
    ("A man is playing a guitar.", "A person is playing a musical instrument.", 3.8),
    ("A man is playing a guitar.", "A woman is cooking dinner.", 0.2),
    ("The dog is running in the park.", "A dog is running through the park.", 4.8),
    ("The dog is running in the park.", "A cat is sleeping on the couch.", 0.5),
    ("The dog is running in the park.", "An animal is moving outdoors.", 3.0),
    ("The stock market fell sharply today.", "Share prices dropped significantly today.", 4.5),
    ("The stock market fell sharply today.", "It is raining heavily outside.", 0.1),
    ("The stock market fell sharply today.", "Financial markets experienced a decline.", 4.0),
    ("She is reading a book in the library.", "A woman reads a book at the library.", 4.7),
    ("She is reading a book in the library.", "He is driving a car on the highway.", 0.2),
    ("She is reading a book in the library.", "Someone is studying quietly indoors.", 2.8),
    ("The chef prepared a delicious meal.", "A cook made a tasty dish.", 4.2),
    ("The chef prepared a delicious meal.", "The mechanic fixed the engine.", 0.3),
    ("Scientists discovered a new species of frog.", "Researchers found a previously unknown frog species.", 4.6),
    ("Scientists discovered a new species of frog.", "The bakery sells fresh bread every morning.", 0.1),
    ("The children played soccer after school.", "Kids played football after school.", 4.3),
    ("The children played soccer after school.", "The elderly man walked slowly down the street.", 0.4),
    ("The airplane landed safely at the airport.", "The plane touched down safely at the airport.", 4.8),
    ("The airplane landed safely at the airport.", "A ship sailed across the ocean.", 1.2),
    ("The company announced record profits this quarter.", "The firm reported its highest earnings ever this quarter.", 4.4),
    ("The company announced record profits this quarter.", "The garden was full of blooming flowers.", 0.1),
    ("He plays basketball every weekend.", "He shoots hoops every weekend.", 4.0),
    ("He plays basketball every weekend.", "She paints landscapes in her free time.", 0.3),
    ("The volcano erupted, spewing lava for miles.", "The volcano's eruption sent lava flowing for miles.", 4.7),
    ("The volcano erupted, spewing lava for miles.", "The library was quiet and peaceful.", 0.1),
    ("A new smartphone was released by the company.", "The company launched a new mobile phone.", 4.5),
    ("A new smartphone was released by the company.", "The recipe calls for two cups of flour.", 0.1),
    ("The team won the championship last night.", "Last night, the team became champions.", 4.6),
    ("The team won the championship last night.", "The weather forecast predicts snow tomorrow.", 0.2),
    ("The doctor examined the patient carefully.", "The physician carefully checked the patient.", 4.6),
    ("The doctor examined the patient carefully.", "The car needs an oil change soon.", 0.2),
    ("Rain is expected across the region tomorrow.", "Tomorrow, the region will likely see rain.", 4.5),
    ("Rain is expected across the region tomorrow.", "The museum exhibit features ancient pottery.", 0.1),
    ("The president gave a speech about the economy.", "The president addressed economic issues in a speech.", 4.3),
    ("The president gave a speech about the economy.", "The puppy chased its tail in circles.", 0.1),
    ("The train arrived ten minutes late.", "The train was ten minutes behind schedule.", 4.5),
    ("The train arrived ten minutes late.", "The chef seasoned the soup with herbs.", 0.1),
    ("A fire broke out in the old warehouse.", "An old warehouse caught fire.", 4.6),
    ("A fire broke out in the old warehouse.", "The students studied for their exams.", 0.2),
]


def forgetting_check(
    encode_before: Callable[[list[str]], np.ndarray],
    encode_after: Callable[[list[str]], np.ndarray],
    pairs: list[tuple[str, str, float]] | None = None,
) -> dict:
    """Spearman correlation of cosine similarity vs. gold score on a small
    generic-English sanity set, before vs. after adaptation.

    A ``delta`` below about -0.05 suggests the adapted embedder
    over-specialized to the corpus/triplets at the expense of general
    semantic competence — the standard fine-tuning regression guard.
    """
    from scipy.stats import spearmanr

    pairs = pairs if pairs is not None else _GENERIC_STS_PAIRS
    texts_a = [p[0] for p in pairs]
    texts_b = [p[1] for p in pairs]
    gold = [p[2] for p in pairs]

    def _sims(encode_fn) -> np.ndarray:
        emb_a = np.asarray(encode_fn(texts_a), dtype=np.float64)
        emb_b = np.asarray(encode_fn(texts_b), dtype=np.float64)
        na = np.linalg.norm(emb_a, axis=1, keepdims=True)
        nb = np.linalg.norm(emb_b, axis=1, keepdims=True)
        na, nb = np.where(na == 0, 1.0, na), np.where(nb == 0, 1.0, nb)
        return np.sum((emb_a / na) * (emb_b / nb), axis=1)

    spearman_before = float(spearmanr(_sims(encode_before), gold).correlation)
    spearman_after = float(spearmanr(_sims(encode_after), gold).correlation)
    return {
        "spearman_before": spearman_before,
        "spearman_after": spearman_after,
        "delta": spearman_after - spearman_before,
    }


# ---------------------------------------------------------------------------
# Clustering-level comparison
# ---------------------------------------------------------------------------

def _match_topics_by_centroid(centroids_a, centroids_b) -> list[tuple[int, int]]:
    """Hungarian match between two centroid sets on cosine similarity.

    Reimplemented locally (rather than imported) so this module stays
    detachable — mirrors ``tritopic.cumulative.evaluation._match_topics_by_centroid``.
    """
    if centroids_a is None or centroids_b is None or len(centroids_a) == 0 or len(centroids_b) == 0:
        return []
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(centroids_a, centroids_b)
    row_ind, col_ind = linear_sum_assignment(-sim)
    return list(zip(row_ind.tolist(), col_ind.tolist()))


def _keyword_overlap_vs(model_a, model_b) -> float:
    from tritopic.utils.metrics import keyword_jaccard

    topics_a = [t for t in model_a.topics_ if t.topic_id != -1]
    topics_b = [t for t in model_b.topics_ if t.topic_id != -1]
    pairs = _match_topics_by_centroid(model_a.topic_embeddings_, model_b.topic_embeddings_)
    if not pairs:
        return float("nan")
    overlaps = [
        keyword_jaccard(topics_a[i].keywords, topics_b[j].keywords)
        for i, j in pairs if i < len(topics_a) and j < len(topics_b)
    ]
    return float(np.mean(overlaps)) if overlaps else float("nan")


def compare_embedders(
    documents: list[str],
    embedders_or_embeddings: dict[str, np.ndarray | Callable[[list[str]], np.ndarray]],
    labels_true: np.ndarray | None = None,
    base_config=None,
    holdout_bank: TripletBank | None = None,
    n_seeds: int = 1,
    baseline: str | None = None,
) -> pd.DataFrame:
    """Fit an identical TriTopic config on each variant's embeddings and
    tabulate metrics side by side — one row per variant, seeds averaged.

    Parameters
    ----------
    documents : list[str]
    embedders_or_embeddings : dict[str, array or callable]
        Either precomputed ``(n_docs, dim)`` embeddings, or a callable
        ``documents -> embeddings`` (e.g. an ``EmbeddingAdapter.encode`` or
        ``EmbeddingEngine.encode`` bound method).
    labels_true : np.ndarray, optional
        Ground truth (20NG / synthetic corpora); enables ARI/NMI/cluster
        accuracy columns.
    base_config : TriTopicConfig, optional
        Reused, deep-copied, for every variant/seed — the comparison is only
        meaningful when every variant is clustered identically.
    holdout_bank : TripletBank, optional
        When given, adds a label-free ``holdout_triplet_acc`` column.
    n_seeds : int
        Leiden-consensus reseeds per variant, averaged (default 1).
    baseline : str, optional
        Variant name to diff keyword overlap against (default: first key).

    Returns
    -------
    pandas.DataFrame, one row per variant.
    """
    from tritopic.core.model import TriTopic, TriTopicConfig
    from tritopic.utils.metrics import compute_ari, compute_nmi, compute_silhouette

    base_config = base_config or TriTopicConfig(verbose=False)
    baseline_name = baseline or next(iter(embedders_or_embeddings))

    resolved: dict[str, np.ndarray] = {
        name: (np.asarray(val(documents)) if callable(val) else np.asarray(val))
        for name, val in embedders_or_embeddings.items()
    }

    avg_rows = []
    first_seed_models: dict[str, "TriTopic"] = {}

    for name, emb in resolved.items():
        seed_rows = []
        for s in range(max(n_seeds, 1)):
            cfg = copy.deepcopy(base_config)
            cfg.random_state = base_config.random_state + s
            cfg.verbose = False

            model = TriTopic(config=cfg)
            t0 = time.perf_counter()
            model.fit(documents, embeddings=emb)
            elapsed = time.perf_counter() - t0

            labels = model.labels_
            mask = labels != -1

            row: dict[str, float] = {
                "n_topics": len([t for t in model.topics_ if t.topic_id != -1]),
                "outlier_ratio": float(np.mean(labels == -1)),
                "silhouette": compute_silhouette(emb, labels),
                "fit_seconds": elapsed,
            }

            if mask.sum() >= 2 and len(np.unique(labels[mask])) >= 2:
                from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

                row["davies_bouldin"] = float(davies_bouldin_score(emb[mask], labels[mask]))
                row["calinski_harabasz"] = float(calinski_harabasz_score(emb[mask], labels[mask]))
            else:
                row["davies_bouldin"] = float("nan")
                row["calinski_harabasz"] = float("nan")

            ev = model.evaluate()
            row["stability"] = ev.get("stability") if ev.get("stability") is not None else float("nan")
            row["coherence_mean"] = ev.get("coherence_mean", float("nan"))
            row["diversity"] = ev.get("diversity", float("nan"))

            if labels_true is not None:
                row["ari"] = compute_ari(labels, labels_true)
                row["nmi"] = compute_nmi(labels, labels_true)
                row["cluster_accuracy"] = hungarian_accuracy(labels, labels_true)

            if holdout_bank is not None:
                row["holdout_triplet_acc"] = triplet_accuracy(emb, holdout_bank.holdout)

            seed_rows.append(row)
            if s == 0:
                first_seed_models[name] = model

        avg_row = {"variant": name}
        for key in seed_rows[0]:
            vals = [r[key] for r in seed_rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
            avg_row[key] = float(np.mean(vals)) if vals else float("nan")
        avg_rows.append(avg_row)

    df = pd.DataFrame(avg_rows)

    if baseline_name in first_seed_models:
        base_model = first_seed_models[baseline_name]
        overlaps = []
        for name in df["variant"]:
            if name == baseline_name:
                overlaps.append(1.0)
            elif name in first_seed_models:
                overlaps.append(_keyword_overlap_vs(first_seed_models[name], base_model))
            else:
                overlaps.append(float("nan"))
        df["keyword_overlap_vs_baseline"] = overlaps

    ordered = ["variant"] + [c for c in df.columns if c != "variant"]
    return df[ordered]
