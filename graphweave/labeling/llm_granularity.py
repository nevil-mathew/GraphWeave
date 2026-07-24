"""
LLM-Guided Granularity Calibration
===================================

Pick a Leiden resolution by asking an LLM to judge same-cluster-vs-different-
cluster triplets and scoring candidate resolutions by agreement with those
judgments — the triplet-query approach from ClusterLLM (Zhang, Wang & Shang,
EMNLP 2023, "ClusterLLM: Large Language Models as a Guide for Text Clustering").

Improvements over the baseline ClusterLLM formulation
------------------------------------------------------
* **Bias-free triplet presentation** — ~50 % of triplets are randomly swapped so
  the LLM sees the same-cluster doc as "C" rather than always "B".  Parse
  failures default to ``None`` (dropped, not counted as "B"), eliminating the
  systematic middle-resolution inflation.
* **Discriminative anchor selection** — oversamples a 4× candidate pool then
  keeps the triplets on which candidate partitions *disagree most*, because only
  those triplets can actually separate the candidates.
* **Two-stage grid refinement** — a coarse geomspace sweep picks the
  neighbourhood; a 5-point fine grid inside that neighbourhood finds the precise
  winner with zero extra LLM calls.
* **Multi-seed candidate scoring** — each candidate resolution is scored across
  ``n_seeds`` Leiden runs (averaged) to reduce single-seed noise.
* **Outlier parity** — candidate partitions apply the same small-cluster
  suppression as the final ``ConsensusLeiden.fit_predict`` so the scored
  partition matches the delivered one.
* **O(n log n) neighbour search** — one global k-NN index replaces per-anchor
  ``NearestNeighbors`` fits, cutting sampling cost from O(anchors × n_docs) to
  O(n_docs × log n_docs).
* **Diagnostics** — pass ``return_diagnostics=True`` to get a per-candidate
  score table useful for spotting a flat signal.

Public entry point: :func:`llm_select_resolution`.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Candidate resolution sweep
# ---------------------------------------------------------------------------

def _candidate_resolutions(
    resolution_range: tuple[float, float], n_candidates: int
) -> list[float]:
    """Geometrically spaced candidate resolutions.

    Matches the ``np.geomspace``-based sweep convention already used
    elsewhere in GraphWeave (e.g. ``GraphWeave.build_hierarchy``).
    """
    lo, hi = resolution_range
    lo = max(lo, 1e-4)  # geomspace requires a nonzero lower bound
    return list(np.geomspace(lo, hi, n_candidates))


# ---------------------------------------------------------------------------
# Small-cluster suppression (mirrors ConsensusLeiden._handle_small_clusters)
# ---------------------------------------------------------------------------

def _suppress_small_clusters(
    labels: np.ndarray,
    min_cluster_size: int,
    node_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Mark clusters smaller than *min_cluster_size* as outliers (−1).

    When *node_weights* is given, "small" is measured by summed weight
    (represented mass), not raw node count, matching
    ``ConsensusLeiden._handle_small_clusters``.
    """
    result = labels.copy()
    if node_weights is not None and len(node_weights) == len(result):
        for cid in np.unique(result):
            if cid != -1 and node_weights[result == cid].sum() < min_cluster_size:
                result[result == cid] = -1
    else:
        unique, counts = np.unique(result, return_counts=True)
        for cid, cnt in zip(unique, counts):
            if cid != -1 and cnt < min_cluster_size:
                result[result == cid] = -1
    return result


# ---------------------------------------------------------------------------
# Candidate partition generation
# ---------------------------------------------------------------------------

def _partition_at_resolution(
    graph,
    resolution: float,
    random_state: int,
    node_weights: np.ndarray | None = None,
    min_cluster_size: int | None = None,
) -> np.ndarray:
    """One single (non-consensus) Leiden run at *resolution* — cheap candidate
    scoring only. The final re-fit at the winning resolution is a full
    ``ConsensusLeiden.fit_predict`` consensus run performed by the caller.

    Mirrors ``ConsensusLeiden.fit_predict``'s choice of partition objective
    (``clustering.py``): when *node_weights* is given (weighted-coreset
    models), uses ``RBERVertexPartition`` with ``node_sizes`` so candidate
    scoring reflects the same objective as the final consensus re-fit;
    otherwise (the default, unweighted path) uses
    ``RBConfigurationVertexPartition`` exactly as before.

    When *min_cluster_size* is given (≥ 2), small clusters are suppressed to
    outliers using :func:`_suppress_small_clusters` so the scored partition
    matches the ``ConsensusLeiden.fit_predict`` result as closely as possible.
    """
    import leidenalg as la

    weights_arr = np.asarray(node_weights, dtype=float) if node_weights is not None else None
    use_node_sizes = weights_arr is not None and len(weights_arr) == graph.vcount()

    if use_node_sizes:
        partition = la.find_partition(
            graph,
            la.RBERVertexPartition,
            weights="weight",
            node_sizes=weights_arr.tolist(),
            resolution_parameter=resolution,
            seed=random_state,
        )
    else:
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=random_state,
        )

    labels = np.array(partition.membership)
    if min_cluster_size is not None and min_cluster_size >= 2:
        labels = _suppress_small_clusters(labels, min_cluster_size, node_weights=weights_arr)
    return labels


# ---------------------------------------------------------------------------
# Triplet sampling — canonical (for tests) and fast (for production)
# ---------------------------------------------------------------------------

def _sample_triplets(
    labels: np.ndarray,
    embeddings: np.ndarray,
    n_triplets: int,
    random_state: int = 42,
) -> list[tuple[int, int, int]]:
    """Sample (anchor, same_cluster_neighbor, diff_cluster_neighbor) index triplets.

    For each sampled anchor A: B = A's nearest neighbor (cosine) *within the
    same cluster* under *labels*; C = A's nearest neighbor *in a different
    cluster*. Deterministic given *random_state*.

    Outlier documents (``label == -1``) are excluded entirely from anchor,
    B, and C candidacy, since "outlier" isn't a coherent cluster identity
    for a same/different judgment.

    Returns an empty list if fewer than 2 non-outlier clusters exist.

    .. note::
        This canonical implementation is kept for unit tests and external
        callers.  The production path inside :func:`llm_select_resolution`
        uses :func:`_sample_triplets_fast` (single global k-NN query) then
        :func:`_sample_triplets_informed` (discriminative selection).
    """
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return []

    rng = np.random.default_rng(random_state)
    pool = np.where(mask)[0]

    if len(pool) > n_triplets:
        anchors = np.sort(rng.choice(pool, size=n_triplets, replace=False))
    else:
        anchors = pool

    triplets: list[tuple[int, int, int]] = []
    for a in anchors:
        same_idx = np.where((labels == labels[a]) & (np.arange(len(labels)) != a))[0]
        diff_idx = np.where((labels != labels[a]) & mask)[0]
        if len(same_idx) == 0 or len(diff_idx) == 0:
            continue

        anchor_vec = embeddings[a : a + 1]

        same_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[same_idx])
        _, same_pos = same_nn.kneighbors(anchor_vec)
        b = int(same_idx[same_pos[0, 0]])

        diff_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(embeddings[diff_idx])
        _, diff_pos = diff_nn.kneighbors(anchor_vec)
        c = int(diff_idx[diff_pos[0, 0]])

        triplets.append((int(a), b, c))

    return triplets


def _sample_triplets_fast(
    labels: np.ndarray,
    embeddings: np.ndarray,
    n_triplets: int,
    random_state: int = 42,
    k_neighbors: int = 64,
) -> list[tuple[int, int, int]]:
    """Fast triplet sampling via a single global k-NN query.

    Builds one ``NearestNeighbors`` index over all non-outlier documents,
    queries all anchors at once, then extracts the nearest same-cluster and
    nearest different-cluster neighbour per anchor from the result — reducing
    cost from O(anchors × n_docs) to O(n_docs × log n_docs).

    Falls back to a per-anchor exact search for anchors whose k-NN window
    does not contain both a same-cluster and a different-cluster neighbour
    (rare: only happens when a cluster is so large it fills the entire k-NN
    window, or so small it has no intra-cluster neighbour in the window).
    """
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return []

    rng = np.random.default_rng(random_state)
    pool_idx = np.where(mask)[0]

    if len(pool_idx) > n_triplets:
        anchors = np.sort(rng.choice(pool_idx, size=n_triplets, replace=False))
    else:
        anchors = pool_idx

    # L2-normalise so euclidean distance ∝ cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = (embeddings / norms).astype(np.float32)

    k = min(k_neighbors, len(pool_idx) - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(normed[pool_idx])
    _, local_indices = nn.kneighbors(normed[anchors])  # (len(anchors), k)

    triplets: list[tuple[int, int, int]] = []
    for i, a in enumerate(anchors):
        b: int | None = None
        c: int | None = None
        for local_ni in local_indices[i]:
            ni = int(pool_idx[local_ni])
            if ni == a:
                continue
            if b is None and labels[ni] == labels[a]:
                b = ni
            elif c is None and labels[ni] != labels[a]:
                c = ni
            if b is not None and c is not None:
                break

        if b is None or c is None:
            continue
        triplets.append((int(a), b, c))

    return triplets


def _sample_triplets_informed(
    partitions: list[np.ndarray],
    embeddings: np.ndarray,
    n_triplets: int,
    random_state: int = 42,
    oversample_factor: int = 4,
) -> list[tuple[int, int, int]]:
    """Sample triplets that discriminate between candidate partitions.

    Builds a 4× oversampled pool from the reference (middle) partition via
    :func:`_sample_triplets_fast`, then ranks candidates by *cross-partition
    disagreement*: how many candidate partitions disagree on whether A
    belongs with B or with C.

    A triplet on which all candidates give the same verdict cannot
    distinguish them regardless of the LLM's answer — those are discarded
    first.  Ties are broken with a tiny random jitter so selection is
    deterministic given *random_state* but not degenerate.

    Falls back to the full oversampled pool when it is already ≤ *n_triplets*.
    """
    reference_labels = partitions[len(partitions) // 2]
    pool_size = n_triplets * oversample_factor

    pool = _sample_triplets_fast(reference_labels, embeddings, pool_size, random_state)

    if len(pool) <= n_triplets:
        return pool

    rng = np.random.default_rng(random_state + 1)
    scores: list[float] = []
    for a, b, c in pool:
        # Count how many partitions place A with B (same cluster)
        n_same_b = sum(1 for lbl in partitions if lbl[a] == lbl[b])
        # Minority vote = disagreement strength (0 = unanimous; higher = more contentious)
        minority = min(n_same_b, len(partitions) - n_same_b)
        scores.append(float(minority) + rng.uniform() * 1e-3)

    order = sorted(range(len(pool)), key=lambda i: -scores[i])
    return [pool[i] for i in order[:n_triplets]]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_triplet_prompt(
    triplets: list[tuple[int, int, int]],
    documents: list[str],
    n_docs_chars: int = 300,
) -> tuple[str, str]:
    """Build the (system, user) prompts for one batch of triplet judgments.

    Each triplet is presented as anchor document A plus two candidates B
    and C; the LLM must say whether A belongs with B or with C.

    .. note::
        The caller is responsible for swap randomisation.  When a triplet has
        been swapped, the caller passes ``(a, diff, same)`` instead of
        ``(a, same, diff)`` so this function sees — and labels — them
        correctly as B and C without needing to know about the swap.
    """
    system_prompt = (
        "You are a text clustering expert. For each numbered item you will see "
        "three short documents: A (an anchor), and two candidates B and C. "
        "Decide whether document A is thematically/topically more similar to "
        "B or to C — that is, if you were grouping these documents into "
        "topics, would A belong in the same group as B, or the same group as "
        "C? Answer for every item, in order. "
        "You always respond with valid JSON and nothing else."
    )

    def _snippet(doc: str) -> str:
        return doc[:n_docs_chars] + "..." if len(doc) > n_docs_chars else doc

    lines = []
    for i, (a, b, c) in enumerate(triplets, 1):
        lines.append(
            f"Item {i}:\n"
            f"  A: {_snippet(documents[a])}\n"
            f"  B: {_snippet(documents[b])}\n"
            f"  C: {_snippet(documents[c])}"
        )
    items_block = "\n\n".join(lines)

    user_prompt = f"""Below are {len(triplets)} items, each with an anchor document A and two candidate documents B and C.

{items_block}

For each item, decide: is A more similar in topic to B, or to C?

Respond ONLY with JSON in this exact format, no other text:
{{"answers": ["B", "C", "B", ...]}}

The "answers" array must have exactly {len(triplets)} entries, one per item in order, each either "B" or "C"."""

    return system_prompt, user_prompt


# JSON Schema for structured output (Google response_schema / OpenAI json_object)
_GRANULARITY_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {"type": "string", "enum": ["B", "C"]},
        }
    },
    "required": ["answers"],
}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_triplet_response(raw: str, n_expected: int) -> list[str | None]:
    """Parse a batch response into a list of "B"/"C"/None answers, length *n_expected*.

    3-tier robust parse:
    1. Locate outermost ``{...}`` object, ``json.loads`` it, read ``["answers"]``.
    2. Regex-extract individual quoted "B"/"C" tokens in document order.
    3. Fallback: mark every unresolved slot as ``None`` (dropped from scoring,
       not biased toward any candidate) and warn.

    Always returns exactly *n_expected* entries.  Short results are padded with
    ``None``; long ones are truncated.  Invalid enum values (not "B"/"C") within
    an otherwise parseable response are also normalised to ``None`` — they are
    equally uninformative.
    """
    answers: list[str | None] | None = None

    # Tier 1: outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end])
            if isinstance(data, dict) and isinstance(data.get("answers"), list):
                answers = [
                    str(x).strip().upper() if str(x).strip().upper() in ("B", "C") else None
                    for x in data["answers"]
                ]
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 2: regex-extract quoted B/C tokens in order
    if answers is None:
        tokens = re.findall(r'"\s*([BC])\s*"', raw, flags=re.IGNORECASE)
        if tokens:
            answers = [t.upper() for t in tokens]

    # Tier 3: nothing usable at all
    if not answers:
        warnings.warn(
            "llm_select_resolution: could not parse LLM triplet response — "
            "dropping all answers in this batch from scoring.",
            UserWarning,
            stacklevel=4,
        )
        return [None] * n_expected

    if len(answers) < n_expected:
        warnings.warn(
            f"llm_select_resolution: LLM returned {len(answers)} answers for "
            f"{n_expected} items — dropping missing entries from scoring.",
            UserWarning,
            stacklevel=4,
        )
        answers = answers + [None] * (n_expected - len(answers))
    elif len(answers) > n_expected:
        answers = answers[:n_expected]

    return answers


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _triplet_agreement(
    labels: np.ndarray,
    triplets: list[tuple[int, int, int]],
    llm_answers: list[str | None],
) -> float:
    """Fraction of triplets where the partition's same/different verdict for
    (A,B) vs (A,C) agrees with the LLM's stated preference.

    For triplet (a, b, c) and partition *labels*:
        same_b = (labels[a] == labels[b])
        same_c = (labels[a] == labels[c])
    A triplet only contributes to the score if:
    - The answer is not ``None`` (unparsed responses are dropped, not biased).
    - Exactly one of ``{same_b, same_c}`` is True — a clean, unambiguous verdict
      from this partition. If both are True (a, b, c all merged into one cluster)
      or both are False (a is separated from both b and c), the partition gives
      no informative signal and the triplet is skipped.

    A candidate with zero informative triplets is genuinely degenerate and
    scores 0.0. Otherwise the score is Laplace/add-one smoothed —
    ``(agreements + 1) / (informative-count + 2)`` — so a candidate
    informative for only one or two triplets can't outrank one informative
    across many triplets at a slightly lower (but statistically much more
    reliable) agreement rate.
    """
    agree = 0
    counted = 0
    for (a, b, c), ans in zip(triplets, llm_answers):
        if ans is None:
            continue  # unparsed — drop rather than bias toward any candidate
        same_b = labels[a] == labels[b]
        same_c = labels[a] == labels[c]
        if same_b == same_c:
            continue  # uninformative: both merged or both separated
        counted += 1
        implied = "B" if same_b else "C"
        if implied == ans:
            agree += 1
    if counted == 0:
        return 0.0
    return (agree + 1) / (counted + 2)


# ---------------------------------------------------------------------------
# Default triplet budget
# ---------------------------------------------------------------------------

def _default_n_triplets(n_docs: int) -> int:
    """Corpus-size-scaled default triplet budget.

    ~24 triplets for small corpora, growing logarithmically to ~120 for
    50K+ document corpora, since sampling coverage of boundary cases
    matters more at scale while LLM cost stays trivial either way.
    """
    if n_docs <= 100:
        return 24
    scaled = 24 * (1 + np.log10(n_docs / 100))
    return int(np.clip(round(scaled), 24, 120))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _n_clusters(partition: np.ndarray) -> int:
    """Number of distinct non-outlier clusters in *partition*."""
    return int(len(np.unique(partition[partition != -1])))


def _score_seeds(
    seed_partitions: list[np.ndarray],
    triplets: list[tuple[int, int, int]],
    canonical_answers: list[str | None],
) -> float:
    """Average triplet-agreement score across multiple seed partitions."""
    return float(np.mean([
        _triplet_agreement(lbl, triplets, canonical_answers)
        for lbl in seed_partitions
    ]))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def llm_select_resolution(
    labeler,
    documents: list[str],
    graph,
    embeddings: np.ndarray,
    resolution_range: tuple[float, float] = (0.1, 2.0),
    n_candidates: int = 6,
    n_triplets: int | None = None,
    random_state: int = 42,
    batch_size: int = 8,
    node_weights: np.ndarray | None = None,
    min_cluster_size: int | None = None,
    n_seeds: int = 3,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict]:
    """Select a Leiden resolution via LLM-judged triplet agreement (ClusterLLM).

    Algorithm
    ---------
    **Stage A — coarse sweep:**

    1. Generate *n_candidates* candidate resolutions (geometrically spaced
       across *resolution_range*).
    2. Run *n_seeds* single-pass (non-consensus) Leiden partitions per
       candidate (averaged for noise reduction); apply small-cluster
       suppression when *min_cluster_size* is given.
    3. Use the middle candidate's first-seed partition as the reference for
       triplet sampling.
    4. Build a 4× oversampled triplet pool via one global k-NN query then
       rank by cross-candidate disagreement, keeping *n_triplets* (or the
       corpus-scaled default) most-discriminative triplets.
    5. Randomly swap ~50 % of triplet presentations (diff-cluster as "B")
       to neutralise LLM position bias; unswap answers before scoring.
    6. Ask the LLM to judge each triplet in batches of *batch_size*.
    7. Score every coarse candidate by average agreement across seeds.

    **Stage B — local fine grid:**

    8. Around the coarse winner, generate a 5-point geomspace fine grid
       between its neighbours on the coarse grid.
    9. Score fine candidates against the *already-collected* LLM answers —
       zero extra API calls.
    10. Return the resolution with the highest score across both stages.

    Parameters
    ----------
    labeler : LLMLabeler (duck-typed)
        Must expose ``call_structured(system_prompt, user_prompt, schema,
        max_tokens=None) -> str``.
    documents : list[str]
        Full corpus of fitted document texts.
    graph : igraph.Graph
        Pre-built multi-view graph, reused as-is for candidate scoring.
    embeddings : np.ndarray, shape (n_docs, dim)
        Document embeddings for triplet neighbour search.
    resolution_range : tuple[float, float]
        Sweep range for candidate resolutions.
    n_candidates : int
        Number of coarse candidate resolutions to score. Must be >= 1.
    n_triplets : int, optional
        Number of triplets to sample and send to the LLM. ``None`` uses
        the corpus-size-scaled default (see :func:`_default_n_triplets`).
    random_state : int
        Seed for triplet sampling, swap randomisation, and Leiden runs.
    batch_size : int
        Triplets per LLM call. Must be >= 1.
    node_weights : np.ndarray, optional
        Per-node representation weight (e.g. how many real documents a
        coreset point stands for), aligned row-wise with *graph*'s
        vertices. When given, candidate partitions use the same
        ``RBERVertexPartition`` + ``node_sizes`` objective as
        ``ConsensusLeiden.fit_predict`` would use for the final re-fit.
    min_cluster_size : int, optional
        When given (≥ 2), candidate partitions suppress small clusters to
        outliers, matching ``ConsensusLeiden.fit_predict``'s post-partition
        cleanup.  Pass ``GraphWeave._effective_min_cluster_size(n_docs)``
        to keep candidate and final-fit partitions consistent.
    n_seeds : int
        Number of single-pass Leiden runs per candidate resolution; their
        agreement scores are averaged to reduce seed-noise. Default 3.
    return_diagnostics : bool
        When ``True``, return ``(best_resolution, diagnostics)`` where
        *diagnostics* is a dict with keys ``n_triplets``, ``n_unparsed``,
        ``reference_resolution``, ``best_resolution``, and ``candidates``
        (list of per-candidate dicts with ``resolution``, ``score``,
        ``n_clusters``, ``stage``).  When ``False`` (default), return
        only the ``float`` best resolution.

    Returns
    -------
    float or (float, dict)
        The winning resolution.  When *return_diagnostics* is ``True``,
        also returns the diagnostics dict.
    """
    if n_candidates <= 0:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # ── Stage A: coarse geomspace sweep ──────────────────────────────────────
    candidates = _candidate_resolutions(resolution_range, n_candidates)
    mid = len(candidates) // 2

    # All seed partitions: shape [n_candidates][n_seeds]
    all_seed_partitions: list[list[np.ndarray]] = [
        [
            _partition_at_resolution(
                graph, res, random_state + seed,
                node_weights=node_weights,
                min_cluster_size=min_cluster_size,
            )
            for seed in range(max(n_seeds, 1))
        ]
        for res in candidates
    ]

    reference_labels = all_seed_partitions[mid][0]

    n_triplets_eff = (
        n_triplets if n_triplets is not None else _default_n_triplets(len(documents))
    )

    # ── Discriminative triplet sampling ──────────────────────────────────────
    flat_partitions = [sp[0] for sp in all_seed_partitions]
    triplets = _sample_triplets_informed(
        flat_partitions, embeddings, n_triplets_eff, random_state
    )

    def _fallback(reason: str, extra: str = "") -> float | tuple[float, dict]:
        mid_res = float(candidates[mid])
        msg = (
            f"llm_select_resolution: {reason} — "
            f"falling back to the middle candidate resolution ({mid_res:.3f})"
            + (f"; {extra}" if extra else "") + "."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        if return_diagnostics:
            diag: dict = {
                "n_triplets": 0,
                "n_unparsed": 0,
                "reference_resolution": float(candidates[mid]),
                "best_resolution": mid_res,
                "candidates": [
                    {"resolution": r, "score": 0.0, "n_clusters": _n_clusters(sp[0]), "stage": "A"}
                    for r, sp in zip(candidates, all_seed_partitions)
                ],
            }
            return mid_res, diag
        return mid_res

    if not triplets:
        return _fallback(
            "no valid triplets could be sampled "
            "(reference partition has fewer than 2 non-outlier clusters)"
        )

    # ── B-position bias mitigation ────────────────────────────────────────────
    # Randomly swap ~50 % of triplets so the LLM sometimes sees diff-cluster
    # as "B" and same-cluster as "C", eliminating position bias toward "B".
    rng_swap = np.random.default_rng(random_state + 999)
    swaps: np.ndarray = rng_swap.random(len(triplets)) < 0.5

    presented_triplets: list[tuple[int, int, int]] = [
        (a, c, b) if bool(swaps[i]) else (a, b, c)
        for i, (a, b, c) in enumerate(triplets)
    ]

    # ── LLM queries ───────────────────────────────────────────────────────────
    raw_answers: list[str | None] = []
    for start in range(0, len(presented_triplets), batch_size):
        batch = presented_triplets[start : start + batch_size]
        system_prompt, user_prompt = _build_triplet_prompt(batch, documents)
        max_tokens = max(128, len(batch) * 20 + 64)
        raw = labeler.call_structured(
            system_prompt, user_prompt, schema=_GRANULARITY_SCHEMA, max_tokens=max_tokens
        )
        raw_answers.extend(_parse_triplet_response(raw, len(batch)))

    # Un-swap: translate presented B/C back to canonical (same=B, diff=C) space
    canonical_answers: list[str | None] = []
    for ans, swapped in zip(raw_answers, swaps):
        if ans is None:
            canonical_answers.append(None)
        elif bool(swapped):
            # Swap was: B←diff, C←same → reverse the mapping
            canonical_answers.append("C" if ans == "B" else "B")
        else:
            canonical_answers.append(ans)

    n_unparsed = sum(1 for a in canonical_answers if a is None)

    # ── Score Stage-A candidates (multi-seed average) ─────────────────────────
    scores_a = [
        _score_seeds(sp, triplets, canonical_answers)
        for sp in all_seed_partitions
    ]

    if max(scores_a) == 0.0:
        return _fallback(
            "no candidate resolution showed a clear preference "
            "(all triplet judgments were uninformative)"
        )

    # ── Stage B: local fine grid around coarse winner ─────────────────────────
    # Coarse winner (tie-break: closest index to mid)
    max_a = max(scores_a)
    tied_a = [i for i, s in enumerate(scores_a) if s == max_a]
    best_a_idx = min(tied_a, key=lambda i: abs(i - mid))

    lo_b = candidates[max(0, best_a_idx - 1)]
    hi_b = candidates[min(len(candidates) - 1, best_a_idx + 1)]

    fine_candidates: list[float] = []
    fine_seed_partitions: list[list[np.ndarray]] = []
    scores_b: list[float] = []

    if hi_b > lo_b:
        n_fine = 5
        fine_grid = list(np.geomspace(lo_b, hi_b, n_fine))
        # Skip candidates already in the coarse sweep to avoid redundant Leiden runs
        fine_unique = [r for r in fine_grid if not any(abs(r - c) < 1e-9 for c in candidates)]
        for res in fine_unique:
            sp = [
                _partition_at_resolution(
                    graph, res, random_state + seed,
                    node_weights=node_weights,
                    min_cluster_size=min_cluster_size,
                )
                for seed in range(max(n_seeds, 1))
            ]
            fine_seed_partitions.append(sp)
            fine_candidates.append(res)
            scores_b.append(_score_seeds(sp, triplets, canonical_answers))

    # ── Select overall winner ─────────────────────────────────────────────────
    all_cands = candidates + fine_candidates
    all_scores = scores_a + scores_b
    all_stages = ["A"] * len(candidates) + ["B"] * len(fine_candidates)
    all_sp = all_seed_partitions + fine_seed_partitions

    max_overall = max(all_scores)
    tied_all = [i for i, s in enumerate(all_scores) if s == max_overall]

    # Prefer Stage A in ties (more Leiden runs sampled the same range → more
    # representative), then pick closest to the middle coarse candidate value.
    mid_res_val = float(candidates[mid])
    a_tied = [i for i in tied_all if all_stages[i] == "A"]
    best_pool = a_tied if a_tied else tied_all
    best_idx = min(best_pool, key=lambda i: abs(all_cands[i] - mid_res_val))
    best_res = float(all_cands[best_idx])

    if not return_diagnostics:
        return best_res

    diag = {
        "n_triplets": len(triplets),
        "n_unparsed": n_unparsed,
        "reference_resolution": float(candidates[mid]),
        "best_resolution": best_res,
        "candidates": [
            {
                "resolution": round(r, 6),
                "score": round(s, 4),
                "n_clusters": _n_clusters(sp[0]),
                "stage": stage,
            }
            for r, s, sp, stage in zip(all_cands, all_scores, all_sp, all_stages)
        ],
    }
    return best_res, diag
