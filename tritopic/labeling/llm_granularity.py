"""
LLM-Guided Granularity Calibration
===================================

Pick a Leiden resolution by asking an LLM to judge same-cluster-vs-different-
cluster triplets and scoring candidate resolutions by agreement with those
judgments — the triplet-query approach from ClusterLLM (Zhang, Wang & Shang,
EMNLP 2023, "ClusterLLM: Large Language Models as a Guide for Text Clustering").

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
    elsewhere in TriTopic (e.g. ``TriTopic.build_hierarchy``).
    """
    lo, hi = resolution_range
    lo = max(lo, 1e-4)  # geomspace requires a nonzero lower bound
    return list(np.geomspace(lo, hi, n_candidates))


def _partition_at_resolution(
    graph,
    resolution: float,
    random_state: int,
    node_weights: np.ndarray | None = None,
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
    return np.array(partition.membership)


# ---------------------------------------------------------------------------
# Triplet sampling
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
    cluster*. Deterministic given *random_state*: which anchors are chosen
    (when ``n_triplets < n_docs``) is driven by ``np.random.default_rng``.

    Outlier documents (``label == -1``) are excluded entirely from anchor,
    B, and C candidacy, since "outlier" isn't a coherent cluster identity
    for a same/different judgment.

    Returns an empty list if fewer than 2 non-outlier clusters exist.
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

def _parse_triplet_response(raw: str, n_expected: int) -> list[str]:
    """Parse a batch response into a list of "B"/"C" answers, length *n_expected*.

    3-tier robust parse:
    1. Locate outermost ``{...}`` object, ``json.loads`` it, read ``["answers"]``.
    2. Regex-extract individual quoted "B"/"C" tokens in document order.
    3. Fallback: default every unresolved slot to "B" (a conservative,
       no-opinion default) and warn.

    Always returns exactly *n_expected* entries: pads short results with
    "B" (with a warning) and truncates long ones.
    """
    answers: list[str] | None = None

    # Tier 1: outermost JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start:end])
            if isinstance(data, dict) and isinstance(data.get("answers"), list):
                answers = [str(x).strip().upper() for x in data["answers"]]
        except (json.JSONDecodeError, ValueError):
            pass

    # Tier 2: regex-extract quoted B/C tokens in order
    if answers is None:
        tokens = re.findall(r'"\s*([BC])\s*"', raw, flags=re.IGNORECASE)
        if tokens:
            answers = [t.upper() for t in tokens]

    # Normalize any stray values to a valid enum member
    if answers is not None:
        answers = [a if a in ("B", "C") else "B" for a in answers]

    # Tier 3: nothing usable at all
    if not answers:
        warnings.warn(
            "llm_select_resolution: could not parse LLM triplet response — "
            "defaulting all answers in this batch to 'B'.",
            UserWarning,
            stacklevel=4,
        )
        return ["B"] * n_expected

    if len(answers) < n_expected:
        warnings.warn(
            f"llm_select_resolution: LLM returned {len(answers)} answers for "
            f"{n_expected} items — padding missing entries with 'B'.",
            UserWarning,
            stacklevel=4,
        )
        answers = answers + ["B"] * (n_expected - len(answers))
    elif len(answers) > n_expected:
        answers = answers[:n_expected]

    return answers


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _triplet_agreement(
    labels: np.ndarray,
    triplets: list[tuple[int, int, int]],
    llm_answers: list[str],
) -> float:
    """Fraction of triplets where the partition's same/different verdict for
    (A,B) vs (A,C) agrees with the LLM's stated preference.

    For triplet (a, b, c) and partition *labels*:
        same_b = (labels[a] == labels[b])
        same_c = (labels[a] == labels[c])
    A triplet only contributes to the score if exactly one of
    ``{same_b, same_c}`` is True — a clean, unambiguous verdict from this
    partition. If both are True (a, b, c all merged into one cluster) or
    both are False (a is separated from both b and c), the partition gives
    no informative signal about the B-vs-C preference, so the triplet is
    skipped for this candidate rather than counted as a disagreement.

    A candidate with zero informative triplets is genuinely degenerate
    relative to the sampled triplets and scores 0.0. Otherwise the score is
    Laplace/add-one smoothed — ``(agreements + 1) / (informative-count + 2)``
    — rather than a raw proportion, so a candidate informative for only one
    or two triplets can't outrank one informative across many triplets at a
    slightly lower (but statistically much more reliable) agreement rate.
    """
    agree = 0
    counted = 0
    for (a, b, c), ans in zip(triplets, llm_answers):
        same_b = labels[a] == labels[b]
        same_c = labels[a] == labels[c]
        if same_b == same_c:
            continue  # both merged or both separated: uninformative, skip
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
) -> float:
    """Select a Leiden resolution via LLM-judged triplet agreement (ClusterLLM).

    Algorithm
    ---------
    1. Generate *n_candidates* candidate resolutions (geometrically spaced
       across *resolution_range*).
    2. Run one single-pass (non-consensus) Leiden partition per candidate —
       cheap, since only the final winner gets a full consensus re-fit later.
    3. Use the middle candidate's partition (by sorted resolution) as the
       reference partition for triplet sampling.
    4. Sample *n_triplets* (or the corpus-scaled default) triplets from that
       reference partition.
    5. Ask the LLM to judge each triplet in batches of *batch_size*.
    6. Score every candidate's partition against all LLM judgments.
    7. Return the resolution with the highest agreement score.

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
        Document embeddings for triplet neighbor search.
    resolution_range : tuple[float, float]
        Sweep range for candidate resolutions.
    n_candidates : int
        Number of candidate resolutions to score. Must be >= 1.
    n_triplets : int, optional
        Number of triplets to sample and send to the LLM. ``None`` uses
        the corpus-size-scaled default (see :func:`_default_n_triplets`).
    random_state : int
        Seed for triplet sampling and single-pass Leiden runs.
    batch_size : int
        Triplets per LLM call. Must be >= 1.
    node_weights : np.ndarray, optional
        Per-node representation weight (e.g. how many real documents a
        coreset point stands for), aligned row-wise with *graph*'s
        vertices. When given, candidate partitions use the same
        ``RBERVertexPartition`` + ``node_sizes`` objective as
        ``ConsensusLeiden.fit_predict`` would use for the final re-fit, so
        candidate scoring matches the weighted Leiden objective. ``None``
        (default) reproduces the unweighted behaviour exactly.

    Returns
    -------
    float
        The winning resolution (one of the exact candidate values
        generated by :func:`_candidate_resolutions`).
    """
    if n_candidates <= 0:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    candidates = _candidate_resolutions(resolution_range, n_candidates)
    partitions = [
        _partition_at_resolution(graph, res, random_state, node_weights=node_weights)
        for res in candidates
    ]

    reference_labels = partitions[len(partitions) // 2]

    n_triplets_eff = n_triplets if n_triplets is not None else _default_n_triplets(len(documents))

    triplets = _sample_triplets(reference_labels, embeddings, n_triplets_eff, random_state)

    if not triplets:
        warnings.warn(
            "llm_select_resolution: no valid triplets could be sampled "
            "(reference partition has fewer than 2 non-outlier clusters) — "
            "falling back to the middle candidate resolution without "
            "querying the LLM.",
            UserWarning,
            stacklevel=2,
        )
        return float(candidates[len(candidates) // 2])

    llm_answers: list[str] = []
    for start in range(0, len(triplets), batch_size):
        batch = triplets[start : start + batch_size]
        system_prompt, user_prompt = _build_triplet_prompt(batch, documents)
        max_tokens = max(128, len(batch) * 20 + 64)
        raw = labeler.call_structured(
            system_prompt, user_prompt, schema=_GRANULARITY_SCHEMA, max_tokens=max_tokens
        )
        llm_answers.extend(_parse_triplet_response(raw, len(batch)))

    scores = [_triplet_agreement(labels, triplets, llm_answers) for labels in partitions]
    max_score = max(scores)
    mid = len(candidates) // 2

    if max_score == 0.0:
        # No candidate showed any informative signal at all — an explicit,
        # documented fallback rather than np.argmax's implicit first-index
        # tie-break, matching the no-triplets-sampled fallback above.
        warnings.warn(
            "llm_select_resolution: no candidate resolution showed a clear "
            "preference (all triplet judgments were uninformative) — "
            "falling back to the middle candidate resolution.",
            UserWarning,
            stacklevel=2,
        )
        return float(candidates[mid])

    tied = [i for i, s in enumerate(scores) if s == max_score]
    best_idx = min(tied, key=lambda i: abs(i - mid))
    return float(candidates[best_idx])
