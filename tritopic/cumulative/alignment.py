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
    from tritopic.utils.gpu import gpu_cosine_similarity

    sim = gpu_cosine_similarity(new_centroids, registry_centroids)  # (k_new, k_reg)
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


def _build_align_prompt(
    new_summaries: list[dict],
    registry_summaries: list[dict],
    decide_rows: set[int],
) -> tuple[str, str]:
    """Build the (system, user) prompts asking the LLM to align new topics to globals.

    Mirrors the proposer style in :meth:`TriTopic._propose_meta_themes`.
    """
    existing_lines = []
    for r in registry_summaries:
        kws = ", ".join(r.get("keywords", [])[:6])
        existing_lines.append(f"  [global_id {int(r['global_id'])}] {r.get('label', '')}\n    keywords: {kws}")
    existing_block = "\n".join(existing_lines)

    new_lines = []
    for row in sorted(decide_rows):
        s = new_summaries[row]
        kws = ", ".join(s.get("keywords", [])[:6])
        new_lines.append(
            f"  [row {row}] {s.get('label', '')} (n={s.get('size', 0)})\n    keywords: {kws}"
        )
    new_block = "\n".join(new_lines)

    system_prompt = (
        "You are a topic-alignment analyst for a longitudinal topic model. Each time a "
        "corpus is re-clustered the new topics are arbitrary; you decide which NEW topics "
        "are the SAME underlying theme as an EXISTING tracked global topic, so every theme "
        "keeps a stable identity over time. Judge by subject and meaning, not exact wording "
        "(vocabulary drifts). You always respond with valid JSON and nothing else."
    )

    user_prompt = f"""You are aligning freshly clustered topics to a registry of tracked themes.

EXISTING GLOBAL TOPICS:
{existing_block}

NEW TOPICS:
{new_block}

RULES:
- For each NEW topic decide whether it is the SAME theme as exactly one EXISTING global topic and give that global_id, OR mark it as a brand-new theme.
- Several NEW topics MAY map to the SAME global_id (a theme that split or overlaps) — this is allowed and encouraged when they share meaning.
- Only use global_id values listed under EXISTING GLOBAL TOPICS. If a NEW topic matches none of them, set "new_theme": true.
- Match on the deeper subject/concern, not surface keyword overlap.

OUTPUT FORMAT (JSON, no other text):
{{
  "assignments": [
    {{"new_local_idx": 0, "global_id": 7}},
    {{"new_local_idx": 2, "new_theme": true}}
  ]
}}"""
    return system_prompt, user_prompt


def _parse_alignment_response(
    raw: str, valid_new_rows: set[int], valid_global_ids: set[int]
) -> dict[int, int | None]:
    """Parse the aligner JSON into ``{new_row_idx -> global_id | None}``.

    ``None`` means "new theme". Robust 3-tier parse modeled on
    :meth:`TriTopic._parse_proposer_response`: ``json.loads`` on the ``{...}``
    substring, then a regex fallback. Invalid rows are dropped; an invalid /
    hallucinated ``global_id`` is treated as a new theme; the first valid mention
    of a row wins.
    """
    import json
    import re

    decisions: dict[int, int | None] = {}

    def _record(row: int, gid: int | None, is_new: bool) -> None:
        if row not in valid_new_rows or row in decisions:
            return
        if is_new or gid is None or gid not in valid_global_ids:
            decisions[row] = None
        else:
            decisions[row] = gid

    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        data = json.loads(raw[start:end])
        for a in data.get("assignments", []) or []:
            if "new_local_idx" not in a:
                continue
            row = int(a["new_local_idx"])
            is_new = bool(a.get("new_theme", False))
            gid = a.get("global_id")
            _record(row, int(gid) if gid is not None else None, is_new)
    except (json.JSONDecodeError, ValueError, TypeError):
        for m in re.finditer(
            r'"new_local_idx"\s*:\s*(\d+)\s*,\s*(?:"global_id"\s*:\s*(\d+)|"new_theme"\s*:\s*true)',
            raw,
        ):
            row = int(m.group(1))
            gid = int(m.group(2)) if m.group(2) is not None else None
            _record(row, gid, gid is None)

    return decisions


def llm_align_topics(
    new_summaries: list[dict],
    registry_summaries: list[dict],
    labeler,
    next_id: int,
    *,
    restrict_rows: set[int] | None = None,
    seed_mapping: dict[int, int] | None = None,
    new_centroids: np.ndarray | None = None,
    registry_centroids: np.ndarray | None = None,
    registry_ids: list[int] | None = None,
    threshold: float = 0.6,
) -> tuple[dict[int, int], int, list[tuple[int, int, float]]]:
    """LLM counterpart to :func:`align_topics` — same ``(mapping, next_id, matches)`` contract.

    Asks an LLM which freshly clustered topics correspond to existing global
    topics (by label + keywords), allowing many-to-one merges and brand-new
    themes. ``new_summaries`` is row-aligned with the new centroids; each entry is
    ``{"label", "keywords", "size"}``. ``registry_summaries`` carries the existing
    global topics as ``{"global_id", "label", "keywords"}``.

    Parameters mirror the cosine path plus:

    restrict_rows
        Only let the LLM (re)decide these new rows (``both`` mode); others keep
        their *seed_mapping* value. ``None`` = decide every row.
    seed_mapping
        Pre-existing ``row -> global_id`` decisions to start from (cosine result
        in ``both`` mode).

    On **any** LLM/parse failure this falls back to the cosine result so a
    recluster never crashes: the completed *seed_mapping* in ``both`` mode, or
    :func:`align_topics` when no seed is available. The returned mapping is always
    a total function over ``range(len(new_summaries))``.
    """
    import warnings

    k_new = len(new_summaries)
    mapping: dict[int, int] = dict(seed_mapping or {})

    def _complete(m: dict[int, int], nid: int) -> tuple[dict[int, int], int]:
        for i in range(k_new):
            if i not in m:
                m[i] = nid
                nid += 1
        return m, nid

    # First epoch / empty registry: sequential fresh IDs, no LLM call.
    if not registry_summaries:
        m, nid = _complete(mapping, next_id)
        return m, nid, []

    valid_global_ids = {int(r["global_id"]) for r in registry_summaries}
    reg_pos = {int(r["global_id"]): p for p, r in enumerate(registry_summaries)}
    decide_rows = (
        set(range(k_new))
        if restrict_rows is None
        else {int(r) for r in restrict_rows if 0 <= int(r) < k_new}
    )

    try:
        if decide_rows:
            system_prompt, user_prompt = _build_align_prompt(
                new_summaries, registry_summaries, decide_rows
            )
            raw = labeler.call_raw(system_prompt, user_prompt, max_tokens=4000)
            decisions = _parse_alignment_response(raw, decide_rows, valid_global_ids)
        else:
            decisions = {}
    except Exception as exc:  # network, parse, missing labeler -> cosine fallback
        warnings.warn(
            f"LLM topic alignment failed ({exc}); falling back to cosine alignment.",
            stacklevel=2,
        )
        if seed_mapping is not None:
            m, nid = _complete(dict(seed_mapping), next_id)
            return m, nid, []
        return align_topics(new_centroids, registry_centroids, registry_ids, threshold, next_id)

    matches: list[tuple[int, int, float]] = []
    for row in sorted(decide_rows):
        if row in decisions:
            gid = decisions[row]
            if gid is not None:
                mapping[row] = gid
                matches.append((row, reg_pos[gid], 1.0))
            else:  # explicit new theme
                mapping[row] = next_id
                next_id += 1
        elif row not in mapping:  # LLM did not mention it and no seed -> fresh ID
            mapping[row] = next_id
            next_id += 1
        # else: row omitted but seeded (both mode) -> keep cosine's guess

    mapping, next_id = _complete(mapping, next_id)
    return mapping, next_id, matches


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
    from tritopic.utils.gpu import gpu_cosine_similarity

    if registry_centroids is None or len(registry_centroids) == 0:
        return np.full(len(embeddings), -1, dtype=int)

    ids = np.asarray(registry_ids)
    sim = gpu_cosine_similarity(embeddings, registry_centroids)
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


def sensitivity_weights(embeddings: np.ndarray) -> np.ndarray:
    """Lightweight-coreset importance distribution (Bachem, Lucic & Krause, KDD'18).

    ``q(x) = ½·(1/n) + ½·‖x-μ‖² / Σ_j‖x_j-μ‖²`` with ``μ`` the mean embedding.
    Points far from the mean — the ones that define cluster boundaries and carry
    rare structure — get sampled more, while the uniform ½ term keeps every point
    reachable. Sampling proportional to ``q`` with the matching unbiased coreset
    weight ``1/(m·q(x))`` yields multiplicative ``(1±ε)`` k-means-style error
    bounds, versus none for uniform sampling.

    Returns a probability vector (sums to 1) over the rows of *embeddings*.
    """
    n = len(embeddings)
    if n == 0:
        return np.ones(0, dtype=float)
    mu = embeddings.mean(axis=0)
    diff = embeddings - mu
    d2 = np.einsum("ij,ij->i", diff, diff)
    total = float(d2.sum())
    if total <= 0:  # all points identical -> uniform
        return np.full(n, 1.0 / n, dtype=float)
    return 0.5 / n + 0.5 * d2 / total


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


def stratified_coreset(
    embeddings: np.ndarray,
    size: int,
    labels: np.ndarray,
    new_count: int,
    random_state: int = 42,
    floor: float = 0.25,
    min_per_cluster: int = 50,
    sampling: str = "recency",
) -> tuple[np.ndarray, np.ndarray]:
    """Coreset that guarantees each current topic a floor of representatives.

    Plain :func:`select_coreset` samples each topic ~its population share, so a
    rare topic (e.g. 0.5% of the corpus) can land below Leiden's
    ``min_cluster_size`` and vanish on the next refit (tail-collapse). This
    allocates a per-topic floor of ``min(min_per_cluster, topic_size)`` first,
    then distributes the remaining budget proportional to topic size, and samples
    *within* each stratum (reusing :func:`recency_weights`).

    ``sampling`` chooses the within-stratum distribution:

    - ``"recency"`` (default) — sample by recency weight; every selected point in
      a stratum gets the same inclusion probability ``m_c / N_c`` and so stands
      for ``N_c / m_c`` docs.
    - ``"sensitivity"`` — sample by the lightweight-coreset distribution
      (:func:`sensitivity_weights`) modulated by recency; the inclusion
      probability is per-point ``≈ m_c · p_i`` and the caller's ``1 / p_i`` is
      the unbiased lightweight-coreset weight. Adds proven k-means error bounds
      over the uniform-within-stratum default.

    *labels* is the current global label per accumulated doc (``-1`` = outlier),
    one per row of *embeddings*. Returns sorted ``indices`` plus the per-selected
    inclusion probability ``p_i``; the caller turns that into an inverse-propensity
    representation weight ``1 / p_i``. Floors are honoured when *size* is large
    enough to admit them; otherwise allocations are scaled down to fit the budget.
    Falls back to recency sampling when no labels.

    Returns
    -------
    indices : np.ndarray   # sorted row indices, len <= size
    probs   : np.ndarray   # inclusion probability per selected row (aligned)
    """
    n = len(embeddings)
    if labels is None or len(labels) != n:
        w = recency_weights(n, new_count, floor)
        if sampling == "sensitivity":
            w = w * sensitivity_weights(embeddings)
        idx = select_coreset(embeddings, size, random_state, w)
        return idx, np.full(len(idx), min(1.0, len(idx) / max(n, 1)))
    if size >= n:
        return np.arange(n), np.ones(n)

    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)
    rw = recency_weights(n, new_count, floor)

    uniq = np.unique(labels)
    rows_by_label = {c: np.where(labels == c)[0] for c in uniq}
    sizes = {c: len(rows_by_label[c]) for c in uniq}

    # Floor allocation, then proportional split of the remainder.
    alloc = {c: min(min_per_cluster, sizes[c]) for c in uniq}
    floor_total = sum(alloc.values())
    if floor_total > size:
        # Too many strata for the budget: scale floors down proportionally.
        scale = size / floor_total
        alloc = {c: max(1, int(alloc[c] * scale)) for c in uniq}
    else:
        remaining = size - floor_total
        total_size = sum(sizes.values())
        for c in uniq:
            extra = int(round(remaining * sizes[c] / max(total_size, 1)))
            alloc[c] = min(sizes[c], alloc[c] + extra)

    # Guarantee the budget invariant (sum <= size): trim the largest allocations.
    while sum(alloc.values()) > size:
        c = max(alloc, key=alloc.get)
        alloc[c] -= 1

    chosen, probs = [], []
    for c in uniq:
        rows = rows_by_label[c]
        m_c = min(alloc[c], len(rows))
        if m_c <= 0:
            continue
        if m_c >= len(rows):
            chosen.append(rows)
            probs.append(np.ones(len(rows)))
            continue

        if sampling == "sensitivity":
            # Lightweight-coreset distribution within the stratum, modulated by
            # recency; record the per-point inclusion probability m_c · p_i so
            # 1/p_i recovers the unbiased coreset weight.
            p = sensitivity_weights(embeddings[rows]) * rw[rows]
            p = p / p.sum()
            sel_pos = rng.choice(len(rows), size=m_c, replace=False, p=p)
            sel = rows[sel_pos]
            chosen.append(sel)
            probs.append(np.clip(m_c * p[sel_pos], 1e-12, 1.0))
        else:
            w = rw[rows]
            w = w / w.sum()
            sel = rng.choice(rows, size=m_c, replace=False, p=w)
            chosen.append(sel)
            probs.append(np.full(len(sel), m_c / len(rows)))

    idx = np.concatenate(chosen)
    pr = np.concatenate(probs)
    order = np.argsort(idx)
    return idx[order], pr[order]


def microcluster_coreset(
    embeddings: np.ndarray,
    n_recent: int,
    k: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid CluStream/BIRCH-style coreset: recent docs raw + summarized history.

    Keeps the newest *n_recent* docs at full weight (1.0) and folds the older
    history into at most *k* micro-clusters, each represented by the **real**
    document nearest its centroid and carrying the micro-cluster's member count
    as its weight. Using a real representative (not a synthetic centroid) keeps
    document text available for keyword extraction; the count weight makes the
    representative stand in for its whole micro-cluster downstream (weighted
    pruning / centroids).

    Unlike :func:`select_coreset`, the working-set size is ``n_recent + k`` —
    bounded by a constant regardless of total accumulated ``N`` (true streaming),
    rather than a fixed fraction of an ever-growing accumulator.

    Returns sorted ``indices`` plus the per-row representation ``weights``.
    """
    try:
        from cuml.cluster import MiniBatchKMeans
    except ImportError:
        from sklearn.cluster import MiniBatchKMeans

    n = len(embeddings)
    n_recent = int(min(max(n_recent, 0), n))
    recent_idx = np.arange(n - n_recent, n)
    old_n = n - n_recent
    if old_n <= 0:
        return recent_idx, np.ones(len(recent_idx), dtype=float)

    old = embeddings[:old_n]
    k_eff = int(min(k, old_n))
    if k_eff <= 1:
        d2 = np.einsum("ij,ij->i", old - old.mean(axis=0), old - old.mean(axis=0))
        reps_idx = [int(np.argmin(d2))]
        reps_w = [float(old_n)]
    else:
        km = MiniBatchKMeans(n_clusters=k_eff, random_state=random_state, n_init=3)
        labels = np.asarray(km.fit_predict(old))
        centers = km.cluster_centers_
        reps_idx, reps_w = [], []
        for c in range(k_eff):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            diff = old[members] - centers[c]
            nearest = members[int(np.argmin(np.einsum("ij,ij->i", diff, diff)))]
            reps_idx.append(int(nearest))
            reps_w.append(float(len(members)))

    idx = np.concatenate([np.asarray(reps_idx, dtype=int), recent_idx])
    w = np.concatenate([np.asarray(reps_w, dtype=float), np.ones(len(recent_idx))])
    order = np.argsort(idx)
    return idx[order], w[order]


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
    try:
        from cuml.cluster import MiniBatchKMeans
    except ImportError:
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
