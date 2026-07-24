"""
Pluggable recluster strategies for cumulative clustering
=========================================================

Each strategy answers one question: **which documents should the new epoch's
:class:`~graphweave.GraphWeave` be fit on?** The orchestrator
(:class:`~graphweave.cumulative.cumulative.CumulativeGraphWeave`) owns everything
else — embedding, global topic-ID alignment, and full-corpus label assignment —
so swapping the engine is a one-line config change and the metrics harness can
benchmark all three head-to-head against the full-batch baseline.

Engines (map directly to the four evaluated approaches):

- ``global_refit`` (#1) — fit on the *entire* accumulator; gold-standard quality.
  Above ``max_inmemory_docs`` it transparently drops into Regime B and fits on a
  recency-weighted coreset (#4) so cost/memory stay bounded.
- ``coreset`` (#4) — always fit on a bounded recency-weighted coreset.
- ``batch_merge`` (#2) — fit on the *newest batch only*; the orchestrator's
  cross-epoch alignment merges the batch's topics into the persistent global set.

Approach #3 (HNSW) is not a strategy here: it already runs *inside* ``fit()`` via
``knn_backend="auto"`` for working sets above 5k docs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphweave.cumulative.alignment import (
    microcluster_coreset,
    recency_weights,
    select_coreset,
    stratified_coreset,
)


@dataclass
class ReclusterContext:
    """Everything a strategy needs to choose its working set."""

    documents: list[str]          # full accumulator, oldest -> newest
    embeddings: np.ndarray        # (N, d) aligned with `documents`
    new_count: int                # size of the most recent batch (tail of accumulator)
    max_inmemory_docs: int        # absolute working-set cap (Regime A/B switch)
    coreset_size: int             # target size for coreset summaries
    random_state: int
    labels: np.ndarray | None = None        # current global label per doc (-1 = outlier)
    coreset_selection: str = "stratified"   # "stratified" | "recency" | "microcluster"
    min_per_cluster: int = 50               # stratified per-topic floor
    coreset_sampling: str = "recency"       # "recency" | "sensitivity" (within-stratum)
    reserve_novel: int = 0                  # force-keep this many recent outliers at full weight


def _recent_outlier_rows(ctx: "ReclusterContext", cap: int) -> np.ndarray:
    """The ``cap`` most-recent outlier (label ``-1``) row indices, newest first.

    Emerging themes surface as recent outliers; force-keeping them (DenStream's
    potential-micro-cluster idea) stops probabilistic sampling from dropping a
    new theme before the next recluster can detect it.
    """
    if cap <= 0 or ctx.labels is None:
        return np.empty(0, dtype=int)
    outliers = np.where(np.asarray(ctx.labels) == -1)[0]
    if len(outliers) == 0:
        return np.empty(0, dtype=int)
    return np.sort(outliers)[-cap:]  # tail == most recent


def _select_reduced(ctx: "ReclusterContext", size: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Pick ``size`` rows for a reduced (Regime B / coreset) working set.

    Stratified when labels are available (guarantees small topics a floor of
    representatives and yields inverse-propensity representation weights);
    otherwise recency-weighted random sampling with no weights (Regime A-like).
    When ``reserve_novel`` is set (stratified path only), the most recent outlier
    docs are force-included at full weight first and the rest of the budget is
    sampled around them.
    """
    if ctx.coreset_selection == "microcluster":
        # Recent docs raw + summarized old history; size-bounded by a constant.
        n_recent = min(ctx.new_count, size)
        return microcluster_coreset(
            ctx.embeddings, n_recent, max(1, size - n_recent), ctx.random_state
        )
    if ctx.coreset_selection == "stratified" and ctx.labels is not None:
        reserved = _recent_outlier_rows(ctx, min(ctx.reserve_novel, size))
        idx, probs = stratified_coreset(
            ctx.embeddings, size - len(reserved), ctx.labels, ctx.new_count,
            ctx.random_state, min_per_cluster=ctx.min_per_cluster,
            sampling=ctx.coreset_sampling,
        )
        # Merge reserved rows at weight 1.0 (they stand only for themselves),
        # letting them override any sampled duplicate.
        row_to_w = {int(i): 1.0 / max(p, 1e-12) for i, p in zip(idx, probs)}
        for r in reserved:
            row_to_w[int(r)] = 1.0
        out_idx = np.array(sorted(row_to_w), dtype=int)
        weights = np.array([row_to_w[int(i)] for i in out_idx], dtype=float)
        return out_idx, weights
    weights_in = recency_weights(len(ctx.documents), ctx.new_count)
    idx = select_coreset(ctx.embeddings, size, ctx.random_state, weights_in)
    return idx, None


@dataclass
class WorkingSet:
    """The documents/embeddings the new epoch model is fit on, plus metadata
    the orchestrator uses to assign global labels and update the registry."""

    documents: list[str]
    embeddings: np.ndarray
    indices: np.ndarray | None    # rows of the accumulator used (None == all)
    regime: str                   # "A" (full) or "B" (reduced)
    is_full_accumulator: bool     # True -> model.labels_ already covers every doc
    covers_full_corpus: bool      # True -> topics represent the whole corpus (replace registry)
    weights: np.ndarray | None = None  # per-doc representation weight (None == uniform)


class ReclusterStrategy:
    """Base class. Subclasses implement :meth:`select_working_set`."""

    name: str = "base"

    def select_working_set(self, ctx: ReclusterContext) -> WorkingSet:  # pragma: no cover
        raise NotImplementedError


class GlobalRefitStrategy(ReclusterStrategy):
    """#1 — refit on the whole accumulator (Regime A); coreset above the cap (Regime B)."""

    name = "global_refit"

    def select_working_set(self, ctx: ReclusterContext) -> WorkingSet:
        n = len(ctx.documents)
        if n <= ctx.max_inmemory_docs:
            return WorkingSet(
                documents=ctx.documents,
                embeddings=ctx.embeddings,
                indices=None,
                regime="A",
                is_full_accumulator=True,
                covers_full_corpus=True,
            )
        # Regime B: bounded coreset still represents the whole corpus.
        idx, weights = _select_reduced(ctx, ctx.max_inmemory_docs)
        return WorkingSet(
            documents=[ctx.documents[i] for i in idx],
            embeddings=ctx.embeddings[idx],
            indices=idx,
            regime="B",
            is_full_accumulator=False,
            covers_full_corpus=True,
            weights=weights,
        )


class CoresetStrategy(ReclusterStrategy):
    """#4 — always fit on a bounded recency-weighted coreset of the accumulator."""

    name = "coreset"

    def select_working_set(self, ctx: ReclusterContext) -> WorkingSet:
        n = len(ctx.documents)
        idx, weights = _select_reduced(ctx, ctx.coreset_size)
        full = len(idx) == n
        return WorkingSet(
            documents=[ctx.documents[i] for i in idx],
            embeddings=ctx.embeddings[idx],
            indices=idx,
            regime="A" if full else "B",
            is_full_accumulator=full,
            covers_full_corpus=True,
            weights=None if full else weights,
        )


class BatchMergeStrategy(ReclusterStrategy):
    """#2 — cluster only the newest batch; the orchestrator merges it into the
    persistent global topic registry via cross-epoch alignment."""

    name = "batch_merge"

    def select_working_set(self, ctx: ReclusterContext) -> WorkingSet:
        n = len(ctx.documents)
        start = max(0, n - ctx.new_count)
        idx = np.arange(start, n)
        return WorkingSet(
            documents=ctx.documents[start:],
            embeddings=ctx.embeddings[start:],
            indices=idx,
            regime="A" if start == 0 else "B",
            is_full_accumulator=(start == 0),
            covers_full_corpus=False,   # batch topics do NOT replace the global set; they accumulate
        )


_REGISTRY: dict[str, type[ReclusterStrategy]] = {
    GlobalRefitStrategy.name: GlobalRefitStrategy,
    CoresetStrategy.name: CoresetStrategy,
    BatchMergeStrategy.name: BatchMergeStrategy,
}

STRATEGY_NAMES = tuple(_REGISTRY.keys())


def make_strategy(name: str) -> ReclusterStrategy:
    """Look up a strategy by name (``global_refit`` / ``coreset`` / ``batch_merge``)."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy {name!r}. Choose one of {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]()
