"""
Pluggable recluster strategies for cumulative clustering
=========================================================

Each strategy answers one question: **which documents should the new epoch's
:class:`~tritopic.TriTopic` be fit on?** The orchestrator
(:class:`~tritopic.cumulative.cumulative.CumulativeTriTopic`) owns everything
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

from tritopic.cumulative.alignment import (
    recency_weights,
    select_coreset,
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
        # Regime B: bounded recency-weighted coreset still represents the whole corpus.
        weights = recency_weights(n, ctx.new_count)
        idx = select_coreset(ctx.embeddings, ctx.max_inmemory_docs, ctx.random_state, weights)
        return WorkingSet(
            documents=[ctx.documents[i] for i in idx],
            embeddings=ctx.embeddings[idx],
            indices=idx,
            regime="B",
            is_full_accumulator=False,
            covers_full_corpus=True,
        )


class CoresetStrategy(ReclusterStrategy):
    """#4 — always fit on a bounded recency-weighted coreset of the accumulator."""

    name = "coreset"

    def select_working_set(self, ctx: ReclusterContext) -> WorkingSet:
        n = len(ctx.documents)
        weights = recency_weights(n, ctx.new_count)
        idx = select_coreset(ctx.embeddings, ctx.coreset_size, ctx.random_state, weights)
        full = len(idx) == n
        return WorkingSet(
            documents=[ctx.documents[i] for i in idx],
            embeddings=ctx.embeddings[idx],
            indices=idx,
            regime="A" if full else "B",
            is_full_accumulator=full,
            covers_full_corpus=True,
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
