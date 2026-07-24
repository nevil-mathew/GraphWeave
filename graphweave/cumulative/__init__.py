"""
Cumulative / batch-wise topic modeling
=======================================

A separate workflow that layers cumulative, batch-wise clustering on top of the
full-batch :class:`~graphweave.GraphWeave` pipeline **without modifying it**.

>>> from graphweave.cumulative import CumulativeGraphWeave, CumulativeConfig
>>> model = CumulativeGraphWeave(CumulativeConfig(strategy="global_refit"))
>>> model.add_batch(batch_1_docs)
>>> model.add_batch(batch_2_docs)        # reclusters when drift crosses the threshold
>>> view = model.bigger_picture(n_levels=3)   # high-level themes across all data
"""

from graphweave.cumulative.cumulative import (
    BatchResult,
    CumulativeConfig,
    CumulativeGraphWeave,
    EpochSummary,
)
from graphweave.cumulative.datasets import (
    StreamingCorpus,
    lsa_embed,
    make_streaming_corpus,
)
from graphweave.cumulative.strategies import STRATEGY_NAMES

__all__ = [
    "CumulativeGraphWeave",
    "CumulativeConfig",
    "BatchResult",
    "EpochSummary",
    "STRATEGY_NAMES",
    "StreamingCorpus",
    "make_streaming_corpus",
    "lsa_embed",
]
