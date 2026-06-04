"""
Cumulative / batch-wise topic modeling
=======================================

A separate workflow that layers cumulative, batch-wise clustering on top of the
full-batch :class:`~tritopic.TriTopic` pipeline **without modifying it**.

>>> from tritopic.cumulative import CumulativeTriTopic, CumulativeConfig
>>> model = CumulativeTriTopic(CumulativeConfig(strategy="global_refit"))
>>> model.add_batch(batch_1_docs)
>>> model.add_batch(batch_2_docs)        # reclusters when drift crosses the threshold
>>> view = model.bigger_picture(n_levels=3)   # high-level themes across all data
"""

from tritopic.cumulative.cumulative import (
    BatchResult,
    CumulativeConfig,
    CumulativeTriTopic,
    EpochSummary,
)
from tritopic.cumulative.datasets import (
    StreamingCorpus,
    lsa_embed,
    make_streaming_corpus,
)
from tritopic.cumulative.strategies import STRATEGY_NAMES

__all__ = [
    "CumulativeTriTopic",
    "CumulativeConfig",
    "BatchResult",
    "EpochSummary",
    "STRATEGY_NAMES",
    "StreamingCorpus",
    "make_streaming_corpus",
    "lsa_embed",
]
