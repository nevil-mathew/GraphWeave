"""LLM-guided embedding adaptation for TriTopic.

Implements ClusterLLM's triplet-query approach (Zhang, Wang & Shang, EMNLP
2023): sample (anchor, candidate-B, candidate-C) document triplets, ask an
LLM which candidate the anchor is more similar to, then adapt the embedder
to those judgments — either a real sentence-transformers fine-tune or a
pure-numpy linear transform on top of frozen embeddings (for API embedders
or CPU-only use). Also includes the cheaper Few-Shot-Clustering-style
extras (LLM keyphrase expansion, post-hoc low-confidence correction) and an
evaluation harness for measuring whether the adaptation actually helped on
your corpus.

Import boundary (detachability)
--------------------------------
This subpackage is meant to be liftable into its own package later. Every
module here imports only numpy/scipy/sklearn/pandas/sentence-transformers/
datasets/torch/stdlib plus other ``tritopic.adaptation`` modules, with two
documented exceptions:

- ``_compat.py`` re-exports triplet-sampling/prompt/parsing helpers from
  ``tritopic.labeling.llm_granularity`` (rather than duplicating ~250
  subtle, already-tested lines). To detach: vendor this one file.
- ``evaluation.py`` and ``pipeline.py`` lazily import
  ``tritopic.core.model``/``tritopic.core.embeddings`` *inside function
  bodies* (never at module scope), because driving an actual TriTopic fit
  is unavoidably their job.

Typical usage
-------------
>>> from tritopic import TriTopic, LLMLabeler
>>> from tritopic.adaptation import adapt_and_refit
>>> model = TriTopic().fit(documents)
>>> labeler = LLMLabeler(provider="anthropic", api_key="...", model="claude-haiku-4-5")
>>> adapted_model, report = adapt_and_refit(model, labeler)

or, mirroring ``tune_resolution_with_llm``'s in-place ergonomics:

>>> model.adapt_embeddings_with_llm(labeler)
"""

from __future__ import annotations

from .adapter import EmbeddingAdapter, LinearAdapter
from .config import AdaptationConfig
from .correction import reassign_low_confidence
from .evaluation import compare_embedders, forgetting_check, hungarian_accuracy, triplet_accuracy
from .keyphrase import generate_keyphrases, keyphrase_expand_embeddings
from .pipeline import adapt_and_refit
from .triplets import TripletBank, TripletJudgment

__all__ = [
    "AdaptationConfig",
    "EmbeddingAdapter",
    "LinearAdapter",
    "TripletBank",
    "TripletJudgment",
    "adapt_and_refit",
    "compare_embedders",
    "triplet_accuracy",
    "hungarian_accuracy",
    "forgetting_check",
    "generate_keyphrases",
    "keyphrase_expand_embeddings",
    "reassign_low_confidence",
]
