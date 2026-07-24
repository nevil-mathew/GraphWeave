"""Utility functions for GraphWeave."""

from graphweave.utils.metrics import (
    compute_coherence,
    compute_diversity,
    compute_stability,
    compute_ari,
    compute_nmi,
    keyword_jaccard,
)
from graphweave.utils.timing import step_timer
from graphweave.utils.gpu import (
    is_gpu_available,
    gpu_count,
    get_device,
    gpu_cosine_similarity,
    gpu_refine_embeddings,
)
from graphweave.utils.quote_verification import extract_quoted_phrases, verify_quotes

__all__ = [
    "compute_coherence",
    "compute_diversity",
    "compute_stability",
    "compute_ari",
    "compute_nmi",
    "keyword_jaccard",
    "step_timer",
    "is_gpu_available",
    "gpu_count",
    "get_device",
    "gpu_cosine_similarity",
    "gpu_refine_embeddings",
    "extract_quoted_phrases",
    "verify_quotes",
]
