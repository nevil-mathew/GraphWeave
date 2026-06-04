"""
GPU utilities for TriTopic
==========================

Thin wrappers that transparently accelerate matrix operations when a CUDA
device is available, with silent CPU fallback when torch is absent or no GPU
is found.  All public functions accept and return plain NumPy arrays so
callers require no torch knowledge.

Usage
-----
>>> from tritopic.utils.gpu import gpu_cosine_similarity, is_gpu_available
>>> sim = gpu_cosine_similarity(embeddings_a, embeddings_b)   # (n, m) ndarray
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def is_gpu_available() -> bool:
    """Return True if torch is importable *and* at least one CUDA device exists."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_count() -> int:
    """Return number of available CUDA devices (0 if none / torch absent)."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def get_device() -> "torch.device":
    """Return the best available torch device (cuda:0 or cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    except ImportError:  # pragma: no cover
        # Return a plain string — callers that reach this branch are also
        # wrapped in try/except, so they never actually use the value.
        return "cpu"  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def gpu_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between rows of *a* and rows of *b*.

    Drop-in replacement for ``sklearn.metrics.pairwise.cosine_similarity``.
    Returns an ``(n, m)`` float32 NumPy array where entry ``[i, j]`` is the
    cosine similarity between ``a[i]`` and ``b[j]``.

    GPU path: normalises rows with ``torch.nn.functional.normalize``, then
    computes ``a_norm @ b_norm.T`` on CUDA.
    CPU fallback: delegates to sklearn when torch / CUDA is unavailable or on
    any runtime error (e.g. OOM).
    """
    try:
        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            raise RuntimeError("no cuda")

        device = torch.device("cuda:0")
        ta = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
        tb = torch.from_numpy(np.asarray(b, dtype=np.float32)).to(device)
        ta = F.normalize(ta, p=2, dim=1)
        tb = F.normalize(tb, p=2, dim=1)
        result = torch.mm(ta, tb.t())
        return result.cpu().numpy()

    except Exception:
        from sklearn.metrics.pairwise import cosine_similarity as _cpu_cs
        return _cpu_cs(a, b)


# ---------------------------------------------------------------------------
# Batched centroid similarity (used in _refine_embeddings)
# ---------------------------------------------------------------------------

def gpu_refine_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    blend_factor: float,
) -> np.ndarray:
    """GPU-accelerated version of TriTopic._refine_embeddings.

    Vectorises the per-topic loop onto the GPU.  Falls back to the caller's
    own NumPy implementation on any error so callers can wrap this in a
    try/except and proceed normally.

    Parameters
    ----------
    embeddings : np.ndarray  (n_docs, d)
    labels : np.ndarray      (n_docs,)  integer topic labels, -1 = outlier
    blend_factor : float     base blend strength

    Returns
    -------
    refined : np.ndarray  (n_docs, d)  L2-normalised
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("no cuda")

    device = torch.device("cuda:0")
    emb = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
    lbl = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    refined = emb.clone()

    unique_labels = torch.unique(lbl)
    unique_labels = unique_labels[unique_labels != -1]

    for label in unique_labels:
        mask = lbl == label
        topic_embs = emb[mask]                         # (k, d)
        centroid = topic_embs.mean(dim=0)              # (d,)
        centroid_norm = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_n = F.normalize(topic_embs, p=2, dim=1)   # (k, d)
        cos_sim = (emb_n * centroid_norm).sum(dim=1)   # (k,)

        per_doc_scale = cos_sim.clamp(0.0, 1.0).sqrt().unsqueeze(1)  # (k, 1)
        per_doc_blend = blend_factor * per_doc_scale                  # (k, 1)
        blended = (1 - per_doc_blend) * topic_embs + per_doc_blend * centroid
        refined[mask] = blended

    refined = F.normalize(refined, p=2, dim=1)
    return refined.cpu().numpy()
