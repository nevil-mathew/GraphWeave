"""Triplet sampling and LLM-judgment collection for embedding adaptation.

Implements ClusterLLM's Stage-1 triplet-query approach (Zhang, Wang & Shang,
EMNLP 2023): sample (anchor, candidate-B, candidate-C) triplets, ask an LLM
which candidate the anchor is more similar to, and keep the judged triplets
for fine-tuning. :class:`TripletBank` owns the disk cache (so repeated runs
pay zero API cost) and the train/holdout split (so evaluation is always on
judgments the fine-tune never saw).

Only imports from ``._compat`` outside the standard scientific stack — see
the package docstring in ``graphweave/adaptation/__init__.py`` for the
detachability boundary this maintains.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ._compat import (
    TRIPLET_SCHEMA,
    build_triplet_prompt,
    parse_triplet_response,
    sample_triplets_fast,
)

_PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------

def sample_triplets_entropy(
    labels: np.ndarray,
    embeddings: np.ndarray,
    probabilities: np.ndarray | None,
    n_triplets: int,
    random_state: int = 42,
    top_frac: float = 0.5,
    k_neighbors: int = 64,
) -> list[tuple[int, int, int]]:
    """ClusterLLM-style entropy-based triplet sampling.

    Anchors are drawn from the *top_frac* highest soft-assignment-entropy
    documents (excluding outliers) — the boundary cases where the current
    clustering is least confident, and where an LLM judgment is most useful.
    B/C candidates are the nearest same-cluster / different-cluster
    neighbours from one global k-NN query (same pattern as
    ``_sample_triplets_fast``, since the anchor set here differs from the
    uniform-random one that function draws internally).

    Falls back to uniform fast sampling when *probabilities* is unavailable
    (e.g. a degenerate fit with no soft assignments).
    """
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return []

    if probabilities is None:
        return sample_triplets_fast(
            labels, embeddings, n_triplets, random_state=random_state, k_neighbors=k_neighbors
        )

    pool_idx = np.where(mask)[0]
    probs = np.clip(np.asarray(probabilities)[pool_idx], 1e-12, 1.0)
    entropy = -(probs * np.log(probs)).sum(axis=1)

    n_pool = len(pool_idx)
    top_k = max(1, int(np.ceil(top_frac * n_pool)))
    high_entropy_order = np.argsort(-entropy)[:top_k]
    candidate_anchors = pool_idx[high_entropy_order]

    rng = np.random.default_rng(random_state)
    if len(candidate_anchors) > n_triplets:
        anchors = np.sort(rng.choice(candidate_anchors, size=n_triplets, replace=False))
    else:
        anchors = np.sort(candidate_anchors)

    return _nearest_same_diff(labels, embeddings, anchors, pool_idx, k_neighbors)


def sample_triplets_hard_margin(
    labels: np.ndarray,
    embeddings: np.ndarray,
    n_triplets: int,
    random_state: int = 42,
    k_neighbors: int = 64,
    oversample_factor: int = 4,
) -> list[tuple[int, int, int]]:
    """Single-partition adaptation of ClusterLLM's "informed" sampling.

    The original informed sampler ranks triplets by disagreement across
    several *candidate resolutions*; a fine-tuning run only has one
    clustering to sample against. Instead, oversample via a global k-NN
    query and keep the triplets with the smallest
    ``cos(anchor, same) - cos(anchor, diff)`` margin — the hardest cases,
    since a wide-margin triplet is already obvious to the embedder and
    teaches it nothing new.
    """
    pool = sample_triplets_fast(
        labels, embeddings, n_triplets * oversample_factor, random_state=random_state,
        k_neighbors=k_neighbors,
    )
    if len(pool) <= n_triplets:
        return pool

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / norms

    rng = np.random.default_rng(random_state + 1)
    jitter = rng.uniform(0, 1e-6, size=len(pool))
    margins = [
        float(np.dot(normed[a], normed[b]) - np.dot(normed[a], normed[c]))
        for a, b, c in pool
    ]
    order = sorted(range(len(pool)), key=lambda i: margins[i] + jitter[i])
    return [pool[i] for i in order[:n_triplets]]


def _nearest_same_diff(
    labels: np.ndarray,
    embeddings: np.ndarray,
    anchors: np.ndarray,
    pool_idx: np.ndarray,
    k_neighbors: int,
) -> list[tuple[int, int, int]]:
    """Nearest same-cluster / different-cluster neighbour per anchor via one
    global k-NN query restricted to *pool_idx* (non-outlier documents)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = (embeddings / norms).astype(np.float32)

    k = min(k_neighbors, len(pool_idx) - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(normed[pool_idx])
    _, local_indices = nn.kneighbors(normed[anchors])

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


# ---------------------------------------------------------------------------
# Content hashing (cache keys + stable train/holdout split)
# ---------------------------------------------------------------------------

def _snippet(doc: str, n_chars: int) -> str:
    return doc[:n_chars]


def _content_hash(anchor_doc: str, b_doc: str, c_doc: str, n_docs_chars: int) -> str:
    """Hash of the *canonical* (anchor, same, diff) triple's text content —
    independent of swap presentation order — so the cache survives
    resampling/reordering and only depends on what the LLM actually saw."""
    payload = "\x1f".join(
        [_PROMPT_VERSION, _snippet(anchor_doc, n_docs_chars), _snippet(b_doc, n_docs_chars),
         _snippet(c_doc, n_docs_chars)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_for_hash(content_hash: str, random_state: int, holdout_frac: float) -> Literal["train", "holdout"]:
    """Deterministic Bernoulli split keyed by content hash — stable across
    cache reloads and resampling, independent of collection order."""
    digest = hashlib.sha256(f"{content_hash}:{random_state}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "holdout" if u < holdout_frac else "train"


# ---------------------------------------------------------------------------
# Judgment record + bank
# ---------------------------------------------------------------------------

@dataclass
class TripletJudgment:
    """One LLM-judged (anchor, positive, negative) training triple.

    ``positive``/``negative`` reflect the LLM's actual judgment (which
    candidate it said the anchor was closer to), not necessarily the
    cluster labels the triplet was originally sampled from — that's the
    whole point of asking the LLM.
    """

    anchor: int
    positive: int
    negative: int
    swapped: bool
    raw_answer: str | None  # canonical "B"/"C", or None if unparsed
    split: Literal["train", "holdout"]


class TripletBank:
    """Collects, caches, and splits LLM triplet judgments for fine-tuning."""

    def __init__(self, cache_path: str | None = None, random_state: int = 42):
        self.cache_path = cache_path
        self.random_state = random_state
        self.judgments: list[TripletJudgment] = []
        self.n_llm_calls = 0
        self.n_cache_hits = 0
        self.n_unparsed = 0

    # -- collection ---------------------------------------------------------

    def collect(
        self,
        labeler,
        documents: list[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray | None = None,
        n_triplets: int = 1000,
        sampling: Literal["entropy", "informed", "fast"] = "entropy",
        entropy_top_frac: float = 0.5,
        k_neighbors: int = 64,
        batch_size: int = 8,
        holdout_frac: float = 0.2,
        n_docs_chars: int = 300,
        max_tokens: int | None = None,
    ) -> "TripletBank":
        """Sample triplets, query the LLM (skipping cache hits), and append
        judged results to :attr:`judgments`.

        *max_tokens*, when given, overrides the auto-computed per-batch token
        budget (``max(128, batch_size * 20 + 64)``, sized for compact
        non-reasoning "B"/"C" answers) for every LLM call. Reasoning-style
        models can spend their whole budget on hidden chain-of-thought before
        ever emitting the answer, surfacing as an empty completion with
        ``finish_reason="length"`` — pass a larger value here if you're
        deliberately using such a model (a plain instruct/flash model that
        doesn't need this is usually the better fix).
        """
        if sampling == "entropy":
            triplets = sample_triplets_entropy(
                labels, embeddings, probabilities, n_triplets,
                random_state=self.random_state, top_frac=entropy_top_frac, k_neighbors=k_neighbors,
            )
        elif sampling == "informed":
            triplets = sample_triplets_hard_margin(
                labels, embeddings, n_triplets, random_state=self.random_state, k_neighbors=k_neighbors,
            )
        elif sampling == "fast":
            triplets = sample_triplets_fast(
                labels, embeddings, n_triplets, random_state=self.random_state, k_neighbors=k_neighbors,
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling!r}")

        if not triplets:
            return self

        content_hashes = [
            _content_hash(documents[a], documents[b], documents[c], n_docs_chars)
            for a, b, c in triplets
        ]

        cache_index = self._load_cache_index()

        rng_swap = np.random.default_rng(self.random_state + 999)
        swaps = rng_swap.random(len(triplets)) < 0.5
        presented = [
            (a, c, b) if bool(swaps[i]) else (a, b, c) for i, (a, b, c) in enumerate(triplets)
        ]

        resolved: list[TripletJudgment | None] = [None] * len(triplets)
        uncached: list[int] = []
        for i, h in enumerate(content_hashes):
            if h in cache_index:
                resolved[i] = cache_index[h]
                self.n_cache_hits += 1
            else:
                uncached.append(i)

        for start in range(0, len(uncached), batch_size):
            batch_positions = uncached[start : start + batch_size]
            batch_presented = [presented[i] for i in batch_positions]
            system_prompt, user_prompt = build_triplet_prompt(
                batch_presented, documents, n_docs_chars=n_docs_chars
            )
            batch_max_tokens = max_tokens if max_tokens is not None else max(128, len(batch_positions) * 20 + 64)
            raw = labeler.call_structured(
                system_prompt, user_prompt, schema=TRIPLET_SCHEMA, max_tokens=batch_max_tokens
            )
            self.n_llm_calls += 1
            answers = parse_triplet_response(raw, len(batch_positions))

            new_records: list[tuple[str, TripletJudgment]] = []
            for pos, ans in zip(batch_positions, answers):
                a, b, c = triplets[pos]
                swapped = bool(swaps[pos])
                if ans is None:
                    canonical = None
                elif swapped:
                    canonical = "C" if ans == "B" else "B"
                else:
                    canonical = ans

                if canonical is None:
                    self.n_unparsed += 1
                    positive, negative = b, c
                elif canonical == "B":
                    positive, negative = b, c
                else:
                    positive, negative = c, b

                split = _split_for_hash(content_hashes[pos], self.random_state, holdout_frac)
                judgment = TripletJudgment(
                    anchor=a, positive=positive, negative=negative,
                    swapped=swapped, raw_answer=canonical, split=split,
                )
                resolved[pos] = judgment
                new_records.append((content_hashes[pos], judgment))

            self._append_cache(new_records)

        self.judgments.extend(j for j in resolved if j is not None)
        return self

    # -- cache ----------------------------------------------------------

    def _load_cache_index(self) -> dict[str, TripletJudgment]:
        if not self.cache_path or not Path(self.cache_path).exists():
            return {}
        index: dict[str, TripletJudgment] = {}
        with open(self.cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                index[rec["hash"]] = TripletJudgment(
                    anchor=rec["anchor"], positive=rec["positive"], negative=rec["negative"],
                    swapped=rec["swapped"], raw_answer=rec["raw_answer"], split=rec["split"],
                )
        return index

    def _append_cache(self, records: list[tuple[str, TripletJudgment]]) -> None:
        if not self.cache_path or not records:
            return
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a") as f:
            for h, j in records:
                f.write(json.dumps({
                    "hash": h, "anchor": j.anchor, "positive": j.positive, "negative": j.negative,
                    "swapped": j.swapped, "raw_answer": j.raw_answer, "split": j.split,
                }) + "\n")

    # -- views ------------------------------------------------------------

    @property
    def train(self) -> list[TripletJudgment]:
        return [j for j in self.judgments if j.split == "train" and j.raw_answer is not None]

    @property
    def holdout(self) -> list[TripletJudgment]:
        return [j for j in self.judgments if j.split == "holdout" and j.raw_answer is not None]

    def to_training_texts(self, documents: list[str]) -> dict[str, list[str]]:
        """``{"anchor": [...], "positive": [...], "negative": [...]}`` from the
        train split, deduplicated to one triplet per anchor to avoid
        MultipleNegativesRankingLoss treating repeated anchors as false
        in-batch negatives."""
        seen_anchors: set[int] = set()
        anchors, positives, negatives = [], [], []
        for j in self.train:
            if j.anchor in seen_anchors:
                continue
            seen_anchors.add(j.anchor)
            anchors.append(documents[j.anchor])
            positives.append(documents[j.positive])
            negatives.append(documents[j.negative])
        return {"anchor": anchors, "positive": positives, "negative": negatives}

    # -- persistence of full bank state ------------------------------------

    def save(self, path: str) -> None:
        data = {
            "random_state": self.random_state,
            "n_llm_calls": self.n_llm_calls,
            "n_cache_hits": self.n_cache_hits,
            "n_unparsed": self.n_unparsed,
            "judgments": [
                {
                    "anchor": j.anchor, "positive": j.positive, "negative": j.negative,
                    "swapped": j.swapped, "raw_answer": j.raw_answer, "split": j.split,
                }
                for j in self.judgments
            ],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "TripletBank":
        data = json.loads(Path(path).read_text())
        bank = cls(random_state=data["random_state"])
        bank.n_llm_calls = data["n_llm_calls"]
        bank.n_cache_hits = data["n_cache_hits"]
        bank.n_unparsed = data["n_unparsed"]
        bank.judgments = [TripletJudgment(**j) for j in data["judgments"]]
        return bank
