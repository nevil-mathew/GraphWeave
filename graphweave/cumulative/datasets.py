"""
Realistic streaming corpora for evaluating cumulative clustering — no model
============================================================================

Quality is only meaningful on data that *looks like the real thing*. These
utilities build a realistic, fully in-memory text stream and embed it with
**LSA** (TF-IDF + TruncatedSVD) — the classic, pre-neural way to turn documents
into dense vectors. That means the whole pipeline (text → embedding → graph →
Leiden → keywords) runs end-to-end **without downloading any embedding model**,
yet on data with the messiness that breaks naive clustering:

- overlapping vocabulary between topics (themes are not cleanly separable),
- imbalanced topic sizes (a few big themes, several small ones),
- background "filler" words shared by everything (stopword-like noise),
- genuine outlier/noise documents,
- optionally an **emerging topic** that only appears partway through the stream
  (to exercise drift detection).

The LSA embedder is fit **once** on the whole corpus (simulating a *fixed,
pre-trained* embedder) and then used to embed every batch in the same space —
exactly how a real sentence-transformer would behave across a stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StreamingCorpus:
    """A realistic batch stream plus its full-corpus view and ground truth."""

    batches: list[list[str]]              # documents per batch
    batch_embeddings: list[np.ndarray]    # LSA embeddings per batch (shared space)
    batch_labels: list[np.ndarray]        # ground-truth topic id per doc (-1 = noise)
    all_documents: list[str]
    all_embeddings: np.ndarray
    all_labels: np.ndarray
    n_topics: int

    @property
    def n_docs(self) -> int:
        return len(self.all_documents)


def lsa_embed(
    documents: list[str],
    dim: int = 48,
    min_df: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Embed documents with LSA (TF-IDF → TruncatedSVD), L2-normalized.

    A model-free, deterministic dense embedder. Returns ``(n_docs, dim)``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    tfidf = TfidfVectorizer(min_df=min_df)
    X = tfidf.fit_transform(documents)
    n_components = int(min(dim, X.shape[1] - 1, X.shape[0] - 1))
    n_components = max(n_components, 2)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    emb = svd.fit_transform(X)
    return normalize(emb).astype(np.float32)


def make_streaming_corpus(
    n_topics: int = 6,
    docs_per_batch: int = 120,
    n_batches: int = 5,
    embed_dim: int = 48,
    vocab_per_topic: int = 18,
    n_background: int = 12,
    doc_len: tuple[int, int] = (14, 36),
    overlap: float = 0.18,
    noise_frac: float = 0.06,
    imbalance: bool = True,
    emerging_topic_at: int | None = None,
    emerging_frac: float = 0.30,
    random_state: int = 42,
) -> StreamingCorpus:
    """Generate a realistic, LSA-embedded text stream.

    Parameters
    ----------
    n_topics : int
        Number of ground-truth topics.
    docs_per_batch, n_batches : int
        Stream shape (total docs ≈ ``docs_per_batch * n_batches``).
    embed_dim : int
        LSA embedding dimensionality.
    vocab_per_topic, n_background : int
        Size of each topic's signature vocabulary and the shared filler pool.
    doc_len : (int, int)
        Min/max document length in tokens.
    overlap : float
        Probability a token is "borrowed" from a neighbouring topic
        (controls how separable topics are; higher = harder).
    noise_frac : float
        Fraction of documents that are pure noise (labelled ``-1``).
    imbalance : bool
        If True, topic prevalence follows a Zipf-like distribution.
    emerging_topic_at : int | None
        If set, the *last* topic only appears from this batch index onward
        (use to test drift detection / new-theme discovery).
    emerging_frac : float
        Once active, the share of each batch made up of the emerging topic.
        Kept prominent so the new theme is actually detectable as drift.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    StreamingCorpus
    """
    rng = np.random.default_rng(random_state)

    # Vocabulary: per-topic signatures + a shared background pool.
    topic_vocab = [
        [f"t{t}w{j}" for j in range(vocab_per_topic)] for t in range(n_topics)
    ]
    background = [f"bg{j}" for j in range(n_background)]

    # Topic prevalence (imbalanced -> Zipf-ish).
    if imbalance:
        weights = 1.0 / (1.0 + np.arange(n_topics))
    else:
        weights = np.ones(n_topics)
    weights = weights / weights.sum()

    def gen_doc(topic: int) -> str:
        length = int(rng.integers(doc_len[0], doc_len[1] + 1))
        words: list[str] = []
        for _ in range(length):
            r = rng.random()
            if r < overlap:
                # borrow from a neighbouring topic (vocabulary overlap)
                nb = (topic + rng.integers(1, n_topics)) % n_topics
                words.append(rng.choice(topic_vocab[nb]))
            elif r < overlap + 0.15:
                words.append(rng.choice(background))  # filler / stopword-like
            else:
                words.append(rng.choice(topic_vocab[topic]))  # on-topic
        return " ".join(words)

    def gen_noise() -> str:
        length = int(rng.integers(doc_len[0], doc_len[1] + 1))
        pool = [w for tv in topic_vocab for w in tv] + background
        return " ".join(rng.choice(pool, size=length))

    # Build each batch. Honour the emerging-topic schedule.
    new_topic = n_topics - 1
    old_topics = list(range(n_topics - 1)) if emerging_topic_at is not None else list(range(n_topics))
    old_w = weights[: len(old_topics)]
    old_w = old_w / old_w.sum()

    batches_docs: list[list[str]] = []
    batches_truth: list[list[int]] = []
    for b in range(n_batches):
        emerging_active = emerging_topic_at is not None and b >= emerging_topic_at

        docs_b: list[str] = []
        truth_b: list[int] = []
        for _ in range(docs_per_batch):
            if rng.random() < noise_frac:
                docs_b.append(gen_noise())
                truth_b.append(-1)
            elif emerging_active and rng.random() < emerging_frac:
                docs_b.append(gen_doc(new_topic))   # the prominent new theme
                truth_b.append(new_topic)
            else:
                t = int(rng.choice(old_topics, p=old_w))
                docs_b.append(gen_doc(t))
                truth_b.append(t)

        # Shuffle within the batch (arrival order is not topic-sorted).
        order = rng.permutation(len(docs_b))
        batches_docs.append([docs_b[i] for i in order])
        batches_truth.append([truth_b[i] for i in order])

    # Fit the LSA embedder ONCE on the whole corpus (a fixed pre-trained embedder),
    # then slice per batch so every batch lives in the same vector space.
    all_documents = [d for b in batches_docs for d in b]
    all_embeddings = lsa_embed(all_documents, dim=embed_dim, random_state=random_state)
    all_labels = np.array([t for b in batches_truth for t in b])

    batch_embeddings: list[np.ndarray] = []
    batch_labels: list[np.ndarray] = []
    start = 0
    for docs_b in batches_docs:
        end = start + len(docs_b)
        batch_embeddings.append(all_embeddings[start:end])
        batch_labels.append(all_labels[start:end])
        start = end

    return StreamingCorpus(
        batches=batches_docs,
        batch_embeddings=batch_embeddings,
        batch_labels=batch_labels,
        all_documents=all_documents,
        all_embeddings=all_embeddings,
        all_labels=all_labels,
        n_topics=n_topics,
    )
