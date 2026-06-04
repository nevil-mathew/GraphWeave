"""
Shared pytest fixtures for the TriTopic test suite.

All heavy work (fitting a TriTopic model) is done once per session via
session-scoped fixtures. Embeddings are precomputed blob vectors so no
embedding model download is required.
"""

import numpy as np
import pytest

from tritopic import TriTopic, TriTopicConfig

_DIM = 32
_N_TOPICS = 3
_PER_TOPIC = 35   # enough docs for min_cluster_size, split-topic, and soft-assignment tests


@pytest.fixture(scope="session")
def fake_documents():
    """105 short fake documents (35 per topic), ordered by topic."""
    rng = np.random.default_rng(seed=0)
    topic_words = [
        ["neural", "network", "deep", "learning", "model", "train", "gradient"],
        ["climate", "carbon", "emission", "warming", "energy", "solar", "wind"],
        ["rocket", "space", "orbit", "nasa", "planet", "launch", "satellite"],
    ]
    docs = []
    for words in topic_words:
        for _ in range(_PER_TOPIC):
            n = int(rng.integers(5, 12))
            docs.append(" ".join(rng.choice(words, size=n)))
    return docs


@pytest.fixture(scope="session")
def _fake_embeddings(fake_documents):
    """Blob embeddings aligned with fake_documents (same order)."""
    rng = np.random.default_rng(seed=99)
    centers = rng.normal(size=(_N_TOPICS, _DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embs = []
    for t in range(_N_TOPICS):
        for _ in range(_PER_TOPIC):
            v = centers[t] + 0.08 * rng.normal(size=_DIM)
            v /= np.linalg.norm(v)
            embs.append(v)
    return np.array(embs, dtype=np.float32)


@pytest.fixture(scope="session")
def fitted_model(fake_documents, _fake_embeddings):
    """A fitted TriTopic model — shared across all tests in the session."""
    cfg = TriTopicConfig(
        use_dim_reduction=False,
        use_iterative_refinement=False,
        use_lexical_view=True,
        n_consensus_runs=3,
        min_cluster_size=5,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )
    model = TriTopic(config=cfg)
    model.fit(fake_documents, embeddings=_fake_embeddings)
    return model
