"""Tests for graphweave.adaptation.keyphrase."""

import json

import numpy as np

from graphweave.adaptation.keyphrase import generate_keyphrases, keyphrase_expand_embeddings


class _CannedKeyphraseLabeler:
    def __init__(self, phrases):
        self.phrases = phrases
        self.calls = 0

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        self.calls += 1
        n = user_prompt.count("Document ")
        return json.dumps({"keyphrases": [self.phrases for _ in range(n)]})


class _DuckEncoder:
    """Deterministic bag-of-length encoder — no real model needed."""

    def encode(self, texts):
        return np.array([[len(t), t.count(" ") + 1] for t in texts], dtype=float)


def test_generate_keyphrases_basic():
    docs = ["the cat sat on the mat", "dogs like to run in parks"]
    result = generate_keyphrases(_CannedKeyphraseLabeler(["animal", "pet"]), docs, n_keyphrases=2)
    assert len(result) == 2
    assert all(kp == ["animal", "pet"] for kp in result)


def test_generate_keyphrases_cache_round_trip(tmp_path):
    docs = ["doc one text", "doc two text"]
    cache_path = str(tmp_path / "kw_cache.jsonl")

    labeler1 = _CannedKeyphraseLabeler(["a", "b"])
    generate_keyphrases(labeler1, docs, n_keyphrases=2, cache_path=cache_path)
    assert labeler1.calls == 1

    labeler2 = _CannedKeyphraseLabeler(["a", "b"])
    generate_keyphrases(labeler2, docs, n_keyphrases=2, cache_path=cache_path)
    assert labeler2.calls == 0


def test_keyphrase_expand_embeddings_weight_zero_reproduces_base():
    docs = ["hello world", "goodbye world"]
    keyphrases = [["greeting"], ["farewell"]]
    encoder = _DuckEncoder()

    base = encoder.encode(docs)
    base_norm = base / np.linalg.norm(base, axis=1, keepdims=True)

    expanded = keyphrase_expand_embeddings(docs, keyphrases, encoder, weight=0.0, mode="average")
    np.testing.assert_allclose(expanded, base_norm, atol=1e-8)


def test_keyphrase_expand_embeddings_empty_keyphrases_unblended():
    """A document with no keyphrases should keep its own embedding even with
    weight > 0 — blending in encode("") would dilute it for no reason."""
    docs = ["hello world", "goodbye world"]
    keyphrases = [["greeting"], []]  # second doc has no keyphrases
    encoder = _DuckEncoder()

    base = encoder.encode(docs)
    base_norm = base / np.linalg.norm(base, axis=1, keepdims=True)

    expanded = keyphrase_expand_embeddings(docs, keyphrases, encoder, weight=0.9, mode="average")
    np.testing.assert_allclose(expanded[1], base_norm[1], atol=1e-8)


def test_keyphrase_expand_embeddings_concat_mode():
    docs = ["hello world"]
    keyphrases = [["greeting", "hi"]]
    encoder = _DuckEncoder()

    # The raw re-encoding of "doc + keyphrases" is not unit norm.
    raw = encoder.encode([docs[0] + "\nKeyphrases: greeting, hi"])

    # normalize=False returns the encoder output untouched.
    unnormed = keyphrase_expand_embeddings(
        docs, keyphrases, encoder, mode="concat_encode", normalize=False
    )
    assert unnormed.shape == (1, 2)
    np.testing.assert_allclose(unnormed, raw, atol=1e-8)

    # normalize defaults to True — same contract as the "average" path.
    out = keyphrase_expand_embeddings(docs, keyphrases, encoder, mode="concat_encode")
    assert out.shape == (1, 2)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=1e-8)
    np.testing.assert_allclose(out, raw / np.linalg.norm(raw, axis=1, keepdims=True), atol=1e-8)
