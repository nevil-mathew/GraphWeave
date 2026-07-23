"""Tests for graphweave.adaptation.pipeline.adapt_and_refit and
GraphWeave.adapt_embeddings_with_llm — the end-to-end plumbing, forced into
linear-adapter mode so no fine-tuning dependencies are required.
"""

import json
import re

import numpy as np
import pytest

from graphweave import GraphWeave, GraphWeaveConfig
from graphweave.adaptation.config import AdaptationConfig
from graphweave.adaptation.pipeline import adapt_and_refit


class _TextOracleLabeler:
    """A perfect triplet-judgment oracle for testing: our synthetic
    documents embed their true topic id as a 'TOPIC_<id>' token, so the
    judgment can be answered exactly by parsing that token back out of the
    document snippets shown in the prompt (rather than needing access to
    the original index arrays, which the LLM interface never exposes)."""

    _PATTERN = re.compile(r"A: (.*?)\n  B: (.*?)\n  C: (.*?)\n\n", re.DOTALL)

    def __init__(self):
        self.calls = 0

    def call_structured(self, system_prompt, user_prompt, schema, max_tokens=None):
        self.calls += 1
        items = self._PATTERN.findall(user_prompt)
        answers = []
        for a_text, b_text, _c_text in items:
            a_topic = re.search(r"TOPIC_(\d+)", a_text).group(1)
            b_topic = re.search(r"TOPIC_(\d+)", b_text).group(1)
            answers.append("B" if a_topic == b_topic else "C")
        return json.dumps({"answers": answers})


def _make_corpus(n_per_topic=35, dim=16, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    docs, labels = [], []
    for t in range(3):
        for i in range(n_per_topic):
            docs.append(f"TOPIC_{t} sample document number {i} about subject area {t}")
            labels.append(t)
    labels = np.array(labels)
    centers = rng.normal(size=(3, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embs = np.vstack([
        centers[t] + noise * rng.normal(size=(n_per_topic, dim)) for t in range(3)
    ])
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return docs, labels, embs.astype(np.float32)


def _fit_model(docs, embs):
    cfg = GraphWeaveConfig(
        use_dim_reduction=False,
        use_iterative_refinement=False,
        n_consensus_runs=3,
        min_cluster_size=5,
        n_neighbors=10,
        random_state=42,
        verbose=False,
    )
    return GraphWeave(config=cfg).fit(docs, embeddings=embs)


_ADAPT_CONFIG = AdaptationConfig(
    adapter_mode="linear",
    n_triplets=200,
    holdout_frac=0.25,
    triplet_sampling="fast",
    linear_epochs=60,
    linear_lr=0.1,
    random_state=42,
    verbose=False,
)


class TestAdaptAndRefit:
    def test_end_to_end_linear_mode(self):
        docs, _labels, embs = _make_corpus()
        model = _fit_model(docs, embs)
        original_embeddings = model.embeddings_.copy()

        new_model, report = adapt_and_refit(
            model, _TextOracleLabeler(), config=_ADAPT_CONFIG, evaluate=False
        )

        assert new_model is not model
        assert new_model._is_fitted
        np.testing.assert_allclose(model.embeddings_, original_embeddings)

        for key in (
            "mode", "adapter", "n_llm_calls", "n_cache_hits", "n_unparsed", "n_train_triplets",
            "n_holdout_triplets", "holdout_triplet_acc_before", "holdout_triplet_acc_after",
        ):
            assert key in report

        assert report["mode"] == "linear"
        assert report["n_holdout_triplets"] > 0
        assert report["holdout_triplet_acc_after"] >= report["holdout_triplet_acc_before"]

    def test_report_adapter_is_saveable(self, tmp_path):
        docs, _labels, embs = _make_corpus()
        model = _fit_model(docs, embs)

        _new_model, report = adapt_and_refit(
            model, _TextOracleLabeler(), config=_ADAPT_CONFIG, evaluate=False
        )

        save_path = str(tmp_path / "saved_adapter")
        report["adapter"].save(save_path)
        assert (tmp_path / "saved_adapter" / "manifest.json").exists()
        assert (tmp_path / "saved_adapter" / "linear.npz").exists()

    def test_raises_on_unfitted_model(self):
        model = GraphWeave()
        with pytest.raises(ValueError):
            adapt_and_refit(model, _TextOracleLabeler())

    def test_explicit_n_topics_carried_through_refit(self):
        docs, _labels, embs = _make_corpus()
        cfg = GraphWeaveConfig(
            use_dim_reduction=False, use_iterative_refinement=False,
            n_consensus_runs=3, min_cluster_size=5, n_neighbors=10,
            random_state=42, verbose=False,
        )
        model = GraphWeave(n_topics=3, config=cfg).fit(docs, embeddings=embs)
        assert model.n_topics == 3

        new_model, _report = adapt_and_refit(
            model, _TextOracleLabeler(), config=_ADAPT_CONFIG, evaluate=False
        )
        assert new_model.n_topics == 3

    def test_metadata_view_warns_when_not_carried_through(self):
        docs, _labels, embs = _make_corpus()
        cfg = GraphWeaveConfig(
            use_dim_reduction=False, use_iterative_refinement=False,
            n_consensus_runs=3, min_cluster_size=5, n_neighbors=10,
            random_state=42, verbose=False, use_metadata_view=True,
        )
        model = GraphWeave(config=cfg).fit(docs, embeddings=embs)

        with pytest.warns(UserWarning, match="metadata"):
            adapt_and_refit(model, _TextOracleLabeler(), config=_ADAPT_CONFIG, evaluate=False)


class TestAdaptEmbeddingsWithLlmDelegate:
    def test_in_place_refit_sets_diagnostics(self):
        docs, _labels, embs = _make_corpus()
        model = _fit_model(docs, embs)

        result = model.adapt_embeddings_with_llm(_TextOracleLabeler(), config=_ADAPT_CONFIG)

        assert result is model
        assert hasattr(model, "adaptation_diagnostics_")
        assert model.adaptation_diagnostics_["mode"] == "linear"
        assert model._is_fitted

    def test_raises_before_fit(self):
        model = GraphWeave()
        with pytest.raises(ValueError):
            model.adapt_embeddings_with_llm(_TextOracleLabeler())
