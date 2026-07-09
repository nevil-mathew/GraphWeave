"""Tests for GraphWeaveConfig and GraphWeave constructor."""

from graphweave import GraphWeave, GraphWeaveConfig


class TestConfigDefaults:
    def test_default_language(self):
        config = GraphWeaveConfig()
        assert config.language == "english"

    def test_default_soft_assignment_method(self):
        config = GraphWeaveConfig()
        assert config.soft_assignment_method == "centroid"

    def test_default_embedding_model(self):
        config = GraphWeaveConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"


class TestLanguageInit:
    def test_language_from_constructor(self):
        model = GraphWeave(language="german", verbose=False)
        assert model.config.language == "german"

    def test_multilingual_auto_selects_bge_m3(self):
        model = GraphWeave(language="multilingual", verbose=False)
        assert model.config.embedding_model == "BAAI/bge-m3"

    def test_multilingual_keeps_custom_model(self):
        model = GraphWeave(
            language="multilingual",
            embedding_model="custom/model",
            verbose=False,
        )
        assert model.config.embedding_model == "custom/model"


class TestLanguagePropagation:
    def test_graph_builder_receives_language(self):
        model = GraphWeave(language="german", verbose=False)
        assert model._graph_builder.language == "german"

    def test_keyword_extractor_receives_language(self):
        model = GraphWeave(language="french", verbose=False)
        assert model._keyword_extractor.language == "french"

    def test_default_language_propagated(self):
        model = GraphWeave(verbose=False)
        assert model._graph_builder.language == "english"
        assert model._keyword_extractor.language == "english"
