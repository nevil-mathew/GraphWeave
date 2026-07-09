"""
GraphWeave: Multi-View Graph Topic Modeling with Iterative Refinement
===================================================================

A state-of-the-art topic modeling library that combines:
- Semantic embeddings (Sentence-BERT, Instructor, BGE)
- Lexical similarity (BM25)
- Metadata context (optional)

With advanced techniques:
- Leiden clustering with consensus
- Mutual kNN + SNN graph construction
- Iterative refinement loop
- LLM-powered topic labeling

Basic usage:
-----------
>>> from graphweave import GraphWeave
>>> model = GraphWeave()
>>> topics = model.fit_transform(documents)
>>> model.visualize()

Author: Nevil Mathew
License: MIT

Originally built on tritopic (MIT License, Copyright (c) 2025 Roman Egger)
— see NOTICE.md for the full third-party license text and provenance.
"""

__version__ = "0.1.0"
__author__ = "Nevil Mathew"

from graphweave.core.model import GraphWeave, GraphWeaveConfig, TopicInfo, ReportTheme
from graphweave.core.graph_builder import GraphBuilder
from graphweave.core.clustering import ConsensusLeiden
from graphweave.core.embeddings import EmbeddingEngine
from graphweave.core.keywords import KeywordExtractor
from graphweave.core.hierarchy import TopicNode, TopicHierarchy
from graphweave.labeling.llm_labeler import LLMLabeler, SimpleLabeler
from graphweave.visualization.plotter import TopicVisualizer
from graphweave.cumulative.cumulative import CumulativeGraphWeave, CumulativeConfig

__all__ = [
    "GraphWeave",
    "GraphWeaveConfig",
    "TopicInfo",
    "ReportTheme",
    "TopicNode",
    "TopicHierarchy",
    "GraphBuilder",
    "ConsensusLeiden",
    "EmbeddingEngine",
    "KeywordExtractor",
    "LLMLabeler",
    "SimpleLabeler",
    "TopicVisualizer",
    "CumulativeGraphWeave",
    "CumulativeConfig",
]
