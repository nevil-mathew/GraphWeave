"""
Cumulative / batch-wise topic modeling for TriTopic
====================================================

``CumulativeTriTopic`` is a **separate** workflow that layers cumulative,
batch-wise clustering on top of the existing full-batch
:class:`~tritopic.TriTopic` — without touching ``fit()`` /
``fit_transform()``. Documents arrive in batches (1k–500k each) and accumulate;
the model periodically **re-clusters** the accumulated corpus (the highest-quality
operation, identical to a full-batch run) and **aligns** topic IDs across epochs so
themes stay stable for longitudinal tracking. Between reclusters, incoming batches
are assigned with :meth:`TriTopic.transform`.

The actual clustering engine is pluggable (see
:mod:`tritopic.cumulative.strategies`): ``global_refit`` (default), ``coreset``,
or ``batch_merge``. The "bigger picture" across accumulated data is delivered by
the existing :meth:`TriTopic.build_hierarchy` and
:meth:`TriTopic.generate_report_themes`, which scale with #topics, not #docs.

Persistence (spilling the accumulator / summaries to disk so they survive a
process or exceed RAM) is intentionally **out of scope** here — that is the DB
phase. This POC keeps everything in memory.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from tritopic.core.embeddings import EmbeddingEngine
from tritopic.core.model import TriTopic, TriTopicConfig
from tritopic.cumulative.alignment import (
    align_topics,
    assign_to_registry,
    identity_mapping,
)
from tritopic.cumulative.strategies import (
    ReclusterContext,
    WorkingSet,
    make_strategy,
)


@dataclass
class CumulativeConfig:
    """Configuration for :class:`CumulativeTriTopic`.

    ``base_config`` is reused (deep-copied) for every recluster, so all the
    full-batch knobs (graph type, consensus, dim-reduction, HNSW backend, …)
    carry over unchanged.
    """

    base_config: TriTopicConfig | None = None
    strategy: Literal["global_refit", "batch_merge", "coreset"] = "global_refit"

    # When to fire a recluster.
    recluster_trigger: Literal["drift", "manual", "schedule"] = "drift"
    novelty_threshold: float = 0.30          # drift: fraction of a batch that lands as outliers
    min_docs_between_recluster: int = 0       # debounce auto-triggers
    schedule_every_n_docs: int | None = None  # for trigger == "schedule"

    # Unbounded-growth control (Regime A -> B switch).
    max_inmemory_docs: int = 300_000          # absolute working-set cap (representative points)
    coreset_size: int = 50_000                # size of the recency-weighted coreset summary

    # Cross-epoch topic alignment (stable global topic IDs).
    align_topics: bool = True
    align_threshold: float = 0.6              # cosine; below this a new topic gets a fresh global ID

    verbose: bool = False


@dataclass
class BatchResult:
    """Returned by :meth:`CumulativeTriTopic.add_batch`."""

    epoch: int
    n_new_docs: int
    n_total_docs: int
    reclustered: bool
    novelty: float | None                     # None on the first batch
    assignments: np.ndarray | None            # global topic ID per *new* doc


@dataclass
class EpochSummary:
    """One row of :attr:`CumulativeTriTopic.history_`."""

    epoch: int
    strategy: str
    regime: str
    n_docs_clustered: int
    n_total_docs: int
    n_topics: int
    n_global_topics: int


class CumulativeTriTopic:
    """Cumulative, batch-wise topic model that reuses the full-batch pipeline.

    Parameters
    ----------
    config : CumulativeConfig, optional
        Cumulative-specific configuration. ``config.base_config`` is the
        full-batch :class:`TriTopicConfig` reused for every recluster.

    Key attributes after one or more batches
    ----------------------------------------
    model_ : TriTopic | None
        The most recent fitted model (used for keywords, hierarchy, themes).
    labels_ : np.ndarray | None
        Global topic ID for every accumulated document (``-1`` = outlier).
    history_ : list[EpochSummary]
        One entry per recluster.
    """

    def __init__(self, config: CumulativeConfig | None = None):
        self.config = config or CumulativeConfig()
        self.base_config = self.config.base_config or TriTopicConfig(verbose=self.config.verbose)

        # One persistent embedding engine so each document is embedded exactly once.
        self._engine = EmbeddingEngine(
            model_name=self.base_config.embedding_model,
            batch_size=self.base_config.embedding_batch_size,
            provider=self.base_config.embedding_provider,
            api_key=self.base_config.embedding_api_key,
            api_batch_size=self.base_config.embedding_api_batch_size,
            output_dim=self.base_config.embedding_output_dim,
            task_type=self.base_config.embedding_task_type,
            batch_delay=self.base_config.embedding_batch_delay,
            prefix=self.base_config.embedding_prefix,
            verbose=self.config.verbose,
        )
        self._strategy = make_strategy(self.config.strategy)

        # Accumulator (in-memory; disk spill is a DB-phase concern).
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None

        # Fitted state.
        self.model_: TriTopic | None = None
        self.labels_: np.ndarray | None = None

        # Global topic registry (persists across epochs).
        self._registry_centroids: np.ndarray | None = None
        self._registry_ids: list[int] = []
        self._registry_counts: np.ndarray | None = None
        self._next_global_id: int = 0
        self._local_id_to_global: dict[int, int] = {}

        # Bookkeeping.
        self._epoch: int = 0
        self._last_batch_len: int = 0
        self._docs_since_recluster: int = 0
        self.history_: list[EpochSummary] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_batch(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
    ) -> BatchResult:
        """Ingest a batch, assign it, and recluster if the trigger fires.

        Parameters
        ----------
        documents : list[str]
            New documents.
        embeddings : np.ndarray, optional
            Pre-computed embeddings for *documents*. If ``None`` they are
            embedded once with the persistent engine.
        metadata : pd.DataFrame, optional
            Accepted for API symmetry; the metadata view is not yet wired into
            cumulative reclustering (DB-phase / future work).

        Returns
        -------
        BatchResult
        """
        if not documents:
            raise ValueError("documents must be a non-empty list of strings.")

        emb = np.asarray(embeddings) if embeddings is not None else self._engine.encode(documents)
        if len(emb) != len(documents):
            raise ValueError(
                f"embeddings length ({len(emb)}) must match documents length ({len(documents)})."
            )

        first = self.model_ is None
        novelty: float | None = None
        pre_assignments: np.ndarray | None = None

        if not first:
            local = self.model_.transform(documents, embeddings=emb)
            pre_assignments = self._map_local_to_global(local)
            novelty = float(np.mean(pre_assignments == -1))

        # Accumulate.
        self._append(documents, emb)
        self._last_batch_len = len(documents)
        self._docs_since_recluster += len(documents)

        do_recluster = self._should_recluster(first=first, novelty=novelty)
        if do_recluster:
            self.recluster()
            assignments = self.labels_[-len(documents):]
        else:
            # No recluster: extend the global labelling with this batch's
            # transform-based assignments so labels_ always spans the full
            # accumulator (these are refreshed at the next recluster).
            assignments = pre_assignments
            if pre_assignments is not None:
                self.labels_ = (
                    pre_assignments
                    if self.labels_ is None
                    else np.concatenate([self.labels_, pre_assignments])
                )

        if self.config.verbose:
            tag = "reclustered" if do_recluster else "assigned"
            nv = f", novelty={novelty:.2f}" if novelty is not None else ""
            print(
                f"[Cumulative] batch +{len(documents)} -> {len(self._documents)} docs "
                f"({tag}, epoch={self._epoch}{nv})"
            )

        return BatchResult(
            epoch=self._epoch,
            n_new_docs=len(documents),
            n_total_docs=len(self._documents),
            reclustered=do_recluster,
            novelty=novelty,
            assignments=assignments,
        )

    def recluster(self) -> "CumulativeTriTopic":
        """Re-cluster the accumulated corpus and re-align topic IDs.

        Callable manually (e.g. ``recluster_trigger="manual"``) or fired
        automatically by :meth:`add_batch`.
        """
        if self._embeddings is None or not self._documents:
            raise ValueError("No data to cluster. Call add_batch() first.")

        ctx = ReclusterContext(
            documents=self._documents,
            embeddings=self._embeddings,
            new_count=self._last_batch_len or len(self._documents),
            max_inmemory_docs=self.config.max_inmemory_docs,
            coreset_size=self.config.coreset_size,
            random_state=self.base_config.random_state,
        )
        ws = self._strategy.select_working_set(ctx)

        new_model = TriTopic(config=copy.deepcopy(self.base_config))
        new_model.fit(ws.documents, embeddings=ws.embeddings)

        self._align_and_assign(new_model, ws)

        self.model_ = new_model
        self._epoch += 1
        self._docs_since_recluster = 0

        n_topics = len([t for t in new_model.topics_ if t.topic_id != -1])
        self.history_.append(
            EpochSummary(
                epoch=self._epoch,
                strategy=self._strategy.name,
                regime=ws.regime,
                n_docs_clustered=len(ws.documents),
                n_total_docs=len(self._documents),
                n_topics=n_topics,
                n_global_topics=len(self._registry_ids),
            )
        )
        return self

    def transform(
        self, documents: list[str], embeddings: np.ndarray | None = None
    ) -> np.ndarray:
        """Assign new documents to **global** topic IDs (without accumulating them)."""
        self._require_fitted()
        local = self.model_.transform(documents, embeddings=embeddings)
        return self._map_local_to_global(local)

    def transform_proba(
        self, documents: list[str], embeddings: np.ndarray | None = None
    ) -> np.ndarray:
        """Soft assignment over the current model's topics (delegates to TriTopic)."""
        self._require_fitted()
        return self.model_.transform_proba(documents, embeddings=embeddings)

    def bigger_picture(
        self,
        labeler=None,
        n_levels: int = 3,
        n_themes: int | None = None,
    ) -> dict:
        """Produce the high-level view across the accumulated corpus.

        Always builds a multi-resolution :class:`TopicHierarchy`. If a
        ``labeler`` (LLMLabeler) is given, also labels topics and synthesizes
        report meta-themes. Both reuse existing :class:`TriTopic` machinery and
        scale with #topics, not #docs.

        Returns
        -------
        dict with keys ``"hierarchy"`` and ``"themes"`` (``themes`` is ``None``
        when no labeler is supplied).
        """
        self._require_fitted()
        hierarchy = self.model_.build_hierarchy(n_levels=n_levels)
        themes = None
        if labeler is not None:
            self.model_.generate_labels(labeler)
            themes = self.model_.generate_report_themes(labeler, n_themes=n_themes)
        return {"hierarchy": hierarchy, "themes": themes}

    def get_topic_info(self, global_ids: bool = True) -> pd.DataFrame:
        """Per-topic DataFrame from the current model, plus a ``GlobalTopic`` column."""
        self._require_fitted()
        df = self.model_.get_topic_info().copy()
        if global_ids and "Topic" in df.columns:
            df["GlobalTopic"] = df["Topic"].map(
                lambda t: self._local_id_to_global.get(int(t), -1) if t != -1 else -1
            )
        return df

    def evaluate(self) -> dict:
        """Quality metrics for the current model + cumulative bookkeeping."""
        self._require_fitted()
        metrics = self.model_.evaluate()
        metrics.update(
            {
                "n_total_docs": len(self._documents),
                "n_epochs": self._epoch,
                "n_global_topics": len(self._registry_ids),
            }
        )
        return metrics

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def visualize(
        self,
        method: Literal["umap", "pacmap"] = "umap",
        color_by: Literal["topic", "custom"] = "topic",
        custom_labels: list[str] | None = None,
        show_outliers: bool = True,
        interactive: bool = True,
        **kwargs,
    ):
        """Visualize all accumulated documents in 2-D, coloured by global topic ID.

        Uses every document ever added (not just the working set of the last
        recluster), so the plot is directly comparable to a full-batch
        :meth:`TriTopic.visualize` run on the same corpus.

        Parameters
        ----------
        method : {"umap", "pacmap"}
            Dimensionality reduction method.
        color_by : {"topic", "custom"}
            Colouring strategy.
        custom_labels : list[str], optional
            One label per accumulated document when ``color_by="custom"``.
        show_outliers : bool
            Whether to show outlier documents (global topic ID == -1).
        interactive : bool
            If True, returns an interactive Plotly figure.
        """
        from tritopic.visualization.plotter import TopicVisualizer

        self._require_fitted()
        visualizer = TopicVisualizer(method=method)
        return visualizer.plot_documents(
            embeddings=self._embeddings,
            labels=self.labels_,
            documents=self._documents,
            topics=self._make_global_topics(),
            show_outliers=show_outliers,
            interactive=interactive,
            **kwargs,
        )

    def visualize_3d(
        self,
        method: Literal["umap", "pacmap"] = "umap",
        show_outliers: bool = True,
        **kwargs,
    ):
        """Visualize all accumulated documents in 3-D, coloured by global topic ID.

        Parameters
        ----------
        method : {"umap", "pacmap"}
            Dimensionality reduction method.
        show_outliers : bool
            Whether to show outlier documents.
        """
        from tritopic.visualization.plotter import TopicVisualizer

        self._require_fitted()
        visualizer = TopicVisualizer(method=method)
        return visualizer.plot_documents_3d(
            embeddings=self._embeddings,
            labels=self.labels_,
            documents=self._documents,
            topics=self._make_global_topics(),
            show_outliers=show_outliers,
            **kwargs,
        )

    def visualize_topics(self, **kwargs):
        """Heatmap / bar chart of the current epoch's topics.

        Note: for ``batch_merge`` strategy this reflects the last batch's topics
        only, not the full global topic set.
        """
        from tritopic.visualization.plotter import TopicVisualizer

        self._require_fitted()
        if self.config.strategy == "batch_merge":
            warnings.warn(
                "visualize_topics() with strategy='batch_merge' shows the last "
                "batch's topics only, not the full global topic set.",
                UserWarning,
                stacklevel=2,
            )
        visualizer = TopicVisualizer()
        return visualizer.plot_topics(topics=self._make_global_topics(), **kwargs)

    def visualize_hierarchy(self, **kwargs):
        """Dendrogram of topic similarity using the current epoch's centroids.

        Note: for ``batch_merge`` strategy this reflects the last batch's topics
        only, not the full global topic set.
        """
        from tritopic.visualization.plotter import TopicVisualizer

        self._require_fitted()
        if self.config.strategy == "batch_merge":
            warnings.warn(
                "visualize_hierarchy() with strategy='batch_merge' shows the last "
                "batch's topics only, not the full global topic set.",
                UserWarning,
                stacklevel=2,
            )
        visualizer = TopicVisualizer()
        return visualizer.plot_hierarchy(
            topic_embeddings=self.model_.topic_embeddings_,
            topics=self._make_global_topics(),
            **kwargs,
        )

    def visualize_topic_map(
        self,
        method: Literal["mds", "pca", "umap"] = "mds",
        **kwargs,
    ):
        """Intertopic distance map — 2-D projection of topic centroids.

        Bubbles sized by topic count; layout driven by centroid cosine distance.

        Note: for ``batch_merge`` strategy this reflects the last batch's topics
        only, not the full global topic set.

        Parameters
        ----------
        method : {"mds", "pca", "umap"}
            Projection used on the centroid matrix.
        """
        from tritopic.visualization.plotter import plot_intertopic_distance_map

        self._require_fitted()
        if self.model_.topic_embeddings_ is None:
            raise ValueError("Topic centroids are not available.")
        if self.config.strategy == "batch_merge":
            warnings.warn(
                "visualize_topic_map() with strategy='batch_merge' shows the last "
                "batch's topics only, not the full global topic set.",
                UserWarning,
                stacklevel=2,
            )
        return plot_intertopic_distance_map(
            topic_embeddings=self.model_.topic_embeddings_,
            topics=self._make_global_topics(),
            method=method,
            **kwargs,
        )

    def visualize_overlap(self, threshold: float = 0.1, **kwargs):
        """Topic co-occurrence heatmap based on soft assignments in the working set.

        Note: soft assignment probabilities are computed over the working set of
        the last recluster, not the full accumulator.
        """
        self._require_fitted()
        return self.model_.visualize_overlap(threshold=threshold, **kwargs)

    def visualize_hierarchy_tree(self, **kwargs):
        """Tree diagram of the topic hierarchy.

        Requires :meth:`bigger_picture` (or ``model_.build_hierarchy()``) to have
        been called first.
        """
        self._require_fitted()
        return self.model_.visualize_hierarchy_tree(**kwargs)

    # ------------------------------------------------------------------ #
    # Topic info
    # ------------------------------------------------------------------ #

    def get_topic(self, global_topic_id: int):
        """Return :class:`TopicInfo` for a global topic ID.

        Parameters
        ----------
        global_topic_id : int
            The stable global topic ID (as seen in :attr:`labels_` and
            ``get_topic_info()["GlobalTopic"]``).
        """
        self._require_fitted()
        global_to_local = {v: k for k, v in self._local_id_to_global.items()}
        local_id = global_to_local.get(global_topic_id)
        if local_id is None:
            raise ValueError(
                f"Global topic {global_topic_id} is not in the current epoch's model. "
                "It may belong to a past epoch that has been superseded."
            )
        return self.model_.get_topic(local_id)

    def get_representative_docs(
        self, global_topic_id: int, n_docs: int = 5
    ) -> list[tuple[int, str]]:
        """Representative documents for a global topic ID, drawn from all accumulated docs.

        Parameters
        ----------
        global_topic_id : int
            Stable global topic ID.
        n_docs : int
            Number of documents to return.

        Returns
        -------
        list[tuple[int, str]]
            ``(accumulator_index, document_text)`` pairs, closest to the topic centroid.
        """
        self._require_fitted()
        mask = self.labels_ == global_topic_id
        indices = np.where(mask)[0]
        if len(indices) == 0:
            raise ValueError(f"No documents found for global topic {global_topic_id}.")

        topic_embeddings = self._embeddings[mask]
        centroid = topic_embeddings.mean(axis=0)
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(centroid.reshape(1, -1), topic_embeddings)[0]
        top_local = np.argsort(sims)[::-1][: min(n_docs, len(indices))]
        top_global = indices[top_local]
        return [(int(idx), self._documents[idx]) for idx in top_global]

    def get_document_topics(
        self, doc_idx: int, top_n: int = 3, method=None
    ) -> list[tuple[int, float]]:
        """Top-N global topic IDs with probabilities for one document.

        Parameters
        ----------
        doc_idx : int
            Index into the **working set** of the last recluster (i.e.
            ``model_.documents_``), *not* the full accumulator. Use
            :attr:`model_` directly to inspect which documents are in scope.
        top_n : int
            Number of top topics to return.
        method : str, optional
            ``"centroid"`` or ``"graph"``. Defaults to the model config.

        Returns
        -------
        list[tuple[int, float]]
            ``(global_topic_id, probability)`` pairs, sorted descending.
        """
        self._require_fitted()
        local_results = self.model_.get_document_topics(doc_idx, top_n=top_n, method=method)
        return [
            (self._local_id_to_global.get(tid, tid), prob)
            for tid, prob in local_results
        ]

    def topic_overlap_matrix(self, threshold: float = 0.1) -> pd.DataFrame:
        """Topic co-occurrence matrix from soft assignments in the working set.

        Parameters
        ----------
        threshold : float
            Minimum probability for a topic to count as active.

        Returns
        -------
        pd.DataFrame
            Symmetric ``(n_topics, n_topics)`` DataFrame labelled by global topic ID.
        """
        self._require_fitted()
        df = self.model_.topic_overlap_matrix(threshold=threshold)
        # Relabel columns/index from local to global IDs.
        rename = {
            f"Topic {local_id}": f"Topic {self._local_id_to_global.get(local_id, local_id)}"
            for local_id in self._local_id_to_global
        }
        return df.rename(index=rename, columns=rename)

    # ------------------------------------------------------------------ #
    # Labeling
    # ------------------------------------------------------------------ #

    def generate_labels(self, labeler, topics=None, dedup: bool = True, dedup_passes: int = 1):
        """Generate LLM labels for the current epoch's topics (delegates to TriTopic).

        Parameters
        ----------
        labeler : LLMLabeler
            Configured LLM labeler instance.
        topics : list[int], optional
            Local topic IDs to label. ``None`` labels all topics.
        dedup : bool
            Whether to de-duplicate colliding labels.
        dedup_passes : int
            Number of de-duplication passes.
        """
        self._require_fitted()
        return self.model_.generate_labels(
            labeler, topics=topics, dedup=dedup, dedup_passes=dedup_passes
        )

    def generate_report_themes(self, labeler, n_themes=None, n_docs_per_theme: int = 12):
        """Synthesize high-level meta-themes from topic labels (delegates to TriTopic).

        Parameters
        ----------
        labeler : LLMLabeler
            Configured LLM labeler instance.
        n_themes : int, optional
            Target number of meta-themes. ``None`` lets the LLM decide.
        n_docs_per_theme : int
            Representative documents per theme fed to the LLM.
        """
        self._require_fitted()
        return self.model_.generate_report_themes(
            labeler, n_themes=n_themes, n_docs_per_theme=n_docs_per_theme
        )

    def regenerate_theme(self, labeler, theme_id: int, new_topic_ids=None, n_docs: int = 12):
        """Re-generate one meta-theme's narrative (delegates to TriTopic).

        Requires :meth:`generate_report_themes` to have been called first.

        Parameters
        ----------
        labeler : LLMLabeler
            Configured LLM labeler instance.
        theme_id : int
            1-indexed theme ID to regenerate.
        new_topic_ids : list[int], optional
            Replace the theme's topic membership before regenerating.
        n_docs : int
            Representative documents fed to the LLM.
        """
        self._require_fitted()
        return self.model_.regenerate_theme(
            labeler, theme_id, new_topic_ids=new_topic_ids, n_docs=n_docs
        )

    # ------------------------------------------------------------------ #
    # Post-fit operations
    # ------------------------------------------------------------------ #

    def reduce_outliers(self, strategy: str = "embeddings", threshold=None) -> "CumulativeTriTopic":
        """Reassign outlier documents in the working set to the nearest topic.

        Mutates the internal TriTopic model in-place. ``self.labels_`` and the
        global topic registry become stale until the next :meth:`recluster`.

        Parameters
        ----------
        strategy : str
            ``"embeddings"`` or ``"neighbors"`` — passed through to TriTopic.
        threshold : float, optional
            Distance threshold passed through to TriTopic.
        """
        self._require_fitted()
        self.model_.reduce_outliers(strategy=strategy, threshold=threshold)
        warnings.warn(
            "reduce_outliers() mutated the internal TriTopic model. "
            "CumulativeTriTopic.labels_ and the global topic registry are now stale. "
            "Call recluster() to re-synchronize.",
            UserWarning,
            stacklevel=2,
        )
        return self

    def reduce_topics(self, n_topics: int) -> "CumulativeTriTopic":
        """Merge working-set topics until ``n_topics`` remain.

        Mutates the internal TriTopic model in-place. ``self.labels_``,
        ``self._local_id_to_global``, and the global topic registry become stale
        until the next :meth:`recluster`.

        Parameters
        ----------
        n_topics : int
            Target number of topics.
        """
        self._require_fitted()
        self.model_.reduce_topics(n_topics)
        warnings.warn(
            "reduce_topics() mutated the internal TriTopic model. "
            "CumulativeTriTopic.labels_ and the global topic registry are now stale. "
            "Call recluster() to re-synchronize.",
            UserWarning,
            stacklevel=2,
        )
        return self

    def merge_topics(self, topics_to_merge: list[int]) -> "CumulativeTriTopic":
        """Merge the specified local topic IDs into one.

        Mutates the internal TriTopic model in-place. ``self.labels_``,
        ``self._local_id_to_global``, and the global topic registry become stale
        until the next :meth:`recluster`.

        Parameters
        ----------
        topics_to_merge : list[int]
            Local topic IDs to merge (the largest ID is kept).
        """
        self._require_fitted()
        self.model_.merge_topics(topics_to_merge)
        warnings.warn(
            "merge_topics() mutated the internal TriTopic model. "
            "CumulativeTriTopic.labels_ and the global topic registry are now stale. "
            "Call recluster() to re-synchronize.",
            UserWarning,
            stacklevel=2,
        )
        return self

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def n_global_topics(self) -> int:
        return len(self._registry_ids)

    @property
    def embeddings_(self) -> np.ndarray | None:
        """All accumulated document embeddings (read-only view)."""
        return self._embeddings

    @property
    def documents_(self) -> list[str]:
        """All accumulated documents."""
        return self._documents

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _append(self, documents: list[str], emb: np.ndarray) -> None:
        self._documents.extend(documents)
        if self._embeddings is None:
            self._embeddings = emb.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, emb])

    def _should_recluster(self, first: bool, novelty: float | None) -> bool:
        if first:
            return True
        if self._docs_since_recluster < self.config.min_docs_between_recluster:
            return False
        trigger = self.config.recluster_trigger
        if trigger == "manual":
            return False
        if trigger == "drift":
            return novelty is not None and novelty > self.config.novelty_threshold
        if trigger == "schedule":
            n = self.config.schedule_every_n_docs
            return n is not None and self._docs_since_recluster >= n
        return False

    def _map_local_to_global(self, local_labels: np.ndarray) -> np.ndarray:
        out = np.empty(len(local_labels), dtype=int)
        for i, lab in enumerate(local_labels):
            out[i] = -1 if lab == -1 else self._local_id_to_global.get(int(lab), -1)
        return out

    def _align_and_assign(self, new_model: TriTopic, ws: WorkingSet) -> None:
        """Map the new model's local topics to stable global IDs, update the
        registry, and compute global labels for the whole accumulator."""
        non_outlier = [t for t in new_model.topics_ if t.topic_id != -1]
        new_local_ids = [t.topic_id for t in non_outlier]
        new_sizes = np.array([t.size for t in non_outlier], dtype=float)
        new_centroids = new_model.topic_embeddings_

        # Degenerate: no real topics this epoch.
        if new_centroids is None or len(new_local_ids) == 0:
            self._local_id_to_global = {}
            self.labels_ = np.full(len(self._documents), -1, dtype=int)
            return

        if self.config.align_topics:
            mapping, self._next_global_id, _ = align_topics(
                new_centroids,
                self._registry_centroids,
                self._registry_ids,
                threshold=self.config.align_threshold,
                next_id=self._next_global_id,
            )
        else:
            mapping, self._next_global_id = identity_mapping(
                len(new_centroids), self._next_global_id
            )

        self._local_id_to_global = {
            new_local_ids[i]: g for i, g in mapping.items()
        }

        if ws.covers_full_corpus:
            self._replace_registry(new_centroids, new_local_ids, new_sizes)
        else:
            self._accumulate_registry(new_centroids, new_local_ids, new_sizes, mapping)

        # Global labels for every accumulated document.
        if (
            ws.is_full_accumulator
            and new_model.labels_ is not None
            and len(new_model.labels_) == len(self._documents)
        ):
            self.labels_ = np.array(
                [self._local_id_to_global.get(int(l), -1) for l in new_model.labels_],
                dtype=int,
            )
        else:
            self.labels_ = assign_to_registry(
                self._embeddings,
                self._registry_centroids,
                self._registry_ids,
                self.base_config.outlier_threshold,
            )

    def _replace_registry(
        self, new_centroids: np.ndarray, new_local_ids: list[int], new_sizes: np.ndarray
    ) -> None:
        """Registry := exactly the new topics (the working set represents the
        whole corpus, so old topics are superseded)."""
        self._registry_centroids = new_centroids.copy()
        self._registry_ids = [self._local_id_to_global[lid] for lid in new_local_ids]
        self._registry_counts = new_sizes.copy()

    def _accumulate_registry(
        self,
        new_centroids: np.ndarray,
        new_local_ids: list[int],
        new_sizes: np.ndarray,
        mapping: dict[int, int],
    ) -> None:
        """Merge batch topics into the persistent registry (approach #2): update
        matched globals via count-weighted running mean, append new globals."""
        bank: dict[int, list] = {}
        if self._registry_centroids is not None:
            for gid, cen, cnt in zip(
                self._registry_ids, self._registry_centroids, self._registry_counts
            ):
                bank[int(gid)] = [cen.astype(float).copy(), float(cnt)]

        for i, lid in enumerate(new_local_ids):
            gid = mapping[i]
            cen = new_centroids[i].astype(float)
            size = float(new_sizes[i])
            if gid in bank:
                old_cen, old_cnt = bank[gid]
                total = old_cnt + size
                merged = (old_cen * old_cnt + cen * size) / max(total, 1e-12)
                norm = np.linalg.norm(merged)
                bank[gid] = [merged / norm if norm > 0 else merged, total]
            else:
                bank[gid] = [cen.copy(), size]

        ids = sorted(bank.keys())
        self._registry_ids = ids
        self._registry_centroids = np.array([bank[g][0] for g in ids])
        self._registry_counts = np.array([bank[g][1] for g in ids], dtype=float)

    def _make_global_topics(self) -> list:
        """Copy model_.topics_ with topic_id remapped to stable global IDs.

        Needed so that visualizers that colour-match on topic_id work correctly
        when we pass self.labels_ (global IDs) alongside the topic list.
        """
        remapped = []
        for t in self.model_.topics_:
            t_copy = copy.copy(t)
            if t.topic_id != -1:
                t_copy.topic_id = self._local_id_to_global.get(t.topic_id, t.topic_id)
            remapped.append(t_copy)
        return remapped

    def _require_fitted(self) -> None:
        if self.model_ is None:
            raise ValueError("No model yet. Call add_batch() first.")

    def __repr__(self) -> str:
        return (
            f"CumulativeTriTopic(strategy={self.config.strategy!r}, epoch={self._epoch}, "
            f"docs={len(self._documents)}, global_topics={self.n_global_topics})"
        )
