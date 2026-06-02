"""
StreamingTriTopic: Incremental Batch Topic Modeling
===================================================

Wraps a TriTopic instance to support streaming arrivals: an initial batch
seeds the theme catalog via the regular single-shot fit; subsequent batches
are routed against per-theme cosine thresholds (calibrated from batch 1),
with low-confidence docs accumulating in a pool that triggers a sub-fit
when full.  New sub-clusters either merge into existing themes
(0.5 * cosine + 0.5 * jaccard >= merge_threshold) or live in an emerging
buffer until they accumulate enough evidence to graduate.

A periodic full-refit checkpoint (every ``refit_every_n_batches``) uses
Hungarian assignment over centroid cosine to preserve stable theme IDs
while correcting long-term centroid drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from tritopic.core.model import TriTopic, TriTopicConfig


@dataclass
class Theme:
    """A live, drifting theme in the streaming catalog."""

    theme_id: int
    centroid: np.ndarray
    effective_n: float
    keywords: list[str]
    keyword_scores: list[float]
    label: str | None
    true_count: int
    historical_max: int
    batches_seen: set[int]
    created_at_batch: int
    assign_threshold: float
    review_threshold: float
    member_sim_mean: float
    member_sim_std: float
    recent_docs: deque = field(default_factory=lambda: deque(maxlen=2000))
    docs_since_keyword_refresh: int = 0


@dataclass
class EmergingCluster:
    """A candidate not yet promoted to a Theme."""

    centroid: np.ndarray
    keywords: list[str]
    keyword_scores: list[float]
    docs: list[str]
    embeddings: np.ndarray
    batches_seen: set[int]


@dataclass
class PoolDoc:
    """An unassigned document waiting for the next reseed."""

    text: str
    embedding: np.ndarray
    batch_id: int


def _jaccard(a: list[str], b: list[str], top_k: int = 10) -> float:
    sa, sb = set(a[:top_k]), set(b[:top_k])
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _coherence(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine of an embedding set, mapped to [0, 1]."""
    if len(embeddings) < 2:
        return 0.0
    sims = cosine_similarity(embeddings)
    iu = np.triu_indices_from(sims, k=1)
    mean_sim = float(sims[iu].mean())
    return (mean_sim + 1.0) / 2.0


def _single_mode_config(cfg: "TriTopicConfig", **overrides: Any) -> "TriTopicConfig":
    """Clone a config and force mode='single' (used for the internal base fit)."""
    from copy import copy as _copy

    out = _copy(cfg)
    out.mode = "single"
    for k, v in overrides.items():
        setattr(out, k, v)
    return out


class StreamingTriTopic:
    """Streaming wrapper around TriTopic.

    Not meant to be constructed directly by users — use
    ``TriTopic(TriTopicConfig(mode="streaming"))`` and call
    ``fit(first_batch)`` / ``add_batch(next_batch)``.
    """

    def __init__(self, config: "TriTopicConfig"):
        self.config = config
        self.base: "TriTopic" | None = None
        self.themes: dict[int, Theme] = {}
        self.emerging: list[EmergingCluster] = []
        self.unassigned_pool: list[PoolDoc] = []
        self.all_docs_history: list[str] = []
        self.all_embs_history: np.ndarray | None = None
        self.all_labels_history: list[int] = []
        self.batch_counter: int = 0
        self._next_theme_id: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def encode(self, documents: list[str]) -> np.ndarray:
        if self.base is None:
            raise RuntimeError("StreamingTriTopic not seeded. Call fit() first.")
        return self.base.encode(documents)

    def transform(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Route documents through the theme catalog.

        Uses per-theme assign_threshold; anything below it becomes -1 (outlier).
        """
        if not self.themes:
            raise ValueError("No themes. Call fit() first.")

        embs = embeddings if embeddings is not None else self.encode(documents)
        theme_ids, centroid_mat, assign_thr, _ = self._theme_arrays()
        sim = cosine_similarity(embs, centroid_mat)
        nearest = sim.argmax(axis=1)
        top_sim = sim[np.arange(len(documents)), nearest]

        labels = theme_ids[nearest]
        for i, s in enumerate(top_sim):
            if s < assign_thr[nearest[i]]:
                labels[i] = -1
        return labels

    def transform_proba(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        from scipy.special import softmax

        if not self.themes:
            raise ValueError("No themes. Call fit() first.")
        embs = embeddings if embeddings is not None else self.encode(documents)
        _, centroid_mat, _, _ = self._theme_arrays()
        sim = cosine_similarity(embs, centroid_mat)
        return softmax(sim * self.config.softmax_temperature, axis=1)

    def fit_first_batch(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Seed the theme catalog with a full TriTopic.fit on batch 1."""
        from tritopic.core.model import TriTopic

        base_cfg = _single_mode_config(self.config)
        self.base = TriTopic(config=base_cfg)
        self.base.fit(documents, embeddings=embeddings)

        emb = self.base.original_embeddings_
        labels = self.base.labels_
        topics = [t for t in self.base.topics_ if t.topic_id != -1]

        base_to_stream: dict[int, int] = {}  # base topic_id -> streaming theme_id
        for t in topics:
            mask = labels == t.topic_id
            member_embs = emb[mask]
            centroid = self.base.topic_embeddings_[
                [i for i, tt in enumerate(topics) if tt.topic_id == t.topic_id][0]
            ]
            sims = cosine_similarity(member_embs, centroid[None, :]).ravel()
            assign_thr = float(np.percentile(sims, self.config.assign_threshold_percentile))
            review_thr = float(np.percentile(sims, self.config.review_threshold_percentile))

            theme_id = self._allocate_theme_id()
            base_to_stream[t.topic_id] = theme_id
            member_docs = [documents[i] for i in np.flatnonzero(mask)]
            ring: deque = deque(member_docs[-2000:], maxlen=2000)

            theme = Theme(
                theme_id=theme_id,
                centroid=centroid.copy(),
                effective_n=float(t.size),
                keywords=list(t.keywords),
                keyword_scores=list(t.keyword_scores),
                label=t.label,
                true_count=int(t.size),
                historical_max=int(t.size),
                batches_seen={0},
                created_at_batch=0,
                assign_threshold=assign_thr,
                review_threshold=review_thr,
                member_sim_mean=float(sims.mean()),
                member_sim_std=float(sims.std()),
                recent_docs=ring,
                docs_since_keyword_refresh=0,
            )
            self.themes[theme_id] = theme

        # Map batch-1 base labels to streaming theme IDs for visualization history.
        self.all_labels_history = [base_to_stream.get(int(l), -1) for l in labels]

        # Route batch-1 outliers into the pool so they can seed emerging clusters.
        for i in np.flatnonzero(labels == -1):
            self.unassigned_pool.append(
                PoolDoc(text=documents[i], embedding=emb[i], batch_id=0)
            )

        # Record history for the periodic refit checkpoint.
        self.all_docs_history = list(documents)
        self.all_embs_history = emb.copy()
        self.batch_counter = 0

        if self.config.verbose:
            print(
                f"[StreamingTriTopic] batch 0 seeded: "
                f"{len(self.themes)} themes, "
                f"{len(self.unassigned_pool)} pool docs from outliers"
            )

    def add_batch(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
    ) -> dict:
        if self.base is None:
            raise RuntimeError("StreamingTriTopic not seeded. Call fit() first.")
        if not documents:
            return {"assignments": [], "emerged_themes": [], "merged": []}

        self.batch_counter += 1
        batch_id = self.batch_counter
        embs = embeddings if embeddings is not None else self.base.encode(documents)

        # Route against current theme catalog.
        theme_ids, centroid_mat, assign_thr, review_thr = self._theme_arrays()
        sim = cosine_similarity(embs, centroid_mat)
        nearest = sim.argmax(axis=1)
        top_sim = sim[np.arange(len(documents)), nearest]

        direct: dict[int, list[int]] = {tid: [] for tid in theme_ids}
        review: dict[int, list[int]] = {tid: [] for tid in theme_ids}
        assignments: list[dict] = []

        for i in range(len(documents)):
            s = float(top_sim[i])
            tid = int(theme_ids[nearest[i]])
            if s >= assign_thr[nearest[i]]:
                direct[tid].append(i)
                assignments.append(
                    {"doc_idx": i, "theme_id": tid, "confidence": s, "status": "assigned"}
                )
            elif s >= review_thr[nearest[i]]:
                review[tid].append(i)
                assignments.append(
                    {"doc_idx": i, "theme_id": tid, "confidence": s, "status": "needs_review"}
                )
            else:
                self.unassigned_pool.append(
                    PoolDoc(text=documents[i], embedding=embs[i], batch_id=batch_id)
                )
                assignments.append(
                    {"doc_idx": i, "theme_id": None, "confidence": s, "status": "unassigned"}
                )

        # Update centroids + counts for themes that got hits.
        for tid, direct_idxs in direct.items():
            review_idxs = review[tid]
            if not direct_idxs and not review_idxs:
                continue
            theme = self.themes[tid]
            if direct_idxs:
                self._update_centroid(theme, embs[direct_idxs])
                # Feed the keyword ring buffer.
                for j in direct_idxs:
                    theme.recent_docs.append(documents[j])
                theme.docs_since_keyword_refresh += len(direct_idxs)
            theme.true_count += len(direct_idxs) + len(review_idxs)
            theme.historical_max = max(theme.historical_max, theme.true_count)
            theme.batches_seen.add(batch_id)

        # Accumulate raw history for the periodic refit and visualization.
        self.all_docs_history.extend(documents)
        self.all_embs_history = (
            embs.copy()
            if self.all_embs_history is None
            else np.vstack([self.all_embs_history, embs])
        )
        self.all_labels_history.extend(
            a["theme_id"] if a["theme_id"] is not None else -1
            for a in assignments
        )

        # Reseed if pool is full.
        merged: list[tuple[int, int]] = []
        if len(self.unassigned_pool) >= self.config.reseed_pool_size:
            merged = self._reseed_pool(batch_id)

        # Promote emerging clusters that meet the gate.
        emerged_themes: list[int] = []
        promoted_indices: list[int] = []
        for idx, ec in enumerate(self.emerging):
            if self._can_promote(ec):
                new_id = self._promote_emerging(ec, batch_id)
                emerged_themes.append(new_id)
                promoted_indices.append(idx)
        for idx in reversed(promoted_indices):
            self.emerging.pop(idx)

        # Periodic keyword refresh.
        if (
            self.config.keyword_refresh_every_n_batches > 0
            and batch_id % self.config.keyword_refresh_every_n_batches == 0
        ):
            self._refresh_keywords()

        # Periodic full refit.
        if (
            self.config.refit_every_n_batches > 0
            and batch_id % self.config.refit_every_n_batches == 0
        ):
            self._periodic_refit()

        if self.config.verbose:
            n_direct = sum(len(v) for v in direct.values())
            n_review = sum(len(v) for v in review.values())
            n_pool = len(documents) - n_direct - n_review
            print(
                f"[StreamingTriTopic] batch {batch_id}: "
                f"{n_direct} assigned, {n_review} review, {n_pool} pooled | "
                f"themes={len(self.themes)} emerging={len(self.emerging)} "
                f"pool={len(self.unassigned_pool)}"
            )

        return {
            "assignments": assignments,
            "emerged_themes": emerged_themes,
            "merged": merged,
        }

    def themes_view(self) -> dict[int, dict]:
        """Snapshot of every theme for diagnostics / demos."""
        return {
            tid: {
                "label": t.label or f"Theme {tid}",
                "keywords": list(t.keywords),
                "true_count": t.true_count,
                "public_count": max(t.true_count, t.historical_max),
                "assign_threshold": t.assign_threshold,
                "review_threshold": t.review_threshold,
                "batches_seen": sorted(t.batches_seen),
                "effective_n": t.effective_n,
            }
            for tid, t in sorted(self.themes.items())
        }

    def visualize(
        self,
        method: str = "umap",
        show_outliers: bool = True,
        interactive: bool = True,
        **kwargs,
    ):
        """Visualize all accumulated docs coloured by their streaming theme."""
        from tritopic.visualization.plotter import TopicVisualizer
        from tritopic.core.model import TopicInfo

        if not self.themes:
            raise ValueError("No themes. Call fit() first.")
        if self.all_embs_history is None:
            raise ValueError("No embedding history. Call fit() first.")

        topics = [
            TopicInfo(
                topic_id=t.theme_id,
                size=t.true_count,
                keywords=list(t.keywords),
                keyword_scores=list(t.keyword_scores),
                representative_docs=[],
                label=t.label,
                centroid=t.centroid,
            )
            for t in sorted(self.themes.values(), key=lambda x: x.theme_id)
        ]

        labels = np.array(self.all_labels_history, dtype=int)
        visualizer = TopicVisualizer(method=method)
        return visualizer.plot_documents(
            embeddings=self.all_embs_history,
            labels=labels,
            documents=self.all_docs_history,
            topics=topics,
            show_outliers=show_outliers,
            interactive=interactive,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _allocate_theme_id(self) -> int:
        tid = self._next_theme_id
        self._next_theme_id += 1
        return tid

    def _theme_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ordered = sorted(self.themes.values(), key=lambda t: t.theme_id)
        ids = np.array([t.theme_id for t in ordered], dtype=int)
        centroids = np.stack([t.centroid for t in ordered])
        assign = np.array([t.assign_threshold for t in ordered], dtype=float)
        review = np.array([t.review_threshold for t in ordered], dtype=float)
        return ids, centroids, assign, review

    def _update_centroid(self, theme: Theme, batch_embs: np.ndarray) -> None:
        n_new = len(batch_embs)
        batch_mean = batch_embs.mean(axis=0)
        old_w = theme.effective_n * self.config.centroid_decay
        new_w = float(n_new)
        theme.centroid = (old_w * theme.centroid + new_w * batch_mean) / (old_w + new_w)
        theme.effective_n = old_w + new_w

    def _reseed_pool(self, batch_id: int) -> list[tuple[int, int]]:
        """Run a sub-fit on the unassigned pool; merge or push to emerging."""
        from tritopic.core.model import TriTopic

        pool = self.unassigned_pool
        pool_docs = [p.text for p in pool]
        pool_embs = np.stack([p.embedding for p in pool])

        sub_cfg = _single_mode_config(
            self.config,
            n_consensus_runs=min(self.config.n_consensus_runs, 5),
            use_iterative_refinement=False,
            verbose=False,
        )
        sub = TriTopic(config=sub_cfg)
        try:
            sub.fit(pool_docs, embeddings=pool_embs)
        except Exception as e:
            if self.config.verbose:
                print(f"[StreamingTriTopic] reseed sub-fit failed ({e}); keeping pool")
            return []

        merged: list[tuple[int, int]] = []
        for sub_topic_idx, t in enumerate([t for t in sub.topics_ if t.topic_id != -1]):
            mask = sub.labels_ == t.topic_id
            c_embs = pool_embs[mask]
            c_docs = [pool_docs[i] for i in np.flatnonzero(mask)]
            # Sub.topic_embeddings_ is ordered by non-outlier topic id, same as topics_.
            c_centroid = sub.topic_embeddings_[sub_topic_idx]

            # Find best existing-theme match.
            best_tid, best_score = None, -1.0
            for tid, theme in self.themes.items():
                cos = float(
                    cosine_similarity(c_centroid[None, :], theme.centroid[None, :])[0, 0]
                )
                jac = _jaccard(t.keywords, theme.keywords, top_k=10)
                score = 0.5 * cos + 0.5 * jac
                if score > best_score:
                    best_score, best_tid = score, tid

            if best_tid is not None and best_score >= self.config.merge_threshold:
                theme = self.themes[best_tid]
                self._update_centroid(theme, c_embs)
                theme.true_count += len(c_docs)
                theme.historical_max = max(theme.historical_max, theme.true_count)
                theme.batches_seen.add(batch_id)
                for d in c_docs:
                    theme.recent_docs.append(d)
                theme.docs_since_keyword_refresh += len(c_docs)
                merged.append((sub_topic_idx, best_tid))
                continue

            # Push as emerging — try to merge with an existing emerging first.
            absorbed = False
            for ec in self.emerging:
                if float(
                    cosine_similarity(c_centroid[None, :], ec.centroid[None, :])[0, 0]
                ) >= self.config.emerging_merge_threshold:
                    new_embs = np.vstack([ec.embeddings, c_embs])
                    ec.centroid = new_embs.mean(axis=0)
                    ec.docs.extend(c_docs)
                    ec.embeddings = new_embs
                    ec.batches_seen.add(batch_id)
                    absorbed = True
                    break
            if not absorbed:
                self.emerging.append(
                    EmergingCluster(
                        centroid=c_centroid.copy(),
                        keywords=list(t.keywords),
                        keyword_scores=list(t.keyword_scores),
                        docs=c_docs,
                        embeddings=c_embs.copy(),
                        batches_seen={batch_id},
                    )
                )

        self.unassigned_pool = []
        return merged

    def _can_promote(self, ec: EmergingCluster) -> bool:
        if len(ec.batches_seen) < self.config.promote_min_batches:
            return False
        if len(ec.docs) < self.config.promote_min_docs:
            return False
        if _coherence(ec.embeddings) < self.config.promote_min_coherence:
            return False
        return True

    def _promote_emerging(self, ec: EmergingCluster, batch_id: int) -> int:
        sims = cosine_similarity(ec.embeddings, ec.centroid[None, :]).ravel()
        assign_thr = float(np.percentile(sims, self.config.assign_threshold_percentile))
        review_thr = float(np.percentile(sims, self.config.review_threshold_percentile))
        tid = self._allocate_theme_id()
        ring: deque = deque(ec.docs[-2000:], maxlen=2000)
        self.themes[tid] = Theme(
            theme_id=tid,
            centroid=ec.centroid.copy(),
            effective_n=float(len(ec.docs)),
            keywords=list(ec.keywords),
            keyword_scores=list(ec.keyword_scores),
            label=None,
            true_count=len(ec.docs),
            historical_max=len(ec.docs),
            batches_seen=set(ec.batches_seen),
            created_at_batch=batch_id,
            assign_threshold=assign_thr,
            review_threshold=review_thr,
            member_sim_mean=float(sims.mean()),
            member_sim_std=float(sims.std()),
            recent_docs=ring,
            docs_since_keyword_refresh=0,
        )
        if self.config.verbose:
            print(f"[StreamingTriTopic] emerging cluster promoted -> theme {tid}")
        return tid

    def _refresh_keywords(self) -> None:
        """Re-extract keywords for themes that have accumulated enough new docs."""
        min_new = self.config.keyword_refresh_min_new_docs
        candidates = [
            t for t in self.themes.values()
            if t.docs_since_keyword_refresh >= min_new and len(t.recent_docs) > 0
        ]
        if not candidates:
            return

        # Pool of recent docs across all candidate themes provides IDF context.
        recent_corpus: list[str] = []
        for t in candidates:
            recent_corpus.extend(t.recent_docs)
        if len(recent_corpus) < 2:
            return

        extractor = self.base._keyword_extractor
        extractor.reset()
        for theme in candidates:
            try:
                kws, scores = extractor.extract(
                    topic_docs=list(theme.recent_docs),
                    all_docs=recent_corpus,
                )
                theme.keywords = list(kws)
                theme.keyword_scores = list(scores)
            except Exception as e:
                if self.config.verbose:
                    print(
                        f"[StreamingTriTopic] keyword refresh failed for "
                        f"theme {theme.theme_id}: {e}"
                    )
            theme.docs_since_keyword_refresh = 0
        # Reset the extractor again so the base model's own state isn't polluted.
        extractor.reset()

    def _periodic_refit(self) -> None:
        """Full re-fit on cumulative history; Hungarian re-ID against current themes."""
        from scipy.optimize import linear_sum_assignment

        from tritopic.core.model import TriTopic

        if self.all_embs_history is None or len(self.all_docs_history) < 2:
            return

        refit_cfg = _single_mode_config(self.config, verbose=False)
        refit = TriTopic(config=refit_cfg)
        try:
            refit.fit(self.all_docs_history, embeddings=self.all_embs_history)
        except Exception as e:
            if self.config.verbose:
                print(f"[StreamingTriTopic] periodic refit failed ({e}); skipping")
            return

        new_topics = [t for t in refit.topics_ if t.topic_id != -1]
        if not new_topics:
            return
        new_centroids = np.stack(
            [refit.topic_embeddings_[i] for i, _ in enumerate(new_topics)]
        )

        existing_ids = sorted(self.themes.keys())
        existing_centroids = np.stack([self.themes[i].centroid for i in existing_ids])

        # Cost = -cosine (Hungarian minimises).
        sim = cosine_similarity(new_centroids, existing_centroids)
        cost = -sim
        # Pad to square for linear_sum_assignment by extending the shorter axis with
        # zeros (= no preference for unmatched slots).
        n_new, n_exist = sim.shape
        size = max(n_new, n_exist)
        padded = np.zeros((size, size))
        padded[:n_new, :n_exist] = cost
        row_ind, col_ind = linear_sum_assignment(padded)

        used_existing: set[int] = set()
        used_new: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if r >= n_new or c >= n_exist:
                continue
            if sim[r, c] < 0.5:
                continue
            tid = existing_ids[c]
            self._reassign_theme_from_refit(tid, new_topics[r], refit, r)
            used_existing.add(c)
            used_new.add(r)

        # Unmatched new topics → new themes.
        for r, t in enumerate(new_topics):
            if r in used_new:
                continue
            self._register_theme_from_refit(t, refit, r)
        # Unmatched existing themes are left alone (counts already capture them).

        if self.config.verbose:
            print(
                f"[StreamingTriTopic] refit done: "
                f"{len(used_existing)} re-IDed, "
                f"{n_new - len(used_new)} new themes added"
            )

    def _reassign_theme_from_refit(
        self,
        theme_id: int,
        topic_info: Any,
        refit_model: "TriTopic",
        topic_idx_in_refit: int,
    ) -> None:
        theme = self.themes[theme_id]
        labels = refit_model.labels_
        emb = refit_model.original_embeddings_
        mask = labels == topic_info.topic_id
        member_embs = emb[mask]
        new_centroid = refit_model.topic_embeddings_[topic_idx_in_refit]
        sims = cosine_similarity(member_embs, new_centroid[None, :]).ravel()
        theme.centroid = new_centroid.copy()
        theme.effective_n = float(topic_info.size)
        theme.keywords = list(topic_info.keywords)
        theme.keyword_scores = list(topic_info.keyword_scores)
        theme.assign_threshold = float(
            np.percentile(sims, self.config.assign_threshold_percentile)
        )
        theme.review_threshold = float(
            np.percentile(sims, self.config.review_threshold_percentile)
        )
        theme.member_sim_mean = float(sims.mean())
        theme.member_sim_std = float(sims.std())
        theme.historical_max = max(theme.historical_max, int(topic_info.size))
        theme.docs_since_keyword_refresh = 0

    def _register_theme_from_refit(
        self,
        topic_info: Any,
        refit_model: "TriTopic",
        topic_idx_in_refit: int,
    ) -> None:
        labels = refit_model.labels_
        emb = refit_model.original_embeddings_
        mask = labels == topic_info.topic_id
        member_embs = emb[mask]
        new_centroid = refit_model.topic_embeddings_[topic_idx_in_refit]
        sims = cosine_similarity(member_embs, new_centroid[None, :]).ravel()
        tid = self._allocate_theme_id()
        member_docs = [
            self.all_docs_history[i] for i in np.flatnonzero(mask)
        ]
        ring: deque = deque(member_docs[-2000:], maxlen=2000)
        self.themes[tid] = Theme(
            theme_id=tid,
            centroid=new_centroid.copy(),
            effective_n=float(topic_info.size),
            keywords=list(topic_info.keywords),
            keyword_scores=list(topic_info.keyword_scores),
            label=topic_info.label,
            true_count=int(topic_info.size),
            historical_max=int(topic_info.size),
            batches_seen={self.batch_counter},
            created_at_batch=self.batch_counter,
            assign_threshold=float(
                np.percentile(sims, self.config.assign_threshold_percentile)
            ),
            review_threshold=float(
                np.percentile(sims, self.config.review_threshold_percentile)
            ),
            member_sim_mean=float(sims.mean()),
            member_sim_std=float(sims.std()),
            recent_docs=ring,
            docs_since_keyword_refresh=0,
        )

    # ------------------------------------------------------------------ #
    # Persistence helpers (used by TriTopic.save / TriTopic.load)
    # ------------------------------------------------------------------ #

    def to_state(self) -> dict:
        return {
            "themes": list(self.themes.values()),
            "emerging": list(self.emerging),
            "unassigned_pool": list(self.unassigned_pool),
            "all_docs_history": list(self.all_docs_history),
            "all_embs_history": self.all_embs_history,
            "all_labels_history": list(self.all_labels_history),
            "batch_counter": self.batch_counter,
            "next_theme_id": self._next_theme_id,
            "base_state": _save_base_to_dict(self.base) if self.base is not None else None,
        }

    @classmethod
    def from_state(cls, state: dict, config: "TriTopicConfig") -> "StreamingTriTopic":
        inst = cls(config)
        inst.themes = {t.theme_id: t for t in state["themes"]}
        inst.emerging = list(state["emerging"])
        inst.unassigned_pool = list(state["unassigned_pool"])
        inst.all_docs_history = list(state["all_docs_history"])
        inst.all_embs_history = state["all_embs_history"]
        inst.all_labels_history = list(state.get("all_labels_history", []))
        inst.batch_counter = int(state["batch_counter"])
        inst._next_theme_id = int(state["next_theme_id"])
        if state["base_state"] is not None:
            inst.base = _load_base_from_dict(state["base_state"])
        return inst


def _save_base_to_dict(base: "TriTopic") -> dict:
    """Pickle-friendly snapshot of the internal base TriTopic.

    We only need enough state to call encode() / use _keyword_extractor.
    """
    return {
        "config": base.config,
        "topics_": base.topics_,
        "labels_": base.labels_,
        "original_embeddings_": base.original_embeddings_,
        "topic_embeddings_": base.topic_embeddings_,
        "_is_fitted": base._is_fitted,
    }


def _load_base_from_dict(d: dict) -> "TriTopic":
    from tritopic.core.model import TriTopic

    cfg = d["config"]
    cfg.mode = "single"
    base = TriTopic(config=cfg)
    base.topics_ = d["topics_"]
    base.labels_ = d["labels_"]
    base.original_embeddings_ = d["original_embeddings_"]
    base.topic_embeddings_ = d["topic_embeddings_"]
    base._is_fitted = d["_is_fitted"]
    return base
