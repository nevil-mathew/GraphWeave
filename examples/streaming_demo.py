"""
StreamingTriTopic demo: synthetic Gaussian blobs over 4 batches.

Generates synthetic documents and embeddings (no real embedding model needed).
Batches 0-1 sample the 5 base distributions; batch 2 injects a brand-new 6th
distribution to exercise the unassigned-pool -> reseed -> emerging -> promote
pipeline.
"""

from __future__ import annotations

import numpy as np

from tritopic import TriTopic, TriTopicConfig

D = 64  # embedding dimensionality (small for speed; real models are 384/768)
N_BASE_TOPICS = 5
RNG = np.random.default_rng(42)

VOCAB = {
    0: ["finance", "markets", "stocks", "bonds", "trading", "investor", "fund", "yield"],
    1: ["sports", "football", "league", "player", "match", "season", "coach", "score"],
    2: ["politics", "election", "policy", "government", "senate", "vote", "bill", "campaign"],
    3: ["tech", "software", "startup", "cloud", "ai", "platform", "developer", "release"],
    4: ["health", "doctor", "hospital", "patient", "study", "disease", "treatment", "vaccine"],
    5: ["climate", "carbon", "emissions", "renewable", "solar", "wind", "warming", "policy"],
}


def _topic_center(topic_id: int) -> np.ndarray:
    g = np.random.default_rng(1000 + topic_id)
    c = g.normal(size=D)
    return c / np.linalg.norm(c) * 5.0


CENTERS = {tid: _topic_center(tid) for tid in VOCAB}


def gen_batch(topic_distribution: dict[int, int], noise: float = 0.4) -> tuple[list[str], np.ndarray]:
    docs: list[str] = []
    embs: list[np.ndarray] = []
    for topic_id, n in topic_distribution.items():
        words = VOCAB[topic_id]
        center = CENTERS[topic_id]
        for _ in range(n):
            length = RNG.integers(8, 16)
            chosen = RNG.choice(words, size=length, replace=True)
            docs.append(" ".join(chosen))
            embs.append(center + RNG.normal(scale=noise, size=D))
    embs_arr = np.stack(embs).astype(np.float32)
    perm = RNG.permutation(len(docs))
    return [docs[i] for i in perm], embs_arr[perm]


def print_themes(model: TriTopic, header: str) -> None:
    view = model._streaming_backend.themes_view()
    print(f"\n=== {header} ===")
    print(f"  themes: {len(view)}, emerging: {len(model._streaming_backend.emerging)}, "
          f"pool: {len(model._streaming_backend.unassigned_pool)}")
    print(f"  {'tid':<4} {'true':<6} {'public':<7} {'assign':<7} {'review':<7} "
          f"{'batches':<14} keywords")
    for tid, info in view.items():
        kw = ", ".join(info["keywords"][:5]) if info["keywords"] else ""
        print(
            f"  {tid:<4} {info['true_count']:<6} {info['public_count']:<7} "
            f"{info['assign_threshold']:<7.3f} {info['review_threshold']:<7.3f} "
            f"{str(info['batches_seen']):<14} {kw}"
        )


def main() -> None:
    cfg = TriTopicConfig(
        mode="streaming",
        use_dim_reduction=False,
        use_iterative_refinement=False,
        n_consensus_runs=3,
        min_cluster_size=10,
        reseed_pool_size=120,
        promote_min_batches=2,
        promote_min_docs=150,
        promote_min_coherence=0.55,
        refit_every_n_batches=0,  # disable periodic refit for a short demo
        keyword_refresh_every_n_batches=1,
        keyword_refresh_min_new_docs=20,
        verbose=True,
    )
    model = TriTopic(config=cfg)

    # Batch 0: 5 base topics, ~120 docs each.
    docs0, embs0 = gen_batch({0: 120, 1: 120, 2: 120, 3: 120, 4: 120})
    model.fit(docs0, embeddings=embs0)
    print_themes(model, "after batch 0 (initial fit)")

    # Batch 1: same 5 topics, lighter draws.
    docs1, embs1 = gen_batch({0: 40, 1: 40, 2: 40, 3: 40, 4: 40})
    model.add_batch(docs1, embeddings=embs1)
    print_themes(model, "after batch 1 (drift)")

    # Batch 2: 6th brand-new topic dominates; small share of existing.
    docs2, embs2 = gen_batch({5: 100, 0: 10, 1: 10})
    model.add_batch(docs2, embeddings=embs2)
    print_themes(model, "after batch 2 (new topic injected)")

    # Batch 3: more of the new topic to push emerging -> promotion.
    docs3, embs3 = gen_batch({5: 100, 2: 10, 3: 10})
    model.add_batch(docs3, embeddings=embs3)
    print_themes(model, "after batch 3 (emerging promoted)")

    print("\nDone. public_count never decreases across the run.")


if __name__ == "__main__":
    main()
