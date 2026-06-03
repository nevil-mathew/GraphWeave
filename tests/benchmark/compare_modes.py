"""
Compare mode='single' vs mode='streaming' across multiple corpus sizes.
Ground truth: 5 topics with distinct embedding centroids and vocabularies.
Metrics: NMI, ARI, topic count, outlier rate, timing.
Sizes tested: 1k, 2k, 5k, 10k docs.
"""
from __future__ import annotations

import time
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tritopic import TriTopic, TriTopicConfig

D = 64
VOCAB = {
    0: ["finance", "markets", "stocks", "bonds", "trading", "investor", "fund", "yield"],
    1: ["sports", "football", "league", "player", "match", "season", "coach", "score"],
    2: ["politics", "election", "policy", "government", "senate", "vote", "bill", "campaign"],
    3: ["tech", "software", "startup", "cloud", "ai", "platform", "developer", "release"],
    4: ["health", "doctor", "hospital", "patient", "study", "disease", "treatment", "vaccine"],
}

def _topic_center(tid: int) -> np.ndarray:
    g = np.random.default_rng(1000 + tid)
    c = g.normal(size=D)
    return c / np.linalg.norm(c) * 2.0

CENTERS = {tid: _topic_center(tid) for tid in VOCAB}

def gen_docs(topic_dist: dict[int, int], noise: float = 1.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    docs, embs, gt = [], [], []
    for tid, n in topic_dist.items():
        words = VOCAB[tid]
        center = CENTERS[tid]
        for _ in range(n):
            chosen = rng.choice(words, size=rng.integers(8, 16), replace=True)
            docs.append(" ".join(chosen))
            embs.append(center + rng.normal(scale=noise, size=D))
            gt.append(tid)
    embs_arr = np.stack(embs).astype(np.float32)
    perm = rng.permutation(len(docs))
    return [docs[i] for i in perm], embs_arr[perm], np.array(gt)[perm]

def eval_labels(pred_labels: np.ndarray, gt: np.ndarray, n_docs: int) -> dict:
    mask = pred_labels != -1
    n_outliers = int((pred_labels == -1).sum())
    if mask.sum() < 2:
        return {"nmi": 0.0, "ari": 0.0, "n_topics": 0, "outlier_pct": 100.0}
    nmi = normalized_mutual_info_score(gt[mask], pred_labels[mask])
    ari = adjusted_rand_score(gt[mask], pred_labels[mask])
    n_topics = len(set(pred_labels[mask]))
    return {
        "nmi": round(nmi, 4),
        "ari": round(ari, 4),
        "n_topics": n_topics,
        "outlier_pct": round(100 * n_outliers / n_docs, 1),
    }

def run_single(docs, embs, gt):
    cfg = TriTopicConfig(
        mode="single",
        use_dim_reduction=False,
        use_iterative_refinement=True,
        n_consensus_runs=10,
        min_cluster_size=5,
        verbose=False,
    )
    m = TriTopic(config=cfg)
    t0 = time.perf_counter()
    m.fit(docs, embeddings=embs)
    elapsed = time.perf_counter() - t0
    return elapsed, eval_labels(m.labels_, gt, len(docs))

def run_streaming(docs, embs, gt, n_batches=4, refit_every=0):
    n = len(docs)
    batch_size = n // n_batches
    reseed = max(50, batch_size // 5)
    promote_docs = max(50, batch_size // 4)
    cfg = TriTopicConfig(
        mode="streaming",
        use_dim_reduction=False,
        use_iterative_refinement=False,
        n_consensus_runs=5,
        min_cluster_size=5,
        reseed_pool_size=reseed,
        promote_min_batches=2,
        promote_min_docs=promote_docs,
        promote_min_coherence=0.55,
        refit_every_n_batches=refit_every,
        keyword_refresh_every_n_batches=2,
        keyword_refresh_min_new_docs=20,
        verbose=False,
    )
    m = TriTopic(config=cfg)
    t0 = time.perf_counter()
    m.fit(docs[:batch_size], embeddings=embs[:batch_size])
    for b in range(1, n_batches):
        lo, hi = b * batch_size, min((b + 1) * batch_size, n)
        m.add_batch(docs[lo:hi], embeddings=embs[lo:hi])
    elapsed = time.perf_counter() - t0
    backend = m._streaming_backend
    labels = np.array(backend.all_labels_history, dtype=int)
    # Trim to actual doc count in case of rounding
    labels = labels[:n]
    return elapsed, eval_labels(labels, gt, n)

# ── Sizes to test ─────────────────────────────────────────────────────────────
SIZES = [1_000, 2_000, 5_000, 10_000]
N_TOPICS_GT = 5

print(f"\n{'='*80}")
print(f"  TriTopic: single vs streaming — quality & speed across corpus sizes")
print(f"  Noise=1.2, cluster separation=2.0, D={D}, GT topics={N_TOPICS_GT}")
print(f"{'='*80}")

rows = []  # (size, mode, time, n_topics, nmi, ari, outlier_pct)

for size in SIZES:
    per_topic = size // N_TOPICS_GT
    dist = {tid: per_topic for tid in range(N_TOPICS_GT)}
    docs, embs, gt = gen_docs(dist, noise=1.2, seed=42)
    actual_n = len(docs)

    print(f"\n  ── {actual_n:,} docs ──────────────────────────────────────────")

    print(f"    [single]              ", end="", flush=True)
    t, m = run_single(docs, embs, gt)
    print(f"done  {t:.1f}s  topics={m['n_topics']}  NMI={m['nmi']:.3f}  ARI={m['ari']:.3f}  outliers={m['outlier_pct']}%")
    rows.append((actual_n, "single", t, m['n_topics'], m['nmi'], m['ari'], m['outlier_pct']))

    print(f"    [streaming no-refit]  ", end="", flush=True)
    t, m = run_streaming(docs, embs, gt, n_batches=4, refit_every=0)
    print(f"done  {t:.1f}s  topics={m['n_topics']}  NMI={m['nmi']:.3f}  ARI={m['ari']:.3f}  outliers={m['outlier_pct']}%")
    rows.append((actual_n, "stream (no refit)", t, m['n_topics'], m['nmi'], m['ari'], m['outlier_pct']))

    print(f"    [streaming refit/2]   ", end="", flush=True)
    t, m = run_streaming(docs, embs, gt, n_batches=4, refit_every=2)
    print(f"done  {t:.1f}s  topics={m['n_topics']}  NMI={m['nmi']:.3f}  ARI={m['ari']:.3f}  outliers={m['outlier_pct']}%")
    rows.append((actual_n, "stream (refit/2)", t, m['n_topics'], m['nmi'], m['ari'], m['outlier_pct']))

# ── Final table ───────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print(f"  FULL RESULTS TABLE")
print(f"{'='*80}")
print(f"  {'Docs':>7}  {'Mode':<22}  {'Time':>7}  {'Topics':>6}  {'NMI':>6}  {'ARI':>6}  {'Outlier%':>9}")
print(f"  {'-'*7}  {'-'*22}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*9}")
prev_size = None
for (size, mode, t, n_topics, nmi, ari, out) in rows:
    if prev_size is not None and size != prev_size:
        print()
    print(f"  {size:>7,}  {mode:<22}  {t:>6.1f}s  {n_topics:>6}  {nmi:>6.3f}  {ari:>6.3f}  {out:>8}%")
    prev_size = size
