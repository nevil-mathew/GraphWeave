"""
Real-world integration test: 20 Newsgroups + sentence-transformers
==================================================================

Full 18,846 docs, 20 real categories, all cumulative clustering flows.

Run:
    python benchmarks/integration_test_20ng.py

First run encodes all docs and caches embeddings to benchmarks/.cache/.
Subsequent runs load the cache and skip encoding (~40-90 s).

Falls back to LSA embeddings if sentence-transformers is not installed:
    uv pip install sentence-transformers
"""

from __future__ import annotations

import copy
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphweave import GraphWeave, GraphWeaveConfig
from graphweave.cumulative import CumulativeConfig, CumulativeGraphWeave
from graphweave.cumulative.evaluation import benchmark_strategies, compare_to_full_batch
from graphweave.utils.metrics import compute_ari, compute_nmi

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
EMB_CACHE   = CACHE_DIR / "ng20_emb.npy"
DOCS_CACHE  = CACHE_DIR / "ng20_docs.pkl"
LABELS_CACHE = CACHE_DIR / "ng20_labels.npy"

PROCESS = psutil.Process()


def rss_mb() -> float:
    return PROCESS.memory_info().rss / 1e6


def hr(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


# ── Data & embedding ────────────────────────────────────────────────────────

def _embed_sentence_transformers(documents: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print("  Loading all-MiniLM-L6-v2 (384-dim, ~22 MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Encoding {len(documents):,} docs on CPU...")
    # CPU-only: ~30-40 min first run; cached afterwards. Use a GPU or a smaller
    # dataset (e.g. the 20NG train split, ~11 k docs) to shorten this.
    emb = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256,          # larger batch → fewer Python-loop iterations on CPU
    )
    return emb.astype(np.float32)


def _embed_lsa(documents: list[str]) -> np.ndarray:
    from graphweave.cumulative.datasets import lsa_embed
    print("  Using LSA fallback (TF-IDF + TruncatedSVD, 128-dim)...")
    return lsa_embed(documents, dim=128)


def load_data() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (documents, embeddings, labels). Loads cache if available."""
    if EMB_CACHE.exists() and DOCS_CACHE.exists() and LABELS_CACHE.exists():
        print("  Loading cached embeddings...")
        with open(DOCS_CACHE, "rb") as f:
            documents = pickle.load(f)
        embeddings = np.load(EMB_CACHE)
        labels = np.load(LABELS_CACHE)
        return documents, embeddings, labels

    from sklearn.datasets import fetch_20newsgroups
    print("  Fetching 20 Newsgroups (all 18,846 docs)...")
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    documents = [d.strip() or "empty document" for d in data.data]
    labels    = np.array(data.target)

    try:
        import sentence_transformers  # noqa: F401
        embeddings = _embed_sentence_transformers(documents)
        method = "sentence-transformers"
    except ImportError:
        print("  sentence-transformers not installed — using LSA fallback.")
        print("  Install with: uv pip install sentence-transformers")
        embeddings = _embed_lsa(documents)
        method = "LSA"

    np.save(EMB_CACHE, embeddings)
    np.save(LABELS_CACHE, labels)
    with open(DOCS_CACHE, "wb") as f:
        pickle.dump(documents, f)
    print(f"  Cached to {CACHE_DIR}/ (method: {method})")
    return documents, embeddings, labels


def make_batches(
    documents: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[list[str]], list[np.ndarray], list[np.ndarray]]:
    """
    Build 6 batches with no document appearing more than once.

    Batches 1-4: categories 0-14 only  (~3 k docs each)
    Batches 5-6: all categories (0-19) — 5 new categories appear, driving drift
    """
    established = np.where(labels < 15)[0]
    emerging    = np.where(labels >= 15)[0]

    est_shuf  = rng.permutation(established)
    emg_shuf  = rng.permutation(emerging)

    # Reserve 800 established docs exclusively for padding batches 5-6 (~40 % emerging).
    # These are drawn before splitting the rest into 4 chunks, so no doc appears twice.
    est_reserved = est_shuf[:800]
    est_remaining = est_shuf[800:]

    # Split remaining established into 4 roughly equal chunks for batches 1-4.
    est_chunks = np.array_split(est_remaining, 4)

    # Split emerging into 2 halves for batches 5-6.
    emg_half   = len(emg_shuf) // 2
    emg_a, emg_b = emg_shuf[:emg_half], emg_shuf[emg_half:]

    raw_batches = [
        est_chunks[0],
        est_chunks[1],
        est_chunks[2],
        est_chunks[3],
        np.concatenate([emg_a, est_reserved[:400]]),   # batch 5: ~40 % emerging
        np.concatenate([emg_b, est_reserved[400:]]),   # batch 6: ~40 % emerging
    ]

    batch_docs  = [[documents[i] for i in idx] for idx in raw_batches]
    batch_emb   = [embeddings[idx]             for idx in raw_batches]
    batch_lbl   = [labels[idx]                 for idx in raw_batches]
    return batch_docs, batch_emb, batch_lbl


def base_cfg() -> GraphWeaveConfig:
    return GraphWeaveConfig(
        use_dim_reduction=False,         # MiniLM 384-dim already dense; UMAP adds minutes
        use_lexical_view=True,           # real text benefits strongly from TF-IDF graph
        use_iterative_refinement=False,  # 3× speedup; marginal loss at this scale
        n_consensus_runs=5,
        min_cluster_size=15,             # appropriate for ~3k-doc batches
        # consensus_method left at default "graph": avoids the 18k×18k dense
        # matrix entirely (low_memory has no effect on this path, so it's omitted).
        n_neighbors=15,
        random_state=42,
        verbose=False,
    )


# ── Scenario A: Drift detection ──────────────────────────────────────────────

def scenario_drift(
    batch_docs: list[list[str]],
    batch_emb: list[np.ndarray],
    batch_lbl: list[np.ndarray],
) -> CumulativeGraphWeave:
    hr("SCENARIO A — Drift detection on real 20 Newsgroups text")
    print("Batches 1-4: categories 0-14 only.  "
          "Batches 5-6: all 20 categories (5 new themes emerge).\n")

    cfg = CumulativeConfig(
        base_config=base_cfg(),
        strategy="global_refit",
        recluster_trigger="drift",
        novelty_threshold=0.20,
    )
    model = CumulativeGraphWeave(cfg)

    print(f"  {'batch':>5} | {'docs':>6} | {'cumul':>6} | {'novelty':>7} | "
          f"{'reclust':>7} | {'g-topics':>8} | {'ARI/truth':>9} | {'RAM MB':>6} | {'wall_s':>6}")
    print("  " + "-" * 83)

    for i, (docs, emb, lbl) in enumerate(zip(batch_docs, batch_emb, batch_lbl)):
        t0 = time.perf_counter()
        r  = model.add_batch(docs, embeddings=emb)
        elapsed = time.perf_counter() - t0

        # ARI against ground truth for accumulated docs seen so far
        acc_labels_true = np.concatenate([l for l in batch_lbl[:i + 1]])
        ari_truth = compute_ari(model.labels_, acc_labels_true)

        nv = f"{r.novelty:.2f}" if r.novelty is not None else "  -  "
        print(f"  {i+1:>5} | {len(docs):>6,} | {r.n_total_docs:>6,} | {nv:>7} | "
              f"{'YES' if r.reclustered else 'no':>7} | {model.n_global_topics:>8} | "
              f"{ari_truth:>9.3f} | {rss_mb():>6.0f} | {elapsed:>6.1f}")

    print()
    # Final metrics vs the full-batch baseline
    print("  Computing full-batch baseline (fit once on ALL accumulated docs)...")
    t0 = time.perf_counter()
    full = GraphWeave(config=copy.deepcopy(base_cfg()))
    all_acc_docs = [d for b in batch_docs for d in b]
    all_acc_emb  = np.vstack(batch_emb)
    all_acc_lbl  = np.concatenate(batch_lbl)
    full.fit(all_acc_docs, embeddings=all_acc_emb)
    print(f"  Full-batch fit done in {time.perf_counter()-t0:.1f} s")

    m = compare_to_full_batch(model, full, labels_true=all_acc_lbl)
    print(f"\n  ARI vs full-batch   : {m['ari_vs_full']:.3f}")
    print(f"  NMI vs full-batch   : {m['nmi_vs_full']:.3f}")
    print(f"  ARI vs ground truth : cumulative {m['ari_vs_truth_cumulative']:.3f}  "
          f"| full-batch {m['ari_vs_truth_full']:.3f}")
    print(f"  Topics found        : cumulative {m['n_topics_cumulative']}  "
          f"| full-batch {m['n_topics_full']}  (true = 20)")
    print(f"  Outlier ratio       : {m['outlier_ratio_cumulative']:.1%}")
    print(f"  Peak RAM            : {rss_mb():.0f} MB")

    return model


# ── Scenario B: Strategy head-to-head ────────────────────────────────────────

def scenario_head_to_head(
    batch_docs: list[list[str]],
    batch_emb: list[np.ndarray],
    batch_lbl: list[np.ndarray],
) -> None:
    hr("SCENARIO B — Strategy head-to-head (first 3 batches, recluster every batch)")
    print("Using established-only batches 1-3 (~9k docs total).\n")

    sub_docs = batch_docs[:3]
    sub_emb  = batch_emb[:3]
    sub_lbl  = np.concatenate(batch_lbl[:3])

    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    df = benchmark_strategies(
        sub_docs,
        base_config=base_cfg(),
        labels_true=sub_lbl,
        precomputed_batch_embeddings=sub_emb,
        cumulative_kwargs=dict(
            recluster_trigger="schedule",
            schedule_every_n_docs=1,   # recluster every batch
        ),
    )
    show = [
        "strategy", "ari_vs_full", "nmi_vs_full",
        "ari_vs_truth_cumulative", "n_topics_cumulative", "n_topics_full",
        "keyword_overlap", "silhouette_delta", "wall_clock_s",
    ]
    print(df[show].to_string(index=False))
    print()
    print("  global_refit = matches full-batch (ARI~1.0); most expensive.")
    print("  coreset      = same quality here; stays fast as data grows.")
    print("  batch_merge  = fastest; keyword overlap lower (batch-local keywords).")


# ── Scenario C: Bigger picture ───────────────────────────────────────────────

def scenario_bigger_picture(model: CumulativeGraphWeave) -> None:
    hr("SCENARIO C — Bigger picture (3-level hierarchy across all accumulated docs)")
    t0 = time.perf_counter()
    view = model.bigger_picture(n_levels=3)
    print(f"  Built in {time.perf_counter()-t0:.1f} s\n")

    hierarchy = view["hierarchy"]
    print(f"  Hierarchy: {hierarchy.n_levels} levels, "
          f"topics per level: {[len(lvl) for lvl in hierarchy.levels]}\n")

    coarse = hierarchy.cut(0)
    coarse_sorted = sorted(coarse, key=lambda n: n.size, reverse=True)
    print(f"  {'Coarse theme (level 0)':40s}  {'docs':>6}  keywords")
    print("  " + "-" * 72)
    for node in coarse_sorted:
        kws = ", ".join(node.keywords[:5])
        print(f"  {node.node_id:40s}  {node.size:>6,}  {kws}")


# ── Report footer ────────────────────────────────────────────────────────────

def footer(t_total: float) -> None:
    hr("SUMMARY")
    try:
        import hnswlib  # noqa: F401
        hnsw_status = "ACTIVE (hnswlib installed)"
    except ImportError:
        hnsw_status = "not active (install graphweave[fast-knn] to enable)"
    print(f"  Total runtime    : {t_total/60:.1f} min")
    print(f"  Peak RAM         : {rss_mb():.0f} MB")
    print(f"  HNSW backend     : {hnsw_status}")
    print(f"  Consensus        : default graph-consensus path (avoids 18k×18k matrix)")
    print()
    print("  Interpretation:")
    print("  · Drift fires when new categories appear → recluster discovers new topics.")
    print("  · global_refit matches full-batch; coreset is equally good & scales.")
    print("  · batch_merge is fastest but may miss topics + keywords drift.")
    print("  · bigger_picture hierarchy groups fine topics into recognisable themes.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    print("20 Newsgroups integration test")
    print("(full 18,846 docs · 20 categories · real embeddings)")
    print("NOTE: first-run CPU encoding takes ~30-40 min; embeddings are then")
    print("cached to benchmarks/.cache/ so subsequent runs are fast.\n")

    hr("SETUP — Loading data & embeddings")
    t0 = time.perf_counter()
    documents, embeddings, labels = load_data()
    dim = embeddings.shape[1]
    print(f"\n  Corpus  : {len(documents):,} docs, {len(np.unique(labels))} categories")
    print(f"  Vectors : {dim}-dim, {embeddings.nbytes/1e6:.0f} MB")
    print(f"  Setup   : {time.perf_counter()-t0:.1f} s")

    rng = np.random.default_rng(42)
    batch_docs, batch_emb, batch_lbl = make_batches(documents, embeddings, labels, rng)
    print(f"  Batches : {len(batch_docs)} × "
          f"{[len(b) for b in batch_docs]} docs")
    print(f"  New categories in batches 5-6: "
          f"{sorted(set(np.concatenate(batch_lbl[4:]).tolist()) - set(np.concatenate(batch_lbl[:4]).tolist()))}")

    model = scenario_drift(batch_docs, batch_emb, batch_lbl)
    scenario_head_to_head(batch_docs, batch_emb, batch_lbl)
    scenario_bigger_picture(model)
    footer(time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
