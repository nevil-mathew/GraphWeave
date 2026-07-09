<p align="center">
  <img src="assets/logo.png" alt="GraphWeave logo" width="180">
</p>

# GraphWeave

**Multi-view graph topic modeling with consensus clustering and iterative refinement**

[![CI](https://github.com/nevil-mathew/topic-extraction-poc/actions/workflows/ci.yml/badge.svg)](https://github.com/nevil-mathew/topic-extraction-poc/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/graphweave.svg)](https://pypi.org/project/graphweave/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

GraphWeave fuses semantic embeddings, lexical similarity, and optional metadata into a single
multi-view graph, then clusters it with consensus Leiden and sharpens the result with an
iterative refinement loop. On the four benchmark datasets in [`run_benchmark.py`](run_benchmark.py)
it reaches a mean NMI of **0.575** against BERTopic's 0.513, NMF's 0.416, and LDA's 0.299, with
**100% corpus coverage** (zero outliers by default). See [Benchmarks](#benchmarks) for the full
breakdown and how to reproduce it.

> GraphWeave began as a fork-in-spirit of [`tritopic`](https://pypi.org/project/tritopic/), an
> MIT-licensed library by Roman Egger, and has since been substantially rewritten and extended.
> See [Acknowledgments](#acknowledgments) and [NOTICE.md](NOTICE.md) for the full story and license text.

---

## Contents

**Getting started**
[Why GraphWeave?](#why-graphweave) ·
[Installation](#installation) ·
[Quick Start](#quick-start) ·
[The Pipeline](#the-pipeline)

**Core configuration**
[Configuration Reference](#configuration-reference) ·
[Choosing min_cluster_size](#choosing-min_cluster_size) ·
[Memory Optimization](#memory-optimization-for-large-datasets) ·
[Adaptive kNN Backend](#adaptive-knn-backend) ·
[Troubleshooting](#troubleshooting)

**Working with topics**
[Dimensionality Reduction](#dimensionality-reduction) ·
[Soft Topic Assignments](#soft-topic-assignments) ·
[Outlier Reduction](#outlier-reduction) ·
[Topic Merging](#topic-merging) ·
[Keyword Extraction](#keyword-extraction)

**LLM-powered features**
[LLM-Powered Labels](#llm-powered-labels) ·
[LLM-Guided Granularity Calibration](#llm-guided-granularity-calibration) ·
[LLM-Guided Embedding Adaptation](#llm-guided-embedding-adaptation) ·
[Report Themes](#report-themes-qualitative-research-output)

**Scaling up**
[Cumulative / Batch-wise Clustering](graphweave/cumulative/README.md)

**Visualizing & evaluating**
[Visualizations](#visualizations) ·
[Evaluation](#evaluation)

**Reference**
[Advanced Usage](#advanced-usage) ·
[API Reference](#api-reference) ·
[Architecture](#architecture) ·
[Comparison with BERTopic](#comparison-with-bertopic) ·
[Benchmarks](#benchmarks)

**Project**
[Citation](#citation) ·
[Acknowledgments](#acknowledgments) ·
[License](#license) ·
[Contributing](#contributing) ·
[Links](#links)

---

## Why GraphWeave?

Most topic models rely on a single signal — either word co-occurrences (LDA, NMF) or embeddings
alone (BERTopic). That limits their ability to separate topics that share vocabulary but differ
semantically, or vice versa.

GraphWeave fuses three complementary views of the corpus into one graph:

1. **Semantic view** — sentence-transformer embeddings capture meaning
2. **Lexical view** — TF-IDF similarity captures surface-level word patterns
3. **Metadata view** — optional categorical/numerical features add domain context

On top of that graph, GraphWeave runs **consensus Leiden clustering** (multiple runs aggregated
via a co-occurrence graph) and **iterative refinement** (embeddings pulled toward cluster
centroids and re-clustered). The result: topics that are more accurate, more coherent, more
stable, and — by default — assign every document to a topic.

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-view graph fusion** | Combines semantic embeddings, TF-IDF lexical similarity, and optional metadata into a single graph, avoiding the "embedding blur" that single-view models suffer from |
| **Mutual kNN + SNN graphs** | Eliminates noise bridges between unrelated documents using bidirectional neighbor checks and shared-neighbor weighting |
| **Consensus Leiden clustering** | Runs the Leiden algorithm multiple times and merges results via a co-occurrence graph, producing dramatically more stable topics than single-run approaches |
| **Iterative refinement** | Alternates between clustering and embedding refinement, pulling documents toward their topic centroids to sharpen boundaries |
| **Bidirectional resolution search** | Automatically finds the Leiden resolution parameter that produces the target number of topics |
| **Dimensionality reduction** | Reduces high-dimensional embeddings (384-768d) to ~10d with UMAP or PaCMAP before graph construction, improving neighbor quality |
| **100% corpus coverage** | Zero outliers by default — every document is assigned to a topic, unlike HDBSCAN-based approaches |
| **Soft topic assignments** | Computes per-document probability distributions over all topics, not just hard labels |
| **Post-fit outlier reduction** | Reassigns outlier documents using centroid similarity or neighbor voting after the model is fitted |
| **Hierarchical topic merging** | Iteratively merges the most similar topic pairs to reach a target count, or manually merges specific topics |
| **Multiple keyword methods** | c-TF-IDF, BM25, and KeyBERT keyword extraction with automatic diversity |
| **LLM-powered labels** | Generates human-readable topic names via Claude, GPT-4, Gemini, or OpenRouter |
| **Interactive visualizations** | 2D and 3D document maps, keyword bar charts, dendrograms, similarity heatmaps, and temporal topic evolution via Plotly |
| **TensorFlow Projector export** | Export embeddings and topic metadata for [projector.tensorflow.org](https://projector.tensorflow.org) with one call |
| **scikit-learn compatible** | Familiar `fit()` / `transform()` / `fit_transform()` API |
| **Save and load** | Full model persistence including fitted reducer, probabilities, and graph state |
| **Cumulative / batch-wise clustering** | Cluster documents that arrive in batches and accumulate over time, with drift-triggered reclustering, stable topic IDs, and a high-level "bigger picture" across all data — see [`graphweave/cumulative/README.md`](graphweave/cumulative/README.md) |

> **Streaming your data in batches?** GraphWeave also ships a separate cumulative /
> batch-wise workflow (`graphweave.cumulative.CumulativeGraphWeave`) that layers on top of
> the full-batch pipeline below **without changing it**. See the dedicated guide:
> [**Cumulative / Batch-wise Clustering →**](graphweave/cumulative/README.md)
> (quality report: [`LEARNING_GUIDE/CUMULATIVE_QUALITY_REPORT.md`](LEARNING_GUIDE/CUMULATIVE_QUALITY_REPORT.md)).

---

## Installation

```bash
# Core installation
pip install graphweave

# With LLM labeling support (Claude / GPT-4 / Gemini)
pip install graphweave[llm]

# With LLM-guided embedding fine-tuning (adds datasets, accelerate)
pip install graphweave[adaptation]

# Full installation (all optional features, including GPU support)
pip install graphweave[full]
```

### From source

```bash
git clone https://github.com/nevil-mathew/topic-extraction-poc.git
cd topic-extraction-poc
pip install -e ".[dev]"
```

### Dependencies

**Core:** numpy, pandas, scipy, scikit-learn, sentence-transformers, leidenalg, igraph, umap-learn, hdbscan, plotly, tqdm, rank-bm25, keybert

**Optional:** anthropic, openai, google-genai (for LLM labeling), pacmap, datamapplot (for advanced visualizations)

**Python:** 3.9, 3.10, 3.11, 3.12, 3.13

---

### GPU Acceleration (optional)

GraphWeave can offload the most compute-intensive steps to one or more CUDA GPUs.
CPU is always the automatic fallback — no code changes required and no errors if
torch / FAISS / cuML are absent.

#### What runs on GPU

| Step | GPU library | Speedup (typical) | Quality change |
|------|-------------|-------------------|----------------|
| Cosine similarity (transform, inference, refinement) | PyTorch | 20–100× | None |
| Embedding refinement loop | PyTorch | 5–20× | None |
| kNN graph construction (≥ 5k docs) | FAISS | 10–50× | None (exact) |
| UMAP dimensionality reduction | RAPIDS cuML | 5–30× | None |
| Coreset MiniBatchKMeans (cumulative mode) | RAPIDS cuML | 5–15× | Near-zero |
| Document embedding | sentence-transformers (already GPU-aware) | varies | None |

#### Install

```bash
# PyTorch + FAISS — covers cosine similarity, refinement loop, and kNN
pip install "graphweave[gpu]"

# On a CUDA machine, swap faiss-cpu for the GPU build (CUDA 11 or 12):
pip install faiss-gpu-cu11   # CUDA 11.x
pip install faiss-gpu-cu12   # CUDA 12.x

# RAPIDS cuML — adds GPU UMAP and GPU MiniBatchKMeans
# Install via conda (cuML has no standard pip wheel):
conda install -c rapidsai -c conda-forge cuml=24.06 cuda-version=12.0
# Once installed, GraphWeave detects cuML automatically — no further config needed.
```

#### Multi-GPU behaviour

| GPUs available | cosine similarity | kNN | UMAP | Embedding |
|----------------|-------------------|-----|------|-----------|
| 0 (CPU only) | sklearn | hnswlib / exact | umap-learn | CPU |
| 1 | torch `cuda:0` | FAISS GPU | cuML UMAP | ST on `cuda:0` |
| N > 1 | torch `cuda:0` | FAISS multi-GPU | cuML UMAP | ST multi-process pool |

All GPU paths are wrapped in `try/except` — a missing package or a CUDA OOM
falls back to the CPU implementation silently.

---

## Quick Start

```python
from graphweave import GraphWeave

documents = [
    "Machine learning is transforming healthcare diagnostics",
    "Deep neural networks achieve superhuman performance in image recognition",
    "Climate change affects biodiversity in tropical regions",
    "Renewable energy adoption accelerates globally",
    "The stock market rallied on strong earnings reports",
    # ... hundreds or thousands of documents
]

model = GraphWeave(verbose=True)
labels = model.fit_transform(documents)

# View discovered topics
print(model.get_topic_info())
```

**Output:**

```
[GraphWeave] Fitting model on 1000 documents
   Config: hybrid graph, iterative mode
   > Encoding 1000 documents (all-MiniLM-L6-v2)...
   > Reducing dimensions to 10d (UMAP)...
   > Building TF-IDF lexical matrix...
   > Running Leiden consensus clustering (10 runs)...
   > Starting iterative refinement (max 5 iterations)...
      Iteration 1 / 5...
      Iteration 2 / 5...
         ARI vs previous: 0.9234
      Iteration 3 / 5...
         ARI vs previous: 0.9812
      Converged at iteration 3
   > Extracting keywords and representative documents...

[OK] Fitting complete!  (total: 4.2 s)
   Found 12 topics
   47 outlier documents (4.7%)
```

### Post-fit refinement

```python
# Reassign outliers to their nearest topic
model.reduce_outliers(strategy="embeddings")

# Merge down to exactly 8 topics
model.reduce_topics(8)

# Access soft assignments
print(model.probabilities_.shape)   # (n_docs, n_topics)
print(model.probabilities_[0])      # probability distribution for doc 0
```

### Predict new documents

```python
new_docs = [
    "The Mars rover discovered ancient water deposits",
    "Baseball playoffs drew record attendance",
]

# Hard labels
new_labels = model.transform(new_docs)

# Soft probabilities
new_proba = model.transform_proba(new_docs)
```

### Save and load

```python
model.save("my_model.pkl")

from graphweave import GraphWeave
loaded = GraphWeave.load("my_model.pkl")
```

---

## The Pipeline

GraphWeave processes documents through a multi-stage pipeline:

```
Documents
    |
    |--- 1. Embedding Engine ----------------\
    |    (Sentence-BERT / BGE / Instructor)   |
    |                                         |
    |--- 1.5 Dim Reduction (UMAP/PaCMAP) ----+--- Multi-View
    |                                         |    Graph Builder
    |--- 2. Lexical Matrix (TF-IDF) ---------+         |
    |                                         |         |
    \--- 3. Metadata Graph (optional) -------/          |
                                                        v
                                          +-------------------------+
                                          |   Consensus Leiden       |
                                          |   (n runs + co-occur.)   |
                                          +------------+------------+
                                                       |
                                          +------------v------------+
                                          |  Iterative Refinement    |
                                          |  (blend toward centroid) |
                                          +------------+------------+
                                                       |
                                          +------------v------------+
                                          |  Keyword Extraction      |
                                          |  (c-TF-IDF / BM25)      |
                                          +------------+------------+
                                                       |
                                          +------------v------------+
                                          |  Topic Centroids +       |
                                          |  Soft Probabilities      |
                                          +-------------------------+
                                                       |
                                           (optional post-fit)
                                                       |
                                     +---------+-------+--------+
                                     |         |                |
                                reduce_    reduce_        merge_
                                outliers   topics         topics
```

**Step 1 - Embeddings:** Documents are encoded into dense vectors using a sentence-transformer model (local) or a cloud embedding API (e.g. Google Gemini). Pre-computed embeddings can also be passed directly.

**Step 1.5 - Dimensionality reduction:** High-dimensional embeddings (384-768d) are projected to ~10 dimensions using UMAP or PaCMAP. This dramatically improves kNN neighbor quality and speeds up graph construction. Full-dimensional embeddings are kept for centroid computation and keyword extraction.

**Step 2 - Lexical matrix:** TF-IDF with n-grams captures surface-level word patterns that embeddings may miss.

**Step 3 - Metadata graph (optional):** Categorical and numerical metadata fields create additional edges between related documents.

**Step 4 - Multi-view graph fusion:** The semantic kNN graph (built on reduced embeddings), lexical graph, and metadata graph are combined with configurable weights into a single igraph Graph. The semantic graph can use mutual kNN, SNN, or a hybrid of both.

**Step 5 - Consensus Leiden clustering:** The Leiden algorithm runs multiple times (default: 10) with different seeds. A sparse co-occurrence matrix records how often each pair of documents lands in the same cluster. The default `consensus_method="graph"` then thresholds this matrix and runs Leiden once more on the resulting weighted graph to produce the consensus partition (Lancichinetti & Fortunato, *Consensus clustering in complex networks*, Sci. Rep. 2:336, 2012). This is far more memory-efficient than the legacy hierarchical-linkage approach and avoids the N×N memory wall entirely. Clusters below `min_cluster_size` are marked as outliers (-1).

**Step 6 - Iterative refinement:** Embeddings are softly blended toward their topic centroid (20% pull), then the graph and clustering are re-run. This loop continues until the Adjusted Rand Index between consecutive iterations exceeds the convergence threshold (default: 0.95), or until `max_iterations` is reached. During iterative refinement, the dimensionality reducer transforms the refined embeddings for each new graph-building pass.

**Step 7 - Keywords and centroids:** c-TF-IDF (or BM25/KeyBERT) extracts representative keywords per topic. Topic centroids are computed as the mean embedding of each topic's documents. Soft probabilities are computed via cosine similarity to centroids passed through softmax.

---

## Configuration Reference

All parameters are set through `GraphWeaveConfig` or as constructor overrides:

```python
from graphweave import GraphWeave, GraphWeaveConfig

config = GraphWeaveConfig(
    # --- Embedding ---
    embedding_model="all-MiniLM-L6-v2",   # sentence-transformers model (local) or API model name
    embedding_batch_size=32,               # local GPU batch size
    embedding_provider="local",            # "local" or "google"
    embedding_api_key=None,                # API key (not shown in repr)
    embedding_api_batch_size=100,          # documents per API request (max 250 for Gemini)
    embedding_output_dim=None,             # MRL output dimensionality (auto-768 for Google)
    embedding_task_type=None,              # task hint for gemini-embedding-001: "CLUSTERING" etc.
    embedding_batch_delay=0.0,             # seconds between API batches (set ~4.0 on free tier)

    # --- Dimensionality Reduction ---
    use_dim_reduction=True,                # reduce before graph building
    reduced_dims=10,                       # target dimensionality
    dim_reduction_method="umap",           # "umap" or "pacmap"
    umap_n_neighbors=15,                   # UMAP/PaCMAP neighbor count
    umap_min_dist=0.0,                     # 0.0 optimized for clustering

    # --- Graph Construction ---
    n_neighbors=15,                        # k for kNN graph
    metric="cosine",                       # distance metric
    graph_type="hybrid",                   # "knn", "mutual_knn", "snn", "hybrid"
    snn_weight=0.5,                        # SNN weight in hybrid mode
    knn_backend="auto",                    # "auto" | "exact" | "hnsw" — see "Adaptive kNN Backend"
    hnsw_small_threshold=5_000,            # below this, "auto" stays on exact sklearn
    hnsw_large_threshold=50_000,           # at/above this, HNSW uses (M=32, ef=400)

    # --- Multi-View Fusion ---
    use_lexical_view=True,                 # include TF-IDF view
    use_metadata_view=False,               # include metadata view
    semantic_weight=0.5,                   # weight for semantic graph
    lexical_weight=0.3,                    # weight for lexical graph
    metadata_weight=0.2,                   # weight for metadata graph

    # --- Clustering ---
    resolution=1.0,                        # Leiden resolution (higher = more topics)
    n_consensus_runs=10,                   # number of Leiden runs for consensus
    min_cluster_size=5,                    # absolute floor: communities smaller than this become outliers (-1)
    min_cluster_fraction=None,             # recommended: 0.005 — see "Choosing min_cluster_size" below
    consensus_method="graph",              # "graph" (default, memory-safe) or "hierarchical"
    consensus_threshold_tau=0.5,           # τ for graph consensus: keep pairs that co-cluster in ≥τ·n_runs runs

    # --- Iterative Refinement ---
    use_iterative_refinement=True,         # enable the refinement loop
    max_iterations=5,                      # maximum refinement iterations
    convergence_threshold=0.95,            # ARI threshold to stop early

    # --- Keyword Extraction ---
    n_keywords=10,                         # keywords per topic
    n_representative_docs=5,               # representative docs per topic
    keyword_method="ctfidf",               # "ctfidf", "bm25", or "keybert"

    # --- Representative-doc sampling (for LLM labelling) ---
    labeling_sample_strategy="centroid",   # "centroid" | "mmr" | "stratified"
    mmr_lambda=0.5,                        # MMR: 1.0=pure relevance, 0.0=pure diversity
    stratified_proportions=(0.6, 0.3, 0.1),# stratified: (close, mid, far) bin weights

    # --- Outlier Handling ---
    outlier_threshold=0.1,                 # cosine similarity threshold for transform()

    # --- Misc ---
    random_state=42,
    verbose=True,
    low_memory=False,                      # only affects hierarchical consensus path; graph mode ignores it
    n_jobs=-1,                             # parallelism: -1 = all cores (kNN, ARI); Leiden runs capped at 4
)

model = GraphWeave(config=config)
```

**Quick overrides** without creating a config object:

```python
model = GraphWeave(
    embedding_model="all-mpnet-base-v2",
    n_neighbors=20,
    use_iterative_refinement=True,
    verbose=True,
    random_state=42,
)
```

You can also modify the config after construction:

```python
model = GraphWeave()
model.config.use_dim_reduction = False       # disable dim reduction
model.config.graph_type = "snn"              # use pure SNN graph
model.config.keyword_method = "bm25"         # switch keyword method
```

---

## Choosing min_cluster_size

After Leiden clustering, any community smaller than `min_cluster_size` is merged into the outlier class (`-1`). Getting this number right matters:

- **Too small** (e.g. `5` on 100K docs): hundreds of micro-communities survive as topics; raw ARI against coarse ground-truth labels collapses.
- **Too large** (e.g. `2000` on 10K-doc batches): all communities get filtered, `topic_embeddings_` becomes empty, and `transform()` crashes with a shape error.

The fundamental problem with an absolute number is **scale sensitivity**: a community of 50 docs is meaningful at 1K documents but negligible at 100K.

### Recommended: use `min_cluster_fraction`

```python
config = GraphWeaveConfig(
    min_cluster_fraction=0.005,   # 0.5% of corpus — the one number to tune
    # min_cluster_size stays at default=5 (acts as absolute floor only)
)
```

At fit time, the effective threshold is `max(min_cluster_size, int(min_cluster_fraction × n_docs))`:

| Corpus size | Effective min_cluster_size |
|---|---|
| 1K docs | 5 (floor) |
| 10K docs (first batch) | 50 |
| 50K docs (coreset) | 250 |
| 120K docs (full fit) | 600 |

This scales naturally across corpus sizes, so cumulative models that grow from a 10K first batch to a 120K full accumulator never need a different constant.

**Tuning guide:**

| `min_cluster_fraction` | Effect |
|---|---|
| `0.002` | Fine-grained — many sub-topics, higher outlier rate |
| `0.005` | Balanced — recommended starting point |
| `0.010` | Coarse — few large topics, lower outlier rate |

**Precedence rules:**
- `min_cluster_fraction=None` (default): only `min_cluster_size` is used (absolute mode).
- `min_cluster_fraction` set: `max(min_cluster_size, fraction × n_docs)` — the absolute value acts as a hard floor.
- Setting `min_cluster_size` to a large value with `min_cluster_fraction` also set: the larger of the two always wins.

### Research basis

This design follows three converging lines of reasoning:

1. **Scale invariance** — standard statistical practice prefers relative thresholds over absolute counts when the input size is variable. An absolute cutoff optimised for one corpus size is arbitrary at another.

2. **HDBSCAN community practice** — McInnes, Healy & Astels (2017) recommend expressing `min_cluster_size` as "roughly 1–5% of dataset size" in practice. GraphWeave uses Leiden, not HDBSCAN, but the same post-clustering size filter has the same scale-sensitivity problem.

3. **Resolution limit in community detection** — Fortunato & Barthélemy (2007) proved that modularity-based community detectors have a resolution limit that scales with graph size: communities become undetectable below a minimum size that grows with N. A fixed absolute filter is inconsistent with this limit as the corpus grows.

---

## Memory Optimization for Large Datasets

GraphWeave handles **100,000+ documents on a laptop** out of the box. This section explains why, and what knobs to reach for if you ever hit a wall.

### The underlying problem

To find stable topics, GraphWeave runs Leiden clustering 10 times and then asks: *"How often did each pair of documents end up in the same cluster?"* This pairwise tally is a **co-occurrence matrix** of size N × N.

A naive implementation would densify that matrix and run hierarchical clustering (`scipy.linkage`) on it. Both steps scale as N²:

| Documents (N) | N × N cells | Dense-path peak memory |
|---|---|---|
| 5,000 | 25 M | ~0.6 GB |
| 20,000 | 400 M | ~10 GB |
| 50,000 | 2.5 B | **~60 GB** (OOM on most machines) |
| 100,000 | 10 B | **~240 GB** (will not fit anywhere) |

### The default: graph consensus

The default `consensus_method="graph"` (Lancichinetti & Fortunato, *Consensus clustering in complex networks*, Sci. Rep. 2:336, 2012) replaces the dense matrix **and** the `scipy.linkage` step with a single Leiden pass on a thresholded sparse co-occurrence **graph**:

1. Keep only document pairs that co-cluster in at least `consensus_threshold_tau` (default **0.5**, i.e. 5 out of 10 runs) of the Leiden runs.
2. Build a weighted graph from those surviving pairs (typically <1% of N²).
3. Run Leiden once on that graph — that is your consensus partition.

| Documents (N) | Dense hierarchical path | Graph consensus (default) |
|---|---|---|
| 20,000 | ~10 GB | **~0.3 GB** |
| 50,000 | ~60 GB | **~0.8 GB** |
| 100,000 | ~240 GB | **~2 GB** |

Quality is at least as good — the LF paper shows graph consensus improves stability and accuracy versus any single Leiden run. You do not need to do anything: the default is on automatically.

> **Implementation note:** The co-occurrence matrix is accumulated in float32 (halving dtype overhead vs float64) and pruned after each Leiden run — entries that can no longer reach the τ threshold are dropped immediately, so the matrix stays sparse throughout rather than growing to its maximum at the final run. Parallel Leiden runs are capped at 4 concurrent threads regardless of `n_jobs`, preventing 10× peak C-level allocations from all runs landing in memory simultaneously.

### When to touch the knobs

| Situation | What to do |
|---|---|
| Any size, default install | **Nothing.** The default is already memory-safe with automatic float32, early pruning, and capped parallelism. |
| You want stricter / looser consensus | Tune `consensus_threshold_tau` in `[0.3, 0.8]`. Higher τ = stricter (fewer, tighter topics). |
| You want bit-for-bit identical results to the legacy hierarchical path | Set `consensus_method="hierarchical"`. See below. |
| You still hit an OOM crash | Lower `n_consensus_runs` (e.g. 5) or lower `consensus_threshold_tau` (e.g. 0.3, more aggressive pruning). |

### Tuning the consensus threshold τ

```python
config = GraphWeaveConfig(
    consensus_method="graph",          # default
    consensus_threshold_tau=0.5,       # default
)
```

- **τ = 0.3** — loose. More edges survive, larger / merged topics, more robust to noisy Leiden runs.
- **τ = 0.5** — balanced. Recommended starting point.
- **τ = 0.7** — strict. Only pairs that almost-always co-clustered survive; produces tighter, more conservative topics.

The LF paper reports results are robust across `τ ∈ [0.3, 0.8]`, so this is a soft dial, not a cliff.

### Legacy hierarchical mode (opt-in)

The old hierarchical-linkage path is still available for backwards compatibility:

```python
config = GraphWeaveConfig(
    consensus_method="hierarchical",   # legacy
    low_memory=True,                   # use sparse co-occurrence (still N² in the worst case)
)
```

For best results in legacy mode, install the `legacy-consensus` extra — it adds `fastcluster`, a C++ replacement for `scipy.linkage` that is ~2-5× faster and avoids a hidden float64 copy:

```bash
pip install graphweave[legacy-consensus]
```

### What does `low_memory=True` still do?

In **graph mode** (default), `low_memory` has **no effect** — float32 is used automatically and early pruning keeps the co-occurrence matrix lean throughout accumulation. No action needed.

In **hierarchical mode**, `low_memory=True` keeps the co-occurrence sparse and builds the condensed distance vector directly from it, saving ~7-20× on peak RAM versus the dense path. Same math, same topics, same `random_state`.

### Still want more headroom?

The main consensus optimizations (float32, early pruning, 4-thread cap) are automatic. If you still need more room, these knobs trade a little quality for memory:

```python
config = GraphWeaveConfig(
    max_iterations=2,            # was 5. Refinement gains are mostly in rounds 1-2.
    n_consensus_runs=5,          # was 10. Fewer runs = smaller co-occurrence matrix peak.
    convergence_threshold=0.90,  # was 0.95. Stops one iteration sooner.
)
```

Lowering `n_consensus_runs` is the strongest lever here: fewer runs means the co-occurrence matrix accumulates fewer non-zeros, and early pruning kicks in sooner. Combined, these typically give an additional 2-3× headroom with under 2% quality loss.

---

## Adaptive kNN Backend

The semantic graph that drives Leiden clustering relies on a kNN search over
document embeddings. For corpora above a few thousand documents the exact
`O(n²)` search becomes the dominant fit cost — and it runs once per
refinement iteration. GraphWeave ships an **adaptive backend** that
chooses between exact and approximate (HNSW) search based on corpus size:

| Corpus size       | Backend (priority)              | Notes                  |
|-------------------|----------------------------------|------------------------|
| < 5,000 docs      | exact (sklearn)                 | —                      |
| ≥ 5,000 docs      | FAISS GPU *(if `gpu` extra + CUDA)* | exact, fastest    |
| ≥ 5,000 docs      | FAISS CPU *(if `gpu` extra, no CUDA)* | exact, fast     |
| ≥ 5,000 docs      | hnswlib HNSW *(if `fast-knn` extra)* | approx, M=16/32  |
| any               | exact (sklearn) fallback        | always available       |

The switch is invisible to everything downstream — mutual-kNN filtering, SNN
computation and multi-view fusion receive identical `(neighbor_id, similarity)`
output regardless of which backend ran.

### Enabling the FAISS path (recommended)

```bash
pip install "graphweave[gpu]"        # includes faiss-cpu + torch
pip install faiss-gpu              # optional: swap in GPU FAISS on CUDA machines
```

### Enabling the HNSW path (CPU-only alternative)

```bash
pip install "graphweave[fast-knn]"
```

When neither extra is installed, `knn_backend="auto"` silently falls back to
the exact sklearn path.

### Configuration

```python
config = GraphWeaveConfig(
    knn_backend="auto",            # "auto" | "exact" | "hnsw"
    hnsw_small_threshold=5_000,    # below this, "auto" stays on exact
    hnsw_large_threshold=50_000,   # at/above this, HNSW switches to (M=32, ef=400)
)
```

- `"auto"` (default): exact below `hnsw_small_threshold`; FAISS (GPU > CPU) or
  HNSW above, depending on which extras are installed.
- `"exact"`: always use sklearn. Use this for reproducibility benchmarks or
  to A/B against the approximate path.
- `"hnsw"`: always use hnswlib (requires the `fast-knn` extra).

With `verbose=True`, each kNN call logs which backend was selected and the
corpus size — useful for confirming the dispatch picked the right branch
on your data.

### Notes

- HNSW uses `space="cosine"`, matching the default `metric="cosine"`. If
  you change `metric`, the adaptive backend falls back to exact.
- The index is built fresh per call — refinement mutates embeddings each
  iteration, so caching across iterations would be wrong.
- `transform()` and `transform_proba()` use centroid similarity, not graph
  kNN — incremental document additions never hit the HNSW index.

---

## Troubleshooting

### "Spectral initialisation failed! ... Falling back to random initialisation!"

You may see this `UserWarning` from UMAP during fitting:

```
UserWarning: Spectral initialisation failed! The eigenvector solver failed.
This is likely due to too small an eigengap.
Consider adding some noise or jitter to your data.
Falling back to random initialisation!
```

**What it means (plain English):** UMAP -- the tool that shrinks your high-dimensional embeddings to ~10 dimensions -- normally starts by computing a smart initial placement using linear algebra ("spectral initialisation"). That math is unstable when your documents are very tightly packed or have lots of near-duplicates. When it fails, UMAP automatically falls back to placing points randomly and then optimizing from there.

**Does it affect my topics?** No. The fallback (random init + UMAP optimization) converges to essentially the same embedding. Your clustering quality is unaffected. The warning is informational, not an error.

**Is reproducibility affected?** No, as long as you set `random_state` in your config. Same seed, same output.

**Can I silence the warning?**

```python
import warnings
warnings.filterwarnings(
    "ignore",
    message="Spectral initialisation failed",
)
```

**Common causes (none are bugs):**
- Many near-duplicate documents in the corpus
- Very tightly clustered embeddings (a corpus on one narrow topic)
- Very small datasets where the graph is fully connected

You can ignore the warning and use the resulting model normally.

### Other common issues

| Symptom | Cause | Fix |
|---|---|---|
| Crash at `Iteration 1...` with no traceback | Out of memory in consensus step | Lower `n_consensus_runs` (e.g. 5) or `consensus_threshold_tau` (e.g. 0.3); `low_memory=True` only helps the legacy `hierarchical` path |
| `ImportError: cannot import name '...' from 'transformers'` in Colab | Colab silently upgraded torch/transformers mid-session | **Runtime -> Restart session**, then rerun |
| Too many tiny topics | `resolution` too high or `min_cluster_size` too low | Lower `resolution` (e.g. 0.8) or set `min_cluster_fraction=0.005` (scales with corpus size) |
| `ValueError: Found array with 0 sample(s)` in `transform()` | `min_cluster_size` too large — all Leiden communities filtered to outliers | Lower `min_cluster_size`, or switch to `min_cluster_fraction=0.005` |
| Too few large topics | `resolution` too low | Raise `resolution` (e.g. 1.3) or set `n_topics=N` |
| 30%+ outliers | HDBSCAN-like over-pruning of small clusters | Call `model.reduce_outliers(strategy="embeddings")` after fit |
| LLM labels are empty / generic | API call failed silently | Retries with backoff automatically; check API key and rate limits |

---

## Dimensionality Reduction

kNN graphs built on high-dimensional embeddings (384-768d) suffer from the curse of dimensionality: distances concentrate and neighbor quality degrades. GraphWeave addresses this by reducing embeddings to a low-dimensional space before graph construction.

```python
model = GraphWeave()
model.config.use_dim_reduction = True        # enabled by default
model.config.reduced_dims = 10               # target dimensions
model.config.dim_reduction_method = "umap"   # or "pacmap"
model.config.umap_n_neighbors = 15
model.config.umap_min_dist = 0.0             # 0.0 is best for clustering

model.fit(documents)

# Reduced embeddings are stored alongside full embeddings
print(model.reduced_embeddings_.shape)  # (n_docs, 10)
print(model.embeddings_.shape)          # (n_docs, 384)  full embeddings kept
```

**How it works:**

- Reduced embeddings are used only for graph construction (kNN neighbor search)
- Full-dimensional embeddings are used for centroid computation, keyword extraction, representative docs, and similarity calculations
- During iterative refinement, refined embeddings are re-projected through the fitted reducer at each iteration
- The fitted reducer is saved with `model.save()` so `transform()` on new documents works correctly

**When to disable it:**

```python
model.config.use_dim_reduction = False
```

Disable if your embeddings are already low-dimensional, or if you want to experiment with raw high-dimensional graph construction.

---

## Soft Topic Assignments

Every document gets a probability distribution over all topics, not just a hard label.

### Training documents

After `fit()`, probabilities are automatically available:

```python
model.fit(documents)

# Shape: (n_documents, n_topics)
print(model.probabilities_.shape)

# Each row sums to ~1.0
print(model.probabilities_[0].sum())  # ~1.0

# Probability distribution for document 0
for i, prob in enumerate(model.probabilities_[0]):
    topic_id = [t.topic_id for t in model.topics_ if t.topic_id != -1][i]
    print(f"  Topic {topic_id}: {prob:.3f}")
```

### New documents

```python
proba = model.transform_proba(["A new document about space exploration"])
# Shape: (1, n_topics)
print(proba)
```

**How it works:** Cosine similarity between document embeddings and topic centroid embeddings, followed by softmax normalization. Probabilities are recomputed automatically after any post-fit operation (outlier reduction, topic merging).

---

## Outlier Reduction

Leiden clustering combined with small-cluster removal can produce 20-40% outliers. `reduce_outliers()` reassigns them post-fit.

### Strategy: embeddings (default)

Each outlier is assigned to the topic whose centroid is most similar, if the similarity exceeds a threshold:

```python
model.fit(documents)
print(f"Outliers before: {(model.labels_ == -1).sum()}")

# Default threshold = config.outlier_threshold (0.1)
model.reduce_outliers(strategy="embeddings")
print(f"Outliers after: {(model.labels_ == -1).sum()}")

# Lower threshold = more aggressive reassignment
model.reduce_outliers(strategy="embeddings", threshold=0.05)
```

### Strategy: neighbors

Each outlier is assigned by majority vote of its k nearest non-outlier neighbors:

```python
model.reduce_outliers(strategy="neighbors")
```

This strategy is threshold-free and works well when outliers are near cluster boundaries.

**After reassignment:** Keywords, centroids, topic sizes, and probabilities are all recomputed automatically.

---

## Topic Merging

### Automatic: reduce to a target count

`reduce_topics()` iteratively merges the two most cosine-similar topic centroids until the target count is reached:

```python
model.fit(documents)
print(f"Topics found: {len([t for t in model.topics_ if t.topic_id != -1])}")

# Reduce to exactly 5 topics
model.reduce_topics(5)
print(f"Topics after: {len([t for t in model.topics_ if t.topic_id != -1])}")
```

At each step, the two most similar centroids are found, and the smaller topic is relabeled to the larger one. After all merges complete, keywords and centroids are re-extracted from scratch.

### Manual: merge specific topics

```python
# Merge topics 2 and 7 into one (the larger one's ID is kept)
model.merge_topics([2, 7])

# Merge three topics together
model.merge_topics([1, 4, 9])
```

This is useful when you inspect topics and find two that clearly cover the same theme.

---

## Keyword Extraction

GraphWeave supports three keyword extraction methods:

### c-TF-IDF (default)

Class-based TF-IDF treats all documents in a topic as a single "class document" and scores terms by their distinctiveness for that topic compared to the corpus. This is the same approach used by BERTopic.

```python
model.config.keyword_method = "ctfidf"
```

### BM25

BM25 scoring is more robust to document length variations than TF-IDF:

```python
model.config.keyword_method = "bm25"
```

### KeyBERT

Embedding-based extraction that finds keywords by comparing candidate n-gram embeddings to the topic embedding. Uses Maximal Marginal Relevance (MMR) for diversity:

```python
model.config.keyword_method = "keybert"
```

### Accessing keywords

```python
# DataFrame view
df = model.get_topic_info()
print(df[["Topic", "Size", "Keywords"]])

# Detailed access for a specific topic
topic = model.get_topic(0)
print(topic.keywords)         # ['machine', 'learning', 'neural', ...]
print(topic.keyword_scores)   # [0.42, 0.38, 0.31, ...]

# Representative documents
docs = model.get_representative_docs(0, n_docs=3)
for idx, text in docs:
    print(f"  Doc {idx}: {text[:100]}...")
```

### Representative document sampling

The docs returned by `get_representative_docs()` are the ones sent to the LLM during `generate_labels()`. The selection algorithm is controlled by `labeling_sample_strategy`:

| Strategy | What it does | When to use |
|---|---|---|
| `"centroid"` *(default)* | Closest-to-centroid only (L2 distance). | Tight, coherent topics. Backward-compatible. |
| `"mmr"` | Maximal Marginal Relevance: greedily picks docs that are close to the centroid **and** diverse from each other. Tuned by `mmr_lambda` (default 0.5). | Topics where centroid-closest docs are near-paraphrases of each other — gives the LLM broader coverage without absorbing outliers. |
| `"stratified"` | Sorts members by distance-to-centroid, splits into close/mid/far bins, samples proportionally per `stratified_proportions` (default 60/30/10). | When you want explicit control over how much "edge" of the topic the LLM sees. |

All three strategies are deterministic. If `get_representative_docs(topic_id, n_docs=N)` is called with `N` larger than `n_representative_docs`, the sampling is re-run on demand over the full topic so the LLM gets exactly `N` docs in the selected style.

---

## LLM-Powered Labels

Generate human-readable topic names using Claude, GPT-4, or Gemini.

### With Claude (Anthropic)

```python
from graphweave import GraphWeave, LLMLabeler

model = GraphWeave()
model.fit(documents)

labeler = LLMLabeler(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-haiku-4-5-20251001",  # fast and cheap
    language="english",                  # output language
    domain_hint="technology news",       # optional domain context
)
model.generate_labels(labeler)

# Topics now have labels and descriptions
df = model.get_topic_info()
print(df[["Topic", "Label", "Description"]])
```

### With GPT-4 (OpenAI)

```python
labeler = LLMLabeler(
    provider="openai",
    api_key="sk-...",
    model="gpt-4o-mini",
    language="german",          # works in any language
)
model.generate_labels(labeler)
```

### With Gemini (Google)

```python
labeler = LLMLabeler(
    provider="google",
    api_key="...",
    model="gemini-2.5-flash",   # default if model omitted
)
model.generate_labels(labeler)
```

### With OpenRouter (or any OpenAI-compatible endpoint)

OpenRouter exposes an OpenAI-compatible API, so it works through the same `openai`
package — no extra dependency. Use the `vendor/model` id and set
`provider="openrouter"`:

```python
labeler = LLMLabeler(
    provider="openrouter",
    api_key="sk-or-...",
    model="anthropic/claude-3.5-haiku",   # any model on openrouter.ai/models
)
model.generate_labels(labeler)
```

The same `labeler` also drives LLM merging / report-theme synthesis via
`model.generate_report_themes(labeler)`. To target a different OpenAI-compatible
server (e.g. a local Ollama instance), pass `provider="openai"` with a custom
`base_url`:

```python
labeler = LLMLabeler(
    provider="openai",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="llama3.1",
)
```

Install the required extra: `pip install graphweave[llm]` (includes all providers).

### Controlling prompt size

By default each LLM call receives up to **5 representative documents**, each truncated to **500 characters**. Both limits are configurable:

```python
labeler = LLMLabeler(
    provider="anthropic",
    api_key="...",
    n_docs=3,           # fewer docs → lower cost / latency
    doc_max_chars=200,  # shorter snippets
)

# Higher quality: more context per topic
labeler = LLMLabeler(
    provider="anthropic",
    api_key="...",
    n_docs=8,
    doc_max_chars=1000,
)
```

> **Note:** `n_docs` draws from the representative documents stored at fit time, which are the docs closest to the topic centroid. If you set `n_docs` higher than `GraphWeaveConfig.n_representative_docs` (default 5), raise that value too:
>
> ```python
> from graphweave import GraphWeave, GraphWeaveConfig
> config = GraphWeaveConfig(n_representative_docs=10)
> model = GraphWeave(config=config)
> model.fit(documents)
>
> labeler = LLMLabeler(provider="anthropic", api_key="...", n_docs=8)
> model.generate_labels(labeler)
> ```

### Simple labeler (no API needed)

```python
from graphweave import SimpleLabeler

labeler = SimpleLabeler(n_words=3)
model.generate_labels(labeler)
# Labels like "Machine & Learning & Neural"
```

### Label specific topics only

```python
model.generate_labels(labeler, topics=[0, 3, 5])
```

### Two label styles: `short` vs `theme`

`LLMLabeler` accepts a `style` parameter that controls how rich the output is:

| Style | Title length | Description length | Use for |
|-------|--------------|-------------------|---------|
| `"short"` *(default, backward-compatible)* | 3–7 words | 1–2 sentences | dashboards, charts, CSV exports |
| `"theme"` | 5–8 words, evocative | 4–6 sentence narrative paragraph with quoted phrases | qualitative research reports |

```python
labeler = LLMLabeler(
    provider="anthropic",
    api_key="sk-ant-...",
    style="theme",                                # report-quality output
    domain_hint="education barriers in rural India",  # anchors register
)
model.generate_labels(labeler)
```

The theme style automatically:
- Passes **15 keywords** (instead of 10) and **8 representative docs** of up to 1200 chars (instead of 5 of 500 chars) for richer context.
- Uses a longer `max_tokens` budget (900) so the narrative isn't truncated.
- Few-shots the prompt with the qualitative-report register so output reads like a Findings section.

### Automatic duplicate-label prevention

Both styles run with **sequential dedup context** by default: each topic is labeled with awareness of all topics labeled before it in the same run, and the LLM is explicitly instructed to name a distinguishing mechanism if its cluster overlaps thematically with an earlier one.

After the main loop, a **cleanup pass** finds any remaining exact-duplicate or shared-prefix labels (e.g., two topics both labeled "Systemic Barriers to Educational Access") and regenerates them with explicit "make these distinct" instructions.

```python
model.generate_labels(labeler)                  # dedup on by default
model.generate_labels(labeler, dedup=False)     # opt out (legacy behavior)
model.generate_labels(labeler, dedup_passes=2)  # extra cleanup pass for tricky datasets
```

### Response caching

`LLMLabeler` caches each response by prompt hash, so re-running `generate_labels` (or the cleanup pass) does not re-pay for identical calls. Disable with `cache=False`.

---

## LLM-Guided Granularity Calibration

An optional third way to pick the Leiden `resolution` parameter, alongside modularity-maximization (the default linear sweep) and binary-search-to-a-target-topic-count (`n_topics=<int>`). This path asks an LLM to judge clustering granularity directly, following the triplet-query approach from **ClusterLLM** (Zhang, Wang & Shang, *"ClusterLLM: Large Language Models as a Guide for Text Clustering"*, EMNLP 2023).

The idea: sample `(A, B, C)` document triplets where `B` is A's nearest neighbor in the same cluster and `C` is A's nearest neighbor in a different cluster (under a reference partition), then ask the LLM "does A belong with B or with C?" for each triplet. Several candidate resolutions are scored by how well their partitions agree with the LLM's answers, and the best-agreeing resolution is kept.

This is **fully opt-in** — it is never invoked by `fit()` or by `n_topics=<int>`, and it does not change the behavior of the existing resolution-search paths.

```python
from graphweave import GraphWeave, LLMLabeler

model = GraphWeave().fit(documents)

labeler = LLMLabeler(provider="anthropic", api_key="...", model="claude-haiku-4-5")
model.tune_resolution_with_llm(labeler)

print(f"Topics after calibration: {len([t for t in model.topics_ if t.topic_id != -1])}")
```

By default, 6 candidate resolutions are scored (geometrically spaced across `resolution_range=(0.1, 2.0)`) against a corpus-size-scaled number of triplets (~24 for small corpora, scaling up to ~80-120 for 50K+ document corpora), sent to the LLM in batches of 8. All of this is configurable:

```python
model.tune_resolution_with_llm(
    labeler,
    resolution_range=(0.05, 3.0),
    n_candidates=8,
    n_triplets=60,     # override the corpus-size default
    batch_size=8,
    random_state=42,
)
```

**How it works (two-stage):**

- **Stage A — coarse sweep:** scores all `n_candidates` resolutions against LLM answers collected once.
- **Stage B — local refinement:** around the Stage A winner, scores a 5-point fine grid reusing the *same* LLM answers — zero extra API calls. This removes grid-quantization artifacts where the hand-tuned resolution falls between two coarse candidates.

**Bias and sampling improvements:**

- ~50 % of triplet presentations are randomly swapped (different-cluster doc shown as "B") so LLM position preference can't systematically favor one resolution. Answers are un-swapped before scoring.
- Triplets are chosen to be *discriminative*: a 4× oversampled pool is ranked by how much the candidates *disagree* on each triplet, and only the most informative are kept. Uninformative triplets (where all candidates agree) are discarded upfront.
- Each candidate is scored across multiple Leiden seeds to reduce run-to-run variance.

**Diagnostics:** after the call, `model.granularity_diagnostics_` holds a per-candidate breakdown:

```python
model.tune_resolution_with_llm(labeler)

import pandas as pd
diag = model.granularity_diagnostics_
df = pd.DataFrame(diag["candidates"])
print(df[["stage", "resolution", "score", "n_clusters"]])
# stage  resolution    score  n_clusters
#     A    0.600000  0.69330          56
#     A    0.865620  0.69690          73   ← winner
#     B    0.720675  0.68960          65
#     ...
```

A flat score column (all values within ~0.02 of each other) means the LLM couldn't strongly prefer any granularity — in that case the calibrated resolution is a weak preference and your hand-tuned value is equally valid.

**Cost**: each triplet costs one small batched LLM call slice (~8 triplets per call). On Claude Haiku 4.5, a full run costs roughly **$0.01 for a small corpus (~24 triplets) up to ~$0.15 for a large one (~500 triplets)** — cheap enough to run after every `fit()` on a new dataset while you're still tuning `resolution_range`.

---

## LLM-Guided Embedding Adaptation

GraphWeave's embedder is fixed and off-the-shelf by default (`all-MiniLM-L6-v2`, or an API model). On a narrow, domain-specific corpus that's the real quality ceiling — the graph, Leiden consensus, and refinement loop can only rearrange whatever geometry the embedder already gives them. `graphweave.adaptation` closes that gap: it asks an LLM to judge a small budget of triplet comparisons on *your* documents, then adapts the embedder to those judgments — the triplet fine-tuning approach from **ClusterLLM** (Zhang, Wang & Shang, EMNLP 2023), with sampling/evaluation ideas from **PRISM** (Douglas, Balci & Aylett-Bullock, WWW 2026) and the LLM keyphrase-expansion / low-confidence-correction techniques from **Viswanathan et al.**, *"Large Language Models Enable Few-Shot Clustering"* (TACL 2024).

It lives in a self-contained `graphweave.adaptation` subpackage — same opt-in philosophy as `tune_resolution_with_llm`: never invoked automatically, and importing it doesn't pull in any extra dependencies until you actually fine-tune.

**Notebooks:** [`notebooks/embedding_adaptation_demo.ipynb`](notebooks/embedding_adaptation_demo.ipynb) proves the pipeline for free (a fake oracle labeler, no API key, no GPU); [`notebooks/embedding_adaptation_kaggle.ipynb`](notebooks/embedding_adaptation_kaggle.ipynb) runs the real thing on your own CSV — OpenRouter for triplet judgments, real `all-MiniLM-L6-v2` fine-tuning on a Kaggle GPU, plus both bonus extras below.

```python
from graphweave import GraphWeave, LLMLabeler

model = GraphWeave().fit(documents)

labeler = LLMLabeler(provider="anthropic", api_key="...", model="claude-haiku-4-5")
model.adapt_embeddings_with_llm(labeler)   # refits in place with the adapted embeddings

print(model.adaptation_diagnostics_["holdout_triplet_acc_before"])
print(model.adaptation_diagnostics_["holdout_triplet_acc_after"])
```

### How it works

1. **Sample triplets** `(A, B, C)` — by default, anchors `A` are the documents the model's *soft assignment is least confident about* (highest entropy across topic probabilities), since those are the boundary cases an LLM judgment helps most. `B`/`C` are `A`'s nearest same-cluster / different-cluster neighbors.
2. **Ask the LLM** which of `B` or `C` the anchor is more similar to, in batches — reusing the same bias-mitigated (swap-randomized), cached, structured-output query machinery as `tune_resolution_with_llm`.
3. **Hold out ~20%** of the judged triplets before any training, so "did this help" can always be measured on judgments the adapter never saw.
4. **Adapt the embedder** with one of two backends (below), trained on `(anchor, positive, negative)` triples the LLM actually judged — not just the pre-existing cluster labels.
5. **Refit** GraphWeave on the adapted embeddings and report before/after metrics.

### Two backends

| Mode | Requires | Works with |
|---|---|---|
| `"linear"` | numpy only | any embedder, including API-based ones (Gemini) — the practical default on CPU-only machines |
| `"finetune"` | `pip install "graphweave[adaptation]"` (adds `datasets`, `accelerate`) | local sentence-transformers models only |
| `"auto"` (default) | — | picks `"finetune"` when possible, else `"linear"` with a warning |

`"linear"` trains an identity-initialized d×d matrix on top of frozen embeddings with a cosine hinge triplet loss, shrunk toward the identity by an L2 penalty so noisy LLM judgments can't push it far from the base geometry — cheap, CPU-friendly, and the only option for embedders you can't fine-tune. `"finetune"` runs a real 1-epoch, low-LR sentence-transformers fine-tune (`MultipleNegativesRankingLoss` by default) — the full ClusterLLM recipe.

```python
from graphweave.adaptation import AdaptationConfig

model.adapt_embeddings_with_llm(
    labeler,
    config=AdaptationConfig(
        adapter_mode="linear",     # or "finetune" / "auto"
        n_triplets=1000,           # ~ ClusterLLM's budget
        triplet_sampling="entropy",
        holdout_frac=0.2,
        cache_path="triplet_cache.jsonl",   # repeat runs pay zero API cost
    ),
)
```

### Did it actually help? (`compare_embedders`)

Fine-tuning without a way to check for regressions is how bugs ship. `compare_embedders` fits an *identical* GraphWeave config on baseline vs. adapted embeddings and tabulates intrinsic metrics (silhouette, Davies–Bouldin, Calinski–Harabasz, consensus stability), held-out triplet accuracy (label-free — works on your own unlabeled corpus), and — when you have ground truth (e.g. a labeled benchmark corpus) — ARI/NMI/cluster accuracy:

```python
from graphweave.adaptation import adapt_and_refit

new_model, report = adapt_and_refit(model, labeler)
print(report["comparison"])
```

`adapt_and_refit` (the non-mutating counterpart to `adapt_embeddings_with_llm`) leaves the original model untouched and returns a new one, so before/after is always available side by side. If held-out triplet accuracy doesn't improve, it warns rather than silently shipping a worse embedder.

### Cheaper extras (no fine-tuning required)

Two additional, independent levers from Viswanathan et al. (TACL 2024) — useful with any embedder, including API-based ones:

```python
from graphweave import EmbeddingEngine
from graphweave.adaptation import generate_keyphrases, keyphrase_expand_embeddings, reassign_low_confidence

# LLM keyphrase expansion: blend per-document keyphrases into the embedding
engine = EmbeddingEngine(model_name=model.config.embedding_model, provider=model.config.embedding_provider)
keyphrases = generate_keyphrases(labeler, documents)
adapted_emb = keyphrase_expand_embeddings(documents, keyphrases, engine)

# Post-hoc correction: ask the LLM to re-adjudicate only the least-confident assignments
corrections = reassign_low_confidence(model, labeler, margin_threshold=0.15, max_docs=200)
```

**Cost**: same batched-query economics as `tune_resolution_with_llm` — a few hundred to ~1000 triplets, ~8 per LLM call, cached to disk. On Claude Haiku 4.5, a full 1000-triplet adaptation run costs well under $1.

**Detachability**: `graphweave.adaptation` is designed to be lifted into its own package later — it imports only numpy/scipy/scikit-learn/pandas/sentence-transformers plus one small compatibility shim into the existing triplet-sampling code, never the rest of GraphWeave's internals.

---

## Report Themes (qualitative research output)

For deliverables like a Findings section in a research report, an 85-topic catalog is too granular. GraphWeave can synthesize all per-topic labels into a small number of **emerging meta-themes** — each a 5–8 word evocative title plus a 4–6 sentence narrative paragraph that quotes participants and names the underlying pattern.

### End-to-end workflow

```python
from graphweave import GraphWeave, GraphWeaveConfig, LLMLabeler

# 1. Fit the model (standard)
model = GraphWeave(
    n_neighbors=10,
    config=GraphWeaveConfig(
        embedding_model="BAAI/bge-m3",
        consensus_method="graph",
        resolution=0.75,
        min_cluster_size=15,
    ),
)
model.fit(documents)

# 2. Generate report-style per-topic themes (the "catalog")
labeler = LLMLabeler(
    provider="anthropic",
    api_key="sk-ant-...",
    style="theme",
    domain_hint="education barriers in rural India",
)
model.generate_labels(labeler)

# 3. Synthesize all 80+ topics into 5-12 meta-themes (the "report")
#    Omit n_themes to let the LLM choose the natural count, or pass an int as a target.
model.generate_report_themes(labeler)
# model.generate_report_themes(labeler, n_themes=8)   # if you want a specific count

# 4. Write the final Markdown report
model.export_report("findings.md")
```

### What `generate_report_themes` does

Two LLM stages:

1. **Proposer call** — sees the title, narrative excerpt, and top keywords of every non-outlier topic in one request. Returns ~`n_themes` groupings, each `{title, topic_ids[]}`. Every topic must be assigned to exactly one meta-theme. The proposer may merge or split groups based on narrative content, not just embedding distance.
2. **Narrative writer** — one call per meta-theme. Given the title and constituent topics, writes the report paragraph using docs sampled proportionally across all member topics, aggregated keywords, and the existing topic narratives as context.

Cost is small: typically 1 proposer call (~5K input tokens) plus N narrative calls (~3K each). With Claude Haiku, a full report of 8 meta-themes is well under $0.20.

### Sample output (`findings.md`)

```markdown
# Emerging Themes

## 1. Aadhar Card as Gatekeeper to Schooling
Parents repeatedly described being turned away from admission because their
child lacked an Aadhar card. The phrase 'no card, no admission' recurred
across both rural and peri-urban accounts. Many families had been waiting
months for documentation, while their children remained out of school...

## 2. Distance to Anganwadi Centre Disrupts Early Learning
Mothers said the nearest centre was 'too far' for small children to walk
unaccompanied, and described carrying younger siblings on the route. Where
centres existed, irregular hours and supply gaps further eroded attendance...
```

### Editing themes after the fact

```python
# Resample the narrative for theme 3 (no topic changes)
model.regenerate_theme(labeler, theme_id=3)

# Move topics 12 and 17 into theme 3 and resample
model.regenerate_theme(labeler, theme_id=3, new_topic_ids=[5, 8, 12, 17])
```

### Programmatic access

```python
for theme in model.report_themes_:
    print(f"{theme.theme_id}. {theme.title}")
    print(f"   ({len(theme.topic_ids)} topics, {theme.total_size} docs)")
    print(f"   {theme.narrative}")
```

### When to use what

| Use case | Method | Output |
|----------|--------|--------|
| Topic catalog for analysts / CSV / charts | `generate_labels(labeler)` with `style="short"` | 85 topics × 5-word label + 1-2 sentence description |
| Rich per-topic catalog for browsing | `generate_labels(labeler)` with `style="theme"` | 85 topics × evocative title + narrative paragraph |
| Findings section of a research report | `generate_report_themes(labeler)` + `export_report()` | 6–10 meta-themes × narrative paragraph, plus appendix linking back to constituent topics |

If the LLM API call fails, the labeler automatically falls back to a keyword-based label (with exponential-backoff retry before giving up).

---

## Visualizations

All visualizations return interactive Plotly figures.

### Document map (2D)

2D scatter plot where each point is a document, colored by topic:

```python
fig = model.visualize(method="umap", show_outliers=True)
fig.show()
fig.write_html("document_map.html")
```

### Document map (3D)

Fully interactive 3D scatter — rotate, zoom, and hover for topic labels and document snippets:

```python
fig = model.visualize_3d()          # UMAP 3D (default)
fig = model.visualize_3d(method="pacmap")
fig = model.visualize_3d(show_outliers=False)
fig.show()
fig.write_html("document_map_3d.html")
```

### Intertopic distance map

A pyLDAvis-style overview: each bubble is a topic (area ∝ document count),
positioned by the 2-D projection of its centroid. Much faster than
`visualize()` because it only projects `n_topics` points rather than every
document — ideal for grasping the topic landscape at a glance.

```python
fig = model.visualize_topic_map()                    # MDS on cosine distances (default)
fig = model.visualize_topic_map(method="pca")        # deterministic
fig = model.visualize_topic_map(method="umap", n_keywords=8, size_scale=1.5)
fig.show()
```

| Parameter | Description |
|---|---|
| `method` | `"mds"` (default, cosine-distance MDS), `"pca"`, or `"umap"` |
| `n_keywords` | Top-N keywords shown in hover text |
| `size_scale` | Multiplier for bubble area |
| `title`, `width`, `height`, `random_state` | Standard Plotly options |

### TensorFlow Embedding Projector export

Export to [projector.tensorflow.org](https://projector.tensorflow.org) for interactive PCA / UMAP / t-SNE exploration in the browser, with topics visible as color labels:

```python
# Recommended: export raw embeddings so the projector can apply its own reduction
vectors_path, metadata_path = model.export_projector("projector_export")

# Or export pre-reduced coordinates directly
vectors_path, metadata_path = model.export_projector("projector_export", embeddings="2d")
vectors_path, metadata_path = model.export_projector("projector_export", embeddings="3d")
```

This writes two TSV files:

- `vectors.tsv` — one document per row, tab-separated floats
- `metadata.tsv` — `topic_id`, `topic_label`, top-5 `keywords`, and a 150-character document snippet per row

**To load in the projector:**
1. Go to [projector.tensorflow.org](https://projector.tensorflow.org) and click **Load**
2. Upload `vectors.tsv` as the tensor file
3. Upload `metadata.tsv` as the metadata file
4. Use **Color by → topic_label** to colour points by topic

The `embeddings` parameter accepts:

| Value | Description |
|---|---|
| `"original"` (default) | Raw embeddings — lets the projector apply PCA/UMAP/t-SNE interactively |
| `"reduced"` | Clustering-space reduced embeddings (`reduced_embeddings_`) |
| `"2d"` | Fresh UMAP/PaCMAP projection to 2D |
| `"3d"` | Fresh UMAP/PaCMAP projection to 3D |

### Topic keywords

Horizontal bar charts showing the top keywords and their scores for each topic:

```python
fig = model.visualize_topics(n_keywords=8)
fig.show()
```

### Topic hierarchy

Dendrogram showing how topics relate to each other based on centroid distances:

```python
fig = model.visualize_hierarchy()
fig.show()
```

### Topic similarity heatmap

Cosine similarity matrix between all topic centroids:

```python
from graphweave import TopicVisualizer

viz = TopicVisualizer()
fig = viz.plot_topic_similarity(model.topic_embeddings_, model.topics_)
fig.show()
```

### Topics over time

Stacked area chart showing topic prevalence over time (requires timestamps):

```python
from graphweave import TopicVisualizer

viz = TopicVisualizer()
fig = viz.plot_topic_over_time(
    labels=model.labels_,
    timestamps=your_timestamps,   # list of datetime-like values
    topics=model.topics_,
)
fig.show()
```

---

## Evaluation

```python
metrics = model.evaluate()
```

Returns a dictionary with:

| Metric | Range | Description |
|---|---|---|
| `coherence_mean` | -1 to 1 | Average NPMI coherence across topics (higher = more coherent keywords) |
| `coherence_std` | 0+ | Standard deviation of coherence across topics |
| `diversity` | 0 to 1 | Proportion of unique keywords across all topics (higher = more distinct topics) |
| `stability` | -1 to 1 | Average pairwise ARI across consensus runs (higher = more reproducible) |
| `n_topics` | 1+ | Number of non-outlier topics |
| `outlier_ratio` | 0 to 1 | Fraction of documents labeled as outliers |

Additional metrics are available as standalone functions:

```python
from graphweave.utils.metrics import (
    compute_coherence,
    compute_diversity,
    compute_stability,
    compute_silhouette,
    compute_downstream_score,
)

# Silhouette score for cluster separation
sil = compute_silhouette(model.embeddings_, model.labels_)

# Downstream classification performance
f1 = compute_downstream_score(
    model.embeddings_, model.labels_, true_labels, task="classification"
)
```

---

## Advanced Usage

### Pre-computed embeddings

Skip the embedding step by passing your own vectors:

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = encoder.encode(documents)

model = GraphWeave()
model.fit(documents, embeddings=embeddings)
```

### Multi-model embeddings

Combine embeddings from multiple models for richer representations:

```python
from graphweave import EmbeddingEngine
from graphweave.core.embeddings import MultiModelEmbedding

multi = MultiModelEmbedding(
    model_names=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
    weights=[0.5, 0.5],
)
embeddings = multi.encode(documents)

model = GraphWeave()
model.fit(documents, embeddings=embeddings)
```

### Metadata-enhanced topics

Documents with shared metadata (source, category, date) get additional graph edges:

```python
import pandas as pd

metadata = pd.DataFrame({
    "source": ["twitter", "news", "twitter", ...],
    "category": ["tech", "science", "tech", ...],
})

model = GraphWeave()
model.config.use_metadata_view = True
model.config.metadata_weight = 0.2

model.fit(documents, metadata=metadata)
```

Categorical columns create edges between documents with matching values. Numerical columns create edges between documents with similar values (similarity > 0.8 after normalization).

### Target number of topics

Use `n_topics` to automatically find the Leiden resolution that produces a specific number of topics:

```python
model = GraphWeave(n_topics=10)
model.fit(documents)
# GraphWeave uses bidirectional resolution search to find ~10 topics
```

### Finding the optimal resolution

The resolution parameter controls how many topics Leiden produces. You can search for the best value:

```python
from graphweave.core.clustering import ConsensusLeiden

# After initial fit
clusterer = ConsensusLeiden()
optimal = clusterer.find_optimal_resolution(
    graph=model.graph_,
    resolution_range=(0.5, 2.0),
    n_steps=10,
    target_n_topics=15,       # optional: aim for ~15 topics
)
print(f"Optimal resolution: {optimal}")

# Re-fit with the optimal resolution
model.config.resolution = optimal
model.fit(documents)
```

### Disabling features

```python
# No iterative refinement (faster, less accurate)
model = GraphWeave(use_iterative_refinement=False)

# No dimensionality reduction
model.config.use_dim_reduction = False

# No lexical view (embeddings only)
model.config.use_lexical_view = False
```

### Complete workflow

```python
from graphweave import GraphWeave, GraphWeaveConfig, LLMLabeler

# 1. Configure
config = GraphWeaveConfig(
    embedding_model="all-mpnet-base-v2",
    n_neighbors=20,
    graph_type="hybrid",
    use_dim_reduction=True,
    reduced_dims=10,
    n_consensus_runs=15,
    use_iterative_refinement=True,
    max_iterations=7,
    convergence_threshold=0.97,
    keyword_method="ctfidf",
    n_keywords=15,
)

# 2. Fit
model = GraphWeave(config=config)
model.fit(documents)

# 3. Reduce outliers
model.reduce_outliers(strategy="embeddings", threshold=0.05)

# 4. Merge to desired granularity
model.reduce_topics(10)

# 5. Label with LLM
labeler = LLMLabeler(provider="anthropic", api_key="...")
model.generate_labels(labeler)

# 6. Evaluate
metrics = model.evaluate()

# 7. Explore
print(model.get_topic_info())
print(f"Probabilities shape: {model.probabilities_.shape}")

fig = model.visualize()
fig.show()

# 8. Save
model.save("production_model.pkl")
```

---

## API Reference

### GraphWeave

The main model class. Follows the scikit-learn fit/transform pattern.

| Method | Description |
|---|---|
| `fit(documents, embeddings?, metadata?)` | Fit the model. Returns `self`. |
| `fit_transform(documents, embeddings?, metadata?)` | Fit and return hard labels. |
| `transform(documents)` | Assign topics to new documents. Returns labels array. |
| `transform_proba(documents)` | Get soft probabilities for new documents. Returns `(n_docs, n_topics)` matrix. |
| `reduce_outliers(strategy?, threshold?)` | Reassign outliers. Strategies: `"embeddings"`, `"neighbors"`. Returns `self`. |
| `reduce_topics(n_topics)` | Merge down to `n_topics` non-outlier topics. Returns `self`. |
| `merge_topics(topics_to_merge)` | Merge specific topic IDs into one. Returns `self`. |
| `get_topic_info()` | DataFrame with Topic, Size, Keywords, Label, Coherence columns. |
| `get_topic(topic_id)` | Get `TopicInfo` for a specific topic. |
| `get_representative_docs(topic_id, n_docs?)` | Get `(index, text)` tuples for a topic's most central documents. |
| `generate_labels(labeler, topics?, dedup?, dedup_passes?)` | Generate LLM labels for topics. Dedup is on by default. |
| `generate_report_themes(labeler, n_themes?, n_docs_per_theme?)` | Synthesize per-topic labels into report-style meta-themes. If `n_themes` is omitted the LLM chooses the natural count (typically 5–12); pass an int to target a specific count ±2. Stores on `model.report_themes_`. |
| `regenerate_theme(labeler, theme_id, new_topic_ids?, n_docs?)` | Resample a single meta-theme's narrative, optionally moving topics in/out. |
| `export_report(path, include_appendix?)` | Write meta-themes as a Markdown report. |
| `evaluate()` | Compute coherence, diversity, stability, and outlier ratio. |
| `visualize(method?, show_outliers?, ...)` | 2D document scatter plot. |
| `visualize_3d(method?, show_outliers?, ...)` | Interactive 3D document scatter plot. |
| `visualize_topic_map(method?, ...)` | Intertopic distance map — 2D projection of topic centroids, bubbles sized by topic count. |
| `export_projector(output_dir?, embeddings?)` | Export `vectors.tsv` + `metadata.tsv` for [projector.tensorflow.org](https://projector.tensorflow.org). Returns `(vectors_path, metadata_path)`. |
| `visualize_topics(n_keywords?, ...)` | Keyword bar charts per topic. |
| `visualize_hierarchy(...)` | Topic dendrogram. |
| `save(path)` | Pickle model to disk (includes all state, reducer, probabilities). |
| `GraphWeave.load(path)` | Class method to load a saved model. |

### Key attributes after fit

| Attribute | Type | Description |
|---|---|---|
| `labels_` | `np.ndarray` | Hard topic assignment per document. -1 = outlier. |
| `probabilities_` | `np.ndarray` | Soft assignments, shape `(n_docs, n_topics)`. Rows sum to ~1. |
| `embeddings_` | `np.ndarray` | Full-dimensional document embeddings (refined if iterative). |
| `reduced_embeddings_` | `np.ndarray` | Low-dimensional embeddings used for graph building. |
| `topic_embeddings_` | `np.ndarray` | Centroid embedding per topic, shape `(n_topics, embed_dim)`. |
| `topics_` | `list[TopicInfo]` | List of `TopicInfo` objects with keywords, scores, centroids. |
| `documents_` | `list[str]` | Stored training documents. |
| `graph_` | `igraph.Graph` | The final fused graph. |

### TopicInfo

| Field | Type | Description |
|---|---|---|
| `topic_id` | `int` | Topic ID (-1 for outliers). |
| `size` | `int` | Number of documents in the topic. |
| `keywords` | `list[str]` | Ranked keywords. |
| `keyword_scores` | `list[float]` | Keyword importance scores. |
| `representative_docs` | `list[int]` | Indices of documents closest to centroid. |
| `label` | `str \| None` | LLM-generated label. |
| `description` | `str \| None` | LLM-generated description. |
| `centroid` | `np.ndarray \| None` | Topic centroid embedding. |
| `coherence` | `float \| None` | NPMI coherence score (after `evaluate()`). |

### Supporting classes

| Class | Module | Purpose |
|---|---|---|
| `GraphWeaveConfig` | `graphweave.core.model` | All configuration parameters (see [Configuration Reference](#configuration-reference)) |
| `EmbeddingEngine` | `graphweave.core.embeddings` | Encode documents with sentence-transformers (local) or Google Gemini API. Supports Instructor, BGE, and API-based models. |
| `MultiModelEmbedding` | `graphweave.core.embeddings` | Combine embeddings from multiple models. |
| `GraphBuilder` | `graphweave.core.graph_builder` | Build kNN, mutual kNN, SNN, hybrid, lexical, and metadata graphs. |
| `ConsensusLeiden` | `graphweave.core.clustering` | Leiden clustering with consensus and resolution search. |
| `HDBSCANClusterer` | `graphweave.core.clustering` | Alternative HDBSCAN clustering. |
| `KeywordExtractor` | `graphweave.core.keywords` | c-TF-IDF, BM25, and KeyBERT keyword extraction. |
| `KeyphraseExtractor` | `graphweave.core.keywords` | Multi-word keyphrase extraction (YAKE). |
| `LLMLabeler` | `graphweave.labeling.llm_labeler` | Generate labels via Claude or GPT-4. |
| `SimpleLabeler` | `graphweave.labeling.llm_labeler` | Rule-based labels from top keywords. |
| `TopicVisualizer` | `graphweave.visualization.plotter` | All Plotly visualizations. |

---

## Architecture

### Graph types

**kNN:** Each document connects to its k nearest neighbors. Simple but includes asymmetric "one-way" connections that can bridge unrelated clusters.

**Mutual kNN:** Only keeps edges where both nodes are in each other's neighborhoods. This removes noise bridges and produces cleaner clusters.

**SNN (Shared Nearest Neighbors):** Edge weight equals the number of shared neighbors between two nodes, normalized by k. This captures structural similarity and is robust against noise.

**Hybrid (default):** Weighted combination of mutual kNN and SNN: `(1 - snn_weight) * mutual_kNN + snn_weight * SNN`. Gives both direct similarity (mutual kNN) and structural similarity (SNN).

### Consensus clustering

Running Leiden once is sensitive to random initialization. GraphWeave runs it `n_consensus_runs` times (default: 10) with different seeds and builds a co-occurrence graph recording how often each document pair was assigned to the same cluster (see [Memory Optimization](#memory-optimization-for-large-datasets) for the default graph-based consensus path, and the legacy dense/hierarchical alternative). The stability score (average pairwise ARI across runs) quantifies how reproducible the clustering is.

### Iterative refinement

After an initial clustering pass, document embeddings are softly blended toward their topic centroid: `refined = 0.8 * original + 0.2 * centroid`, then L2-normalized. The full pipeline (graph + clustering) re-runs on the refined embeddings. This process converges when consecutive partitions have ARI >= 0.95 (configurable). The effect is tighter, more separated topic clusters.

### Supported embedding models

#### Local models (sentence-transformers)

Any model from the [sentence-transformers](https://www.sbert.net/) library works. Recommended choices:

| Model | Dimensions | Speed | Quality | Notes |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default. Best speed/quality tradeoff. |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality, 2x slower. |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Best | State-of-the-art for English. |
| `BAAI/bge-m3` | 1024 | Slow | Best | Multilingual support. |
| `hkunlp/instructor-large` | 768 | Slow | Best | Task-specific with instructions. |

#### API-based embedding (Google Gemini)

To use Google Gemini embeddings instead of a local model, set `embedding_provider="google"`.
Requires: `pip install 'graphweave[llm]'`

```python
from graphweave import GraphWeave, GraphWeaveConfig

config = GraphWeaveConfig(
    embedding_provider="google",
    embedding_api_key="YOUR_GOOGLE_API_KEY",
    # Optional tuning:
    embedding_output_dim=768,        # Matryoshka compression (128–3072); default 768
    embedding_api_batch_size=100,    # docs per request (max 250)
    embedding_batch_delay=4.0,       # set ~4.0 on Gemini free tier (5–15 RPM)
)
model = GraphWeave(config=config)
model.fit(documents)
```

**Supported Gemini models:**

| Model | Default dims | MRL range | task_type | Notes |
|---|---|---|---|---|
| `gemini-embedding-2` | 768* | 128–3072 | via prompt prefix | Default. Best quality, 8192 tok/text. |
| `gemini-embedding-001` | 768 | — | ✅ (`CLUSTERING` etc.) | Stable, 2048 tok/text. |

\* `gemini-embedding-2` output defaults to 768 dims (Matryoshka truncation). Pass `embedding_output_dim=3072` for full resolution — same API cost, higher memory.

**With `gemini-embedding-2` (default, best quality):**
```python
config = GraphWeaveConfig(
    embedding_provider="google",
    embedding_api_key="YOUR_GOOGLE_API_KEY",
    embedding_model="gemini-embedding-2",    # explicit; also the default
    embedding_output_dim=768,                # Matryoshka default (pass 3072 for full resolution)
    embedding_api_batch_size=100,            # docs per request (max 250)
    embedding_batch_delay=4.0,               # ~4.0 for free tier; 0.0 for paid
)
model = GraphWeave(config=config)
model.fit(documents)
```

**With `gemini-embedding-001` (stable, supports task_type):**
```python
config = GraphWeaveConfig(
    embedding_provider="google",
    embedding_api_key="YOUR_GOOGLE_API_KEY",
    embedding_model="gemini-embedding-001",
    embedding_task_type="CLUSTERING",        # optimises embeddings for topic modeling
)
model = GraphWeave(config=config)
model.fit(documents)
```

For `gemini-embedding-2`, `embedding_task_type` is automatically applied as a prompt prefix (the model does not accept it as an API parameter).

---

## Comparison with BERTopic

| Aspect | BERTopic | GraphWeave |
|---|---|---|
| **Graph construction** | kNN only | Mutual kNN + SNN hybrid |
| **Dimensionality reduction** | UMAP (for clustering) | UMAP/PaCMAP (configurable) |
| **Clustering** | HDBSCAN (single run) | Leiden with consensus (n runs) |
| **Stability** | Low (varies between runs) | High (consensus + stability score) |
| **Input signals** | Embeddings only | Semantic + Lexical + Metadata |
| **Refinement** | None | Iterative embedding refinement |
| **Coverage** | ~80% (19.2% outliers avg.) | **100%** (0% outliers) |
| **Soft assignments** | Via HDBSCAN probabilities | Cosine similarity + softmax |
| **Outlier reduction** | 4 strategies | 2 strategies (embeddings, neighbors) |
| **Topic merging** | Hierarchical | Hierarchical + manual merge |
| **Keyword extraction** | c-TF-IDF | c-TF-IDF, BM25, or KeyBERT |
| **LLM labels** | Via representation model | Built-in Claude/GPT-4 support |
| **NMI (benchmark avg.)** | 0.513 | **0.575 (+12.1%)** |
| **Coherence (benchmark avg.)** | 0.233 | **0.341 (+46.4%)** |

---

## Benchmarks

Evaluated on four standard text classification datasets against BERTopic, LDA (scikit-learn), and NMF (scikit-learn). Each configuration was run with 3 random seeds across multiple topic counts (k). Metrics: NMI against ground-truth labels, NPMI coherence, and coverage (1 - outlier fraction).

### Overall Results

| Model | Mean NMI | Mean Coherence (NPMI) | Mean Coverage | Wins (NMI) |
|---|---|---|---|---|
| **GraphWeave** | **0.575** | **0.341** | **1.000** | **4/4 datasets** |
| BERTopic | 0.513 | 0.233 | 0.808 | 0/4 |
| NMF | 0.416 | 0.330 | 1.000 | 0/4 |
| LDA | 0.299 | 0.161 | 1.000 | 0/4 |

GraphWeave achieves the **highest NMI on every single dataset** while maintaining 100% corpus coverage (zero outliers). BERTopic's HDBSCAN leaves 19.2% of documents unassigned on average.

### Per-Dataset NMI

| Dataset | Docs | k range | GraphWeave | BERTopic | NMF | LDA |
|---|---|---|---|---|---|---|
| 20 Newsgroups | 2,000 | 10-50 | **0.532** | 0.519 | 0.319 | 0.158 |
| BBC News | 1,225 | 3-20 | **0.702** | 0.642 | 0.648 | 0.505 |
| AG News | 2,000 | 3-20 | **0.527** | 0.380 | 0.191 | 0.027 |
| Arxiv | 2,000 | 5-25 | **0.540** | 0.511 | 0.505 | 0.508 |

### Per-Dataset Coherence (NPMI)

| Dataset | GraphWeave | BERTopic | NMF | LDA |
|---|---|---|---|---|
| 20 Newsgroups | **0.413** | 0.223 | 0.374 | 0.256 |
| BBC News | **0.380** | 0.082 | 0.336 | 0.154 |
| AG News | 0.269 | 0.161 | **0.325** | 0.092 |
| Arxiv | 0.303 | **0.466** | 0.277 | 0.150 |

### Methodology

- All embeddings: `all-MiniLM-L6-v2` (384 dimensions)
- BERTopic: default HDBSCAN settings with UMAP reduction
- NMF / LDA: scikit-learn implementations with TF-IDF input
- GraphWeave: default settings (hybrid graph, consensus Leiden, iterative refinement)
- 3 random seeds per configuration, results averaged
- Full reproduction script: [`run_benchmark.py`](run_benchmark.py)

```bash
# Full reproduction (needs network + `pip install -e ".[benchmark]"`): downloads
# 20 Newsgroups / BBC News / AG News / Arxiv and all-MiniLM-L6-v2, then runs
# GraphWeave, BERTopic, NMF, and LDA across each dataset's documented k-range.
python run_benchmark.py

# Fast, no-download smoke test (synthetic data) — proves the harness runs
# end-to-end; this is what CI runs on every push, NOT a reproduction of the
# numbers above.
python run_benchmark.py --quick
```

---

## Citation

If you use GraphWeave in academic work, please cite the software and the methods it builds on.

### Software

```bibtex
@software{graphweave2026,
  author    = {Mathew, Nevil},
  title     = {GraphWeave: Multi-View Graph Topic Modeling with Iterative Refinement},
  year      = {2026},
  url       = {https://github.com/nevil-mathew/topic-extraction-poc}
}
```

GraphWeave began as a derivative of `tritopic`; if you're citing the underlying method
lineage rather than this specific codebase, consider also citing the original:

```bibtex
@software{tritopic2025,
  author    = {Egger, Roman},
  title     = {TriTopic: Tri-Modal Graph Topic Modeling with Iterative Refinement},
  year      = {2025},
  publisher = {PyPI},
  url       = {https://github.com/SmartVisions-AI/tritopic}
}
```

### Underlying methods

```bibtex
@article{traag2019leiden,
  author  = {Traag, V. A. and Waltman, L. and van Eck, N. J.},
  title   = {From {L}ouvain to {L}eiden: Guaranteeing Well-Connected Communities},
  journal = {Scientific Reports},
  volume  = {9},
  pages   = {5233},
  year    = {2019},
  doi     = {10.1038/s41598-019-41695-0},
  url     = {https://www.nature.com/articles/s41598-019-41695-0}
}

@article{lancichinetti2012consensus,
  author  = {Lancichinetti, Andrea and Fortunato, Santo},
  title   = {Consensus Clustering in Complex Networks},
  journal = {Scientific Reports},
  volume  = {2},
  pages   = {336},
  year    = {2012},
  doi     = {10.1038/srep00336},
  url     = {https://www.nature.com/articles/srep00336},
  note    = {Foundation for the default `consensus_method="graph"` path.}
}

@article{mullner2013fastcluster,
  author  = {M{\"u}llner, Daniel},
  title   = {fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for {R} and {P}ython},
  journal = {Journal of Statistical Software},
  volume  = {53},
  number  = {9},
  pages   = {1--18},
  year    = {2013},
  doi     = {10.18637/jss.v053.i09},
  url     = {https://danifold.net/fastcluster.html},
  note    = {Used by the legacy `consensus_method="hierarchical"` path when installed via the `legacy-consensus` extra.}
}

@article{mcinnes2018umap,
  author  = {McInnes, Leland and Healy, John and Melville, James},
  title   = {{UMAP}: Uniform Manifold Approximation and Projection for Dimension Reduction},
  journal = {arXiv preprint arXiv:1802.03426},
  year    = {2018},
  url     = {https://arxiv.org/abs/1802.03426}
}

@inproceedings{zhang2023clusterllm,
  author    = {Zhang, Yuwei and Wang, Zihan and Shang, Jingbo},
  title     = {{ClusterLLM}: Large Language Models as a Guide for Text Clustering},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {13903--13920},
  year      = {2023},
  url       = {https://arxiv.org/abs/2305.14871},
  note      = {Triplet-query approach underlying LLM-guided granularity calibration and embedding adaptation.}
}

@inproceedings{douglas2026prism,
  author    = {Douglas, Connor and Balci, Utkucan and Aylett-Bullock, Joseph},
  title     = {{PRISM}: LLM-Guided Semantic Clustering for High-Precision Topics},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  pages     = {8701--8704},
  year      = {2026},
  doi       = {10.1145/3774904.3792944},
  url       = {https://arxiv.org/abs/2604.03180},
  note      = {Sampling/evaluation design underlying LLM-guided embedding adaptation.}
}

@article{viswanathan2024fewshot,
  author  = {Viswanathan, Vijay and Gashteovski, Kiril and Lawrence, Carolin and Wu, Tongshuang and Neubig, Graham},
  title   = {Large Language Models Enable Few-Shot Clustering},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {12},
  pages   = {321--333},
  year    = {2024},
  url     = {https://arxiv.org/abs/2307.00524},
  note    = {Keyphrase expansion and low-confidence correction techniques.}
}

@article{mcinnes2017hdbscan,
  author  = {McInnes, Leland and Healy, John and Astels, Steve},
  title   = {hdbscan: Hierarchical density based clustering},
  journal = {Journal of Open Source Software},
  volume  = {2},
  number  = {11},
  pages   = {205},
  year    = {2017},
  doi     = {10.21105/joss.00205},
  note    = {Basis for the min\_cluster\_size / min\_cluster\_fraction scale-invariance argument.}
}

@article{fortunato2007resolution,
  author  = {Fortunato, Santo and Barthelemy, Marc},
  title   = {Resolution limit in community detection},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {104},
  number  = {1},
  pages   = {36--41},
  year    = {2007},
  doi     = {10.1073/pnas.0605965104},
  note    = {Basis for the min\_cluster\_size / min\_cluster\_fraction scale-invariance argument.}
}
```

## Acknowledgments

GraphWeave's starting codebase came from [`tritopic`](https://pypi.org/project/tritopic/), an
MIT-licensed topic modeling library published to PyPI by **Roman Egger** (SmartVisions AI). The
original GitHub repository ([SmartVisions-AI/tritopic](https://github.com/SmartVisions-AI/tritopic))
didn't include browsable source, so the codebase here started from the published package.

Since then, most core modules have been substantially rewritten (several have roughly doubled
in size or more), and entire subsystems were added that don't exist upstream: cumulative /
batch-wise streaming clustering, LLM-guided embedding adaptation, LLM-guided granularity
calibration, quote verification for LLM-generated narratives, and the adaptive FAISS/HNSW kNN
backend.

The MIT License permits this kind of derivative work freely; it only requires that the original
copyright notice be preserved. That notice — the full original license text plus a longer
provenance note — lives in [`NOTICE.md`](NOTICE.md).

## License

MIT License. See [LICENSE](LICENSE) for details, and [NOTICE.md](NOTICE.md) for the
inherited third-party notice from the original `tritopic` project.

## Contributing

Contributions welcome! Please open an issue or pull request on [GitHub](https://github.com/nevil-mathew/topic-extraction-poc).

## Links

- **Repository:** [GitHub](https://github.com/nevil-mathew/topic-extraction-poc)
- **PyPI:** [graphweave](https://pypi.org/project/graphweave/)
- **Issues:** [Bug reports & feature requests](https://github.com/nevil-mathew/topic-extraction-poc/issues)
- **Prior work:** [tritopic on PyPI](https://pypi.org/project/tritopic/) / [SmartVisions-AI/tritopic on GitHub](https://github.com/SmartVisions-AI/tritopic)
