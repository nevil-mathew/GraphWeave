# GraphWeave Cheat Sheet

Quick reference for common tasks and concepts.

---

## 🚀 Quick Start

```python
from graphweave import GraphWeave, GraphWeaveConfig

# For large datasets (recommended) — graph consensus is the default
# and is memory-safe out of the box, no config needed
config = GraphWeaveConfig()
model = GraphWeave(config)
model.fit(documents)

# Access results
print(f"Topics: {len(np.unique(model.labels_))}")
print(f"Stability: {model.stability_score_:.3f}")
```

---

## 📊 What's Happening?

### The Pipeline (5 Steps)

```
1. EMBEDDING      → Convert text to vectors
2. GRAPH BUILDING → Compute document similarity
3. LEIDEN 10x     → Cluster with different random seeds
4. CONSENSUS      → Find agreement between 10 runs
5. REFINE         → Improve for next iteration
                    (Repeat steps 2-5 for 5 iterations)
```

### Memory Spike Locations

```
Default consensus_method="graph" (Lancichinetti-Fortunato):
  ✓ Sparse co-occurrence + Leiden on thresholded graph
  └─ Consensus-step peak: <1 GB at 43k docs (README benchmark: ~0.3 GB @ 20k, ~0.8 GB @ 50k)
  └─ scipy.linkage is NOT called
  └─ Reference: Sci. Rep. 2:336 (2012)

Legacy consensus_method="hierarchical":
  ✗ DANGEROUS: Co-occurrence matrix (step 4)
    └─ Dense: 43k × 43k = 14 GB ← OOM risk
    └─ Sparse + condensed: ~7 GB ✓ (with low_memory=True)

  ✗ MODERATE: linkage (after step 4)
    └─ fastcluster: 2-4 GB workspace (if installed)
    └─ scipy fallback: 6-10 GB workspace

✓ SAFE: All other steps < 3 GB
```

---

## 🎯 Configuration Cheat Sheet

| Goal | Setting | Code |
|------|---------|------|
| **Large dataset** | *(nothing to set)* | Default `consensus_method="graph"` is already memory-safe |
| **Fast processing** | max_iterations | `max_iterations=3` |
| **More topics** | resolution | `resolution=1.5` |
| **Fewer topics** | resolution | `resolution=0.5` |
| **Better quality** | n_neighbors | `n_neighbors=25` |
| **Faster clustering** | n_neighbors | `n_neighbors=10` |
| **Stop early** | convergence_threshold | `convergence_threshold=0.85` |
| **More consensus** | n_runs | Increase Leiden runs (memory cost) |
| **Memory-safe consensus** | consensus_method | `consensus_method="graph"` (default) |
| **Stricter agreement** | consensus_threshold_tau | `consensus_threshold_tau=0.7` (more conservative) |
| **Looser agreement** | consensus_threshold_tau | `consensus_threshold_tau=0.3` (more edges retained) |
| **Legacy linkage** | consensus_method | `consensus_method="hierarchical"` (small N only) |
| **Diverse LLM context** | labeling_sample_strategy | `labeling_sample_strategy="mmr"` (avoids near-paraphrase docs) |
| **Edge-aware LLM context** | labeling_sample_strategy | `labeling_sample_strategy="stratified"` (60/30/10 close/mid/far) |
| **MMR relevance/diversity** | mmr_lambda | `mmr_lambda=0.7` (more relevance) / `0.3` (more diversity) |
| **Adaptive kNN (default)** | knn_backend | `knn_backend="auto"` — exact under 5k, hnswlib HNSW above (needs `pip install graphweave[fast-knn]`) |
| **Exact kNN (reproducible)** | knn_backend | `knn_backend="exact"` (sklearn NearestNeighbors at every size) |
| **Force HNSW** | knn_backend | `knn_backend="hnsw"` (always hnswlib; requires `fast-knn` extra) |
| **HNSW size thresholds** | hnsw_small_threshold / hnsw_large_threshold | defaults `5_000` / `50_000` — below = exact, between = `M=16, ef=200`, at/above = `M=32, ef=400` |

---

## 📈 Interpreting Results

### Stability Score

```
model.stability_score_

0.90-1.0  ✓✓✓ Excellent (Leiden runs very consistent)
0.80-0.89 ✓✓  Good (Leiden runs mostly agree)
0.70-0.79 ✓   OK (Some variation, acceptable)
0.60-0.69 ⚠️  Weak (Concerning variation)
<0.60     ✗✗  Bad (Leiden runs very different)
```

### Iteration History (ARI)

```
model._iteration_history

Iteration 1: ARI = None         (baseline)
Iteration 2: ARI = 0.85         (good jump)
Iteration 3: ARI = 0.92         (convergence starting)
Iteration 4: ARI = 0.93         (small improvement)
Iteration 5: ARI = 0.94         (converged - stop here)

Good sign: ARI increases then plateaus
Bad sign: ARI decreases or stays low
```

---

## 💾 Memory Quick Estimate

With the default `consensus_method="graph"`, none of this applies — consensus-step peak memory
stays under 1 GB through tens of thousands of docs and only ~2 GB even at 100k+ (see README's
"Memory Optimization for Large Datasets" for the exact benchmark table). The math below is for
the **legacy** `consensus_method="hierarchical"` path only:

```
Legacy hierarchical path, with low_memory=False:

Memory = (N × N × 8 bytes) / 1e9 GB

Examples:
  10k docs:   10k × 10k × 8 / 1e9 = 0.8 GB ✓
  20k docs:   20k × 20k × 8 / 1e9 = 3.2 GB ✓
  43k docs:   43k × 43k × 8 / 1e9 = 14.7 GB ✗
  50k docs:   50k × 50k × 8 / 1e9 = 20 GB ✗✗

Rule of thumb (legacy hierarchical path only): >30k docs → MUST use low_memory=True.
On the default graph-consensus path this doesn't apply — there's no N×N densification.
```

---

## 🔍 Debugging Checklist

```
Problem: Out of Memory
□ Confirm you're on the default consensus_method="graph" (memory-safe already)
□ If deliberately using consensus_method="hierarchical", set low_memory=True
□ Reduce max_iterations
□ Reduce dataset size
□ Close other applications

Problem: Unstable Clustering (stability < 0.7)
□ Check data quality
□ Adjust resolution (try 0.8, 1.0, 1.2)
□ Increase n_neighbors
□ Check if n_clusters seems reasonable

Problem: Too Many Topics
□ Lower resolution
□ Increase min_cluster_size

Problem: Too Few Topics
□ Raise resolution
□ Decrease min_cluster_size
□ Check convergence threshold

Problem: Slow Processing
□ Reduce max_iterations
□ Reduce n_neighbors (15→10)
□ Reduce reduced_dims (50→30)
□ Disable unused features (use_lexical_view=False)
```

---

## 📝 Key Concepts

### Leiden Algorithm
A **clustering algorithm** that groups similar items. Non-deterministic = different runs may produce different results. GraphWeave runs it 10 times to find consensus.

### Co-Occurrence Matrix
Tracks "how many times did documents A and B end up in same cluster?" across all 10 Leiden runs. Used to find consensus clustering.

### Stability Score
How well the 10 Leiden runs agree (0-1 scale). Higher = more robust clusters.

### ARI (Adjusted Rand Index)
Compares two clusterings (0-1 scale). Shows how much clustering changed between iterations.

### Iterative Refinement
Pull documents toward their cluster centers after each iteration. Improves embeddings, leads to better clusters next iteration.

### Blend Factor
Controls how much to refine embeddings. Decreases over iterations (start aggressive, end gentle).

---

## 🐛 Common Errors

### `MemoryError: Unable to allocate X GB`
```python
# This shouldn't happen on the default consensus_method="graph" path.
# If you're intentionally on the legacy hierarchical path, fix with:
config = GraphWeaveConfig(consensus_method="hierarchical", low_memory=True)
```

### `IndexError in co_occurrence matrix`
```python
# Usually means corrupted graph or duplicate documents
# Try: Remove duplicates, check document validity
```

### Clusters are all the same label
```python
# Resolution too low, graph too weak, or bad embeddings
# Try: Increase resolution, increase n_neighbors
config = GraphWeaveConfig(resolution=1.5, n_neighbors=25)
```

---

## 📊 Monitoring Memory

```python
import psutil
import os

process = psutil.Process(os.getpid())

def print_memory():
    rss = process.memory_info().rss / 1e9
    print(f"Memory: {rss:.1f} GB")

print_memory()
model = GraphWeave(config)
model.fit(documents)
print_memory()
```

---

## ⚡ Performance Tips

These use the default `consensus_method="graph"` — `low_memory` is not needed and has no
effect on this path.

```python
# Fastest (but lower quality)
config = GraphWeaveConfig(
    max_iterations=2,
    n_neighbors=10,
    reduced_dims=30,
    use_lexical_view=False,
)

# Balanced (recommended)
config = GraphWeaveConfig(
    max_iterations=3,
    n_neighbors=15,
    reduced_dims=50,
)

# Best quality (slower)
config = GraphWeaveConfig(
    max_iterations=5,
    n_neighbors=30,
    reduced_dims=100,
)
```

---

## 📚 File Reference

```
Where to find what:

Iterative Refinement:
  → graphweave/core/model.py
  → _refine_embeddings() method

Leiden Consensus Clustering:
  → graphweave/core/clustering.py
  → ConsensusLeiden class
  → _compute_consensus() method (default: graph consensus path)

Legacy hierarchical / low_memory paths:
  → graphweave/core/clustering.py
  → only reachable via consensus_method="hierarchical"

Graph Building:
  → graphweave/core/graph_builder.py
  → kNN, SNN, mutual_knn methods

Embedding Adaptation:
  → graphweave/adaptation/
  → LinearAdapter (pure-numpy) and EmbeddingAdapter (sentence-transformers fine-tuning)
```

---

## ✅ Pre-Flight Checklist

Before running on large dataset:

```
□ On the default consensus_method="graph"? (no memory config needed)
□ Checked RAM available?
□ Set max_iterations reasonably?
□ Disabled unused features?
□ Set convergence_threshold?
□ Have monitoring ready (psutil)?
□ Know expected number of topics?
□ Checked document quality?
```

---

## 🔗 Links

- **Main README**: [LEARNING_GUIDE/README.md](README.md)
- **Leiden Paper**: https://www.nature.com/articles/s41598-019-41695-0
- **scipy.linkage**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- **UMAP**: https://umap-learn.readthedocs.io/

---

## 💡 Remember

1. **Default `consensus_method="graph"` is memory-safe out of the box** — no config needed,
   even at 40k+ documents. `low_memory=True` only matters if you opt into the legacy
   `consensus_method="hierarchical"` path.
2. **Stability > 0.8**: Good sign ✓
3. **ARI increasing**: Converging correctly ✓
4. **Iterations usually converge**: By iteration 3-4, diminishing returns
5. **Different data = different resolution**: Experiment with resolution parameter
