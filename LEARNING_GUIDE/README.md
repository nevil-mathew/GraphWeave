# GraphWeave 2.3.0: Complete Learning Guide

A comprehensive guide to understanding how GraphWeave works, where memory issues occur, and how to optimize for large datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [The Memory Problem](#the-memory-problem)
3. [GraphWeave Pipeline](#graphweave-pipeline)
4. [Leiden Consensus Clustering](#leiden-consensus-clustering)
5. [Co-Occurrence Matrix (The Memory Bottleneck, Legacy Hierarchical Path)](#co-occurrence-matrix-the-memory-bottleneck-legacy-hierarchical-path)
6. [Iterative Refinement](#iterative-refinement)
7. [Memory Optimization](#memory-optimization)
8. [Measuring Stability](#measuring-stability)
9. [Quick Reference](#quick-reference)

---

## Overview

**GraphWeave** is a topic modeling framework that uses:
- **Embeddings**: Convert documents to numerical vectors
- **Graph Building**: Create similarity networks (kNN/SNN)
- **Leiden Clustering**: Find document clusters (with consensus)
- **Iterative Refinement**: Improve embeddings based on clustering results
- **LLM Labeling**: Generate human-readable cluster names

The main challenge: **Memory usage during consensus clustering** can cause Out-Of-Memory (OOM) crashes on large datasets.

---

## The Memory Problem

> **This describes why the co-occurrence matrix was a bottleneck before GraphWeave 2.3.0.**
> Since 2.3.0, the default `consensus_method="graph"` (see "The Solution" below) avoids the
> problem entirely — peak extra memory stays under 1 GB through tens of thousands of docs. Read
> on for the historical context that motivates the default, or jump straight to "The Solution".

### The Issue (legacy hierarchical path, `low_memory=False`)

When processing **43,000 documents** on the legacy `consensus_method="hierarchical"` path with
`low_memory=False`, memory usage spikes to **25+ GB** during clustering.

```
Before clustering:     2-3 GB (normal)
During clustering:     25+ GB (SPIKE)
After clustering:      2-3 GB (freed)
```

### Why It Happens (legacy hierarchical path)

The **consensus clustering step** builds a co-occurrence matrix that tracks which documents clustered together across multiple Leiden runs. On the legacy `consensus_method="hierarchical"` path with `low_memory=False`, this becomes a **43,000 × 43,000 dense table**:

- **Cells**: 43,000 × 43,000 = 1.8 billion
- **Memory per cell**: 8 bytes (float64)
- **Total**: 14.4 GB just for this matrix

Add in scipy.linkage workspace and temporary arrays, and you hit 25-30 GB.

### The Solution (default since 2.3.0): graph consensus

You don't need to configure anything — `consensus_method="graph"` is the
**default**. It replaces the N×N co-occurrence densification *and* the
`scipy.linkage` step with a single Leiden pass on a thresholded sparse
co-occurrence graph (Lancichinetti & Fortunato, *Consensus clustering in
complex networks*, Sci. Rep. 2:336, 2012 —
[nature.com/articles/srep00336](https://www.nature.com/articles/srep00336)).

```python
from graphweave import GraphWeave, GraphWeaveConfig

config = GraphWeaveConfig()   # consensus_method="graph" is already the default
model = GraphWeave(config)
model.fit(documents)          # peak extra memory well under 1 GB, even at 43k docs
```

Tune agreement strictness with `consensus_threshold_tau` (default 0.5 — keep
pairs that co-cluster in ≥50% of runs) if you want stricter/looser consensus;
it has no memory cost either way.

### Legacy path: `consensus_method="hierarchical"` + `low_memory`

Before 2.3.0, the only lever was `low_memory=True`, which kept the
co-occurrence matrix sparse instead of densifying it:

```python
config = GraphWeaveConfig(
    consensus_method="hierarchical",
    low_memory=True,
)
```

This reduces memory from **25 GB → 10 GB** — still far worse than the
default graph path's <1 GB. Only use this path if you need reproducibility
against older GraphWeave/tritopic results, or specifically want
`scipy.linkage`-based hierarchical clustering. Install
`graphweave[legacy-consensus]` to pull in `fastcluster`, which replaces
`scipy.linkage` with a C++ implementation (Θ(N²) time, no hidden float64
copy) when you do use this path.

**On the default graph path, `low_memory` has no effect at all** — it only
applies inside the hierarchical path above.

---

## GraphWeave Pipeline

Here's what happens when you call `model.fit(documents)`:

### Step 1: Embedding (5-10 GB)
```
Documents → AI Model → Vectors (43,000 docs × 768 dimensions)
"The meeting was productive" → [0.23, -0.45, 0.67, ...]
```

### Step 2: Dimension Reduction (2-3 GB temporary)
```
768 dimensions → UMAP reduction → 50 dimensions
(Keeps important information, smaller for faster clustering)
```

### Step 3: Iterative Refinement Loop (5 iterations)

Each iteration:

```
a) Build Graph
   └─ Compute similarity between documents (kNN/SNN)
   
b) LEIDEN CLUSTERING (Memory spike here!)
   ├─ Run Leiden 10 times with different random seeds
   ├─ Get 10 different cluster assignments
   ├─ Build co-occurrence matrix
   └─ Find consensus clustering
   
c) Refine Embeddings
   └─ Pull documents closer to their cluster centers
   
d) Check Convergence
   └─ Compare with previous iteration (ARI score)
   └─ Stop if converged
```

**Adaptive kNN backend.** Step (a) does a kNN search over document
embeddings to seed the similarity graph. As of 2.3.0, `knn_backend="auto"`
(default) switches strategies by corpus size:

- **< 5,000 docs** → exact sklearn `NearestNeighbors` (no change vs. earlier
  versions; results are byte-for-byte reproducible).
- **5,000 – 49,999 docs** → hnswlib HNSW with `M=16`, `ef=200`.
- **≥ 50,000 docs** → hnswlib HNSW with `M=32`, `ef=400`.

The dispatch is invisible to the rest of the pipeline — mutual-kNN
filtering, SNN counting and multi-view fusion get the same `(neighbor_id,
similarity)` shape regardless of which backend ran. Install hnswlib via
`pip install graphweave[fast-knn]`; if it isn't available the adaptive path
silently falls back to exact at every size. Force the exact path with
`knn_backend="exact"` for reproducibility benchmarks.

### Step 4: Label Generation (2-3 GB temporary)
```
Clusters → LLM → Human-readable labels
"Topic 1: Business Development" (based on cluster contents)
```

**Representative-doc sampling.** Before calling the LLM, GraphWeave picks `n_representative_docs` per topic to send as context. The selection is controlled by `labeling_sample_strategy`:

- `"centroid"` *(default)* — closest-to-centroid only. Best for tight, coherent topics.
- `"mmr"` — Maximal Marginal Relevance. Picks docs that are close to the centroid **and** diverse from each other (tuned by `mmr_lambda`, default 0.5). Use when centroid-closest docs are near-paraphrases and you want broader coverage without absorbing outliers.
- `"stratified"` — sorts members by distance-to-centroid, bins into close/mid/far, samples per `stratified_proportions` (default 60/30/10). Use when you want explicit control over how much "edge" of the topic the LLM sees.

All three are deterministic. If `get_representative_docs(topic_id, n_docs=N)` is called with `N` larger than was precomputed, the sampling re-runs over the full topic in the configured style.

### Memory Timeline

```
Start:              2 GB
After embedding:    5-10 GB
After UMAP:         7-12 GB
Iteration 1:        25 GB (peak) → 7 GB
Iteration 2:        25 GB (peak) → 7 GB
Iteration 3:        25 GB (peak) → 7 GB
Iteration 4:        25 GB (peak) → 7 GB
Iteration 5:        25 GB (peak) → 7 GB
After labeling:     3-5 GB
```

---

## Leiden Consensus Clustering

### What is Leiden?

Leiden is a **clustering algorithm** that groups similar items in a graph. It's non-deterministic, meaning:

```
Same input + same seed = same output
Same input + different seed = possibly different output
```

This randomness is a feature—it helps avoid bad local optimizations.

### Why Consensus?

Instead of running Leiden once, GraphWeave runs it **10 times with different random seeds** to find the most robust clusters:

```
Run 1 (seed=42):  [Topic A, Topic A, Topic B, Topic B, Topic C]
Run 2 (seed=43):  [Topic A, Topic B, Topic A, Topic B, Topic C]
Run 3 (seed=44):  [Topic A, Topic A, Topic B, Topic B, Topic C]
...
Run 10 (seed=51): [Topic A, Topic A, Topic B, Topic B, Topic C]
```

**Question**: Which clustering is best?

**Answer**: Find the consensus—the clustering that most runs agree on.

### The Random Seed Parameter

```python
# In ConsensusLeiden class:
for run in range(self.n_runs):  # 10 runs
    seed = self.random_state + run  # seed = 42, 43, 44, ..., 51
    partition = la.find_partition(graph, seed=seed)
```

Each seed produces slightly different results, capturing different "views" of the clustering.

### Stability Score

After consensus clustering, GraphWeave computes a **stability score**:

```python
model.stability_score_  # 0.0 to 1.0
```

This measures how well the 10 runs agree:

```
0.90 = 10 runs produce nearly identical clusters ✓ (robust)
0.70 = 10 runs mostly agree
0.50 = 10 runs have mixed results ✗ (unstable)
```

Access it:

```python
print(f"Stability: {model.stability_score_:.3f}")
if model.stability_score_ < 0.7:
    print("⚠️ Clustering is unstable!")
```

---

## Co-Occurrence Matrix (The Memory Bottleneck, Legacy Hierarchical Path)

> This whole section describes the pre-2.3.0 / legacy hierarchical path. The default
> `consensus_method="graph"` never builds this matrix — see "The Memory Problem" above.

### What Is It?

A **co-occurrence matrix** tracks how often pairs of documents end up in the same cluster across all 10 Leiden runs.

### Simple Example

Imagine 6 documents, 3 Leiden runs:

```
Run 1: [A, A, B, B, C, C]  (Doc1 & Doc2 in cluster A, Doc3 & Doc4 in B, etc.)
Run 2: [A, B, A, B, C, C]  (Doc1 & Doc3 in A, Doc2 & Doc4 in B, etc.)
Run 3: [A, A, B, B, C, C]  (Same as Run 1)
```

**Co-occurrence matrix** (how many times each pair clustered together):

```
        Doc1  Doc2  Doc3  Doc4  Doc5  Doc6
Doc1     -     2     1     0     0     0      ← Doc1 & Doc2: 2/3 runs
Doc2     2     -     0     1     0     0
Doc3     1     0     -     2     0     0
Doc4     0     1     2     -     0     0
Doc5     0     0     0     0     -     3      ← Doc5 & Doc6: 3/3 runs
Doc6     0     0     0     0     3     -
```

### Converting to Distance

```python
distance = 1 - (co-occurrence / total_runs)

Doc1 & Doc2: distance = 1 - (2/3) = 0.33   (close together)
Doc1 & Doc4: distance = 1 - (0/3) = 1.0    (far apart)
Doc5 & Doc6: distance = 1 - (3/3) = 0.0    (identical)
```

Lower distance = documents cluster together more often.

### Building the Co-Occurrence Matrix

**Step 1**: Create a one-hot indicator matrix for each run

```
For Run 1 with clusters [A, A, B, B, C]:

Indicator matrix M (6 docs × 3 topics):
       Topic A  Topic B  Topic C
Doc1     1        0        0
Doc2     1        0        0
Doc3     0        1        0
Doc4     0        1        0
Doc5     0        0        1
Doc6     0        0        1
```

**Step 2**: Compute M @ M.T (who is in the same topic)

```
M @ M.T =
       Doc1  Doc2  Doc3  Doc4  Doc5  Doc6
Doc1    1     1     0     0     0     0
Doc2    1     1     0     0     0     0
Doc3    0     0     1     1     0     0
Doc4    0     0     1     1     0     0
Doc5    0     0     0     0     1     1
Doc6    0     0     0     0     1     1
```

**Step 3**: Accumulate across all 10 runs

```
co_occur = sum of (M @ M.T) for all 10 runs

Result: Co-occurrence matrix with counts
```

### The Memory Problem

For **43,000 documents**:

```
Matrix dimensions: 43,000 × 43,000 = 1.8 billion cells
Memory per cell: 8 bytes (float64)
Total memory: 1.8 billion × 8 = 14.4 GB

This is HUGE! Your system might only have 16 GB total.
```

### Dense Path (low_memory=False)

```python
# In graphweave/core/clustering.py
co_occur_dense = co_occur.toarray() / n_runs  # ← Creates full 14 GB matrix!
condensed = squareform(distance)              # ← Creates another 7 GB!

# PEAK: 20+ GB (plus scipy workspace)
```

### Sparse Path (low_memory=True)

```python
# In graphweave/core/clustering.py
condensed = np.ones(n_pairs, dtype=np.float64)  # ← Only allocate needed space (7 GB)

# Fill only the non-zero entries from sparse matrix
mask = coo.row < coo.col
condensed[idx] = 1.0 - v

del co_occur, coo  # ← Free sparse matrix before scipy uses memory

# PEAK: 10 GB (much better!)
```

---

## Iterative Refinement

### Why Multiple Iterations?

Each iteration improves the embeddings based on what the clustering revealed. This feedback loop converges toward better clusters.

### The Process

```
Iteration 1:
├─ Cluster using original embeddings
└─ Result: OK clusters, some noise

Iteration 2:
├─ Refine embeddings by pulling documents toward their cluster centers
├─ Cluster with refined embeddings
└─ Result: Better clusters

Iteration 3:
├─ Refine further
├─ Cluster again
└─ Result: Even better

Iteration 4-5:
├─ Fine-tuning (diminishing returns)
└─ Result: Converged
```

### The Refinement Algorithm

For each cluster, compute its center (centroid) and pull documents toward it:

```python
# For each cluster:
centroid = average of all documents in cluster

# For each document in cluster:
similarity_to_center = cosine_similarity(doc, centroid)

# Documents close to center: pull HARD (confident)
# Documents far from center: pull GENTLY (might be wrong)

refined_embedding = (1 - blend) × original + blend × centroid
```

### Blend Factor (Decays Over Iterations)

```
blend_factor = 0.3 - 0.2 × (iteration / max_iterations)

Iteration 1: blend = 0.30  (aggressive refinement)
Iteration 2: blend = 0.225
Iteration 3: blend = 0.15
Iteration 4: blend = 0.075
Iteration 5: blend = 0.00  (no change)
```

Why decaying? Early iterations might be wrong, so refine gently. Later iterations are converged, so refine less.

### Convergence: The ARI Score

Each iteration compares with the previous using **Adjusted Rand Index (ARI)**:

```
ARI = 1.0: Identical clustering (converged)
ARI = 0.9: Nearly identical (converged)
ARI = 0.8: Very similar
ARI = 0.5: Moderate agreement
ARI = 0.0: Random/no agreement
```

Your typical output:

```
Iteration 1: ARI = None (first baseline)
Iteration 2: ARI = 0.8717
Iteration 3: ARI = 0.9176  (improvement)
Iteration 4: ARI = 0.9335  (small improvement)
Iteration 5: ARI = 0.9404  (tiny improvement - converged)
```

Access iteration history:

```python
print(model._iteration_history)
# [
#   {'iteration': 1, 'ari': None, 'n_topics': 15},
#   {'iteration': 2, 'ari': 0.8717, 'n_topics': 15},
#   ...
# ]
```

---

## Memory Optimization

### Why Memory Matters

```
Your computer RAM: 16 GB
GraphWeave needs: 25 GB (without optimization)
Result: Out-Of-Memory crash ✗
```

### Solution 0 (do this first): use the default graph consensus

```python
from graphweave import GraphWeave, GraphWeaveConfig

config = GraphWeaveConfig()   # consensus_method="graph" is already the default
model = GraphWeave(config)
model.fit(documents)
```

**Result**: 25 GB → well under 1 GB ✓ — no further tuning needed for memory alone.

### Solution 1 (legacy path only): use low_memory=True

Only relevant if you've opted into `consensus_method="hierarchical"`:

```python
config = GraphWeaveConfig(
    consensus_method="hierarchical",
    low_memory=True,  # ← Use sparse co-occurrence matrix
)
model = GraphWeave(config)
model.fit(documents)
```

**Result**: 25 GB → 10 GB ✓ (still worse than the default graph path)

### Solution 2: Reduce Iterations

More iterations = more clustering = more memory spikes.

```python
config = GraphWeaveConfig(
    max_iterations=3,  # ← Instead of 5
    convergence_threshold=0.85  # ← Stop earlier if converged
)
```

**Result**: 5 spikes × 25 GB → 3 spikes × 25 GB (less frequent)

### Solution 3: Reduce Dataset Size

If using 43,000 documents:

```python
# Sample down for testing
documents_sample = documents[:20000]
model.fit(documents_sample)
```

### Solution 4: Reduce Leiden Runs

```python
# In your clustering setup (if accessible)
# Instead of 10 runs, use fewer
n_runs = 5  # ← Faster, less memory
```

### Solution 5: Disable Features You Don't Need

```python
config = GraphWeaveConfig(
    use_lexical_view=False,  # ← Don't build TF-IDF matrix
    use_metadata=False,      # ← Don't use metadata
)
```

### Memory Comparison

```
Configuration                          Peak Memory  Time
──────────────────────────────────────────────────────────
Default graph consensus (50k docs)     <1 GB        10 min ✓ (default since 2.3.0)
+ max_iterations=3                     <1 GB        6 min ✓

Legacy consensus_method="hierarchical" (50k docs):
  low_memory=False                     25 GB        Crash ✗
  + low_memory=True                    10 GB        10 min ✓
  + max_iterations=3                   7-10 GB      6 min ✓
  + sample 20k docs                    15 GB        5 min ✓
  All optimizations                    5 GB         3 min ✓✓
```

---

## Measuring Stability

### What to Monitor

After fitting:

```python
model = GraphWeave(config)
model.fit(documents)

# 1. Stability Score (Leiden consensus quality)
print(f"Stability: {model.stability_score_:.3f}")

# 2. Iteration History (Convergence)
for h in model._iteration_history:
    print(f"Iteration {h['iteration']}: ARI={h['ari']:.4f}, Topics={h['n_topics']}")

# 3. Final Clusters
print(f"Found {len(np.unique(model.labels_))} clusters")
```

### What's Good?

```
Stability > 0.80:      ✓ Robust clustering
ARI in iterations:     ✓ Should increase (convergence)
Found N topics:        ✓ Should match your expectation
```

### What's Concerning?

```
Stability < 0.60:      ✗ Leiden runs disagree (unstable)
ARI decreases:         ✗ Refinement going wrong
Very few topics:       ⚠️ Resolution too low
Very many topics:      ⚠️ Resolution too high
```

### If Unstable

```python
# Try increasing Leiden runs (if memory allows)
from graphweave.core.clustering import ConsensusLeiden

clusterer = ConsensusLeiden(n_runs=15)  # ← More runs

# Or adjust resolution
config = GraphWeaveConfig(
    resolution=0.8  # ← Try different value
)
```

---

## Quick Reference

### Installation & Basic Usage

```python
from graphweave import GraphWeave, GraphWeaveConfig

# Optimized for large datasets — consensus_method="graph" (default) is
# already memory-safe; low_memory only matters on the legacy hierarchical path
config = GraphWeaveConfig(
    max_iterations=5,             # Refinement iterations
    convergence_threshold=0.90,   # Stop early if converged
    verbose=True                  # Show progress
)

model = GraphWeave(config)
model.fit(documents)

# Results
clusters = model.labels_          # Cluster assignments (-1 = outlier)
topics = model.topic_words_       # Top words per topic
labels = model.topic_labels_      # LLM-generated labels
```

### Memory Profiling

```python
import psutil
import os

process = psutil.Process(os.getpid())

print(f"Before: {process.memory_info().rss / 1e9:.1f} GB")
model.fit(documents)
print(f"After: {process.memory_info().rss / 1e9:.1f} GB")
```

### Debugging Stability Issues

```python
# 1. Check if Leiden runs agree
print(f"Stability Score: {model.stability_score_:.3f}")

# 2. Check convergence
ari_scores = [h['ari'] for h in model._iteration_history if h['ari']]
print(f"ARI progression: {ari_scores}")

# 3. Check cluster distribution
unique, counts = np.unique(model.labels_, return_counts=True)
print(f"Cluster sizes: {dict(zip(unique, counts))}")

# 4. If unstable, increase runs or adjust resolution
```

### Key Configuration Parameters

| Parameter | Impact | Default | Notes |
|-----------|--------|---------|-------|
| `consensus_method` | Consensus memory/algo | `"graph"` | Default is already memory-safe; `"hierarchical"` is the legacy path |
| `low_memory` | Memory usage | False | Only matters if `consensus_method="hierarchical"`; no effect on the default `"graph"` path |
| `max_iterations` | Convergence | 5 | More = slower but better, 3 often enough |
| `convergence_threshold` | Early stop | 0.95 | Lower = stop earlier (saves time) |
| `resolution` | Num topics | auto | Higher = more clusters, lower = fewer |
| `n_neighbors` | Graph density | 15 | Higher = slower, more connections |
| `reduced_dims` | Graph speed | 50 | Lower = faster but less info |
| `min_cluster_fraction` | Min cluster size (scale-invariant) | None | Alternative to `min_cluster_size`; expressed as a fraction of corpus size, see main README |

---

## Common Issues & Solutions

### Out of Memory Error

```
Error: MemoryError during clustering
Check first: are you on consensus_method="hierarchical"? That's the only
path where this should happen. Switch to the default consensus_method="graph"
(or add low_memory=True if you must stay on the hierarchical path).
```

### Instability (Stability < 0.6)

```
Symptom: Different runs produce very different clusters
Causes: 
  - Resolution parameter too extreme
  - Graph too sparse/dense
  - Dataset has natural noise
Solutions:
  - Adjust resolution (try 0.5-1.5)
  - Increase n_neighbors (try 20-30)
  - Check data quality
```

### Convergence Not Improving

```
Symptom: ARI score doesn't increase after iteration 2
Causes:
  - Embeddings already optimal
  - Refinement blend too weak
Solutions:
  - Try reducing max_iterations (stop earlier)
  - This is normal - refinement has limits
```

### Too Many or Too Few Clusters

```
Symptom: Got 200 topics instead of 50
Solutions:
  - Lower resolution (0.5 instead of 1.0)
  - Increase min_cluster_size
  
Symptom: Got 3 topics instead of 50
Solutions:
  - Raise resolution (1.5 instead of 1.0)
  - Decrease min_cluster_size
```

---

## Technical Details

### File Locations in GraphWeave

```text
graphweave/
├─ core/
│  ├─ model.py              ← Main GraphWeave class + iterative refinement
│  ├─ clustering.py         ← ConsensusLeiden + co-occurrence matrix
│  ├─ graph_builder.py      ← kNN, SNN, lexical graphs
│  └─ embeddings.py         ← Embedding models
└─ ...
```

### Memory Timeline for 43k Documents

```
Phase               Memory    Notes
─────────────────────────────────────
Initial            2 GB      Python overhead
Load documents     3 GB
Embeddings         10 GB     AI model output
UMAP reduction     12 GB     (temporary both versions)
Iteration 1-5
  Graph build      2 GB      (temporary)
  Leiden runs      3 GB      (temporary)
  Consensus:       <1 GB     ← default consensus_method="graph"
                   25 GB     ← legacy hierarchical path, low_memory=False
                   10 GB     ← legacy hierarchical path, low_memory=True
  Refinement       4 GB      (temporary)
Cleanup            3 GB
```

### Complexity Analysis

```
Operation                        Time Complexity    Space Complexity
──────────────────────────────────────────────────────────────────
Building embeddings              O(d × n)           O(d × n)
UMAP reduction                  O(n log n)         O(n)
Leiden clustering                O(m + n log n)     O(m)  (m=edges)
Consensus (graph, default)       O(m log n)         O(m)  (thresholded co-cluster graph)
Co-occurrence (sparse, legacy)   O(10n + k)         O(k)  (k=non-zero)
Co-occurrence (dense, legacy)    O(10n²)            O(n²) ← BOTTLENECK
Hierarchical clustering (legacy) O(n² log n)        O(n²)
Iterative refinement             O(n × iterations)  O(n)
```

---

## Further Reading

- Leiden Algorithm: https://www.nature.com/articles/s41598-019-41695-0
- Hierarchical Clustering: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- UMAP: https://umap-learn.readthedocs.io/
- Topic Modeling: https://en.wikipedia.org/wiki/Topic_model

---

## Newer Features (see main README for full detail)

This guide focuses on the core pipeline and memory model. A few things added since this guide
was first written are documented in `README.md` rather than duplicated here:

| Feature | What it does | README section |
|---------|---------------|-----------------|
| LLM-guided embedding adaptation | Fine-tune embeddings via LLM-judged triplets (`graphweave.adaptation`); pure-numpy `LinearAdapter` or full `EmbeddingAdapter` | "LLM-Guided Embedding Adaptation" |
| LLM-guided granularity calibration | Let an LLM help pick `resolution`/cluster granularity instead of guessing | "LLM-Guided Granularity Calibration" |
| Report Themes | Synthesize all per-topic labels into a handful of narrative meta-themes for a findings report (`generate_report_themes`, `export_report`) | "Report Themes (qualitative research output)" |
| `llm_max_tokens` | Override auto-computed token budget for LLM calls — needed for reasoning models that burn budget on hidden chain-of-thought | "LLM-Powered Labels" |

## Summary

**GraphWeave** is a powerful topic modeling framework. The key things to understand:

1. **Default consensus (`consensus_method="graph"`, 2.3.0+)** is memory-safe out of the box —
   peak extra memory stays under 1 GB regardless of corpus size.
2. **Legacy path**: the old co-occurrence-matrix bottleneck and `low_memory=True` only apply if
   you explicitly opt into `consensus_method="hierarchical"`.
3. **Iterative refinement**: Embeddings improve over iterations (usually converges by iteration 3-4)
4. **Stability**: Check `model.stability_score_` to ensure robust clustering
5. **Configuration**: Adjust resolution, iterations, and features based on your dataset

For 30k+ documents on the default path, there's nothing special to configure for memory. Only
reach for `low_memory=True` if you've deliberately switched to `consensus_method="hierarchical"`.
