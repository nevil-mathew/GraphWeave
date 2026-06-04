# Cumulative / Batch-wise Clustering for TriTopic

Cluster documents that **arrive in batches and accumulate over time** — and get a
single, high-level "bigger picture" of themes across *all* the data so far,
without re-thinking your whole pipeline.

This is a **separate, additive** layer on top of TriTopic. It **does not touch**
the existing full-batch `fit()` / `fit_transform()` — those behave exactly as
before. Everything here lives under `tritopic.cumulative`.

```python
from tritopic.cumulative import CumulativeTriTopic, CumulativeConfig

model = CumulativeTriTopic(CumulativeConfig(strategy="global_refit"))
model.add_batch(batch_1_docs)          # first batch -> clusters
model.add_batch(batch_2_docs)          # reclusters only if the data drifted
themes = model.bigger_picture(n_levels=3)   # high-level view across everything
print(model.get_topic_info())
```

---

## Table of contents
1. [When to use this](#when-to-use-this)
2. [How it works (mental model)](#how-it-works-mental-model)
3. [Install](#install)
4. [Quick start](#quick-start)
   - [Bring your own embeddings (recommended)](#a-bring-your-own-embeddings-recommended)
   - [Let it embed for you](#b-let-it-embed-for-you)
5. [The clustering engines (strategies)](#the-clustering-engines-strategies)
6. [When does it recluster? (triggers)](#when-does-it-recluster-triggers)
7. [The "bigger picture" across all data](#the-bigger-picture-across-all-data)
8. [Scaling to millions of docs (Regime A → B)](#scaling-to-millions-of-docs-regime-a--b)
9. [Measuring quality vs full-batch](#measuring-quality-vs-full-batch)
10. [Real-world integration test (20 Newsgroups)](#real-world-integration-test-20-newsgroups)
11. [API reference](#api-reference)
12. [Configuration reference](#configuration-reference)
13. [FAQ & gotchas](#faq--gotchas)
14. [What's intentionally out of scope (DB phase)](#whats-intentionally-out-of-scope-db-phase)

---

## When to use this

Use **cumulative** clustering when:
- Documents come in **batches** (a few thousand to ~500k each), at **unpredictable**
  times, and **pile up** — total corpus grows into the millions.
- You want **themes across the accumulated corpus**, not isolated per-batch results.
- **Quality matters more than latency** and it's **not real-time**.

Use the original **full-batch** `TriTopic.fit(documents)` when you already have the
whole dataset in hand and just want to cluster it once. (Still fully supported.)

---

## How it works (mental model)

```
            ┌─────────────────────────────────────────────────────────┐
 batch ───► │ add_batch():                                            │
            │   1. embed the new docs (once)                          │
            │   2. assign them to existing topics  (cheap: transform) │
            │   3. measure "novelty" = share that fit no known topic  │
            │   4. if novelty is high (or you ask) ─► recluster()     │
            └─────────────────────────────────────────────────────────┘
                                  │ recluster()
                                  ▼
      re-run the full TriTopic pipeline on the accumulated data,
      then ALIGN the new topic IDs to the previous ones so a theme
      keeps the SAME global ID across time (stable tracking).
```

Two ideas do the heavy lifting:
- **Recluster only when needed.** Between reclusters, incoming batches are placed by
  cheap nearest-topic assignment. A full recluster (the expensive, gold-standard step)
  fires only when the data actually changes.
- **Stable global topic IDs.** Each recluster produces fresh IDs; we match them to the
  previous topics by centroid similarity (Hungarian assignment), so "Topic 7" stays
  "Topic 7" over time and genuinely-new themes get fresh IDs.

---

## Install

The cumulative layer adds **no new dependencies** — it reuses TriTopic's. For a
**model-free** setup (you supply embeddings; great for testing/benchmarks), you only
need the torch-free scientific stack:

```bash
pip install numpy pandas scipy scikit-learn igraph leidenalg joblib tqdm plotly
# or, from the repo:  pip install -e .
```

To embed text *inside* the library (path B below), also install an embedder:

```bash
pip install sentence-transformers        # local models
# or use the Google/API provider via TriTopicConfig(embedding_provider=...)
```

---

## Quick start

### A. Bring your own embeddings (recommended)

Embed once with whatever model you like, then pass vectors in. This is the
highest-quality, most efficient path (no re-embedding, no surprises).

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from tritopic import TriTopicConfig
from tritopic.cumulative import CumulativeTriTopic, CumulativeConfig

encoder = SentenceTransformer("all-MiniLM-L6-v2")

cfg = CumulativeConfig(
    base_config=TriTopicConfig(verbose=False),  # all the usual TriTopic knobs
    strategy="global_refit",                     # default engine
    recluster_trigger="drift",                   # recluster when themes shift
    novelty_threshold=0.25,                       # 25% "doesn't fit" -> recluster
)
model = CumulativeTriTopic(cfg)

for batch_docs in stream_of_batches:             # your iterable of list[str]
    emb = encoder.encode(batch_docs, normalize_embeddings=True)
    result = model.add_batch(batch_docs, embeddings=emb)
    print(f"epoch={result.epoch} new={result.n_new_docs} "
          f"reclustered={result.reclustered} novelty={result.novelty}")

# Inspect the current state (covers ALL accumulated docs)
print(model.get_topic_info())          # per-topic keywords/sizes + GlobalTopic id
print("global topics:", model.n_global_topics)
labels = model.labels_                  # global topic id per accumulated doc (-1 = outlier)
```

### B. Let it embed for you

If you don't pass `embeddings=`, the model embeds with the engine configured in
`base_config` (defaults to `sentence-transformers` `all-MiniLM-L6-v2`):

```python
model = CumulativeTriTopic(CumulativeConfig())
model.add_batch(["doc one ...", "doc two ...", ...])   # embedded internally, once
```

> **No embedding model handy?** See `tritopic.cumulative.datasets.lsa_embed` and
> `make_streaming_corpus` for a model-free (TF-IDF + SVD) way to embed real text —
> used by the tests and the quality report.

---

## The clustering engines (strategies)

All three are selected with `CumulativeConfig(strategy=...)` and share the same API.
They differ only in **what data each recluster is fit on**:

| `strategy` | What it clusters on recluster | Quality | Speed / memory | Use when |
|---|---|---|---|---|
| `"global_refit"` *(default)* | **everything** accumulated (or a coreset above the cap) | **best — matches full-batch** | highest cost | you want the best answer; corpus fits in the in-memory budget |
| `"coreset"` | a bounded, recency-weighted representative sample | ~full-batch in tests | **flat cost as data grows** | the corpus is too big to refit in RAM |
| `"batch_merge"` | only the **newest batch**, merged into the global topic set | good but can miss themes; keywords drift | **fastest** | speed beats completeness |

Approach **#3 (HNSW)** from the design isn't a separate strategy — it runs *inside*
`fit()` automatically (`knn_backend="auto"`) for working sets above ~5k docs.

See the head-to-head numbers in
[`LEARNING_GUIDE/CUMULATIVE_QUALITY_REPORT.md`](../../LEARNING_GUIDE/CUMULATIVE_QUALITY_REPORT.md).

---

## When does it recluster? (triggers)

Set with `CumulativeConfig(recluster_trigger=...)`. The **first** batch always
clusters. After that:

| `recluster_trigger` | Fires a recluster when… | Key knobs |
|---|---|---|
| `"drift"` *(default)* | the batch's **novelty** (share of docs that fit no known topic) exceeds `novelty_threshold` | `novelty_threshold` (default 0.30) |
| `"schedule"` | `schedule_every_n_docs` docs have accumulated since the last recluster | `schedule_every_n_docs` |
| `"manual"` | never automatically — **you** call `model.recluster()` | — |

`min_docs_between_recluster` debounces auto-triggers (avoid reclustering too often).
You can always force one with `model.recluster()` regardless of the trigger.

```python
# Drift-based (adaptive): recluster only when the data changes
CumulativeConfig(recluster_trigger="drift", novelty_threshold=0.25)

# Scheduled (predictable cost): every 50k new docs
CumulativeConfig(recluster_trigger="schedule", schedule_every_n_docs=50_000)

# Manual: you decide
cfg = CumulativeConfig(recluster_trigger="manual")
...
model.add_batch(docs, embeddings=emb)
model.recluster()        # when you want it
```

---

## The "bigger picture" across all data

After any number of batches, get the high-level view. This reuses TriTopic's existing
machinery and **scales with the number of topics, not the number of documents**.

```python
view = model.bigger_picture(n_levels=3)
hierarchy = view["hierarchy"]            # TopicHierarchy: coarse -> fine themes
for node in hierarchy.cut(0):            # coarsest level = the big themes
    print(node.node_id, node.keywords[:5], "size:", node.size)
```

Add an LLM labeler to also get human-readable **meta-themes** (report-style):

```python
from tritopic import LLMLabeler
labeler = LLMLabeler(provider="anthropic", api_key="...")

view = model.bigger_picture(labeler=labeler, n_levels=3, n_themes=8)
for theme in view["themes"]:             # list[ReportTheme]
    print(theme.title, "->", theme.topic_ids)
```

---

## Scaling to millions of docs (Regime A → B)

A single max-size batch (~500k docs × 768-dim) is already ~1.5 GB of embeddings, and
the corpus only grows. The system handles this with two regimes, switched on the
working-set size (`max_inmemory_docs`):

- **Regime A** — corpus ≤ `max_inmemory_docs`: refit on **everything** (gold-standard).
- **Regime B** — corpus > `max_inmemory_docs`: refit on a **bounded recency-weighted
  coreset / micro-cluster summary** of history, so **cost and memory stay ~flat** no
  matter how much data accumulates. Old themes survive as summaries (not windowed away).

```python
CumulativeConfig(
    strategy="global_refit",     # transparently uses a coreset above the cap
    max_inmemory_docs=300_000,   # the absolute working-set cap
    coreset_size=50_000,         # summary size in Regime B
)
```

`model.history_[-1].regime` tells you which regime the last recluster used (`"A"`/`"B"`).

---

## Measuring quality vs full-batch

The full-batch `fit()` on all accumulated docs is the **quality ceiling**. Compare
against it directly:

```python
import copy
from tritopic import TriTopic, TriTopicConfig
from tritopic.cumulative.evaluation import compare_to_full_batch, benchmark_strategies

# One model vs the baseline (same docs, same order):
full = TriTopic(config=copy.deepcopy(base_cfg)); full.fit(all_docs, embeddings=all_emb)
metrics = compare_to_full_batch(model, full, labels_true=optional_ground_truth)
#  -> ari_vs_full, nmi_vs_full, topic_count_drift, keyword_overlap,
#     silhouette_*, ari_vs_truth_* (if labels given), ...

# All engines, head-to-head, same batch stream:
df = benchmark_strategies(
    batches,                          # list[list[str]]
    base_config=base_cfg,
    labels_true=ground_truth,         # optional
    precomputed_batch_embeddings=batch_embs,   # optional (skip embedding)
)
print(df)                             # one row per strategy
```

Run the ready-made synthetic report (no model download):

```bash
python benchmarks/cumulative_quality_report.py
```

Metric cheat-sheet (full glossary in the quality report):

| Metric | Plain meaning | Good |
|---|---|---|
| `ari_vs_full` / `nmi_vs_full` | same grouping as re-running on all data? | → 1.0 |
| `ari_vs_truth_*` | recovered the *real* topics? | higher |
| `topic_count_drift` | difference in #topics vs full-batch | → 0 |
| `keyword_overlap` | matched topics described by the same words? | → 1.0 |
| `silhouette_*` | clusters tight & well-separated? | higher |
| `novelty` (per batch) | share of new docs fitting no topic (drift signal) | low = stable |

---

## Real-world integration test (20 Newsgroups)

A full integration test against the **20 Newsgroups** dataset (18,846 real docs,
20 labeled categories) exercises every flow — drift detection, all 3 strategies,
the hierarchy, and `evaluate()` — with a real `all-MiniLM-L6-v2` embedding model.

**Install the embedding model once** (~1-2 GB download):

```bash
# from the repo root
uv pip install sentence-transformers
# or:  pip install "tritopic[integration]"
```

**Run the full integration test script** (first run encodes 18k docs on CPU — ~30-40 min
without a GPU; subsequent runs load cached embeddings and finish in ~5-10 min):

```bash
python benchmarks/integration_test_20ng.py
```

The first run caches embeddings to `benchmarks/.cache/`. What it validates:
- **Drift detection** — batches 1-4 contain 15 categories; batches 5-6 introduce 5
  new categories → novelty spikes → recluster fires → new topics discovered.
- **Strategy comparison** — `global_refit` vs `coreset` vs `batch_merge` on real text.
- **Bigger picture** — 3-level hierarchy on 18k docs with real embeddings.
- **Memory** — `low_memory=True` avoids the 18k×18k dense matrix (~2.7 GB).

**Run the lightweight pytest version** (real 20NG text, LSA embeddings, no torch, ~15 s):

```bash
pytest tests/test_integration_20ng.py -v -m "not slow"
```

**Run the HNSW path test** (6k docs, sentence-transformers or LSA, ~2 min):

```bash
pytest tests/test_integration_20ng.py -v -m slow
```

---

## API reference

### `CumulativeTriTopic(config: CumulativeConfig | None = None)`

| Method | Description |
|---|---|
| `add_batch(documents, embeddings=None, metadata=None) -> BatchResult` | Ingest a batch, assign it, recluster if the trigger fires. |
| `recluster() -> self` | Force a recluster of the accumulated corpus + re-align topic IDs. |
| `transform(documents, embeddings=None) -> np.ndarray` | Assign **new** docs to **global** topic IDs (does not accumulate them). |
| `transform_proba(documents, embeddings=None) -> np.ndarray` | Soft assignment over the current model's topics. |
| `bigger_picture(labeler=None, n_levels=3, n_themes=None) -> dict` | `{"hierarchy": TopicHierarchy, "themes": list[ReportTheme] \| None}`. |
| `get_topic_info(global_ids=True) -> pd.DataFrame` | Per-topic table (`Topic`, `Size`, `Keywords`, …) + a `GlobalTopic` column. |
| `evaluate() -> dict` | Coherence/diversity/stability of the current model + cumulative bookkeeping. |

| Property | Meaning |
|---|---|
| `labels_` | Global topic ID for **every accumulated doc** (`-1` = outlier). |
| `model_` | The most recent fitted `TriTopic` (used for keywords/hierarchy/themes). |
| `n_global_topics` | Number of stable global topics. |
| `epoch` | How many reclusters have happened. |
| `history_` | `list[EpochSummary]` — one entry per recluster. |
| `documents_` / `embeddings_` | The accumulated corpus (read-only). |

### `BatchResult`
`epoch`, `n_new_docs`, `n_total_docs`, `reclustered` (bool), `novelty` (float | None,
`None` on the first batch), `assignments` (global IDs for this batch's docs).

### `EpochSummary`
`epoch`, `strategy`, `regime` (`"A"`/`"B"`), `n_docs_clustered`, `n_total_docs`,
`n_topics`, `n_global_topics`.

### Evaluation & data helpers
`tritopic.cumulative.evaluation`: `compare_to_full_batch(...)`, `benchmark_strategies(...)`.
`tritopic.cumulative.datasets`: `make_streaming_corpus(...)`, `lsa_embed(...)`,
`StreamingCorpus`.

---

## Configuration reference

`CumulativeConfig` fields:

| Field | Default | What it does |
|---|---|---|
| `base_config` | `None` → `TriTopicConfig()` | The full-batch config reused for every recluster (graph type, consensus, HNSW backend, dim-reduction, embedder, …). |
| `strategy` | `"global_refit"` | Clustering engine: `global_refit` / `coreset` / `batch_merge`. |
| `recluster_trigger` | `"drift"` | `drift` / `schedule` / `manual`. |
| `novelty_threshold` | `0.30` | Drift trigger: outlier share above this → recluster. |
| `min_docs_between_recluster` | `0` | Debounce auto-triggers. |
| `schedule_every_n_docs` | `None` | For `schedule` trigger. |
| `max_inmemory_docs` | `300_000` | Working-set cap; above it → Regime B (summarized). |
| `coreset_size` | `50_000` | Size of the Regime-B coreset/summary. |
| `align_topics` | `True` | Keep stable global topic IDs across epochs (Hungarian alignment). |
| `align_threshold` | `0.6` | Cosine below which a new topic is treated as genuinely new. |
| `verbose` | `False` | Print per-batch progress. |

---

## FAQ & gotchas

**Do topic IDs stay stable over time?** Yes — with `align_topics=True` (default), a
theme keeps the same global ID across reclusters; new themes get fresh IDs. Lower
`align_threshold` to merge more aggressively, raise it to split more readily.

**Does `labels_` always cover every doc?** Yes. At a recluster it's recomputed for the
whole corpus; between reclusters, new batches are appended using cheap transform-based
assignment (refreshed at the next recluster).

**Why didn't a rare new topic trigger a recluster?** If a new theme trickles in slowly,
its novelty may stay below `novelty_threshold`. Lower the threshold, or add a
`schedule` safety-net recluster. (This exact tradeoff is shown in the quality report.)

**`batch_merge` found fewer topics / odd keywords?** Expected — it only clusters the
newest batch, so themes absent from that batch can be missed and per-topic keywords are
computed from one batch. Use `global_refit`/`coreset` when completeness matters.

**Does this change the existing `fit()`?** No. The full-batch path in
`tritopic/core/model.py` is untouched; this is a separate package.

**Multi-view / metadata?** `add_batch(metadata=...)` is accepted for API symmetry but
the metadata view isn't wired into cumulative reclustering yet (future work).

---

## What's intentionally out of scope (DB phase)

This is an **in-memory POC**. Deliberately deferred to a later "DB phase":
- Persisting the accumulator / topic registry / HNSW index to disk (surviving a
  process, exceeding RAM).
- A real document/embedding store.
- Global c-TF-IDF over retained raw text across batches.

These don't affect the clustering logic here — they're about durability and scale
beyond a single process.
