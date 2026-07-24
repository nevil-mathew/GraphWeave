# Cumulative Clustering — Quality Report (plain English)

**What this is:** an honest, reproducible check of how good the new cumulative
(batch-wise) clustering is, compared to the original full-batch `fit()`.

**How it was tested without an embedding model:** real clustering quality needs
real-ish text, but downloading a neural embedder is heavy. So we generate
messy, realistic text (overlapping vocabulary between topics, a few big topics
and several small ones, filler words, noise documents, and a brand-new topic
that shows up partway through) and turn it into vectors with **LSA**
(TF-IDF → TruncatedSVD) — the classic, pre-neural way to embed documents. Same
pipeline as production (text → vector → graph → Leiden → keywords), just with a
cheap, deterministic, offline embedder.

Reproduce everything:

```bash
python benchmarks/cumulative_quality_report.py        # the report below
pytest tests/test_cumulative.py tests/test_cumulative_realistic.py -q   # 22 tests
```

---

## The numbers (600 docs, 5 batches, 6 real topics)

### 1. Stable themes — is cheap assignment good enough between reclusters?
When the topics don't change, the system reclusters once and then just **assigns**
new batches to existing topics (cheap), instead of re-running the whole pipeline.

| Measure | Result | Meaning |
|---|---|---|
| Reclusters fired | **1 of 5 batches** | 4 batches were placed cheaply, no full rerun |
| Agreement with full-batch (ARI) | **0.95** | almost the same answer as re-running on everything |
| Accuracy vs ground truth (ARI) | **0.93** (full-batch: 0.98) | recovers the real topics well |
| Topics found | **5** (true: 6) | missed the *rarest* topic — see below |

**Takeaway:** you get ~95% of the full-batch answer for a small fraction of the
compute. The one miss: the rarest topic trickled in so slowly it never tripped
the "new stuff" alarm, so it stayed merged. That's a **dial**, not a bug — lower
the novelty threshold or schedule an occasional full recluster to catch it.

### 2. A new theme appears mid-stream — does the system notice?
The 6th topic only starts appearing at batch 3.

```
batch | new-topic docs | novelty | reclustered | topics
  0   |        0       |   -     |    True     |   5
  1   |        0       |  0.01   |    False    |   5
  2   |        0       |  0.02   |    False    |   5
  3   |       35       |  0.29   |    True     |   6   <- caught it
  4   |       39       |  0.00   |    False    |   6
```

**Takeaway:** while data was stable, "novelty" (the share of incoming docs that
fit no known topic) stayed near zero and **no expensive recluster ran**. The
moment the new theme arrived, ~29% of the batch didn't fit anything → the alarm
tripped → a recluster fired and **discovered the new topic (5 → 6)**. Batch 4
was calm again. The system spends compute **only when the data actually changes**.

### 3. The three engines, head-to-head (reclustering every batch)

| engine | ARI vs full | ARI vs truth | topics | keyword overlap | time |
|---|---|---|---|---|---|
| **global_refit** | **1.00** | 0.97 | 6 | 1.00 | 1.8s |
| **coreset** | **1.00** | 0.97 | 6 | 1.00 | 2.1s |
| **batch_merge** | 0.99 | 0.97 | 6 | **0.60** | **0.7s** |

**Takeaway:**
- **global_refit** (default) — re-clusters everything; **matches the full-batch
  baseline exactly**. Best quality, highest cost.
- **coreset** — clusters a bounded representative sample; ~same quality here, and
  it's the path that keeps **memory and time flat** as the corpus grows to
  millions of docs (you can't hold everything in RAM forever).
- **batch_merge** — clusters only the newest batch and merges; **~2.5× faster**,
  but its keywords drift (overlap 0.60 vs 1.00) because each topic's keywords are
  computed from one batch, and it can miss themes that aren't in the latest batch.

---

## What each metric means (in one line)

| Metric | Plain meaning | Good value |
|---|---|---|
| **ARI vs full** | "Did we get the same grouping as re-running on all data?" | → 1.0 |
| **NMI vs full** | Same idea, information-theoretic | → 1.0 |
| **ARI vs truth** | "Did we recover the *real* topics?" | higher |
| **Topic count drift** | Difference in number of topics vs full-batch | → 0 |
| **Keyword overlap** | Do matched topics describe themselves with the same words? | → 1.0 |
| **Silhouette** | Are clusters internally tight and well-separated? | higher |
| **Novelty / outlier rate** | Share of new docs that fit no known topic (drift signal) | low = stable |

---

## Bottom line

- **Default to `global_refit` + drift-triggered reclustering.** You get
  full-batch quality, but the expensive recluster only runs when new themes
  actually appear — otherwise incoming batches are assigned cheaply.
- **Switch to `coreset`** once the accumulated corpus is too large to refit in
  RAM (per the plan's "Regime B"). Quality stayed effectively identical here.
- **Use `batch_merge` only when speed beats completeness** — it's the fastest but
  the most likely to miss a theme and the most prone to keyword drift.
- **Tune the novelty threshold** to trade cost vs. how fast you want rare/slow
  emerging topics to be caught; add a periodic recluster as a safety net.

*All results above are from `benchmarks/cumulative_quality_report.py` on
LSA-embedded synthetic-but-realistic text — no embedding model, fully
deterministic.*
