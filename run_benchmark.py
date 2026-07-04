"""
Benchmark reproduction script: TriTopic vs BERTopic vs NMF vs LDA
====================================================================

Reproduces the headline numbers quoted in ``README.md``'s "Benchmarks"
section (Mean NMI 0.575 vs. BERTopic 0.513 / NMF 0.416 / LDA 0.299, etc.).
This is the ``run_benchmark.py`` the README's Methodology subsection links
to as the "Full reproduction script".

Two modes:

Full reproduction (real datasets, real sentence-transformer embeddings —
needs network access and ``pip install -e ".[benchmark]"`` plus
``sentence-transformers``):

    python run_benchmark.py
    python run_benchmark.py --datasets 20ng bbc --seeds 3 --k-grid 3

Fast smoke test (synthetic in-memory corpus, no downloads, seconds not
minutes). This is what CI runs on every push to catch pipeline breakage —
it does NOT reproduce the published numbers, it only proves the four model
adapters and the metrics still run end-to-end:

    python run_benchmark.py --quick

Datasets (full mode)
---------------------
- 20ng      20 Newsgroups, via scikit-learn (no auth needed)
- bbc       BBC News, via `datasets` (SetFit/bbc-news, 1,225 train rows)
- ag_news   AG News, via `datasets` (fancyzhx/ag_news)
- arxiv     Arxiv abstracts, via `datasets` (ccdv/arxiv-classification)

Each dataset is subsampled once (fixed --sample-seed) to the doc count in
the README table so every model/seed/k combination sees the same corpus.
Embeddings (all-MiniLM-L6-v2) are computed once per dataset and cached to
benchmarks/.cache/, exactly like benchmarks/integration_test_20ng.py.

Methodology (full mode, matches README):
- Embeddings: all-MiniLM-L6-v2 (384-dim), shared by TriTopic and BERTopic.
- NMF / LDA: TF-IDF (NMF) / raw counts (LDA) input, scikit-learn defaults.
- Each model is forced to the same target topic count k via its own
  "reduce to k" mechanism (TriTopic: n_topics=k: BERTopic: nr_topics=k;
  NMF/LDA: n_components=k), evaluated across a small grid of k values
  spanning the dataset's documented k-range and averaged over --seeds
  random seeds.
- Metrics: NMI vs. ground-truth category labels, NPMI coherence of each
  topic's keywords, and coverage (1 - outlier fraction).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from tritopic import TriTopic, TriTopicConfig
from tritopic.utils.metrics import compute_coherence, compute_nmi

CACHE_DIR = Path(__file__).parent / "benchmarks" / ".cache"
RESULTS_DIR = Path(__file__).parent / "benchmarks" / "results"

try:
    from bertopic import BERTopic
    HAVE_BERTOPIC = True
except ImportError:
    HAVE_BERTOPIC = False

try:
    import datasets as hf_datasets
    HAVE_HF_DATASETS = True
except ImportError:
    HAVE_HF_DATASETS = False


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ── Dataset loading ──────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    name: str
    n_docs: int
    k_range: tuple[int, int]
    loader: str  # method name on this module, resolved via globals()


DATASET_SPECS: dict[str, DatasetSpec] = {
    "20ng": DatasetSpec("20 Newsgroups", n_docs=2000, k_range=(10, 50), loader="load_20ng"),
    "bbc": DatasetSpec("BBC News", n_docs=1225, k_range=(3, 20), loader="load_bbc_news"),
    "ag_news": DatasetSpec("AG News", n_docs=2000, k_range=(3, 20), loader="load_ag_news"),
    "arxiv": DatasetSpec("Arxiv", n_docs=2000, k_range=(5, 25), loader="load_arxiv"),
}


def _subsample_stratified(
    texts: list[str], labels: np.ndarray, n_docs: int, seed: int
) -> tuple[list[str], np.ndarray]:
    if n_docs >= len(texts):
        return texts, labels
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(texts))
    idx_sample, _ = train_test_split(
        idx, train_size=n_docs, stratify=labels, random_state=seed
    )
    return [texts[i] for i in idx_sample], labels[idx_sample]


def load_20ng(n_docs: int, seed: int) -> tuple[list[str], np.ndarray]:
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = [d.strip() or "empty document" for d in data.data]
    labels = np.array(data.target)
    return _subsample_stratified(texts, labels, n_docs, seed)


def load_bbc_news(n_docs: int, seed: int) -> tuple[list[str], np.ndarray]:
    if not HAVE_HF_DATASETS:
        raise ImportError("BBC News requires `pip install datasets` (tritopic[benchmark]).")
    ds = hf_datasets.load_dataset("SetFit/bbc-news", split="train")
    texts = list(ds["text"])
    labels = np.array(ds["label"])
    return _subsample_stratified(texts, labels, n_docs, seed)


def load_ag_news(n_docs: int, seed: int) -> tuple[list[str], np.ndarray]:
    if not HAVE_HF_DATASETS:
        raise ImportError("AG News requires `pip install datasets` (tritopic[benchmark]).")
    ds = hf_datasets.load_dataset("fancyzhx/ag_news", split="train")
    texts = list(ds["text"])
    labels = np.array(ds["label"])
    return _subsample_stratified(texts, labels, n_docs, seed)


def load_arxiv(n_docs: int, seed: int) -> tuple[list[str], np.ndarray]:
    if not HAVE_HF_DATASETS:
        raise ImportError("Arxiv requires `pip install datasets` (tritopic[benchmark]).")
    ds = hf_datasets.load_dataset("ccdv/arxiv-classification", "no_ref", split="train")
    texts = [t[:4000] for t in ds["text"]]  # abstracts+body can be long; cap for speed
    labels = np.array(ds["label"])
    return _subsample_stratified(texts, labels, n_docs, seed)


def load_dataset(key: str, sample_seed: int) -> tuple[list[str], np.ndarray]:
    spec = DATASET_SPECS[key]
    loader = globals()[spec.loader]
    return loader(spec.n_docs, sample_seed)


def embed(key: str, texts: list[str], sample_seed: int) -> np.ndarray:
    """all-MiniLM-L6-v2 embeddings, cached per dataset to benchmarks/.cache/.

    The cache key includes sample_seed so a rerun with a different
    subsampling seed can't silently reuse embeddings for a different set
    of documents (only doc count and dataset key were used previously).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{key}_n{len(texts)}_seed{sample_seed}_emb.npy"
    if cache_path.exists():
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    print(f"  Encoding {len(texts):,} docs with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256)
    emb = emb.astype(np.float32)
    np.save(cache_path, emb)
    return emb


# ── Model adapters: each returns (labels, keywords_per_topic) ───────────────

def _top_words(vectorizer, component_row: np.ndarray, n: int = 10) -> list[str]:
    vocab = vectorizer.get_feature_names_out()
    top_idx = np.argsort(component_row)[::-1][:n]
    return [vocab[i] for i in top_idx]


def run_tritopic(docs: list[str], embeddings: np.ndarray, k: int, seed: int):
    config = TriTopicConfig(random_state=seed, verbose=False)
    model = TriTopic(n_topics=k, config=config)
    model.fit(docs, embeddings=embeddings)
    keywords = [t.keywords for t in model.topics_ if t.topic_id != -1]
    return model.labels_, keywords


def run_bertopic(docs: list[str], embeddings: np.ndarray, k: int, seed: int):
    from hdbscan import HDBSCAN
    from umap import UMAP

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean"),
        nr_topics=k,
        calculate_probabilities=False,
        verbose=False,
    )
    labels, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    labels = np.asarray(labels)
    keywords = [
        [w for w, _ in topic_model.get_topic(tid)][:10]
        for tid in sorted(set(labels.tolist()))
        if tid != -1
    ]
    return labels, keywords


def run_nmf(docs: list[str], k: int, seed: int):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english", max_features=20_000)
    X = vectorizer.fit_transform(docs)
    nmf = NMF(n_components=k, random_state=seed, init="nndsvda", max_iter=300)
    W = nmf.fit_transform(X)
    labels = np.argmax(W, axis=1)
    keywords = [_top_words(vectorizer, comp) for comp in nmf.components_]
    return labels, keywords


def run_lda(docs: list[str], k: int, seed: int):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english", max_features=20_000)
    X = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=k, random_state=seed, max_iter=20)
    W = lda.fit_transform(X)
    labels = np.argmax(W, axis=1)
    keywords = [_top_words(vectorizer, comp) for comp in lda.components_]
    return labels, keywords


MODEL_RUNNERS = {
    "TriTopic": lambda docs, emb, k, seed: run_tritopic(docs, emb, k, seed),
    "BERTopic": lambda docs, emb, k, seed: run_bertopic(docs, emb, k, seed),
    "NMF": lambda docs, emb, k, seed: run_nmf(docs, k, seed),
    "LDA": lambda docs, emb, k, seed: run_lda(docs, k, seed),
}


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(labels: np.ndarray, keywords: list[list[str]], docs: list[str], true_labels: np.ndarray) -> dict:
    coverage = float(np.mean(labels != -1))
    nmi = compute_nmi(labels, true_labels)
    coherences = [compute_coherence(kw, docs) for kw in keywords if len(kw) >= 2]
    coherence = float(np.mean(coherences)) if coherences else 0.0
    return {"nmi": nmi, "coherence": coherence, "coverage": coverage}


def k_grid_for(k_range: tuple[int, int], n_points: int) -> list[int]:
    lo, hi = k_range
    if n_points <= 1:
        return [int(round((lo + hi) / 2))]
    return sorted({int(round(v)) for v in np.linspace(lo, hi, n_points)})


def run_full_benchmark(dataset_keys: list[str], seeds: int, k_grid_points: int, sample_seed: int) -> dict:
    models = dict(MODEL_RUNNERS)
    if not HAVE_BERTOPIC:
        warnings.warn("bertopic not installed — skipping BERTopic (`pip install tritopic[benchmark]`).")
        models.pop("BERTopic")

    all_results: dict[str, dict[str, list[dict]]] = {}

    for key in dataset_keys:
        spec = DATASET_SPECS[key]
        hr(f"Dataset: {spec.name} ({key})")
        docs, true_labels = load_dataset(key, sample_seed)
        print(f"  {len(docs):,} docs, {len(np.unique(true_labels))} ground-truth categories")
        embeddings = embed(key, docs, sample_seed)
        k_values = k_grid_for(spec.k_range, k_grid_points)
        print(f"  k values: {k_values}  |  seeds: {seeds}")

        per_model: dict[str, list[dict]] = {name: [] for name in models}
        for k in k_values:
            for seed in range(seeds):
                for name, runner in models.items():
                    try:
                        labels, keywords = runner(docs, embeddings, k, seed)
                    except Exception as exc:  # a single failed config shouldn't sink the run
                        warnings.warn(f"{name} failed at k={k} seed={seed}: {exc}")
                        continue
                    metrics = evaluate(labels, keywords, docs, true_labels)
                    per_model[name].append(metrics)
                    print(f"    k={k:>3} seed={seed}  {name:<10} "
                          f"NMI={metrics['nmi']:.3f} coherence={metrics['coherence']:.3f} "
                          f"coverage={metrics['coverage']:.3f}")
        all_results[key] = per_model

    return all_results


def run_quick_smoke() -> dict:
    """Synthetic, in-memory, no-download sanity check of the same code paths."""
    from tritopic.cumulative.datasets import make_streaming_corpus

    hr("QUICK SMOKE TEST (synthetic corpus, no downloads)")
    corp = make_streaming_corpus(n_topics=5, docs_per_batch=200, n_batches=1, random_state=7)
    docs, embeddings, true_labels = corp.all_documents, corp.all_embeddings, corp.all_labels
    print(f"  {len(docs)} synthetic docs, {corp.n_topics} true topics")

    models = dict(MODEL_RUNNERS)
    if not HAVE_BERTOPIC:
        print("  bertopic not installed — skipping BERTopic for the smoke test.")
        models.pop("BERTopic")

    per_model: dict[str, list[dict]] = {name: [] for name in models}
    for k in (5,):
        for seed in (0,):
            for name, runner in models.items():
                labels, keywords = runner(docs, embeddings, k, seed)
                metrics = evaluate(labels, keywords, docs, true_labels)
                per_model[name].append(metrics)
                print(f"  {name:<10} NMI={metrics['nmi']:.3f} "
                      f"coherence={metrics['coherence']:.3f} coverage={metrics['coverage']:.3f}")

    print("\nSmoke test passed: all model adapters and metrics ran end-to-end.")
    print("These numbers are NOT the published benchmark — run without --quick for that.")
    return {"quick": per_model}


# ── Reporting ─────────────────────────────────────────────────────────────────

def _mean_metric(entries: list[dict], key: str) -> float:
    return float(np.mean([e[key] for e in entries])) if entries else float("nan")


def print_report(all_results: dict[str, dict[str, list[dict]]]) -> str:
    lines = []
    hr("RESULTS")

    model_names = sorted({name for per_model in all_results.values() for name in per_model})
    overall = {name: {"nmi": [], "coherence": [], "coverage": []} for name in model_names}
    for per_model in all_results.values():
        for name, entries in per_model.items():
            for metric in ("nmi", "coherence", "coverage"):
                overall[name][metric].extend(e[metric] for e in entries)

    lines.append("### Overall Results\n")
    lines.append("| Model | Mean NMI | Mean Coherence (NPMI) | Mean Coverage |")
    lines.append("|---|---|---|---|")
    for name in model_names:
        nmi = np.mean(overall[name]["nmi"]) if overall[name]["nmi"] else float("nan")
        coh = np.mean(overall[name]["coherence"]) if overall[name]["coherence"] else float("nan")
        cov = np.mean(overall[name]["coverage"]) if overall[name]["coverage"] else float("nan")
        lines.append(f"| {name} | {nmi:.3f} | {coh:.3f} | {cov:.3f} |")

    lines.append("\n### Per-Dataset NMI\n")
    lines.append("| Dataset | " + " | ".join(model_names) + " |")
    lines.append("|---|" + "---|" * len(model_names))
    for dataset_key, per_model in all_results.items():
        row = [DATASET_SPECS[dataset_key].name if dataset_key in DATASET_SPECS else dataset_key]
        for name in model_names:
            row.append(f"{_mean_metric(per_model.get(name, []), 'nmi'):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    report = "\n".join(lines)
    print(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_SPECS), default=list(DATASET_SPECS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--k-grid", type=int, default=3, help="number of k values sampled across each dataset's documented k-range")
    parser.add_argument("--sample-seed", type=int, default=20, help="seed used to subsample each dataset to its documented doc count")
    parser.add_argument("--quick", action="store_true", help="fast synthetic smoke test, no downloads, no README numbers")
    parser.add_argument("--output", type=str, default=None, help="path to write the Markdown report (default: benchmarks/results/<timestamp>.md, skipped for --quick)")
    args = parser.parse_args()

    if args.quick:
        run_quick_smoke()
        return

    all_results = run_full_benchmark(args.datasets, args.seeds, args.k_grid, args.sample_seed)
    report = print_report(all_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        import datetime
        out_path = RESULTS_DIR / f"benchmark_{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
