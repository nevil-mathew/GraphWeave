"""
Compare TriTopic vs BERTopic on the same corpus.

Usage:
    # Fit both models first, then:
    from compare_bertopic import compare_models
    compare_models(tritopic_model, bertopic_model, docs, ground_truth_labels=None)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# ── Coherence & diversity helpers ─────────────────────────────────────────────

def _pmi_coherence(keywords: list[str], docs: list[str], top_n: int = 10) -> float:
    """Simple PMI-based coherence: mean co-occurrence score over keyword pairs."""
    kws = [w.lower() for w in keywords[:top_n]]
    if len(kws) < 2:
        return 0.0
    doc_sets = [set(d.lower().split()) for d in docs]
    n = len(doc_sets)
    scores = []
    for i in range(len(kws)):
        for j in range(i + 1, len(kws)):
            p_i  = sum(kws[i] in d for d in doc_sets) / n
            p_j  = sum(kws[j] in d for d in doc_sets) / n
            p_ij = sum(kws[i] in d and kws[j] in d for d in doc_sets) / n
            if p_ij > 0 and p_i > 0 and p_j > 0:
                scores.append(np.log(p_ij / (p_i * p_j)))
    return float(np.mean(scores)) if scores else 0.0


def _topic_diversity(all_keyword_lists: list[list[str]], top_n: int = 10) -> float:
    """Fraction of unique words in the top-N keywords across all topics."""
    words = [w.lower() for kws in all_keyword_lists for w in kws[:top_n]]
    return len(set(words)) / len(words) if words else 0.0


# ── Extract keywords from BERTopic ─────────────────────────────────────────────

def _bertopic_keyword_lists(bt_model, top_n: int = 10) -> dict[int, list[str]]:
    """Return {topic_id: [word, ...]} for all non-outlier topics."""
    info = bt_model.get_topic_info()
    result = {}
    for tid in info["Topic"]:
        if tid == -1:
            continue
        words_scores = bt_model.get_topic(tid)
        if words_scores:
            result[tid] = [w for w, _ in words_scores[:top_n]]
    return result


# ── Extract keywords from TriTopic ────────────────────────────────────────────

def _tritopic_keyword_lists(tt_model, top_n: int = 10) -> dict[int, list[str]]:
    info = tt_model.get_topic_info()
    result = {}
    for _, row in info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        kws = row["All_Keywords"][:top_n] if row["All_Keywords"] else []
        result[tid] = kws
    return result


# ── Topic alignment via Jaccard ───────────────────────────────────────────────

def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def _align_topics(
    kws_a: dict[int, list[str]],
    kws_b: dict[int, list[str]],
) -> list[tuple[int, int, float]]:
    """
    Hungarian-match topics from model A to model B by Jaccard keyword overlap.
    Returns [(tid_a, tid_b, jaccard), ...] sorted by jaccard desc.
    """
    ids_a = list(kws_a.keys())
    ids_b = list(kws_b.keys())
    if not ids_a or not ids_b:
        return []

    cost = np.zeros((len(ids_a), len(ids_b)))
    for i, ta in enumerate(ids_a):
        for j, tb in enumerate(ids_b):
            cost[i, j] = 1 - _jaccard(kws_a[ta], kws_b[tb])

    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = [(ids_a[r], ids_b[c], 1 - cost[r, c]) for r, c in zip(row_ind, col_ind)]
    return sorted(pairs, key=lambda x: -x[2])


# ── Main comparison ───────────────────────────────────────────────────────────

def compare_models(
    tritopic_model,
    bertopic_model,
    docs: list[str],
    ground_truth_labels: np.ndarray | list | None = None,
    top_n_keywords: int = 10,
    show_alignment_top: int = 15,
) -> dict:
    """
    Print a comparison report and return all metrics as a dict.

    Parameters
    ----------
    tritopic_model  : fitted TriTopic instance
    bertopic_model  : fitted BERTopic instance
    docs            : the documents both models were fitted on
    ground_truth_labels : optional integer array of true topic IDs
    top_n_keywords  : keyword list depth for coherence/diversity/alignment
    show_alignment_top  : how many aligned topic pairs to print
    """
    sep = "=" * 72

    # ── 1. Labels ─────────────────────────────────────────────────────────────
    tt_labels = np.array(tritopic_model.labels_)
    bt_labels = np.array(bertopic_model.topics_)

    # ── 2. TriTopic built-in metrics ──────────────────────────────────────────
    tt_eval = tritopic_model.evaluate()

    # ── 3. BERTopic metrics (recomputed to be on same scale) ──────────────────
    bt_kws = _bertopic_keyword_lists(bertopic_model, top_n=top_n_keywords)
    bt_coherences = []
    for tid, kws in bt_kws.items():
        doc_subset = [docs[i] for i, l in enumerate(bt_labels) if l == tid]
        if doc_subset:
            bt_coherences.append(_pmi_coherence(kws, doc_subset, top_n=top_n_keywords))

    bt_diversity  = _topic_diversity(list(bt_kws.values()), top_n=top_n_keywords)
    bt_outlier    = float((bt_labels == -1).mean())
    bt_n_topics   = len(bt_kws)

    # Also recompute TriTopic coherence with same function for fair comparison
    tt_kws = _tritopic_keyword_lists(tritopic_model, top_n=top_n_keywords)
    tt_coherences_fair = []
    for tid, kws in tt_kws.items():
        doc_subset = [docs[i] for i, l in enumerate(tt_labels) if l == tid]
        if doc_subset:
            tt_coherences_fair.append(_pmi_coherence(kws, doc_subset, top_n=top_n_keywords))

    tt_diversity_fair = _topic_diversity(list(tt_kws.values()), top_n=top_n_keywords)

    # ── 4. Cross-model agreement ───────────────────────────────────────────────
    # Only compare docs assigned by both models
    both_assigned = (tt_labels != -1) & (bt_labels != -1)
    cross_ari = adjusted_rand_score(tt_labels[both_assigned], bt_labels[both_assigned]) \
        if both_assigned.sum() > 1 else float("nan")
    cross_nmi = normalized_mutual_info_score(tt_labels[both_assigned], bt_labels[both_assigned]) \
        if both_assigned.sum() > 1 else float("nan")

    # ── 5. Ground truth (optional) ────────────────────────────────────────────
    gt_metrics = {}
    if ground_truth_labels is not None:
        gt = np.array(ground_truth_labels)
        tt_mask = tt_labels != -1
        bt_mask = bt_labels != -1
        gt_metrics = {
            "tt_nmi_vs_gt":  normalized_mutual_info_score(gt[tt_mask], tt_labels[tt_mask]),
            "tt_ari_vs_gt":  adjusted_rand_score(gt[tt_mask], tt_labels[tt_mask]),
            "bt_nmi_vs_gt":  normalized_mutual_info_score(gt[bt_mask], bt_labels[bt_mask]),
            "bt_ari_vs_gt":  adjusted_rand_score(gt[bt_mask], bt_labels[bt_mask]),
        }

    # ── 6. Topic alignment ────────────────────────────────────────────────────
    aligned = _align_topics(tt_kws, bt_kws)
    mean_jaccard = float(np.mean([j for _, _, j in aligned])) if aligned else 0.0

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TriTopic  vs  BERTopic  —  Comparison Report")
    print(sep)

    print(f"\n  {'Metric':<30}  {'TriTopic':>12}  {'BERTopic':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    rows_table = [
        ("Topics found",        tt_eval["n_topics"],          bt_n_topics),
        ("Outlier rate",        f"{tt_eval['outlier_ratio']:.1%}", f"{bt_outlier:.1%}"),
        ("Coherence (PMI, mean)", f"{np.mean(tt_coherences_fair):.4f}" if tt_coherences_fair else "N/A",
                                   f"{np.mean(bt_coherences):.4f}"    if bt_coherences    else "N/A"),
        ("Diversity (unique kws)", f"{tt_diversity_fair:.4f}",    f"{bt_diversity:.4f}"),
        ("Stability (consensus)",  f"{tt_eval.get('stability') or 'N/A'}", "N/A"),
    ]
    for label, tt_val, bt_val in rows_table:
        print(f"  {label:<30}  {str(tt_val):>12}  {str(bt_val):>12}")

    print(f"\n  Cross-model agreement (docs assigned by both)")
    print(f"  {'ARI  (TriTopic ↔ BERTopic):':<38}  {cross_ari:>8.4f}")
    print(f"  {'NMI  (TriTopic ↔ BERTopic):':<38}  {cross_nmi:>8.4f}")
    print(f"  Docs used for cross-model agreement: {both_assigned.sum():,} / {len(tt_labels):,}")

    if gt_metrics:
        print(f"\n  vs Ground Truth")
        print(f"  {'Model':<12}  {'NMI':>8}  {'ARI':>8}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}")
        print(f"  {'TriTopic':<12}  {gt_metrics['tt_nmi_vs_gt']:>8.4f}  {gt_metrics['tt_ari_vs_gt']:>8.4f}")
        print(f"  {'BERTopic':<12}  {gt_metrics['bt_nmi_vs_gt']:>8.4f}  {gt_metrics['bt_ari_vs_gt']:>8.4f}")

    print(f"\n  Topic Alignment  (TriTopic ↔ BERTopic, top {show_alignment_top} by Jaccard)")
    print(f"  Mean keyword Jaccard across aligned pairs: {mean_jaccard:.4f}")
    print(f"  {'TT topic':<10}  {'BT topic':<10}  {'Jaccard':>8}  TT keywords  ↔  BT keywords")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*40}")
    for tt_id, bt_id, jac in aligned[:show_alignment_top]:
        tt_top5 = ", ".join(tt_kws.get(tt_id, [])[:5])
        bt_top5 = ", ".join(bt_kws.get(bt_id, [])[:5])
        print(f"  {tt_id:<10}  {bt_id:<10}  {jac:>8.3f}  [{tt_top5}]  ↔  [{bt_top5}]")

    print(f"\n{sep}\n")

    return {
        "tritopic": {
            "n_topics": tt_eval["n_topics"],
            "outlier_ratio": tt_eval["outlier_ratio"],
            "coherence_mean": np.mean(tt_coherences_fair) if tt_coherences_fair else None,
            "diversity": tt_diversity_fair,
            "stability": tt_eval.get("stability"),
        },
        "bertopic": {
            "n_topics": bt_n_topics,
            "outlier_ratio": bt_outlier,
            "coherence_mean": np.mean(bt_coherences) if bt_coherences else None,
            "diversity": bt_diversity,
        },
        "cross_model": {"ari": cross_ari, "nmi": cross_nmi},
        "alignment": {"mean_jaccard": mean_jaccard, "pairs": aligned},
        **gt_metrics,
    }


# ── File-based comparison (for separate-notebook workflows) ───────────────────

def compare_from_files(
    tt_labels_path: str,
    tt_keywords_path: str,
    bt_labels_path: str,
    bt_keywords_path: str,
    docs_path: str | None = None,
    ground_truth_path: str | None = None,
    top_n_keywords: int = 10,
    show_alignment_top: int = 15,
) -> dict:
    """
    Compare without needing the original model objects.

    Expected files
    --------------
    tt_labels.npy       : np.save("tt_labels.npy", tt_model.labels_)
    tt_keywords.json    : {topic_id_str: [word, ...], ...}
    bt_labels.npy       : np.save("bt_labels.npy", bt_model.topics_)
    bt_keywords.json    : {topic_id_str: [word, ...], ...}
    docs.json           : json.dump(docs, open("docs.json","w"))  [optional, for coherence]
    gt_labels.npy       : np.save("gt_labels.npy", gt)           [optional]
    """
    import json

    tt_labels = np.load(tt_labels_path)
    bt_labels = np.load(bt_labels_path)
    tt_kws    = {int(k): v for k, v in json.load(open(tt_keywords_path)).items() if int(k) != -1}
    bt_kws    = {int(k): v for k, v in json.load(open(bt_keywords_path)).items() if int(k) != -1}
    docs      = json.load(open(docs_path)) if docs_path else []
    gt        = np.load(ground_truth_path) if ground_truth_path else None

    sep = "=" * 72

    # ── Coherence & diversity ─────────────────────────────────────────────────
    def _coherences(kws_dict, labels):
        out = []
        for tid, kws in kws_dict.items():
            subset = [docs[i] for i, l in enumerate(labels) if l == tid] if docs else []
            if subset:
                out.append(_pmi_coherence(kws, subset, top_n=top_n_keywords))
        return out

    tt_coh = _coherences(tt_kws, tt_labels)
    bt_coh = _coherences(bt_kws, bt_labels)
    tt_div = _topic_diversity(list(tt_kws.values()), top_n=top_n_keywords)
    bt_div = _topic_diversity(list(bt_kws.values()), top_n=top_n_keywords)

    # ── Cross-model agreement ─────────────────────────────────────────────────
    both = (tt_labels != -1) & (bt_labels != -1)
    cross_ari = adjusted_rand_score(tt_labels[both], bt_labels[both]) if both.sum() > 1 else float("nan")
    cross_nmi = normalized_mutual_info_score(tt_labels[both], bt_labels[both]) if both.sum() > 1 else float("nan")

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_metrics = {}
    if gt is not None:
        tt_m, bt_m = tt_labels != -1, bt_labels != -1
        gt_metrics = {
            "tt_nmi_vs_gt": normalized_mutual_info_score(gt[tt_m], tt_labels[tt_m]),
            "tt_ari_vs_gt": adjusted_rand_score(gt[tt_m], tt_labels[tt_m]),
            "bt_nmi_vs_gt": normalized_mutual_info_score(gt[bt_m], bt_labels[bt_m]),
            "bt_ari_vs_gt": adjusted_rand_score(gt[bt_m], bt_labels[bt_m]),
        }

    # ── Alignment ─────────────────────────────────────────────────────────────
    aligned = _align_topics(tt_kws, bt_kws)
    mean_jac = float(np.mean([j for _, _, j in aligned])) if aligned else 0.0

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TriTopic  vs  BERTopic  —  Comparison Report (file-based)")
    print(sep)

    print(f"\n  {'Metric':<30}  {'TriTopic':>12}  {'BERTopic':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    rows = [
        ("Topics found",          len(tt_kws),                     len(bt_kws)),
        ("Outlier rate",          f"{(tt_labels==-1).mean():.1%}", f"{(bt_labels==-1).mean():.1%}"),
        ("Coherence (PMI, mean)", f"{np.mean(tt_coh):.4f}" if tt_coh else "need docs",
                                   f"{np.mean(bt_coh):.4f}" if bt_coh else "need docs"),
        ("Diversity (unique kws)", f"{tt_div:.4f}",                f"{bt_div:.4f}"),
    ]
    for label, tt_val, bt_val in rows:
        print(f"  {label:<30}  {str(tt_val):>12}  {str(bt_val):>12}")

    print(f"\n  Cross-model agreement  (docs assigned by both: {both.sum():,}/{len(tt_labels):,})")
    print(f"  {'ARI:':<10}  {cross_ari:.4f}")
    print(f"  {'NMI:':<10}  {cross_nmi:.4f}")

    if gt_metrics:
        print(f"\n  vs Ground Truth")
        print(f"  {'Model':<12}  {'NMI':>8}  {'ARI':>8}")
        print(f"  {'TriTopic':<12}  {gt_metrics['tt_nmi_vs_gt']:>8.4f}  {gt_metrics['tt_ari_vs_gt']:>8.4f}")
        print(f"  {'BERTopic':<12}  {gt_metrics['bt_nmi_vs_gt']:>8.4f}  {gt_metrics['bt_ari_vs_gt']:>8.4f}")

    print(f"\n  Topic Alignment  (mean Jaccard: {mean_jac:.4f})")
    print(f"  {'TT topic':<10}  {'BT topic':<10}  {'Jaccard':>8}  TT keywords  ↔  BT keywords")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*44}")
    for tt_id, bt_id, jac in aligned[:show_alignment_top]:
        tt5 = ", ".join(tt_kws.get(tt_id, [])[:5])
        bt5 = ", ".join(bt_kws.get(bt_id, [])[:5])
        print(f"  {tt_id:<10}  {bt_id:<10}  {jac:>8.3f}  [{tt5}]  ↔  [{bt5}]")

    print(f"\n{sep}\n")
    return {
        "tritopic":    {"n_topics": len(tt_kws), "outlier_ratio": float((tt_labels==-1).mean()), "coherence_mean": float(np.mean(tt_coh)) if tt_coh else None, "diversity": tt_div},
        "bertopic":    {"n_topics": len(bt_kws), "outlier_ratio": float((bt_labels==-1).mean()), "coherence_mean": float(np.mean(bt_coh)) if bt_coh else None, "diversity": bt_div},
        "cross_model": {"ari": cross_ari, "nmi": cross_nmi},
        "alignment":   {"mean_jaccard": mean_jac, "pairs": aligned},
        **gt_metrics,
    }


# ── Quick standalone demo ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5:
        compare_from_files(*sys.argv[1:5])
    elif len(sys.argv) == 6:
        compare_from_files(*sys.argv[1:5], docs_path=sys.argv[5])
    else:
        print("Usage: python compare_bertopic.py tt_labels.npy tt_keywords.json bt_labels.npy bt_keywords.json [docs.json]")
