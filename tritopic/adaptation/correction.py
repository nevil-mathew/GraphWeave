"""Post-hoc LLM correction of low-confidence topic assignments — the
cheapest lever from Viswanathan et al. (TACL 2024, "Large Language Models
Enable Few-Shot Clustering"): only the documents the model itself is least
sure about get a second opinion.

Operates purely by calling methods/attributes on the ``model`` object
passed in (duck-typed) — no import of ``tritopic.core`` needed, so this
stays inside the package's import boundary.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "string"}},
    "required": ["choice"],
}


def _snippet(doc: str, n_chars: int) -> str:
    return doc[:n_chars] + "..." if len(doc) > n_chars else doc


def _build_correction_prompt(
    doc: str, candidates: list[tuple[int, str, list[str]]], n_docs_chars: int
) -> tuple[str, str]:
    options = "\n".join(
        f"{i}: {label} (keywords: {', '.join(kws[:5])})" for i, (_, label, kws) in enumerate(candidates)
    )
    system_prompt = (
        "You are a text clustering expert. Given a document and a list of "
        "candidate topics, decide which topic (if any) the document best "
        'belongs to. Respond with the option index, or "none" if none fit '
        "well. You always respond with valid JSON and nothing else."
    )
    user_prompt = f"""Document:
{_snippet(doc, n_docs_chars)}

Candidate topics:
{options}

Respond ONLY with JSON in this exact format, no other text:
{{"choice": "<index or \\"none\\">"}}"""
    return system_prompt, user_prompt


def _parse_choice(raw: str, n_candidates: int) -> int | None:
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return None
    choice = str(data.get("choice", "")).strip().lower()
    if choice == "none":
        return None
    try:
        idx = int(choice)
    except ValueError:
        return None
    return idx if 0 <= idx < n_candidates else None


def reassign_low_confidence(
    model,
    labeler,
    margin_threshold: float = 0.15,
    top_k: int = 3,
    max_docs: int = 200,
    batch_size: int = 8,  # chunks the iteration only — each document still gets its own LLM call
    n_docs_chars: int = 300,
    dry_run: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ask an LLM to re-adjudicate the model's least-confident assignments.

    Selects documents whose soft-assignment margin (top1 - top2 probability)
    is below *margin_threshold* (lowest margin first, capped at
    *max_docs*), shows the LLM each one plus its top-*top_k* candidate
    topics (label + keywords), and applies any accepted reassignment unless
    *dry_run*. Refreshes centroids/probabilities after applying changes;
    keywords are left stale (re-run keyword extraction yourself if needed).

    Returns
    -------
    pandas.DataFrame with columns ``doc_index, old_topic, new_topic,
    margin, applied``.
    """
    if model.probabilities_ is None:
        raise ValueError("Model has no probabilities_ — fit the model first.")

    proba = model.probabilities_
    topics = [t for t in model.topics_ if t.topic_id != -1]
    topic_ids = [t.topic_id for t in topics]  # matches probabilities_ column order

    sorted_proba = np.sort(proba, axis=1)[:, ::-1]
    margins = sorted_proba[:, 0] - sorted_proba[:, 1] if proba.shape[1] > 1 else sorted_proba[:, 0]

    candidate_docs = np.where(margins < margin_threshold)[0]
    candidate_docs = candidate_docs[np.argsort(margins[candidate_docs])][:max_docs]

    rows = []
    for start in range(0, len(candidate_docs), batch_size):
        for doc_idx in candidate_docs[start : start + batch_size]:
            order = np.argsort(proba[doc_idx])[::-1][:top_k]
            candidates = [
                (topic_ids[j], topics[j].label or str(topic_ids[j]), topics[j].keywords)
                for j in order
            ]
            system_prompt, user_prompt = _build_correction_prompt(
                model.documents_[doc_idx], candidates, n_docs_chars
            )
            raw = labeler.call_structured(
                system_prompt, user_prompt, schema=CORRECTION_SCHEMA, max_tokens=64
            )
            choice = _parse_choice(raw, len(candidates))

            old_topic = int(model.labels_[doc_idx])
            new_topic = old_topic
            applied = False
            if choice is not None:
                new_topic = candidates[choice][0]
                if new_topic != old_topic and not dry_run:
                    model.labels_[doc_idx] = new_topic
                    applied = True

            rows.append({
                "doc_index": int(doc_idx),
                "old_topic": old_topic,
                "new_topic": new_topic,
                "margin": float(margins[doc_idx]),
                "applied": applied,
            })

    if not dry_run and any(r["applied"] for r in rows):
        # A reassignment can empty a topic out entirely; recomputing a
        # centroid over zero documents would average an empty slice (NaN),
        # so drop any now-empty topic before refreshing centroids/probabilities.
        emptied = [
            t.topic_id for t in model.topics_
            if t.topic_id != -1 and not np.any(model.labels_ == t.topic_id)
        ]
        if emptied:
            model.topics_ = [t for t in model.topics_ if t.topic_id not in emptied]
        model._compute_topic_centroids()
        model._compute_probabilities()

    return pd.DataFrame(rows)
