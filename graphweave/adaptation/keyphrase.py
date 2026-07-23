"""LLM keyphrase expansion — the no-fine-tuning-needed lever from
Viswanathan et al. (TACL 2024, "Large Language Models Enable Few-Shot
Clustering"): generate per-document keyphrases with an LLM, then blend them
into the embedding. Works with any embedder, including API-based ones that
can't be fine-tuned.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

KEYPHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "keyphrases": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}
    },
    "required": ["keyphrases"],
}


def _snippet(doc: str, n_chars: int) -> str:
    return doc[:n_chars] + "..." if len(doc) > n_chars else doc


def _build_keyphrase_prompt(
    documents: list[str], n_keyphrases: int, n_docs_chars: int
) -> tuple[str, str]:
    system_prompt = (
        "You are a text analysis expert. For each numbered document, extract "
        f"{n_keyphrases} short keyphrases (1-3 words each) that best capture its "
        "topic. You always respond with valid JSON and nothing else."
    )
    lines = [f"Document {i}:\n{_snippet(doc, n_docs_chars)}" for i, doc in enumerate(documents, 1)]
    items_block = "\n\n".join(lines)
    user_prompt = f"""Below are {len(documents)} documents.

{items_block}

For each document, extract exactly {n_keyphrases} keyphrases.

Respond ONLY with JSON in this exact format, no other text:
{{"keyphrases": [["phrase1", "phrase2", ...], ...]}}

The "keyphrases" array must have exactly {len(documents)} entries, one per document in order, each an array of {n_keyphrases} short strings."""
    return system_prompt, user_prompt


def _parse_keyphrase_response(raw: str, n_expected: int, n_keyphrases: int) -> list[list[str]]:
    start, end = raw.find("{"), raw.rfind("}") + 1
    data = None
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, dict) and isinstance(parsed.get("keyphrases"), list):
                data = parsed["keyphrases"]
        except (json.JSONDecodeError, ValueError):
            pass

    result: list[list[str]] = []
    for i in range(n_expected):
        if data is not None and i < len(data) and isinstance(data[i], list):
            phrases = [str(p).strip() for p in data[i] if str(p).strip()]
        else:
            phrases = []
        result.append(phrases[:n_keyphrases])
    return result


def generate_keyphrases(
    labeler,
    documents: list[str],
    n_keyphrases: int = 5,
    batch_size: int = 8,
    n_docs_chars: int = 300,
    cache_path: str | None = None,
    random_state: int = 42,
) -> list[list[str]]:
    """Batched, cached LLM keyphrase extraction — one phrase list per document."""
    cache: dict[str, list[str]] = {}
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["hash"]] = rec["keyphrases"]
                except (json.JSONDecodeError, KeyError):
                    continue  # skip a malformed/partial line rather than losing the whole cache

    def _hash(doc: str) -> str:
        return hashlib.sha256(f"v1|{n_keyphrases}|{doc[:n_docs_chars]}".encode("utf-8")).hexdigest()

    hashes = [_hash(doc) for doc in documents]
    results: list[list[str] | None] = [cache.get(h) for h in hashes]
    uncached = [i for i, r in enumerate(results) if r is None]

    for start in range(0, len(uncached), batch_size):
        batch_positions = uncached[start : start + batch_size]
        batch_docs = [documents[i] for i in batch_positions]
        system_prompt, user_prompt = _build_keyphrase_prompt(batch_docs, n_keyphrases, n_docs_chars)
        max_tokens = max(128, len(batch_docs) * (n_keyphrases * 8 + 20))
        raw = labeler.call_structured(
            system_prompt, user_prompt, schema=KEYPHRASE_SCHEMA, max_tokens=max_tokens
        )
        parsed = _parse_keyphrase_response(raw, len(batch_docs), n_keyphrases)

        new_lines = []
        for pos, phrases in zip(batch_positions, parsed):
            results[pos] = phrases
            new_lines.append({"hash": hashes[pos], "keyphrases": phrases})

        if cache_path and new_lines:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "a") as f:
                for rec in new_lines:
                    f.write(json.dumps(rec) + "\n")

    return [r if r is not None else [] for r in results]


def keyphrase_expand_embeddings(
    documents: list[str],
    keyphrases: list[list[str]],
    encoder,
    weight: float = 0.5,
    mode: Literal["average", "concat_encode"] = "average",
    normalize: bool = True,
) -> np.ndarray:
    """Expand document embeddings with LLM-generated keyphrases.

    *encoder* is duck-typed (``.encode(list[str]) -> np.ndarray``), so any
    ``EmbeddingEngine`` — local or API-based — works directly.

    - ``"average"``: ``l2norm((1-weight) * enc(doc) + weight * enc(keyphrases))``
      for documents with keyphrases; documents with an empty keyphrase list
      are left as their own (normalized) embedding — blending in
      ``enc("")`` would dilute them with a meaningless embedding-of-nothing.
    - ``"concat_encode"``: re-encode ``doc + "\\nKeyphrases: " + phrases``
    """
    if mode == "concat_encode":
        texts = [
            doc + ("\nKeyphrases: " + ", ".join(kws) if kws else "")
            for doc, kws in zip(documents, keyphrases)
        ]
        concat = np.asarray(encoder.encode(texts), dtype=np.float64)
        if not normalize:
            return concat
        norms = np.linalg.norm(concat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return concat / norms

    if mode != "average":
        raise ValueError(f"Unknown mode: {mode!r}")

    kw_positions = [i for i, kws in enumerate(keyphrases) if kws]
    if kw_positions:
        phrase_texts = ["; ".join(keyphrases[i]) for i in kw_positions]
        # One encode() call for docs + keyphrase texts, not two — halves
        # request overhead for API-backed encoders.
        all_emb = np.asarray(encoder.encode(documents + phrase_texts), dtype=np.float64)
        doc_emb = all_emb[: len(documents)]
        phrase_emb = all_emb[len(documents):]
        combined = doc_emb.copy()
        for local_i, doc_i in enumerate(kw_positions):
            combined[doc_i] = (1 - weight) * doc_emb[doc_i] + weight * phrase_emb[local_i]
    else:
        combined = np.asarray(encoder.encode(documents), dtype=np.float64)

    if not normalize:
        return combined
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return combined / norms
