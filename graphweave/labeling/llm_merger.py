"""
LLM-based Topic Merging
=======================

Ask an LLM to group semantically similar or micro topics into natural merged
clusters and assign a label to each group — all in a single API call.

Public entry point: :func:`llm_merge_topics_call`.
"""

from __future__ import annotations

import json
import re
import warnings


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_merge_prompt(
    topics_data: list[dict],
    n_topics: int | None,
    include_docs: bool,
    n_docs: int,
    doc_max_chars: int,
) -> tuple[str, str]:
    """Build the (system, user) prompts for LLM-driven topic merging.

    Parameters
    ----------
    topics_data:
        List of dicts, each ``{"topic_id": int, "size": int, "keywords": list[str],
        "representative_docs": list[str]}``.
    n_topics:
        Target number of output groups. ``None`` = let the LLM decide.
    include_docs:
        Whether to include representative doc snippets in the prompt.
    n_docs:
        Max docs per topic (from ``labeler.n_docs``).
    doc_max_chars:
        Max chars per doc snippet (from ``labeler.doc_max_chars``).
    """
    system_prompt = (
        "You are a topic modeling expert. You will be given a list of topics, each "
        "described by a size (number of documents) and its most representative keywords. "
        "Your job is to group these topics into natural, coherent themes by merging "
        "micro-topics (very small or overly narrow ones) and semantically duplicative "
        "topics together. For each resulting group you must also suggest a concise, "
        "descriptive label. "
        "You always respond with valid JSON and nothing else."
    )

    # Build the topic list block
    topic_lines: list[str] = []
    for t in topics_data:
        kw_str = ", ".join(t["keywords"])
        line = f"[{t['topic_id']}] (size={t['size']}) keywords: {kw_str}"
        if include_docs and t.get("representative_docs"):
            docs_block = ""
            for i, doc in enumerate(t["representative_docs"][:n_docs], 1):
                snippet = doc[:doc_max_chars] + "..." if len(doc) > doc_max_chars else doc
                docs_block += f"\n    Doc {i}: {snippet}"
            line += docs_block
        topic_lines.append(line)

    topic_block = "\n".join(topic_lines)

    if n_topics is not None:
        constraint = (
            f"\nTarget: merge into exactly {n_topics} groups. "
            f"If there are already fewer than {n_topics} topics, leave them as-is.\n"
        )
    else:
        constraint = (
            "\nMerge topics that are clearly micro-topics (tiny and overly narrow) or "
            "semantically duplicative. Keep topics that represent genuinely distinct themes "
            "as separate groups (possibly of size 1 — that is fine).\n"
        )

    user_prompt = f"""Below is a list of topics from a topic model. Each line shows the topic ID, how many documents belong to it, and its most representative keywords.

Topics:
{topic_block}
{constraint}
Rules:
- Every topic ID listed above must appear in exactly one group.
- Do NOT invent topic IDs that are not listed above.
- The "label" for each group should be 3-7 words, title case, and clearly describe the merged theme.
- Single-element groups are fine for topics that stand alone.

Respond ONLY with a JSON array in this exact format, no other text:
[
  {{"label": "Theme Label Here", "topic_ids": [0, 3, 7]}},
  {{"label": "Another Theme", "topic_ids": [1]}},
  ...
]"""

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_merge_response(
    raw: str, valid_ids: set[int]
) -> list[dict]:
    """Parse the LLM merge response into a list of ``{"label", "topic_ids"}`` dicts.

    3-tier robust parse:
    1. Full JSON array parse.
    2. Regex extraction of JSON array.
    3. Fallback: return each topic as its own singleton group (no-op).

    Unknown topic IDs are dropped with a warning. Any valid ID not covered by
    the LLM response is appended as a singleton so no topic is ever lost.
    """
    groups: list[dict] = []

    if not raw or not raw.strip():
        warnings.warn(
            "llm_merge_topics: empty LLM response — returning all topics as singletons (no merge).",
            UserWarning,
            stacklevel=4,
        )
        return [{"label": "", "topic_ids": [tid]} for tid in sorted(valid_ids)]

    def _try_parse(text: str) -> list[dict] | None:
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # Tier 1: locate outermost JSON array
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start != -1 and end > start:
        parsed = _try_parse(raw[start:end])
        if parsed is not None:
            groups = parsed

    # Tier 2: regex fallback — extract individual {label, topic_ids} objects
    if not groups:
        for m in re.finditer(
            r'\{\s*"label"\s*:\s*"([^"]+)"\s*,\s*"topic_ids"\s*:\s*(\[[^\]]*\])',
            raw,
        ):
            try:
                ids = json.loads(m.group(2))
                groups.append({"label": m.group(1), "topic_ids": ids})
            except (json.JSONDecodeError, ValueError):
                continue

    # Validate and clean
    covered: set[int] = set()
    clean: list[dict] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        label = g.get("label", "")
        raw_ids = g.get("topic_ids", [])
        if not isinstance(raw_ids, list):
            continue
        valid_group_ids = []
        for tid in raw_ids:
            try:
                tid = int(tid)
            except (TypeError, ValueError):
                continue
            if tid not in valid_ids:
                warnings.warn(
                    f"llm_merge_topics: LLM returned unknown topic_id {tid!r} — skipping.",
                    UserWarning,
                    stacklevel=4,
                )
                continue
            if tid in covered:
                continue  # first mention wins
            valid_group_ids.append(tid)
            covered.add(tid)
        if valid_group_ids:
            clean.append({"label": str(label) if label else "Merged Topic", "topic_ids": valid_group_ids})

    # Append any missing topic as a singleton (safety net)
    for tid in sorted(valid_ids - covered):
        warnings.warn(
            f"llm_merge_topics: topic_id {tid} was not assigned by the LLM — keeping as singleton.",
            UserWarning,
            stacklevel=4,
        )
        clean.append({"label": "", "topic_ids": [tid]})

    # Tier 3 fallback: LLM returned nothing usable
    if not clean:
        warnings.warn(
            "llm_merge_topics: could not parse LLM response — returning all topics as singletons (no merge).",
            UserWarning,
            stacklevel=4,
        )
        return [{"label": "", "topic_ids": [tid]} for tid in sorted(valid_ids)]

    return clean


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def llm_merge_topics_call(
    labeler,
    topics_data: list[dict],
    n_topics: int | None,
    include_docs: bool,
    use_structured_output: bool,
) -> list[dict]:
    """Ask the LLM to group topics and return merge groups with labels.

    Parameters
    ----------
    labeler:
        A :class:`~graphweave.labeling.LLMLabeler` instance.
    topics_data:
        List of ``{"topic_id", "size", "keywords", "representative_docs"}`` dicts.
    n_topics:
        Target group count, or ``None`` for LLM-decided natural merging.
    include_docs:
        Whether to include doc snippets in the prompt.
    use_structured_output:
        Use provider-native JSON enforcement where supported (Google, OpenAI,
        OpenRouter). Anthropic always falls back to prompt + parse.

    Returns
    -------
    list[dict]
        Each entry: ``{"label": str, "topic_ids": list[int]}``.
    """
    valid_ids = {int(t["topic_id"]) for t in topics_data}

    system_prompt, user_prompt = _build_merge_prompt(
        topics_data,
        n_topics,
        include_docs,
        labeler.n_docs,
        labeler.doc_max_chars,
    )

    # Generous token budget: one line per topic + some overhead
    max_tokens = max(512, len(topics_data) * 30 + 256)

    if use_structured_output:
        raw = labeler.call_structured(
            system_prompt,
            user_prompt,
            schema=_MERGE_SCHEMA,
            max_tokens=max_tokens,
        )
    else:
        raw = labeler.call_raw(system_prompt, user_prompt, max_tokens=max_tokens)

    return _parse_merge_response(raw, valid_ids)


# JSON Schema for structured output (Google response_schema / OpenAI json_schema)
_MERGE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "label":     {"type": "string"},
            "topic_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["label", "topic_ids"],
    },
}
