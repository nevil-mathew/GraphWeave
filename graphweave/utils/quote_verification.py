"""
Quote verification for LLM-generated report narratives
=========================================================

``GraphWeave.generate_report_themes`` asks an LLM to write report-style
narratives that quote short phrases from the source documents (see
``graphweave/labeling/llm_labeler.py``). The LLM is instructed to only quote
text that appears in the documents it was shown, but nothing enforces that —
a fabricated quote attributed to a participant is a real integrity risk for a
report meant to be read by "program staff, funders, and the public" (the
system prompt's own words).

These utilities extract quoted phrases from a narrative and check each one
against the source documents actually shown to the LLM, so unverifiable
quotes can be surfaced to a human reviewer instead of silently shipping.
"""

from __future__ import annotations

import re
import unicodedata

# Matches phrases wrapped in straight or curly single/double quotes. The
# quote mark must be adjacent to a word boundary (start/end of string,
# whitespace, or punctuation) on the outside and text on the inside, so
# possessives and contractions ("parents'", "don't") aren't mistaken for a
# quote delimiter — a straight `'` there sits between two letters, not at a
# phrase boundary.
_BOUNDARY = r"\s\(\[\{—\-"
_QUOTE_PATTERN = re.compile(
    rf"(?:(?<=^)|(?<=[{_BOUNDARY}]))['‘\"“](.{{3,200}}?)['’\"”](?=$|[{_BOUNDARY}.,!?;:)\]}}])"
)


def _normalize(text: str) -> str:
    """Lowercase, fold curly quotes/whitespace so substring checks aren't
    defeated by cosmetic differences between the LLM's output and the source
    text (e.g. a typographic apostrophe vs a straight one)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def extract_quoted_phrases(narrative: str) -> list[str]:
    """Return the quoted phrases (without their quote marks) found in a
    narrative paragraph, e.g. ``"parents called the school 'too far'"`` ->
    ``["too far"]``.

    Skips matches that are almost the whole narrative — those are usually a
    quote-style mismatch (the writer quoted the entire sentence) rather than
    the intended short evidentiary phrase, and aren't worth verifying.
    """
    quotes = []
    for match in _QUOTE_PATTERN.finditer(narrative):
        phrase = match.group(1).strip()
        if phrase and len(phrase) < 0.8 * len(narrative):
            quotes.append(phrase)
    return quotes


def verify_quotes(narrative: str, source_documents: list[str]) -> list[str]:
    """Return the quoted phrases in ``narrative`` that do not appear
    verbatim (case/whitespace/quote-style insensitive) in any of
    ``source_documents``.

    An empty return means every quote was traced back to a source document.
    A non-empty return does not necessarily mean the LLM fabricated the
    quote — it may have paraphrased slightly — but it means the phrase
    could not be automatically verified and should be checked before the
    report goes out.
    """
    if not source_documents:
        return extract_quoted_phrases(narrative)

    normalized_docs = [_normalize(doc) for doc in source_documents]
    unverified = []
    for phrase in extract_quoted_phrases(narrative):
        normalized_phrase = _normalize(phrase)
        if not any(normalized_phrase in doc for doc in normalized_docs):
            unverified.append(phrase)
    return unverified
