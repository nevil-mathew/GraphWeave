"""Tests for graphweave.utils.quote_verification.

generate_report_themes() asks an LLM to quote short phrases from source
documents when writing report narratives. These tests check the utility
that flags quotes the LLM produced but that don't actually appear in the
documents it was shown — the real risk being a fabricated participant quote
slipping into a report unnoticed.
"""

from graphweave.utils.quote_verification import extract_quoted_phrases, verify_quotes


class TestExtractQuotedPhrases:
    def test_single_quoted_phrase(self):
        narrative = "Parents described the school as 'too far' to reach on foot."
        assert extract_quoted_phrases(narrative) == ["too far"]

    def test_multiple_quoted_phrases(self):
        narrative = "Families said admission was 'denied without Aadhar', and called it 'unfair'."
        assert extract_quoted_phrases(narrative) == ["denied without Aadhar", "unfair"]

    def test_curly_quotes(self):
        narrative = "The center was described as ‘always closed’ by residents."
        assert extract_quoted_phrases(narrative) == ["always closed"]

    def test_no_quotes_returns_empty(self):
        narrative = "Parents described the school as too far to reach on foot."
        assert extract_quoted_phrases(narrative) == []

    def test_bare_apostrophes_not_treated_as_quotes(self):
        narrative = "Parents' frustration grew because they don't have documentation."
        assert extract_quoted_phrases(narrative) == []

    def test_whole_narrative_wrapped_in_quotes_is_skipped(self):
        narrative = "'This entire narrative paragraph is quoted for some reason as one long block.'"
        assert extract_quoted_phrases(narrative) == []


class TestVerifyQuotes:
    def test_verbatim_quote_is_verified(self):
        docs = ["The parent said the school was too far to walk to every day."]
        narrative = "One parent called the school 'too far' to walk to."
        assert verify_quotes(narrative, docs) == []

    def test_case_and_quote_style_insensitive(self):
        docs = ["Families said admission was DENIED WITHOUT AADHAR this year."]
        narrative = "Families described admission as ‘denied without aadhar’."
        assert verify_quotes(narrative, docs) == []

    def test_fabricated_quote_is_flagged(self):
        docs = ["The parent said the school was too far to walk to every day."]
        narrative = "One parent said the teachers were 'completely indifferent to our needs'."
        assert verify_quotes(narrative, docs) == ["completely indifferent to our needs"]

    def test_no_source_documents_flags_all_quotes(self):
        narrative = "One parent called the school 'too far' to walk to."
        assert verify_quotes(narrative, []) == ["too far"]

    def test_no_quotes_in_narrative_returns_empty(self):
        docs = ["Some document text."]
        narrative = "The narrative makes no quoted claims at all."
        assert verify_quotes(narrative, docs) == []

    def test_quote_verified_against_any_of_several_documents(self):
        docs = [
            "Irrelevant document about something else entirely.",
            "A second document where the phrase too far actually appears.",
        ]
        narrative = "Respondents said the walk was 'too far'."
        assert verify_quotes(narrative, docs) == []
