"""Tests for the optional LLM-based topic-alignment layer (tritopic.cumulative).

No real API is ever called: a fake labeler exposing only ``call_raw`` stands in
for :class:`~tritopic.labeling.llm_labeler.LLMLabeler`. Unit tests exercise
``llm_align_topics`` directly; integration tests drive ``CumulativeTriTopic`` with
the fake labeler on the config (precomputed blob embeddings, no model download).
"""

import json
import re

import numpy as np
import pytest

from tritopic import TriTopicConfig
from tritopic.cumulative import CumulativeConfig, CumulativeTriTopic
from tritopic.cumulative.alignment import (
    align_topics,
    llm_align_topics,
    _parse_alignment_response,
)

from test_cumulative import make_dataset, light_config  # reuse fixtures/helpers  # noqa: F401


# --------------------------------------------------------------------------- #
# Fake labelers (only call_raw is needed)
# --------------------------------------------------------------------------- #
class FakeLabeler:
    """Returns a canned string, or raises if given an Exception."""

    def __init__(self, response):
        self._resp = response
        self.calls: list[tuple[str, str]] = []
        self.domain_hint = None

    def call_raw(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


class RowEchoLabeler:
    """Parses ``[row N]`` out of the prompt and assigns each a fixed decision.

    Lets integration tests make deterministic claims without knowing the
    clustering's row order. ``decision="new_theme"`` marks every new topic as a
    brand-new theme; ``decision=<int gid>`` folds every new topic into that global.
    """

    def __init__(self, decision="new_theme"):
        self.decision = decision
        self.calls: list[tuple[str, str]] = []
        self.domain_hint = None

    def call_raw(self, system_prompt, user_prompt, max_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        rows = [int(r) for r in re.findall(r"\[row (\d+)\]", user_prompt)]
        out = []
        for r in rows:
            if self.decision == "new_theme":
                out.append({"new_local_idx": r, "new_theme": True})
            else:
                out.append({"new_local_idx": r, "global_id": int(self.decision)})
        return json.dumps({"assignments": out})


def _new(label, kws=("a", "b"), size=10):
    return {"label": label, "keywords": list(kws), "size": size}


def _reg(gid, label, kws=("a", "b")):
    return {"global_id": gid, "label": label, "keywords": list(kws)}


# --------------------------------------------------------------------------- #
# Unit tests: llm_align_topics
# --------------------------------------------------------------------------- #
class TestLlmAlignTopics:
    def test_one_to_one_match(self):
        labeler = FakeLabeler(
            json.dumps({"assignments": [
                {"new_local_idx": 0, "global_id": 10},
                {"new_local_idx": 1, "global_id": 11},
            ]})
        )
        mapping, next_id, matches = llm_align_topics(
            [_new("cats"), _new("dogs")],
            [_reg(10, "felines"), _reg(11, "canines")],
            labeler,
            next_id=12,
        )
        assert mapping == {0: 10, 1: 11}
        assert next_id == 12  # nothing new minted
        assert len(matches) == 2

    def test_many_to_one_merge(self):
        labeler = FakeLabeler(
            json.dumps({"assignments": [
                {"new_local_idx": 0, "global_id": 10},
                {"new_local_idx": 1, "global_id": 10},
            ]})
        )
        mapping, next_id, _ = llm_align_topics(
            [_new("kittens"), _new("cats")],
            [_reg(10, "felines")],
            labeler,
            next_id=11,
        )
        assert mapping == {0: 10, 1: 10}
        assert next_id == 11

    def test_new_theme_mints_fresh_id(self):
        labeler = FakeLabeler(
            json.dumps({"assignments": [
                {"new_local_idx": 0, "global_id": 10},
                {"new_local_idx": 1, "new_theme": True},
            ]})
        )
        mapping, next_id, _ = llm_align_topics(
            [_new("cats"), _new("rockets")],
            [_reg(10, "felines")],
            labeler,
            next_id=11,
        )
        assert mapping == {0: 10, 1: 11}
        assert next_id == 12

    def test_dropped_row_still_mapped(self):
        # LLM only mentions row 0; row 1 must still get a (fresh) global id.
        labeler = FakeLabeler(
            json.dumps({"assignments": [{"new_local_idx": 0, "global_id": 10}]})
        )
        mapping, next_id, _ = llm_align_topics(
            [_new("cats"), _new("dogs")],
            [_reg(10, "felines")],
            labeler,
            next_id=11,
        )
        assert set(mapping) == {0, 1}          # total function over all rows
        assert mapping[0] == 10
        assert mapping[1] == 11 and next_id == 12

    def test_hallucinated_global_id_becomes_new_theme(self):
        labeler = FakeLabeler(
            json.dumps({"assignments": [{"new_local_idx": 0, "global_id": 999}]})
        )
        mapping, next_id, _ = llm_align_topics(
            [_new("cats")], [_reg(10, "felines")], labeler, next_id=11,
        )
        assert mapping == {0: 11} and next_id == 12  # 999 not in registry -> fresh

    def test_empty_registry_skips_llm(self):
        labeler = FakeLabeler("should not be used")
        mapping, next_id, matches = llm_align_topics(
            [_new("cats"), _new("dogs")], [], labeler, next_id=5,
        )
        assert mapping == {0: 5, 1: 6}
        assert next_id == 7 and matches == []
        assert labeler.calls == []             # no API call on the first epoch

    def test_malformed_but_recoverable_json(self):
        # Prose-wrapped valid JSON is still parsed via the {...} substring.
        labeler = FakeLabeler(
            'Sure, here you go:\n{"assignments": [{"new_local_idx": 0, "global_id": 10}]}\nHope that helps!'
        )
        mapping, _, _ = llm_align_topics(
            [_new("cats")], [_reg(10, "felines")], labeler, next_id=11,
        )
        assert mapping == {0: 10}

    def test_llm_failure_falls_back_to_cosine(self):
        # On exception, result must equal align_topics(...) on the same inputs.
        new_centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        registry_centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        registry_ids = [10, 11]

        labeler = FakeLabeler(RuntimeError("boom"))
        with pytest.warns(UserWarning, match="falling back to cosine"):
            mapping, next_id, _ = llm_align_topics(
                [_new("cats"), _new("dogs")],
                [_reg(10, "felines"), _reg(11, "canines")],
                labeler,
                next_id=12,
                new_centroids=new_centroids,
                registry_centroids=registry_centroids,
                registry_ids=registry_ids,
                threshold=0.6,
            )
        exp_map, exp_next, _ = align_topics(
            new_centroids, registry_centroids, registry_ids, threshold=0.6, next_id=12
        )
        assert mapping == exp_map and next_id == exp_next


class TestParseAlignmentResponse:
    def test_regex_fallback_on_broken_json(self):
        raw = '{"assignments": [{"new_local_idx": 0, "global_id": 10}, {"new_local_idx": 1, "new_theme": true},'  # truncated
        out = _parse_alignment_response(raw, valid_new_rows={0, 1}, valid_global_ids={10})
        assert out == {0: 10, 1: None}

    def test_invalid_rows_and_ids_dropped(self):
        raw = json.dumps({"assignments": [
            {"new_local_idx": 5, "global_id": 10},   # row 5 not allowed
            {"new_local_idx": 0, "global_id": 99},   # gid 99 not allowed -> new theme
        ]})
        out = _parse_alignment_response(raw, valid_new_rows={0, 1}, valid_global_ids={10})
        assert out == {0: None}


# --------------------------------------------------------------------------- #
# Integration tests: CumulativeTriTopic with a fake labeler on the config
# --------------------------------------------------------------------------- #
class TestLlmAlignmentIntegration:
    def _cfg(self, light_config, **kw):
        return CumulativeConfig(
            base_config=light_config,
            strategy="global_refit",
            recluster_trigger="manual",
            **kw,
        )

    def test_llm_method_drives_assignment(self, light_config):
        # Every new topic is marked new_theme -> epoch-2 ids must be disjoint from
        # epoch-1 ids (cosine would have REUSED them on identical data).
        labeler = RowEchoLabeler(decision="new_theme")
        cum = CumulativeTriTopic(self._cfg(light_config, align_method="llm", align_labeler=labeler))
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=7)

        cum.add_batch(docs, embeddings=embs)          # epoch 1: registry empty, no LLM call
        ids1 = {int(x) for x in cum.labels_ if x != -1}
        assert labeler.calls == []                    # confirmed: first epoch skips the LLM

        cum.recluster()                               # epoch 2: registry non-empty -> LLM
        ids2 = {int(x) for x in cum.labels_ if x != -1}
        assert len(labeler.calls) >= 1
        assert ids1.isdisjoint(ids2)                  # all minted fresh, per the LLM
        assert len(cum.labels_) == len(docs)

    def test_both_mode_locks_strong_cosine_rows(self, light_config):
        # Identical data => cosine matches every topic strongly (sim >= high), so
        # the ambiguous band is empty and the LLM is never consulted; ids stay stable.
        labeler = RowEchoLabeler(decision="new_theme")
        cum = CumulativeTriTopic(self._cfg(light_config, align_method="both", align_labeler=labeler))
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=7)
        cum.add_batch(docs, embeddings=embs)
        ids1 = {int(x) for x in cum.labels_ if x != -1}
        cum.recluster()
        ids2 = {int(x) for x in cum.labels_ if x != -1}

        assert labeler.calls == []                    # cost control: no LLM call needed
        assert ids1 == ids2                           # cosine kept the stable ids

    def test_missing_labeler_falls_back_to_cosine(self, light_config):
        cum = CumulativeTriTopic(self._cfg(light_config, align_method="llm", align_labeler=None))
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=7)
        cum.add_batch(docs, embeddings=embs)
        ids1 = {int(x) for x in cum.labels_ if x != -1}
        with pytest.warns(UserWarning, match="requires config.align_labeler"):
            cum.recluster()
        ids2 = {int(x) for x in cum.labels_ if x != -1}
        assert ids1 == ids2                           # behaves exactly like cosine

    def test_registry_summaries_populated_and_keyed(self, light_config):
        labeler = RowEchoLabeler(decision="new_theme")
        cum = CumulativeTriTopic(self._cfg(light_config, align_method="llm", align_labeler=labeler))
        docs, embs, _ = make_dataset([0, 1, 2, 3], seed=7)
        cum.add_batch(docs, embeddings=embs)

        assert cum._registry_summaries                # non-empty
        assert set(cum._registry_summaries) == set(cum._registry_ids)
        assert len(cum._registry_ids) == len(set(cum._registry_ids))  # unique ids
        for s in cum._registry_summaries.values():
            assert "label" in s and "keywords" in s

    def test_align_disabled_is_identity_regardless_of_method(self, light_config):
        labeler = RowEchoLabeler(decision="new_theme")
        cum = CumulativeTriTopic(
            self._cfg(light_config, align_topics=False, align_method="llm", align_labeler=labeler)
        )
        docs, embs, _ = make_dataset([0, 1], seed=1)
        cum.add_batch(docs, embeddings=embs)
        cum.recluster()
        assert labeler.calls == []                    # identity path never calls the LLM
        assert cum.labels_ is not None
