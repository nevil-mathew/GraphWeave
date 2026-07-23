"""Import boundary between ``graphweave.adaptation`` and the rest of GraphWeave.

This is the *only* file in the subpackage allowed to import from outside
``graphweave.adaptation`` (aside from ``evaluation.py``/``pipeline.py``, which
lazily import :class:`graphweave.core.model.GraphWeave` inside function bodies to
drive the actual clustering). To lift this subpackage out into its own
package later: vendor this one file (the five re-exports below), point it at
a local copy of the triplet-sampling/prompt/parsing logic, and nothing else
in ``graphweave/adaptation/`` needs to change.
"""

from __future__ import annotations

from graphweave.labeling.llm_granularity import (
    _build_triplet_prompt as build_triplet_prompt,
)
from graphweave.labeling.llm_granularity import (
    _GRANULARITY_SCHEMA as TRIPLET_SCHEMA,
)
from graphweave.labeling.llm_granularity import (
    _parse_triplet_response as parse_triplet_response,
)
from graphweave.labeling.llm_granularity import (
    _sample_triplets_fast as sample_triplets_fast,
)
from graphweave.labeling.llm_granularity import (
    _sample_triplets_informed as sample_triplets_informed,
)

__all__ = [
    "build_triplet_prompt",
    "TRIPLET_SCHEMA",
    "parse_triplet_response",
    "sample_triplets_fast",
    "sample_triplets_informed",
]
