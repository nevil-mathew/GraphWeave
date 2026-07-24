"""Configuration for LLM-guided embedding adaptation.

Kept deliberately separate from :class:`graphweave.core.model.GraphWeaveConfig` —
see the ``graphweave.adaptation`` package docstring for the detachability
rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class AdaptationConfig:
    """Settings for :func:`graphweave.adaptation.adapt_and_refit` and friends."""

    # --- triplet collection ---
    n_triplets: int = 1000
    triplet_sampling: Literal["entropy", "informed", "fast"] = "entropy"
    entropy_top_frac: float = 0.5
    llm_batch_size: int = 8
    n_docs_chars: int = 300
    holdout_frac: float = 0.2
    cache_path: str | None = None
    # Overrides the auto-computed per-batch token budget (~20 tokens/triplet) for
    # triplet-judgment LLM calls. Needed for reasoning-style models, which can burn the
    # whole budget on hidden chain-of-thought before emitting the answer (surfaces as an
    # empty completion with finish_reason="length"). None keeps the compact auto-sizing —
    # fine for any plain instruct/flash model, which is the better fix for this task anyway.
    llm_max_tokens: int | None = None

    # --- fine-tuning (sentence-transformers backend) ---
    adapter_mode: Literal["auto", "finetune", "linear"] = "auto"
    loss: Literal["mnrl", "triplet"] = "mnrl"
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    train_batch_size: int = 32
    freeze_layers: int = 0
    max_seq_length: int | None = None
    output_dir: str | None = None
    device: str | None = None

    # --- linear-adapter backend (pure numpy) ---
    linear_epochs: int = 50
    linear_lr: float = 0.05
    linear_margin: float = 0.1
    linear_l2: float = 1e-3
    linear_batch_size: int = 256

    # --- hygiene / misc ---
    random_state: int = 42
    verbose: bool = True
