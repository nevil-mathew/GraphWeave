"""End-to-end orchestration: adapt an embedder to LLM triplet judgments,
then refit GraphWeave on the adapted embeddings.

This module (along with ``evaluation.py``) is the one documented exception
to the package's import boundary: driving GraphWeave and building an
``EmbeddingEngine`` necessarily requires importing from ``graphweave.core``.
Those imports are lazy (inside the function body) so the rest of the
subpackage stays independent of them.
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING

from .adapter import EmbeddingAdapter
from .config import AdaptationConfig
from .evaluation import compare_embedders, forgetting_check, triplet_accuracy

if TYPE_CHECKING:
    import pandas as pd

    from graphweave.core.model import GraphWeave


def adapt_and_refit(
    model: "GraphWeave",
    labeler,
    config: AdaptationConfig | None = None,
    evaluate: bool = True,
    labels_true=None,
    metadata: "pd.DataFrame | None" = None,
) -> tuple["GraphWeave", dict]:
    """Adapt *model*'s embedder to LLM triplet judgments and refit a **new**
    GraphWeave model with the adapted embeddings.

    *model* itself is left untouched — this always returns an independent,
    freshly-fitted model, which is what the before/after comparison needs.

    Parameters
    ----------
    model : GraphWeave
        A freshly fitted model (needs ``documents_``, ``labels_``, and
        ``original_embeddings_``/``embeddings_`` — not persisted by
        save()/load()).
    labeler : LLMLabeler (duck-typed)
        Must expose ``call_structured``.
    config : AdaptationConfig, optional
    evaluate : bool
        When True (default), also runs :func:`compare_embedders` on
        baseline vs. adapted embeddings and attaches it as
        ``report["comparison"]``.
    labels_true : np.ndarray, optional
        Ground-truth labels, if known (e.g. a labeled benchmark corpus —
        never available for real unsupervised use). Passed straight through
        to the internal :func:`compare_embedders` call so ``report["comparison"]``
        gets ARI/NMI/cluster-accuracy columns; has no other effect.
    metadata : pd.DataFrame, optional
        The same fit-time metadata *model* was originally fit with. Required
        when ``model.config.use_metadata_view`` is True — it is not persisted
        on the model, so the refit can't reconstruct the metadata view
        without it.

    Returns
    -------
    (GraphWeave, dict)
        The new fitted model and a report dict with the trained
        ``EmbeddingAdapter`` (``report["adapter"].save(path)`` persists the
        fine-tuned/linear weights — do this before an ephemeral session like
        a Kaggle kernel ends), triplet/LLM-call counts, held-out triplet
        accuracy before/after, and (for the fine-tune backend) a forgetting
        check.
    """
    if not getattr(model, "_is_fitted", False):
        raise ValueError("Model not fitted. Call fit() first.")
    if model.documents_ is None or model.labels_ is None:
        raise ValueError(
            "adapt_and_refit requires the fit-time documents and labels. "
            "These are not persisted by save()/load() — call this on a "
            "freshly fit() model, not a reloaded one."
        )

    from graphweave.core.embeddings import EmbeddingEngine
    from graphweave.core.model import GraphWeave

    config = config or AdaptationConfig()
    documents = model.documents_
    base_embeddings = (
        model.original_embeddings_ if model.original_embeddings_ is not None else model.embeddings_
    )
    probabilities = getattr(model, "probabilities_", None)

    is_local = model.config.embedding_provider == "local"
    # Reuse the model's own engine so the baseline/adapted embeddings are
    # produced in the exact same space the model was fit in. Rebuilding it
    # from a subset of config fields (as before) silently dropped
    # api_batch_size/output_dim/task_type/batch_delay/prefix, which for API
    # providers can change the output dimensionality or embedding space.
    base_encoder = getattr(model, "_embedding_engine", None)
    if base_encoder is None:
        base_encoder = EmbeddingEngine(
            model_name=model.config.embedding_model,
            batch_size=model.config.embedding_batch_size,
            provider=model.config.embedding_provider,
            api_key=model.config.embedding_api_key,
            api_batch_size=model.config.embedding_api_batch_size,
            output_dim=model.config.embedding_output_dim,
            task_type=model.config.embedding_task_type,
            batch_delay=model.config.embedding_batch_delay,
            prefix=model.config.embedding_prefix,
            verbose=False,
        )

    adapter = EmbeddingAdapter(
        labeler=labeler,
        base_encoder=base_encoder,
        base_model_name=model.config.embedding_model,
        is_local=is_local,
        config=config,
        embedding_prefix=model.config.embedding_prefix,
    )

    bank = adapter.collect_triplets(
        documents, base_embeddings, model.labels_, probabilities=probabilities
    )
    holdout_acc_before = triplet_accuracy(base_embeddings, bank.holdout)

    adapter.finetune(documents, embeddings=base_embeddings, bank=bank)

    new_embeddings = (
        adapter.linear_.transform(base_embeddings)
        if adapter.mode_ == "linear"
        else adapter.encode(documents)
    )
    holdout_acc_after = triplet_accuracy(new_embeddings, bank.holdout)

    if bank.holdout and holdout_acc_after <= holdout_acc_before:
        warnings.warn(
            "adapt_and_refit: held-out triplet accuracy did not improve "
            f"({holdout_acc_before:.3f} -> {holdout_acc_after:.3f}). The adapted "
            "embeddings may not be better for this corpus — check n_triplets, "
            "epochs, and LLM judgment quality before trusting them.",
            UserWarning,
            stacklevel=2,
        )

    if model.config.use_metadata_view and metadata is None:
        raise ValueError(
            "adapt_and_refit: the original model used use_metadata_view=True, but "
            "the fit-time metadata DataFrame is not persisted on the model, so it "
            "can't be reconstructed automatically. Pass the same metadata used for "
            "the original fit() call via the metadata= argument to preserve the "
            "metadata view on refit."
        )

    new_model = GraphWeave(n_topics=model.n_topics, config=copy.deepcopy(model.config))
    new_model.fit(
        documents,
        embeddings=new_embeddings,
        metadata=metadata,
        sample_weights=getattr(model, "sample_weights_", None),
    )

    report: dict = {
        "mode": adapter.mode_,
        "adapter": adapter,  # call adapter.save(path) to persist the fine-tuned/linear weights
        "n_llm_calls": bank.n_llm_calls,
        "n_cache_hits": bank.n_cache_hits,
        "n_unparsed": bank.n_unparsed,
        "n_train_triplets": len(bank.train),
        "n_holdout_triplets": len(bank.holdout),
        "holdout_triplet_acc_before": holdout_acc_before,
        "holdout_triplet_acc_after": holdout_acc_after,
    }

    if adapter.mode_ == "finetune":
        try:
            report["forgetting"] = forgetting_check(base_encoder.encode, adapter.encode)
        except Exception as e:  # best-effort regression guard; never fail the pipeline on it
            warnings.warn(f"forgetting_check failed: {e}", UserWarning, stacklevel=2)

    if evaluate:
        report["comparison"] = compare_embedders(
            documents,
            {"baseline": base_embeddings, "adapted": new_embeddings},
            labels_true=labels_true,
            holdout_bank=bank,
            base_config=copy.deepcopy(model.config),
        )

    return new_model, report
