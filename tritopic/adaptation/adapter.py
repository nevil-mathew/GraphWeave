"""Embedding adapters: a pure-numpy linear transform and a real
sentence-transformers fine-tune, both trained on LLM-judged triplets.

Neither class imports anything from ``tritopic.core`` — callers hand in a
duck-typed ``base_encoder`` (anything exposing ``.encode(list[str]) ->
np.ndarray``, e.g. :class:`tritopic.core.embeddings.EmbeddingEngine`) so this
module stays detachable. See the package docstring for the boundary.
"""

from __future__ import annotations

import json
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import AdaptationConfig
from .triplets import TripletBank

if TYPE_CHECKING:
    from .triplets import TripletJudgment


# ---------------------------------------------------------------------------
# Linear adapter: identity-initialized d×d transform, trained on triplets
# ---------------------------------------------------------------------------

class LinearAdapter:
    """A learned d×d linear map on top of frozen embeddings.

    Initialized to the identity so an untrained adapter reproduces the base
    embeddings exactly. Trained with mini-batch gradient descent on a cosine
    hinge triplet loss, shrunk toward the identity by an L2 penalty so noisy
    LLM judgments can't push it far from the base geometry.

    Works with any embedder (including API-based ones like Gemini, whose
    weights can't be fine-tuned) and needs no dependencies beyond numpy.
    """

    def __init__(self, dim: int, random_state: int = 42):
        self.dim = dim
        self.random_state = random_state
        self.W = np.eye(dim, dtype=np.float64)
        self.history_: list[float] = []

    def fit(
        self,
        embeddings: np.ndarray,
        judgments: "list[TripletJudgment]",
        epochs: int = 50,
        lr: float = 0.05,
        margin: float = 0.1,
        l2: float = 1e-3,
        batch_size: int = 256,
    ) -> "LinearAdapter":
        """Minimize ``mean(max(0, margin - cos(a',p') + cos(a',n'))) +
        l2 * ||W - I||^2`` where ``x' = l2norm(x @ W)``."""
        triples = np.array(
            [(j.anchor, j.positive, j.negative) for j in judgments], dtype=np.int64
        )
        self.history_ = []
        if len(triples) == 0:
            return self

        embeddings = np.asarray(embeddings, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        identity = np.eye(self.dim)

        for _ in range(epochs):
            order = rng.permutation(len(triples))
            epoch_losses = []
            for start in range(0, len(triples), batch_size):
                batch = triples[order[start : start + batch_size]]
                a_idx, p_idx, n_idx = batch[:, 0], batch[:, 1], batch[:, 2]
                a, p, n = embeddings[a_idx], embeddings[p_idx], embeddings[n_idx]

                A, P, N = a @ self.W, p @ self.W, n @ self.W
                A_norm = np.linalg.norm(A, axis=1, keepdims=True)
                P_norm = np.linalg.norm(P, axis=1, keepdims=True)
                N_norm = np.linalg.norm(N, axis=1, keepdims=True)
                A_norm = np.where(A_norm == 0, 1e-12, A_norm)
                P_norm = np.where(P_norm == 0, 1e-12, P_norm)
                N_norm = np.where(N_norm == 0, 1e-12, N_norm)

                A_hat, P_hat, N_hat = A / A_norm, P / P_norm, N / N_norm
                cos_ap = np.sum(A_hat * P_hat, axis=1, keepdims=True)
                cos_an = np.sum(A_hat * N_hat, axis=1, keepdims=True)

                margins = margin - cos_ap + cos_an
                active = (margins > 0).astype(np.float64)
                epoch_losses.append(float(np.mean(np.maximum(0.0, margins))))

                # d cos(x,y) / dx = (y_hat - cos * x_hat) / ||x||
                dcos_ap_dA = (P_hat - cos_ap * A_hat) / A_norm
                dcos_ap_dP = (A_hat - cos_ap * P_hat) / P_norm
                dcos_an_dA = (N_hat - cos_an * A_hat) / A_norm
                dcos_an_dN = (A_hat - cos_an * N_hat) / N_norm

                dL_dA = active * (-dcos_ap_dA + dcos_an_dA)
                dL_dP = active * (-dcos_ap_dP)
                dL_dN = active * dcos_an_dN

                grad_W = (a.T @ dL_dA + p.T @ dL_dP + n.T @ dL_dN) / len(batch)
                grad_W += 2 * l2 * (self.W - identity)

                self.W -= lr * grad_W

            self.history_.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

        return self

    def transform(self, embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
        out = np.asarray(embeddings, dtype=np.float64) @ self.W
        if not normalize:
            return out
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return out / norms

    def save(self, path: str) -> None:
        p = str(path) if str(path).endswith(".npz") else f"{path}.npz"
        np.savez(p, W=self.W, random_state=np.array(self.random_state))

    @classmethod
    def load(cls, path: str) -> "LinearAdapter":
        p = str(path) if str(path).endswith(".npz") else f"{path}.npz"
        data = np.load(p)
        adapter = cls(dim=data["W"].shape[0], random_state=int(data["random_state"]))
        adapter.W = data["W"]
        return adapter


# ---------------------------------------------------------------------------
# EmbeddingAdapter: orchestrates triplet collection + one of two backends
# ---------------------------------------------------------------------------

class EmbeddingAdapter:
    """Collects LLM triplet judgments and adapts an embedder to them.

    Two backends behind ``config.adapter_mode``:

    - ``"finetune"``: real sentence-transformers fine-tuning (local models
      only; requires the ``[adaptation]`` extra).
    - ``"linear"``: a :class:`LinearAdapter` on top of frozen embeddings
      (works with any embedder, pure numpy, CPU-friendly).
    - ``"auto"`` (default): finetune when possible, else linear with a
      warning explaining why.
    """

    def __init__(
        self,
        labeler=None,
        base_encoder=None,
        base_model_name: str = "all-MiniLM-L6-v2",
        is_local: bool = True,
        config: AdaptationConfig | None = None,
    ):
        self.labeler = labeler
        self.base_encoder = base_encoder
        self.base_model_name = base_model_name
        self.is_local = is_local
        self.config = config or AdaptationConfig()

        self.mode_: str | None = None
        self.model_ = None
        self.linear_: LinearAdapter | None = None
        self.bank_: TripletBank | None = None

    def collect_triplets(
        self,
        documents: list[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> TripletBank:
        cfg = self.config
        bank = TripletBank(cache_path=cfg.cache_path, random_state=cfg.random_state)
        bank.collect(
            self.labeler, documents, embeddings, labels, probabilities=probabilities,
            n_triplets=cfg.n_triplets, sampling=cfg.triplet_sampling,
            entropy_top_frac=cfg.entropy_top_frac, batch_size=cfg.llm_batch_size,
            holdout_frac=cfg.holdout_frac, n_docs_chars=cfg.n_docs_chars,
        )
        self.bank_ = bank
        return bank

    @staticmethod
    def _finetune_deps_error() -> str | None:
        """None if the sentence-transformers Trainer API is usable, else a
        human-readable reason — shared by the explicit-mode check and the
        auto-mode fallback so both give the same actionable guidance."""
        try:
            import accelerate  # noqa: F401
            import datasets  # noqa: F401
            import sentence_transformers

            if int(sentence_transformers.__version__.split(".")[0]) < 3:
                return "sentence-transformers >= 3.0 required for the Trainer API"
        except ImportError as e:
            return str(e)
        return None

    def _resolve_mode(self) -> str:
        mode = self.config.adapter_mode
        if mode == "finetune" and not self.is_local:
            raise ValueError(
                "adapter_mode='finetune' requires a local sentence-transformers "
                "embedder; API-based providers (e.g. Google Gemini) cannot be "
                "fine-tuned. Use adapter_mode='linear' or 'auto' instead."
            )
        if mode == "finetune":
            dep_error = self._finetune_deps_error()
            if dep_error is not None:
                raise ImportError(
                    f"adapter_mode='finetune' was requested but is unusable ({dep_error}). "
                    'Install with: pip install "tritopic[adaptation]", or use '
                    "adapter_mode='linear' or 'auto' instead."
                )
            return "finetune"
        if mode != "auto":
            return mode
        if not self.is_local:
            return "linear"
        dep_error = self._finetune_deps_error()
        if dep_error is not None:
            warnings.warn(
                f"Falling back to linear adapter mode ({dep_error}). For real fine-tuning "
                'install with: pip install "tritopic[adaptation]"',
                UserWarning,
            )
            return "linear"
        return "finetune"

    def finetune(
        self,
        documents: list[str],
        embeddings: np.ndarray | None = None,
        bank: TripletBank | None = None,
    ) -> "EmbeddingAdapter":
        bank = bank or self.bank_
        if bank is None:
            raise ValueError("No triplets to train on — call collect_triplets() first.")

        self.mode_ = self._resolve_mode()

        if self.mode_ == "linear":
            if embeddings is None:
                if self.base_encoder is None:
                    raise ValueError(
                        "linear adapter mode needs embeddings or a base_encoder to compute them"
                    )
                embeddings = self.base_encoder.encode(documents)
            self.linear_ = LinearAdapter(dim=embeddings.shape[1], random_state=self.config.random_state)
            self.linear_.fit(
                embeddings, bank.train, epochs=self.config.linear_epochs,
                lr=self.config.linear_lr, margin=self.config.linear_margin,
                l2=self.config.linear_l2, batch_size=self.config.linear_batch_size,
            )
        else:
            self._finetune_sentence_transformer(documents, bank)

        return self

    def _finetune_sentence_transformer(self, documents: list[str], bank: TripletBank) -> None:
        try:
            import datasets
        except ImportError:
            raise ImportError(
                'Fine-tuning requires the "datasets" package. '
                'Install with: pip install "tritopic[adaptation]"'
            )
        try:
            import accelerate  # noqa: F401
        except ImportError:
            raise ImportError(
                'Fine-tuning requires the "accelerate" package. '
                'Install with: pip install "tritopic[adaptation]"'
            )

        from sentence_transformers import SentenceTransformer

        try:
            from sentence_transformers.sentence_transformer import (
                SentenceTransformerTrainer,
                SentenceTransformerTrainingArguments,
                losses,
            )
        except ImportError:
            from sentence_transformers import (
                SentenceTransformerTrainer,
                SentenceTransformerTrainingArguments,
                losses,
            )

        model = SentenceTransformer(self.base_model_name, device=self.config.device)
        if self.config.max_seq_length is not None:
            model.max_seq_length = self.config.max_seq_length

        if self.config.freeze_layers > 0:
            n_frozen = 0
            for name, param in model.named_parameters():
                if any(f"layer.{i}." in name or f"layers.{i}." in name
                       for i in range(self.config.freeze_layers)):
                    param.requires_grad_(False)
                    n_frozen += 1
            if n_frozen == 0:
                warnings.warn(
                    "freeze_layers was set but no matching parameter names were "
                    "found — no layers were frozen.",
                    UserWarning,
                )

        train_texts = bank.to_training_texts(documents)
        if not train_texts["anchor"]:
            raise ValueError("No training triplets available (bank.train is empty).")
        train_dataset = datasets.Dataset.from_dict(train_texts)

        loss = (
            losses.MultipleNegativesRankingLoss(model)
            if self.config.loss == "mnrl"
            else losses.TripletLoss(model)
        )

        output_dir = self.config.output_dir or tempfile.mkdtemp(prefix="tritopic_adapt_")
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            seed=self.config.random_state,
            logging_steps=50,
            report_to="none",
            save_strategy="no",
        )
        trainer = SentenceTransformerTrainer(
            model=model, args=args, train_dataset=train_dataset, loss=loss
        )
        trainer.train()
        self.model_ = model

    def encode(self, documents: list[str], normalize: bool = True) -> np.ndarray:
        if self.mode_ is None:
            raise ValueError("EmbeddingAdapter not fitted — call finetune() first.")
        if self.mode_ == "finetune":
            return self.model_.encode(
                documents, batch_size=32, normalize_embeddings=normalize, convert_to_numpy=True
            )
        if self.base_encoder is None:
            raise ValueError("linear adapter mode needs base_encoder to encode new documents")
        base = self.base_encoder.encode(documents)
        return self.linear_.transform(base, normalize=normalize)

    def save(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"mode": self.mode_, "base_model_name": self.base_model_name}
        if self.mode_ == "finetune":
            self.model_.save(str(out / "model"))
        elif self.mode_ == "linear":
            self.linear_.save(str(out / "linear"))
        (out / "manifest.json").write_text(json.dumps(manifest))

    @classmethod
    def load(cls, path: str, base_encoder=None) -> "EmbeddingAdapter":
        out = Path(path)
        manifest = json.loads((out / "manifest.json").read_text())
        adapter = cls(base_model_name=manifest.get("base_model_name", "all-MiniLM-L6-v2"))
        adapter.mode_ = manifest["mode"]
        if adapter.mode_ == "finetune":
            from sentence_transformers import SentenceTransformer

            adapter.model_ = SentenceTransformer(str(out / "model"))
        else:
            adapter.linear_ = LinearAdapter.load(str(out / "linear"))
            adapter.base_encoder = base_encoder
        return adapter
