# common/interfaces.py
"""
Shared interfaces and typed contracts for generative pipelines and evaluation.

What this module provides
-------------------------
1) Typed dataclasses that mirror the JSON produced by `evaluate_model_suite`
   so we have a single source of truth across GAN/VAE (and future models).
2) Small Protocol-based "interfaces" for:
      - Train / synth pipelines
      - Conditional generators / decoders
      - Preview grid saving helpers
3) Convenience helpers:
      - Minimal validation of an evaluation summary dict
      - Safe (de)serialization to and from dict/JSON-friendly objects
      - Enforce Keras-3 weight filename suffix (`.weights.h5`)

These interfaces are intentionally lightweight and framework-agnostic;
they keep import costs basically zero and avoid tight coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------
# Typing aliases
# ---------------------------------------------------------------------
JSONDict = Dict[str, Any]
Array = np.ndarray
PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Evaluation summary schema (mirrors eval/val_common output)
# ---------------------------------------------------------------------
@dataclass
class ImageCounts:
    """How many images were used in evaluation (informational)."""
    train_real: int
    val_real: int
    test_real: int
    synthetic: Optional[int] = None


@dataclass
class GenerativeMetrics:
    """Unconditional / class-conditional generative quality & diagnostics."""
    fid: Optional[float] = None
    cfid_macro: Optional[float] = None
    cfid_per_class: Optional[List[float]] = None
    js: Optional[float] = None
    kl: Optional[float] = None
    diversity: Optional[float] = None
    fid_domain: Optional[str] = None  # e.g., "inception", "custom", etc.


@dataclass
class ClasswiseMetrics:
    """Per-class classification metrics (aligned by class index)."""
    precision: List[float]
    recall: List[float]
    f1: List[float]
    support: List[int]


@dataclass
class UtilityMetrics:
    """Aggregate classification utility metrics (macro-style where appropriate)."""
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    macro_auprc: float
    recall_at_1pct_fpr: float
    ece: Optional[float] = None
    brier: Optional[float] = None
    per_class: Optional[ClasswiseMetrics] = None


@dataclass
class EvaluationSummary:
    """
    Top-level evaluation record written to JSON by the standardized evaluator.
    """
    model: str
    seed: int
    images: ImageCounts
    generative: GenerativeMetrics
    utility_real_only: UtilityMetrics
    utility_real_plus_synth: UtilityMetrics
    deltas_RS_minus_R: Optional[Dict[str, float]] = None  # e.g., {"accuracy": -0.001, ...}

    # --- Serialization helpers ---
    def to_dict(self) -> JSONDict:
        return asdict(self)

    @staticmethod
    def from_dict(d: JSONDict) -> "EvaluationSummary":
        # Defensive parsing to keep this resilient to missing sub-objects
        images = ImageCounts(**d.get("images", {}))
        generative = GenerativeMetrics(**d.get("generative", {}))
        def _parse_um(x: JSONDict) -> UtilityMetrics:
            pc = x.get("per_class")
            per_class = ClasswiseMetrics(**pc) if isinstance(pc, dict) else None
            payload = {k: v for k, v in x.items() if k != "per_class"}
            return UtilityMetrics(per_class=per_class, **payload)
        uro = _parse_um(d.get("utility_real_only", {}))
        urs = _parse_um(d.get("utility_real_plus_synth", {}))
        return EvaluationSummary(
            model=d.get("model", "UNKNOWN"),
            seed=int(d.get("seed", 0)),
            images=images,
            generative=generative,
            utility_real_only=uro,
            utility_real_plus_synth=urs,
            deltas_RS_minus_R=d.get("deltas_RS_minus_R"),
        )


# ---------------------------------------------------------------------
# Protocols (lightweight "interfaces")
# ---------------------------------------------------------------------
class ConditionalGeneratorLike(Protocol):
    """
    Minimal surface for a conditional generator/decoder used in sampling.
    Expected to accept [noise, one_hot_labels] and return images as np.ndarray.
    """
    def predict(self, inputs: List[Array], verbose: int = 0) -> Array:  # [-1,1] or [0,1] depending on model
        ...


class PreviewGridSaver(Protocol):
    """
    A helper callable used to produce a preview grid from a trained model.
    Implementations typically live in `gan.sample` or `vae.sample`.
    """
    def __call__(
        self,
        model: ConditionalGeneratorLike,
        num_classes: int,
        latent_dim: int,
        *,
        n: int = 9,
        path: PathLike,
        **kwargs: Any,
    ) -> Path:
        ...


class TrainSynthesizePipeline(Protocol):
    """
    Minimal pipeline surface shared by GAN/VAE orchestrators.
    """
    def train(self, x_train: Array, y_train: Array, **kwargs: Any) -> Tuple[Any, Any]:
        """
        Train a model (or a pair of models). Return a tuple that typically contains
        the components you’ll want to reuse (e.g., (G, D) for GANs or (ENC, DEC) for VAEs).
        """
        ...

    def synthesize(self, model: Optional[ConditionalGeneratorLike] = None) -> Tuple[Array, Array]:
        """
        Generate a balanced per-class synthetic dataset and return (x_synth, y_synth).
        Implementations commonly also persist per-class `.npy` dumps under ARTIFACTS.synthetic.
        """
        ...


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------
REQUIRED_TOP_KEYS = {
    "model",
    "seed",
    "images",
    "generative",
    "utility_real_only",
    "utility_real_plus_synth",
}

REQUIRED_UTILITY_KEYS = {
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "macro_auprc",
    "recall_at_1pct_fpr",
}

def validate_eval_summary(d: JSONDict) -> Tuple[bool, List[str]]:
    """
    Lightweight structural validation of an evaluation summary dict.
    Returns (is_valid, problems).
    """
    problems: List[str] = []

    missing = REQUIRED_TOP_KEYS - set(d.keys())
    if missing:
        problems.append(f"Missing top-level keys: {sorted(missing)}")

    for sect in ("utility_real_only", "utility_real_plus_synth"):
        if sect in d and isinstance(d[sect], dict):
            missing_u = REQUIRED_UTILITY_KEYS - set(d[sect].keys())
            if missing_u:
                problems.append(f"Missing keys in '{sect}': {sorted(missing_u)}")
        else:
            problems.append(f"Section '{sect}' is missing or not a dict")

    ok = len(problems) == 0
    return ok, problems


def cli_brief(summary: EvaluationSummary) -> str:
    """
    Human-friendly one-liner for logs/CLI.
    """
    g = summary.generative
    uro = summary.utility_real_only
    urs = summary.utility_real_plus_synth
    return (
        f"Model={summary.model} Seed={summary.seed} | "
        f"FID={g.fid:.4f} cFID={g.cfid_macro:.4f} | "
        f"Acc(R)={uro.accuracy:.4f} Acc(RS)={urs.accuracy:.4f}"
        if (g.fid is not None and g.cfid_macro is not None)
        else f"Model={summary.model} Seed={summary.seed} | "
             f"Acc(R)={uro.accuracy:.4f} Acc(RS)={urs.accuracy:.4f}"
    )


# ---------------------------------------------------------------------
# Keras-3 filename helper
# ---------------------------------------------------------------------
def ensure_weights_suffix(path: PathLike) -> Path:
    """
    Keras 3 requires `save_weights()` paths to end with `.weights.h5`.
    This helper appends the suffix if missing and returns a `Path`.
    """
    p = Path(path)
    if not p.name.endswith(".weights.h5"):
        p = p.with_name(p.name + ".weights.h5")
    return p


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "JSONDict",
    "Array",
    "PathLike",
    "ImageCounts",
    "GenerativeMetrics",
    "ClasswiseMetrics",
    "UtilityMetrics",
    "EvaluationSummary",
    "ConditionalGeneratorLike",
    "PreviewGridSaver",
    "TrainSynthesizePipeline",
    "validate_eval_summary",
    "cli_brief",
    "ensure_weights_suffix",
]
