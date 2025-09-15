# vae/sample.py
"""
Sampling utilities for the Conditional VAE (cVAE).

What this provides
------------------
- Rebuilds the decoder architecture via `vae.models.build_models(...)` and (optionally)
  loads weights from a checkpoint.
- Generates synthetic samples in two convenient modes:
    * Balanced: N samples per class
    * Custom: arbitrary counts per class and/or class subset
- Saves per-class `.npy` arrays for traceability, plus combined `x_synth.npy` / `y_synth.npy`.
- Optionally saves a quick preview grid `.png`.
- Ships a helper `save_grid_from_decoder(...)` that your training script can call
  to snapshot a grid during/after training.

Conventions
-----------
- Decoder outputs are assumed in [-1, 1] (tanh). We rescale to [0, 1] before saving.
- `config.yaml` should define: IMG_SHAPE, NUM_CLASSES, LATENT_DIM (with sensible defaults).

Examples
--------
# 1) Balanced generation: 1,000 per class into default artifacts dir
python -m vae.sample --samples-per-class 1000

# 2) Total 5,000 samples, evenly split across classes 0,1,3 only
python -m vae.sample --num-samples 5000 --balanced --classes 0 1 3

# 3) Explicit per-class counts
python -m vae.sample --per-class 0:100 2:50 7:250

# 4) Use specific weights and write a 1×12 grid
python -m vae.sample --weights artifacts/vae/checkpoints/D_best.weights.h5 --grid 1 12
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from vae.models import build_models


# -----------------------------
# Utilities & setup
# -----------------------------
def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def enable_gpu_memory_growth() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}")

def ensure_artifact_dirs(root: Path) -> None:
    (root / "artifacts" / "vae" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "vae" / "synthetic").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "vae" / "summaries").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "tensorboard").mkdir(parents=True, exist_ok=True)


# -----------------------------
# CLI parsing helpers
# -----------------------------
def parse_per_class(entries: list[str]) -> dict[int, int]:
    """
    Parse items like ["0:100", "3:50"] -> {0:100, 3:50}
    """
    out: dict[int, int] = {}
    for item in entries:
        try:
            k_str, v_str = item.split(":")
            k, v = int(k_str), int(v_str)
            if v <= 0:
                raise ValueError
            out[k] = v
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid --per-class entry '{item}'. Use 'label:count' with positive count."
            )
    return out

def parse_classes(entries: list[str]) -> list[int]:
    try:
        return [int(x) for x in entries]
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --classes entry; must be integers.")


# -----------------------------
# Latent + image helpers
# -----------------------------
def sample_latents(n: int, dim: int, truncation: float | None = None) -> np.ndarray:
    """
    Sample latent vectors z ~ N(0, I). If `truncation` is provided, clip values to [-t, t].
    """
    z = np.random.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    if truncation is not None and truncation > 0:
        t = float(truncation)
        z = np.clip(z, -t, t)
    return z

def save_grid(
    images01: np.ndarray,
    img_shape: tuple[int, int, int],
    rows: int,
    cols: int,
    out_path: Path,
    titles: list[str] | None = None,
) -> None:
    """
    Save a grid of images in [0,1]. Supports 1 or 3 channels.
    """
    H, W, C = img_shape
    n = min(images01.shape[0], rows * cols)
    plt.figure(figsize=(cols * 1.6, rows * 1.6))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        im = images01[i].reshape(H, W, C)
        if C == 1:
            plt.imshow(im.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            plt.imshow(np.clip(im, 0.0, 1.0))
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Build & load decoder
# -----------------------------
def build_and_load_decoder(
    *,
    latent_dim: int,
    num_classes: int,
    img_shape: tuple[int, int, int],
    weights_path: Optional[Path],
    lr: float = 2e-4,
    beta_1: float = 0.5,
    beta_kl: float = 1.0,
) -> tf.keras.Model:
    """
    Rebuild the decoder via `vae.models.build_models` and load weights if provided.
    """
    mdict = build_models(
        img_shape=img_shape,
        latent_dim=latent_dim,
        num_classes=num_classes,
        lr=lr,
        beta_1=beta_1,
        beta_kl=beta_kl,
    )
    D = mdict["decoder"]

    if weights_path is not None and weights_path.exists():
        D.load_weights(str(weights_path))
        log(f"Loaded decoder weights: {weights_path}")
    else:
        if weights_path is not None:
            log(f"WARNING: weights not found at {weights_path}. Proceeding with untrained decoder.")
        else:
            log("WARNING: no weights path provided. Proceeding with untrained decoder.")
    return D


# -----------------------------
# Main sampling logic
# -----------------------------
def generate_for_classes(
    *,
    decoder: tf.keras.Model,
    latent_dim: int,
    img_shape: tuple[int, int, int],
    class_counts: dict[int, int],
    num_classes: int,
    save_dir: Path,
    save_per_class_npy: bool = True,
    truncation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate per-class images and return concatenated arrays:
        x_synth_01: (N, H, W, C) in [0,1]
        y_synth_oh: (N, num_classes) one-hot
    Also optionally saves per-class .npy dumps for traceability.
    """
    H, W, C = img_shape
    xs, ys = [], []

    for cls, count in class_counts.items():
        if cls < 0 or cls >= num_classes:
            raise ValueError(f"Class {cls} is out of range [0, {num_classes-1}]")
        if count <= 0:
            continue

        z = sample_latents(count, latent_dim, truncation=truncation)
        y = tf.keras.utils.to_categorical(np.full((count,), cls), num_classes).astype(np.float32)

        g = decoder.predict([z, y], verbose=0)          # expected in [-1, 1]
        g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)        # -> [0,1]

        xs.append(g01.reshape(-1, H, W, C))
        ys.append(y)

        if save_per_class_npy:
            np.save(save_dir / f"gen_class_{cls}.npy", g01)
            np.save(save_dir / f"labels_class_{cls}.npy", np.full((count,), cls, dtype=np.int32))
            log(f"Saved class {cls} -> {count} samples to {save_dir}")

    if not xs:
        return np.empty((0, H, W, C), dtype=np.float32), np.empty((0, num_classes), dtype=np.float32)

    x_synth = np.concatenate(xs, axis=0)
    y_synth = np.concatenate(ys, axis=0)
    return x_synth, y_synth


# -----------------------------
# Public helper used by app/main.py
# -----------------------------
def save_grid_from_decoder(
    decoder: tf.keras.Model,
    num_classes: int,
    latent_dim: int,
    *,
    n: int = 9,
    path: Optional[Path | str] = None,
    per_class: bool = True,
    seed: int = 42,
    conditional: bool = True,  # kept for API symmetry; decoder is conditional here
) -> Path:
    """
    Generate a 1×n preview from a conditional decoder and save to disk.

    - Assumes decoder outputs [-1, 1]; rescales to [0, 1].
    - Reuses `save_grid(images01, img_shape, rows, cols, out_path, titles)`.
    """
    rng = np.random.default_rng(seed)
    n = int(max(1, n))

    # sample noise
    z = rng.normal(0.0, 1.0, size=(n, latent_dim)).astype(np.float32)

    # choose labels
    labels = (np.arange(n) % num_classes) if per_class else rng.integers(0, num_classes, size=n)
    y = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    # decode [-1,1] -> [0,1]
    imgs = decoder.predict([z, y], verbose=0)
    imgs01 = np.clip((imgs + 1.0) / 2.0, 0.0, 1.0)

    out_path = Path(path) if path is not None else Path("preview.png")
    save_grid(
        images01=imgs01,
        img_shape=imgs01.shape[1:],  # (H, W, C)
        rows=1,
        cols=n,
        out_path=out_path,
        titles=[f"class {int(l)}" for l in labels],
    )
    return out_path


# -----------------------------
# CLI
# -----------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sample synthetic data from a trained Conditional VAE decoder.")
    ap.add_argument("--config", type=str,
                    default=str((Path(__file__).resolve().parents[1] / "config.yaml")),
                    help="Path to config.yaml")
    ap.add_argument("--weights", type=str, default=None,
                    help="Path to decoder weights (default: artifacts/vae/checkpoints/D_best.weights.h5)")

    # Sampling modes
    ap.add_argument("--samples-per-class", type=int, default=None,
                    help="If set, generate this many samples per selected class (balanced).")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Total number of samples to generate. Use with --balanced or a single --classes.")
    ap.add_argument("--balanced", action="store_true",
                    help="When using --num-samples, split evenly across classes (or selected subset).")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Subset of class IDs to sample, e.g. --classes 0 1 3 (defaults to all classes).")
    ap.add_argument("--per-class", nargs="+", default=None,
                    help="Explicit per-class counts, e.g. --per-class 0:100 2:50")

    # Output & preview
    ap.add_argument("--outdir", type=str, default=None,
                    help="Override output directory (default: artifacts/vae/synthetic/<timestamp>)")
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional subfolder tag under synthetic/, e.g. 'exp1'")
    ap.add_argument("--grid", nargs=2, type=int, default=None,
                    help="Save a preview grid: ROWS COLS, e.g., --grid 1 12")
    ap.add_argument("--grid-path", type=str, default=None,
                    help="Override grid output path (default: <outdir>/preview_grid.png)")
    ap.add_argument("--no-save-per-class", action="store_true",
                    help="Do not save per-class .npy dumps (still returns arrays).")

    # Sampling behavior
    ap.add_argument("--truncation", type=float, default=None,
                    help="Clip latent z to [-t, t]. If omitted, use full N(0,1).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    args = ap.parse_args(argv)

    # Setup
    root = Path(__file__).resolve().parents[1]
    ensure_artifact_dirs(root)
    set_seeds(args.seed)
    enable_gpu_memory_growth()

    # Config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    IMG_SHAPE   = tuple(cfg.get("IMG_SHAPE", (40, 40, 1)))
    NUM_CLASSES = int(cfg.get("NUM_CLASSES", 9))
    LATENT_DIM  = int(cfg.get("LATENT_DIM", 100))
    LR          = float(cfg.get("LR", 2e-4))
    BETA_1      = float(cfg.get("BETA_1", 0.5))
    BETA_KL     = float(cfg.get("BETA_KL", 1.0))

    # Output directory
    default_root = root / "artifacts" / "vae" / "synthetic"
    if args.outdir:
        samples_root = Path(args.outdir)
    else:
        samples_root = default_root
    if args.tag:
        samples_dir = samples_root / args.tag
    else:
        samples_dir = samples_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Decoder weights
    default_w = root / "artifacts" / "vae" / "checkpoints" / "D_best.weights.h5"
    weights_path = Path(args.weights) if args.weights else default_w

    # Build + load decoder
    log("Building decoder...")
    D = build_and_load_decoder(
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        img_shape=IMG_SHAPE,
        weights_path=weights_path,
        lr=LR,
        beta_1=BETA_1,
        beta_kl=BETA_KL,
    )

    # Decide class counts
    if args.per_class:
        class_counts = parse_per_class(args.per_class)
        selected = sorted(class_counts.keys())
    else:
        selected = parse_classes(args.classes) if args.classes else list(range(NUM_CLASSES))
        if args.samples_per_class is not None:
            class_counts = {c: int(args.samples_per_class) for c in selected}
        elif args.num_samples is not None:
            if not args.balanced and len(selected) == 1:
                class_counts = {selected[0]: int(args.num_samples)}
            elif args.balanced:
                per = int(math.floor(args.num_samples / len(selected)))
                if per <= 0:
                    raise ValueError("num-samples too small for the number of selected classes.")
                class_counts = {c: per for c in selected}
            else:
                raise ValueError("When using --num-samples, you must pass --balanced or a single class via --classes.")
        else:
            # default: 1,000 per class across all classes
            class_counts = {c: 1000 for c in selected}
            log("No counts provided; defaulting to 1000 per class.")

    log(f"Sampling plan: {class_counts}")

    # Generate
    x_synth, y_synth = generate_for_classes(
        decoder=D,
        latent_dim=LATENT_DIM,
        img_shape=IMG_SHAPE,
        class_counts=class_counts,
        num_classes=NUM_CLASSES,
        save_dir=samples_dir,
        save_per_class_npy=not args.no_save_per_class,
        truncation=args.truncation,
    )
    log(f"Generated synthetic arrays: x {x_synth.shape}, y {y_synth.shape}")

    # Save combined arrays
    np.save(samples_dir / "x_synth.npy", x_synth)
    np.save(samples_dir / "y_synth.npy", y_synth)

    # Optional grid
    if args.grid:
        rows, cols = args.grid
        n = min(rows * cols, x_synth.shape[0])
        titles = None
        try:
            y_int = np.argmax(y_synth, axis=1)
            titles = [str(y_int[i]) for i in range(n)]
        except Exception:
            pass
        grid_path = Path(args.grid_path) if args.grid_path else (samples_dir / "preview_grid.png")
        save_grid(x_synth[:n], IMG_SHAPE, rows, cols, grid_path, titles=titles)
        log(f"Saved preview grid to {grid_path}")

    # Metadata
    meta = {
        "timestamp": now_ts(),
        "img_shape": IMG_SHAPE,
        "num_classes": NUM_CLASSES,
        "latent_dim": LATENT_DIM,
        "weights": str(weights_path),
        "classes": sorted(list(class_counts.keys())),
        "class_counts": class_counts,
        "seed": args.seed,
        "truncation": args.truncation,
        "outdir": str(samples_dir),
        "x_synth_path": str(samples_dir / "x_synth.npy"),
        "y_synth_path": str(samples_dir / "y_synth.npy"),
    }
    with open(samples_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Wrote metadata.json to {samples_dir}")

    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
