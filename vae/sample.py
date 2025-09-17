# vae/sample.py

"""
Sampling utilities + unified-CLI synth entrypoint for the Conditional VAE (cVAE).

What this provides
------------------
- Rebuild decoder via `vae.models.build_models(...)` and (optionally) load weights.
- Generate synthetic samples either via:
    * Standalone CLI (balanced / custom counts, optional preview grid), or
    * Unified Orchestrator: `synth(cfg, output_root, seed)` → PNGs + manifest.
- Saves per-class `.npy` (optional in CLI) and combined `x_synth.npy` / `y_synth.npy`.

Conventions
-----------
- Decoder outputs tanh in [-1, 1]; we rescale to [0, 1] for saving/PNGs.
- Config keys supported (with fallbacks): IMG_SHAPE, NUM_CLASSES, LATENT_DIM, LR, BETA_1, BETA_KL.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

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


def _cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# -----------------------------
# Image I/O helpers
# -----------------------------
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


def save_grid(
    images01: np.ndarray,
    img_shape: Tuple[int, int, int],
    rows: int,
    cols: int,
    out_path: Path,
    titles: List[str] | None = None,
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
    img_shape: Tuple[int, int, int],
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
    dec = mdict["decoder"]

    if weights_path is not None and weights_path.exists():
        dec.load_weights(str(weights_path))
        log(f"Loaded decoder weights: {weights_path}")
    else:
        if weights_path is not None:
            log(f"[warn] weights not found at {weights_path}. Using untrained decoder.")
        else:
            log("[warn] no weights path provided. Using untrained decoder.")
    return dec


# -----------------------------
# Latent helpers & generation
# -----------------------------
def sample_latents(n: int, dim: int, truncation: float | None = None) -> np.ndarray:
    """
    Sample z ~ N(0, I). If `truncation` is provided, clip values to [-t, t].
    """
    z = np.random.normal(0.0, 1.0, size=(n, dim)).astype(np.float32)
    if truncation is not None and truncation > 0:
        t = float(truncation)
        z = np.clip(z, -t, t)
    return z


def _decode_to_01(decoder: tf.keras.Model, z: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Decoder outputs tanh in [-1,1]; rescale to [0,1].
    """
    g = decoder.predict([z, y_onehot], verbose=0)   # [-1,1]
    return np.clip((g + 1.0) / 2.0, 0.0, 1.0)


# -----------------------------
# Unified Orchestrator entrypoint
# -----------------------------
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    """
    Generate S PNGs/class into {output_root}/{class}/{seed}/... and return a manifest.

    Expects (optionally) a decoder checkpoint at:
      artifacts/vae/checkpoints/D_best.weights.h5
    """
    # Resolve shapes & counts (support legacy spellings)
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    LATENT_DIM = int(_cfg_get(cfg, "LATENT_DIM", _cfg_get(cfg, "vae.latent_dim", 100)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))
    LR = float(_cfg_get(cfg, "LR", _cfg_get(cfg, "vae.lr", 2e-4)))
    BETA_1 = float(_cfg_get(cfg, "BETA_1", _cfg_get(cfg, "vae.beta_1", 0.5)))
    BETA_KL = float(_cfg_get(cfg, "BETA_KL", _cfg_get(cfg, "vae.beta_kl", 1.0)))

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    default_weights = artifacts_root / "vae" / "checkpoints" / "D_best.weights.h5"
    weights_path = Path(_cfg_get(cfg, "ARTIFACTS.vae_decoder_weights", str(default_weights)))

    set_seeds(int(seed))
    tf.keras.utils.set_random_seed(int(seed))

    # Build + (optionally) load decoder
    decoder = build_and_load_decoder(
        latent_dim=LATENT_DIM,
        num_classes=K,
        img_shape=(H, W, C),
        weights_path=weights_path if Path(weights_path).exists() else None,
        lr=LR,
        beta_1=BETA_1,
        beta_kl=BETA_KL,
    )

    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict] = []

    # Generate per class
    for k in range(K):
        z = sample_latents(S, LATENT_DIM)
        y = tf.keras.utils.to_categorical(np.full((S,), k), num_classes=K).astype(np.float32)
        imgs01 = _decode_to_01(decoder, z, y)  # (S, H, W, C)

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(S):
            p = cls_dir / f"vae_{j:05d}.png"
            _save_png(imgs01[j], p)
            paths.append({"path": str(p), "label": int(k)})
        per_class_counts[str(k)] = int(S)

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
        "created_at": now_ts(),
    }
    return manifest


# -----------------------------
# Standalone generation used directly via CLI
# -----------------------------
def parse_per_class(entries: List[str]) -> Dict[int, int]:
    """Parse items like ['0:100', '3:50'] -> {0:100, 3:50}."""
    out: Dict[int, int] = {}
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


def parse_classes(entries: List[str]) -> List[int]:
    try:
        return [int(x) for x in entries]
    except Exception:
        raise argparse.ArgumentTypeError("Invalid --classes entry; must be integers.")


def save_grid_from_decoder(
    decoder: tf.keras.Model,
    num_classes: int,
    latent_dim: int,
    *,
    n: int = 9,
    path: Optional[Path | str] = None,
    per_class: bool = True,
    seed: int = 42,
) -> Path:
    """
    Generate a 1×n preview from a conditional decoder and save to disk.
    """
    rng = np.random.default_rng(seed)
    n = int(max(1, n))

    z = rng.normal(0.0, 1.0, size=(n, latent_dim)).astype(np.float32)
    labels = (np.arange(n) % num_classes) if per_class else rng.integers(0, num_classes, size=n)
    y = tf.keras.utils.to_categorical(labels, num_classes=num_classes).astype(np.float32)

    imgs01 = _decode_to_01(decoder, z, y)

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
                raise ValueError("When using --num-samples, pass --balanced or a single class via --classes.")
        else:
            class_counts = {c: 1000 for c in selected}
            log("No counts provided; defaulting to 1000 per class.")

    log(f"Sampling plan: {class_counts}")

    # Generate & (optionally) save per-class npy
    xs, ys = [], []
    H, W, C = IMG_SHAPE
    for cls, count in class_counts.items():
        z = sample_latents(count, LATENT_DIM, truncation=args.truncation)
        y = tf.keras.utils.to_categorical(np.full((count,), cls), NUM_CLASSES).astype(np.float32)
        g01 = _decode_to_01(D, z, y)

        xs.append(g01.reshape(-1, H, W, C))
        ys.append(y)

        if not args.no_save_per_class:
            np.save(samples_dir / f"gen_class_{cls}.npy", g01)
            np.save(samples_dir / f"labels_class_{cls}.npy", np.full((count,), cls, dtype=np.int32))
            log(f"Saved class {cls} -> {count} samples to {samples_dir}")

    x_synth = np.concatenate(xs, axis=0) if xs else np.empty((0, H, W, C), dtype=np.float32)
    y_synth = np.concatenate(ys, axis=0) if ys else np.empty((0, NUM_CLASSES), dtype=np.float32)
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


__all__ = ["build_and_load_decoder", "save_grid_from_decoder", "synth"]


if __name__ == "__main__":
    raise SystemExit(main())
