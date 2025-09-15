# vae/train.py
"""
Train the Conditional VAE (cVAE) and save checkpoints + a small preview grid.

Usage
-----
# From repo root (or any directory), run:
python -m vae.train --config config.yaml

What this script does
---------------------
1) Reads a YAML config (see keys below).
2) Loads dataset from four .npy files under DATA_DIR:
   - train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
3) Converts inputs for training (decoder is tanh → expects [-1, 1]).
4) Trains the VAE via `VAEPipeline(cfg)` with TensorBoard-friendly logging.
5) Saves a 1×N preview grid from the decoder to ARTIFACTS.summaries.

Expected config keys (with sensible defaults if missing)
--------------------------------------------------------
SEED: 42
DATA_DIR: "USTC-TFC2016_malware"
IMG_SHAPE: [40, 40, 1]
NUM_CLASSES: 9
LATENT_DIM: 100
EPOCHS: 2000
BATCH_SIZE: 256
LR: 2e-4
BETA_1: 0.5
BETA_KL: 1.0
VAL_FRACTION: 0.5
ARTIFACTS:
  checkpoints: artifacts/vae/checkpoints
  synthetic:   artifacts/vae/synthetic
  summaries:   artifacts/vae/summaries
  tensorboard: artifacts/tensorboard
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import yaml

# Ensure sibling packages are importable (vae/, eval/, etc.)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    import sys
    sys.path.insert(0, str(ROOT))

from vae.pipeline import VAEPipeline
from vae.sample import save_grid_from_decoder as save_grid


# ---------------------------
# Small utilities
# ---------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_yaml(path: Path) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def ensure_dirs(cfg: Dict) -> None:
    arts = cfg.get("ARTIFACTS", {})
    for key in ("checkpoints", "synthetic", "summaries", "tensorboard"):
        p = arts.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes)

def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
):
    """
    Reads four .npy files:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Returns x in [0,1], y as one-hot, and splits provided test into (val, test).
    """
    H, W, C = img_shape
    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    def to_01_hwc(x):
        x = x.astype("float32")
        if x.max() > 1.5:  # if data is 0..255
            x = x / 255.0
        x = x.reshape((-1, H, W, C))
        return np.clip(x, 0.0, 1.0)

    x_train01 = to_01_hwc(x_train)
    x_test01  = to_01_hwc(x_test)

    y_train = one_hot(y_train, num_classes)
    y_test  = one_hot(y_test,  num_classes)

    n_val = int(len(x_test01) * val_fraction)
    x_val01, y_val = x_test01[:n_val], y_test[:n_val]
    x_test01, y_test = x_test01[n_val:], y_test[n_val:]

    return x_train01, y_train, x_val01, y_val, x_test01, y_test

def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map [0,1] → [-1,1] (tanh decoder convention)."""
    return (x01 - 0.5) * 2.0


# ---------------------------
# TensorBoard-friendly logging callback
# ---------------------------
def make_log_cb(tboard_dir: Path | None):
    writer = None
    if tboard_dir:
        writer = tf.summary.create_file_writer(str(tboard_dir))

    def cb(epoch: int, train_loss: float, recon_loss: float, kl_loss: float, val_loss: float):
        print(
            f"[epoch {epoch:05d}] "
            f"train={train_loss:.4f} | recon={recon_loss:.4f} | KL={kl_loss:.4f} | val={val_loss:.4f}"
        )
        if writer:
            with writer.as_default():
                tf.summary.scalar("loss/train_total", train_loss, step=epoch)
                tf.summary.scalar("loss/train_recon", recon_loss, step=epoch)
                tf.summary.scalar("loss/train_kl",    kl_loss,    step=epoch)
                tf.summary.scalar("loss/val_total",   val_loss,   step=epoch)
                writer.flush()
    return cb


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Conditional VAE (cVAE)")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    # Be nice to GPUs
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    args = parse_args()
    cfg = load_yaml(Path(args.config))

    # Sensible defaults
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("ARTIFACTS", {})
    cfg["ARTIFACTS"].setdefault("checkpoints", "artifacts/vae/checkpoints")
    cfg["ARTIFACTS"].setdefault("synthetic",   "artifacts/vae/synthetic")
    cfg["ARTIFACTS"].setdefault("summaries",   "artifacts/vae/summaries")
    cfg["ARTIFACTS"].setdefault("tensorboard", "artifacts/tensorboard")

    print(f"[config] Using {Path(args.config).resolve()}")

    set_seed(int(cfg.get("SEED", 42)))
    ensure_dirs(cfg)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    # Load dataset in [0,1]; split test → (val, test)
    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # Map to [-1,1] for tanh decoder
    x_train_m11 = to_minus1_1(x_train01)
    x_val_m11   = to_minus1_1(x_val01)

    # Attach a logging callback (TensorBoard) for VAEPipeline
    cfg["LOG_CB"] = make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))

    # Train
    pipe = VAEPipeline(cfg)
    enc, dec = pipe.train(
        x_train=x_train_m11, y_train=y_train,
        x_val=x_val_m11,     y_val=y_val,
    )

    # Save a small decoder preview grid
    preview_path = save_grid(
        dec, num_classes, cfg["LATENT_DIM"],
        n=min(9, num_classes),
        path=Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png",
        conditional=True,
    )
    print(f"Saved preview grid to {preview_path}")


if __name__ == "__main__":
    main()
