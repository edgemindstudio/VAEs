# common/data.py

"""
Common dataset utilities for generative and evaluation pipelines.

Features
--------
- Robust .npy loader for malware image dataset:
    train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
- Safe normalization to [0,1] (with automatic 0..255 detection)
- Consistent shaping to channels-last (H, W, C)
- One-hot label conversion with idempotency
- Split provided test set into (val, test)
- Optional tf.data dataset helpers for training loops

Typical usage
-------------
from common.data import (
    load_dataset_npy,          # -> x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh
    to_minus1_1,               # map [0,1] -> [-1,1] for tanh decoders / GANs
    make_tf_dataset,           # build a tf.data.Dataset efficiently
    describe_labels,           # quick per-class counts
)

Contract
--------
- Images returned by `load_dataset_npy` are in [0, 1] (float32).
- Labels are returned as one-hot (float32).
- Input arrays may be (N, H, W), (N, H, W, C), or (N, H*W*C); we coerce to (N, H, W, C).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Basic conversions
# ---------------------------------------------------------------------
def to_01(x: np.ndarray) -> np.ndarray:
    """
    Ensure image array is float32 in [0, 1].
    If `x` appears to be 0..255, scale by 255. Otherwise clamp to [0,1].
    """
    x = x.astype("float32", copy=False)
    if np.nanmax(x) > 1.5:  # heuristic: data is likely 0..255
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map images from [0,1] -> [-1,1]."""
    return (x01 - 0.5) * 2.0


def to_01_from_minus1_1(xm11: np.ndarray) -> np.ndarray:
    """Map images from [-1,1] -> [0,1]."""
    return np.clip((xm11 + 1.0) / 2.0, 0.0, 1.0)


# ---------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------
def _reshape_to_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Coerce `x` to (N, H, W, C) given img_shape=(H,W,C).

    Acceptable inputs:
      - (N, H, W, C) -> unchanged (validated)
      - (N, H, W)    -> add channel axis if C == 1
      - (N, H*W*C)   -> reshape to (N, H, W, C)
    """
    H, W, C = img_shape

    if x.ndim == 4 and tuple(x.shape[1:]) == (H, W, C):
        return x.astype("float32", copy=False)

    if x.ndim == 3 and x.shape[1:] == (H, W) and C == 1:
        return x.astype("float32", copy=False)[..., None]

    if x.ndim == 2 and x.shape[1] == H * W * C:
        return x.astype("float32", copy=False).reshape((-1, H, W, C))

    raise ValueError(
        f"Cannot coerce array of shape {x.shape} to (N,{H},{W},{C}). "
        "Expected (N,H,W,C), (N,H,W) with C=1, or (N,H*W*C)."
    )


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Idempotent one-hot:
      - If y already looks like (N, K) with K==num_classes, return as float32.
      - Else assume y are integer class ids and convert.
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32", copy=False)
    return tf.keras.utils.to_categorical(y.astype("int64").ravel(), num_classes).astype("float32")


def describe_labels(y_oh: np.ndarray) -> Dict[int, int]:
    """
    Quick class histogram from one-hot labels: {class_id: count}.
    """
    if y_oh.ndim != 2:
        raise ValueError(f"Expected one-hot labels of shape (N, K), got {y_oh.shape}")
    y_ids = np.argmax(y_oh, axis=1)
    vals, counts = np.unique(y_ids, return_counts=True)
    return {int(k): int(v) for k, v in zip(vals, counts)}


# ---------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------
def _load_required_files(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load four required .npy files from `data_dir`.
    Raises a clear error if any are missing.
    """
    data_dir = Path(data_dir)
    req = ["train_data.npy", "train_labels.npy", "test_data.npy", "test_labels.npy"]
    missing = [p for p in req if not (data_dir / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required dataset files under {data_dir}: {missing}. "
            "Expected: train_data.npy, train_labels.npy, test_data.npy, test_labels.npy"
        )

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test = np.load(data_dir / "test_data.npy")
    y_test = np.load(data_dir / "test_labels.npy")
    return x_train, y_train, x_test, y_test


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from four .npy files in `data_dir`, coerce shapes, normalize to [0,1],
    convert labels to one-hot, and split provided test set into (val, test).

    Returns:
      x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh   (all float32)
    """
    H, W, C = img_shape

    x_train, y_train, x_test, y_test = _load_required_files(Path(data_dir))

    # Normalize to [0,1] then force channels-last
    x_train01 = _reshape_to_hwc(to_01(x_train), (H, W, C))
    x_test01  = _reshape_to_hwc(to_01(x_test),  (H, W, C))

    # Labels → one-hot
    y_train_oh = one_hot(y_train, num_classes)
    y_test_oh  = one_hot(y_test,  num_classes)

    # Split test -> (val, test)
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    n_val = int(round(len(x_test01) * float(val_fraction)))
    x_val01, y_val_oh = x_test01[:n_val], y_test_oh[:n_val]
    x_test01, y_test_oh = x_test01[n_val:], y_test_oh[n_val:]

    return x_train01, y_train_oh, x_val01, y_val_oh, x_test01, y_test_oh


# ---------------------------------------------------------------------
# tf.data helpers (optional)
# ---------------------------------------------------------------------
def make_tf_dataset(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    shuffle_buffer: int = 10240,
    cache: bool = False,
    drop_remainder: bool = False,
    prefetch: bool = True,
    augment: Optional[callable] = None,
) -> tf.data.Dataset:
    """
    Build a performant tf.data pipeline from (x[, y]) arrays.

    - Caches in-memory if `cache=True`.
    - Shuffles with buffer `shuffle_buffer` if `shuffle=True`.
    - Applies optional `augment` callable: fn(x, y)->(x', y') or fn(x)->x'.
    - Prefetches AUTOTUNE by default.

    Notes:
      - This function does NOT change the numerical range. Prepare with `to_minus1_1`
        or other mapping beforehand if your model expects a specific range.
    """
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x, y))

    if cache:
        ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    if augment is not None:
        # Support both fn(x)->x' and fn(x,y)->(x',y')
        def _aug(*args):
            out = augment(*args)
            return out
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


# ---------------------------------------------------------------------
# Convenience wrappers used by training scripts
# ---------------------------------------------------------------------
def load_for_training_01(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
):
    """
    Shorthand that returns arrays in [0,1] (common for evaluation):
      x_train01, y_train, x_val01, y_val, x_test01, y_test
    """
    return load_dataset_npy(data_dir, img_shape, num_classes, val_fraction)


def load_for_training_minus1_1(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
):
    """
    Shorthand that maps images to [-1,1] (common for GANs / tanh decoders):
      x_train_m11, y_train, x_val_m11, y_val, x_test_m11, y_test
    """
    x_train01, y_train, x_val01, y_val, x_test01, y_test = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction
    )
    return (
        to_minus1_1(x_train01),
        y_train,
        to_minus1_1(x_val01),
        y_val,
        to_minus1_1(x_test01),
        y_test,
    )
