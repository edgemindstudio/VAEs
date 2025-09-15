# vae/pipeline.py
"""
Conditional VAE (cVAE) pipeline: training + synthesis.

What this module does
---------------------
- Builds encoder/decoder via vae.models.build_models(...).
- Trains cVAE with a custom loop (MSE recon + beta_KL * KL), optional early stopping,
  and Keras 3–friendly checkpoints:
    * E_epoch_XXXX.weights.h5, D_epoch_XXXX.weights.h5
    * E_best.weights.h5,       D_best.weights.h5
    * E_last.weights.h5,       D_last.weights.h5
- Generates class-balanced synthetic images to [ARTIFACTS.synthetic]/:
    * gen_class_<k>.npy (float32 in [0,1])
    * labels_class_<k>.npy (int labels)
    * x_synth.npy, y_synth.npy (concatenated convenience dumps)

Expected inputs (from app/main.py)
----------------------------------
- Images passed to `train()` must be in [-1, 1] (tanh decoder convention).
- Labels are one-hot with shape (N, NUM_CLASSES).
- The optional logging callback in cfg["LOG_CB"] is called as:
    cb(epoch, train_loss, recon_loss, kl_loss, val_loss)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import tensorflow as tf

from vae.models import build_models


# ---------------------------- small helpers ----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


# ----------------------------- the pipeline ----------------------------

class VAEPipeline:
    """Training + synthesis orchestration for a conditional VAE."""

    DEFAULTS = {
        "IMG_SHAPE": (40, 40, 1),
        "NUM_CLASSES": 9,
        "LATENT_DIM": 100,
        "EPOCHS": 200,
        "BATCH_SIZE": 256,
        "LR": 2e-4,
        "BETA_1": 0.5,
        "BETA_KL": 1.0,                 # weight on KL term
        "LOG_EVERY": None,              # default computed as max(20, EPOCHS//40)
        "EARLY_STOPPING_PATIENCE": 10,  # epochs without val improvement
        "SAMPLES_PER_CLASS": 1000,      # synthesis size per class
        "ARTIFACTS": {
            "checkpoints": "artifacts/vae/checkpoints",
            "synthetic":   "artifacts/vae/synthetic",
        },
    }

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        d = self.DEFAULTS

        # Core hyperparams
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d["IMG_SHAPE"]))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d["NUM_CLASSES"]))
        self.latent_dim: int = int(self.cfg.get("LATENT_DIM", d["LATENT_DIM"]))
        self.epochs: int = int(self.cfg.get("EPOCHS", d["EPOCHS"]))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d["BATCH_SIZE"]))
        self.lr: float = float(self.cfg.get("LR", d["LR"]))
        self.beta_1: float = float(self.cfg.get("BETA_1", d["BETA_1"]))
        self.beta_kl: float = float(self.cfg.get("BETA_KL", d["BETA_KL"]))
        self.log_every: int = int(self.cfg.get("LOG_EVERY", max(20, self.epochs // 40)))
        self.patience: int = int(self.cfg.get("EARLY_STOPPING_PATIENCE", d["EARLY_STOPPING_PATIENCE"]))
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d["SAMPLES_PER_CLASS"]))

        # Artifacts
        arts = self.cfg.get("ARTIFACTS", d["ARTIFACTS"])
        self.ckpt_dir = Path(arts.get("checkpoints", d["ARTIFACTS"]["checkpoints"]))
        self.synth_dir = Path(arts.get("synthetic",   d["ARTIFACTS"]["synthetic"]))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional external logger
        self.log_cb = self.cfg.get("LOG_CB", None)

        # Build models (encoder/decoder are used directly; "vae" is not required for training here)
        mdict = build_models(
            img_shape=self.img_shape,
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            lr=self.lr,
            beta_1=self.beta_1,
            beta_kl=self.beta_kl,
        )
        self.encoder: tf.keras.Model = mdict["encoder"]
        self.decoder: tf.keras.Model = mdict["decoder"]

        # Single optimizer across both nets (sufficient for standard cVAE)
        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta_1)

    # -------------------------- training core --------------------------

    @tf.function
    def _train_step(self, x_batch, y_batch) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """One gradient step; returns (total, recon, kl)."""
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x_batch, y_batch], training=True)
            x_recon = self.decoder([z, y_batch], training=True)

            # MSE recon over pixels, averaged over batch
            recon_loss = tf.reduce_mean(tf.reduce_mean(tf.square(x_batch - x_recon), axis=[1, 2, 3]))
            kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(
                1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = recon_loss + self.beta_kl * kl_loss

        vars_ = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(total_loss, vars_)
        self.opt.apply_gradients(zip(grads, vars_))
        return total_loss, recon_loss, kl_loss

    @tf.function
    def _val_step(self, x_batch, y_batch) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Validation loss computation; no gradients."""
        z_mean, z_log_var, z = self.encoder([x_batch, y_batch], training=False)
        x_recon = self.decoder([z, y_batch], training=False)

        recon_loss = tf.reduce_mean(tf.reduce_mean(tf.square(x_batch - x_recon), axis=[1, 2, 3]))
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = recon_loss + self.beta_kl * kl_loss
        return total_loss, recon_loss, kl_loss

    def _save_epoch_ckpt(self, epoch: int) -> None:
        e_path = self.ckpt_dir / f"E_epoch_{epoch:04d}.weights.h5"
        d_path = self.ckpt_dir / f"D_epoch_{epoch:04d}.weights.h5"
        self.encoder.save_weights(str(e_path))
        self.decoder.save_weights(str(d_path))

    def _save_best_ckpt(self) -> None:
        self.encoder.save_weights(str(self.ckpt_dir / "E_best.weights.h5"))
        self.decoder.save_weights(str(self.ckpt_dir / "D_best.weights.h5"))

    def _save_last_ckpt(self) -> None:
        self.encoder.save_weights(str(self.ckpt_dir / "E_last.weights.h5"))
        self.decoder.save_weights(str(self.ckpt_dir / "D_last.weights.h5"))

    def train(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        Train cVAE with a NumPy-backed loop.

        Args
        ----
        x_train: float32 images in [-1, 1], shape (N, H, W, C)
        y_train: one-hot labels, shape (N, num_classes)
        x_val, y_val: optional validation split (same conventions)

        Returns
        -------
        (encoder, decoder) with final weights saved to *last* and best to *best*.
        """
        H, W, C = self.img_shape
        assert x_train.shape[1:] == (H, W, C), f"Expected train images {H}x{W}x{C}"
        assert y_train.shape[1] == self.num_classes, "Label one-hot size mismatch"

        steps_per_epoch = max(1, math.ceil(len(x_train) / self.batch_size))
        have_val = (x_val is not None) and (y_val is not None)

        best_val = float("inf")
        patience_ctr = 0

        for epoch in range(1, self.epochs + 1):
            # shuffle each epoch
            perm = np.random.permutation(len(x_train))
            train_losses, recon_losses, kl_losses = [], [], []

            for step in range(steps_per_epoch):
                sl = slice(step * self.batch_size, (step + 1) * self.batch_size)
                idx = perm[sl]
                xb = x_train[idx].astype(np.float32)
                yb = y_train[idx].astype(np.float32)

                t, r, k = self._train_step(xb, yb)
                train_losses.append(_to_float(t))
                recon_losses.append(_to_float(r))
                kl_losses.append(_to_float(k))

            # validation
            val_loss = float("nan")
            if have_val:
                # one pass over validation in mini-batches
                v_losses = []
                bs = self.batch_size
                for i in range(0, len(x_val), bs):
                    xv = x_val[i:i + bs].astype(np.float32)
                    yv = y_val[i:i + bs].astype(np.float32)
                    tv, _, _ = self._val_step(xv, yv)
                    v_losses.append(_to_float(tv))
                val_loss = float(np.mean(v_losses)) if v_losses else float("nan")

            # logging callback
            if self.log_cb:
                self.log_cb(
                    epoch,
                    float(np.mean(train_losses)),
                    float(np.mean(recon_losses)),
                    float(np.mean(kl_losses)),
                    val_loss,
                )
            else:
                # minimal console print if no TensorBoard callback was supplied
                print(f"[epoch {epoch:05d}] "
                      f"train={np.mean(train_losses):.4f} | "
                      f"recon={np.mean(recon_losses):.4f} | "
                      f"KL={np.mean(kl_losses):.4f} | "
                      f"val={val_loss:.4f}")

            # periodic checkpoints
            if epoch == 1 or (epoch % self.log_every == 0):
                self._save_epoch_ckpt(epoch)

            # best/early-stopping
            improved = have_val and (val_loss < best_val - 1e-6)
            if improved:
                best_val = val_loss
                patience_ctr = 0
                self._save_best_ckpt()
            elif have_val:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print("Early stopping reached.")
                    break

        # final "last" snapshot
        self._save_last_ckpt()
        return self.encoder, self.decoder

    # ------------------------------ synth ------------------------------

    def synthesize(self, decoder: Optional[tf.keras.Model] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class-balanced synthetic dataset and write it under self.synth_dir.

        Returns
        -------
        (x_synth, y_synth) where:
          - x_synth: float32 in [0,1], shape (num_classes * SAMPLES_PER_CLASS, H, W, C)
          - y_synth: one-hot,     shape (same, num_classes)
        """
        H, W, C = self.img_shape
        K = self.num_classes
        n_pc = self.samples_per_class

        G = decoder if decoder is not None else self.decoder

        xs, ys = [], []
        _ensure_dir(self.synth_dir)

        for cls in range(K):
            z = np.random.normal(0.0, 1.0, size=(n_pc, self.latent_dim)).astype(np.float32)
            y = tf.keras.utils.to_categorical([cls] * n_pc, K).astype(np.float32)

            g = G.predict([z, y], verbose=0)               # [-1, 1]
            g01 = np.clip((g + 1.0) / 2.0, 0.0, 1.0)       # -> [0, 1]

            xs.append(g01.reshape(-1, H, W, C))
            ys.append(y)

            # per-class traceability
            np.save(self.synth_dir / f"gen_class_{cls}.npy", g01)
            np.save(self.synth_dir / f"labels_class_{cls}.npy", np.full((n_pc,), cls, dtype=np.int32))

        x_synth = np.concatenate(xs, axis=0) if xs else np.empty((0, H, W, C), dtype=np.float32)
        y_synth = np.concatenate(ys, axis=0) if ys else np.empty((0, K), dtype=np.float32)

        # convenience dumps
        np.save(self.synth_dir / "x_synth.npy", x_synth)
        np.save(self.synth_dir / "y_synth.npy", y_synth)

        print(f"[synthesize] {x_synth.shape[0]} samples ({n_pc} per class) -> {self.synth_dir}")
        return x_synth, y_synth
