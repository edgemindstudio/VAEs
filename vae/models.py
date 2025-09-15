# vae/models.py

"""
Model builders for the Conditional VAE (cVAE).

What this module provides
-------------------------
- Sampling: reparameterization layer.
- build_encoder(...): image + one-hot label -> (z_mean, z_log_var, z).
- build_decoder(...): (z, one-hot label) -> reconstructed image in [-1, 1].
- build_models(...): returns {"encoder","decoder","vae"} where "vae" is a
  subclassed Keras Model with custom train_step/test_step (Keras 3 compliant).

Conventions
-----------
- Training images are in [-1, 1] (tanh decoder). Rescale to [0, 1] in eval code.
- Labels are one-hot vectors of length `num_classes`.
- When saving weights with Keras 3 `save_weights()`, filenames must end with `.weights.h5`.
"""

from __future__ import annotations

from typing import Dict, Tuple
import math

import tensorflow as tf
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras import ops as K  # Keras 3 math ops (safe for KerasTensors)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
class Sampling(layers.Layer):
    """Reparameterization: z = mean + exp(0.5 * log_var) * eps."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = K.clip(z_log_var, -10.0, 10.0)  # numerical stability
        eps = keras.random.normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * eps


def _broadcast_labels(img: K.Tensor, onehot: K.Tensor) -> K.Tensor:
    """
    Broadcast label [B, C] -> spatial map [B, H, W, C] using Keras ops.
    """
    y = K.expand_dims(onehot, axis=1)  # (B,1,C)
    y = K.expand_dims(y, axis=1)       # (B,1,1,C)
    ones = K.ones_like(img[..., :1])   # (B,H,W,1)
    return y * ones                    # (B,H,W,C)


def _kl_per_example(z_mean: K.Tensor, z_log_var: K.Tensor) -> K.Tensor:
    """KL(N(mu, sigma) || N(0, I)) per-sample -> shape [B]."""
    return -0.5 * K.sum(1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)


# ---------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------
def build_encoder(
    *,
    img_shape: Tuple[int, int, int],
    latent_dim: int,
    num_classes: int,
) -> Model:
    """Conditional encoder: concat image with broadcast label, Conv → Dense → (mu, log_var, z)."""
    x_in = layers.Input(shape=img_shape, name="image_input")
    y_in = layers.Input(shape=(num_classes,), name="label_input")

    y_map = layers.Lambda(lambda t: _broadcast_labels(t[0], t[1]), name="label_broadcast")([x_in, y_in])
    x = layers.Concatenate(name="concat_img_label")([x_in, y_map])

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z         = Sampling(name="z")([z_mean, z_log_var])

    return Model([x_in, y_in], [z_mean, z_log_var, z], name="cEncoder")


def build_decoder(
    *,
    latent_dim: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    lr: float = 2e-4,
    beta_1: float = 0.5,
) -> Model:
    """
    Conditional decoder: concat (z, one-hot) → Dense/Reshape → ConvT → tanh output in [-1, 1].
    Returns a compiled decoder (MSE) for convenience; training happens via the VAE wrapper.
    """
    H, W, C = img_shape

    z_in = layers.Input(shape=(latent_dim,), name="z_input")
    y_in = layers.Input(shape=(num_classes,), name="label_input")

    x = layers.Concatenate(name="concat_z_label")([z_in, y_in])

    fh, fw = max(1, math.ceil(H / 4)), max(1, math.ceil(W / 4))
    ch = 64

    x = layers.Dense(fh * fw * ch, activation="relu")(x)
    x = layers.Reshape((fh, fw, ch))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Resizing(H, W, interpolation="bilinear")(x)  # guarantee exact size
    x_out = layers.Conv2D(C, 3, padding="same", activation="tanh", name="x_recon")(x)

    dec = Model([z_in, y_in], x_out, name="cDecoder")
    dec.compile(optimizer=Adam(learning_rate=lr, beta_1=beta_1), loss="mse")
    return dec


# ---------------------------------------------------------------------
# Subclassed VAE with custom train_step/test_step (Keras 3 friendly)
# ---------------------------------------------------------------------
class ConditionalVAE(Model):
    def __init__(self, encoder: Model, decoder: Model, beta_kl: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta_kl = float(beta_kl)

        # Track metrics nicely
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        # Keras resets these at each epoch automatically
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs, training=False):
        x, y = inputs  # x: image in [-1,1]; y: one-hot labels
        z_mean, z_log_var, z = self.encoder([x, y], training=training)
        x_recon = self.decoder([z, y], training=training)
        return x_recon  # plain forward; losses are computed in train/test step

    def compute_losses(self, x, y):
        z_mean, z_log_var, z = self.encoder([x, y], training=True)
        x_recon = self.decoder([z, y], training=True)

        # Reconstruction (MSE over pixels)
        recon_per_ex = K.mean(K.square(x - x_recon), axis=(1, 2, 3))  # (B,)
        recon_loss   = K.mean(recon_per_ex)                            # scalar

        # KL term
        kl_per_ex = _kl_per_example(z_mean, z_log_var)
        kl_loss   = K.mean(kl_per_ex)

        total = recon_loss + self.beta_kl * kl_loss
        return total, recon_loss, kl_loss

    def train_step(self, data):
        x, y = data  # dataset should yield (image, onehot_label)
        with tf.GradientTape() as tape:
            total, recon, kl = self.compute_losses(x, y)
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        total, recon, kl = self.compute_losses(x, y)
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(),
                "recon_loss": self.recon_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result()}


def build_models(
    *,
    img_shape: Tuple[int, int, int],
    latent_dim: int,
    num_classes: int,
    lr: float = 2e-4,
    beta_1: float = 0.5,
    beta_kl: float = 1.0,
) -> Dict[str, Model]:
    """
    Returns:
      - encoder: conditional encoder
      - decoder: conditional decoder
      - vae    : ConditionalVAE (subclassed) compiled with Adam
    """
    enc = build_encoder(img_shape=img_shape, latent_dim=latent_dim, num_classes=num_classes)
    dec = build_decoder(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_shape=img_shape,
        lr=lr,
        beta_1=beta_1,
    )

    vae = ConditionalVAE(enc, dec, beta_kl=beta_kl, name="cVAE")
    vae.compile(optimizer=Adam(learning_rate=lr, beta_1=beta_1))
    return {"encoder": enc, "decoder": dec, "vae": vae}


__all__ = ["Sampling", "build_encoder", "build_decoder", "ConditionalVAE", "build_models"]
