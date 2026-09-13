"""Evaluation metrics. Never mutates model state."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from minigpt.data.loader import PackedDataset
from minigpt.model import MiniGPT
from minigpt.train import loss_fn


def evaluate(model: MiniGPT, val_ds: PackedDataset, batch_size: int, n_batches: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(n_batches):
        x, y = val_ds.get_batch(batch_size, rng)
        loss, _ = loss_fn(model, (jnp.array(x), jnp.array(y)), True)
        losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    return {"val_loss": mean_loss, "perplexity": math.exp(mean_loss)}
