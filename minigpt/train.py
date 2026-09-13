"""Training loop. Owns the optimizer, device sharding, and checkpoint cadence.
Fixes vs. train.ipynb: masks padding out of the loss, reconciles the three
conflicting hyperparameter sets that used to live in helper.py/train.ipynb/README
into one YAML config, and supports resume via CheckpointManager."""

from __future__ import annotations

import argparse
import json
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from minigpt.checkpoint import make_manager, restore_latest, save
from minigpt.config import Config
from minigpt.data.loader import PackedDataset
from minigpt.model import MiniGPT

_PAD_ID = 0  # stories are packed contiguously; a genuine pad only occurs in the smoke path


def _compute_dtype(name: str):
    return {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[name]


def make_optimizer(cfg: Config, total_steps: int):
    warmup_steps = max(1, int(total_steps * cfg.train.warmup_frac))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.train.peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=cfg.train.end_lr,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.train.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=cfg.train.weight_decay, b2=cfg.train.adam_b2),
    )
    if cfg.train.compute_dtype == "float16":
        # T4 has no bf16 — fp16 needs dynamic loss scaling to avoid silent NaNs.
        tx = optax.apply_if_finite(tx, max_consecutive_errors=10)
    return tx


def loss_fn(model: MiniGPT, batch, deterministic: bool):
    x, y = batch
    logits = model(x, deterministic=deterministic)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    mask = (y != _PAD_ID).astype(losses.dtype)
    loss = jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss, logits


@nnx.jit
def train_step(model: MiniGPT, optimizer: nnx.Optimizer, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, _logits), grads = grad_fn(model, batch, False)
    optimizer.update(grads)
    return loss


@nnx.jit
def eval_step(model: MiniGPT, batch):
    loss, _ = loss_fn(model, batch, True)
    return loss


def train(cfg: Config, resume: bool = False):
    print(f"Config: {cfg}")
    rngs = nnx.Rngs(0)
    model = MiniGPT(cfg.model, rngs=rngs)
    n_params = model.num_params()
    print(f"Model params: {n_params:,}")

    data_dir = f"{cfg.out_dir}/{cfg.name}/data"
    train_ds = PackedDataset(f"{data_dir}/train.bin", cfg.model.maxlen)
    val_ds = PackedDataset(f"{data_dir}/val.bin", cfg.model.maxlen)

    total_steps = cfg.train.max_steps or (len(train_ds) // cfg.train.batch_size) * cfg.train.num_epochs
    tx = make_optimizer(cfg, total_steps)
    optimizer = nnx.Optimizer(model, tx)

    # Multi-GPU data parallel: shard the batch across all visible devices.
    n_devices = jax.local_device_count()
    if n_devices > 1:
        mesh = jax.sharding.Mesh(jax.devices(), ("data",))
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    else:
        data_sharding = None

    ckpt_dir = f"{cfg.out_dir}/{cfg.name}/checkpoints"
    manager = make_manager(ckpt_dir, cfg.train.ckpt_max_to_keep)
    start_step = 0
    if resume:
        # Pass the raw optax opt_state pytree, not nnx.state(optimizer) — the
        # latter would re-wrap the model's params (redundant with `model` below)
        # and pull in the PRNGKey-typed RNG stream nnx.Dropout carries, which
        # Orbax cannot serialize.
        step, model, restored_opt_state = restore_latest(manager, model, optimizer.opt_state)
        if step is not None:
            optimizer.opt_state = restored_opt_state
            start_step = step + 1
            print(f"Resumed from step {step}")

    rng = np.random.default_rng(cfg.data.seed)
    log_path = f"{cfg.out_dir}/{cfg.name}/metrics.jsonl"
    t0 = time.time()

    for step in range(start_step, total_steps):
        x, y = train_ds.get_batch(cfg.train.batch_size, rng)
        x, y = jnp.array(x), jnp.array(y)
        if data_sharding is not None:
            x = jax.device_put(x, data_sharding)
            y = jax.device_put(y, data_sharding)

        loss = train_step(model, optimizer, (x, y))

        if step % cfg.train.log_every == 0:
            elapsed = time.time() - t0
            rec = {"step": step, "loss": float(loss), "elapsed_s": round(elapsed, 1)}
            print(json.dumps(rec))
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            assert jnp.isfinite(loss), f"Non-finite loss at step {step} — check compute_dtype/loss scaling"

        if step % cfg.train.eval_every == 0 and step > 0:
            vx, vy = val_ds.get_batch(min(cfg.train.batch_size, len(val_ds) or 1), rng)
            val_loss = eval_step(model, (jnp.array(vx), jnp.array(vy)))
            print(json.dumps({"step": step, "val_loss": float(val_loss)}))

        if step % cfg.train.ckpt_every == 0 and step > 0:
            save(manager, step, model, optimizer.opt_state)

    save(manager, total_steps - 1, model, optimizer.opt_state)
    print(f"Training complete: {total_steps} steps in {time.time() - t0:.1f}s")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="override max_steps for a quick smoke run")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.steps is not None:
        cfg = Config(
            name=cfg.name,
            model=cfg.model,
            data=cfg.data,
            train=type(cfg.train)(**{**cfg.train.__dict__, "max_steps": args.steps}),
            out_dir=cfg.out_dir,
        )
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
