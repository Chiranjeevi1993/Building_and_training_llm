"""Checkpointing. The ONLY module in minigpt/ that talks to Orbax or Azure Blob.
Uses the modern orbax.checkpoint.CheckpointManager (the old notebooks used the
deprecated PyTreeCheckpointer API against an unpinned orbax-checkpoint version)."""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import orbax.checkpoint as ocp


def make_manager(ckpt_dir: str | Path, max_to_keep: int = 3) -> ocp.CheckpointManager:
    ckpt_dir = Path(ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(ckpt_dir, options=options)


def save(manager: ocp.CheckpointManager, step: int, model: nnx.Module, opt_state) -> None:
    # Only nnx.Param leaves — nnx.state(model) would also pull in the PRNGKey-typed
    # RNG stream that nnx.Dropout carries internally, which Orbax's StandardSave
    # cannot serialize (it isn't a plain array).
    manager.save(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardSave(nnx.state(model, nnx.Param)),
            opt_state=ocp.args.StandardSave(opt_state),
        ),
    )
    manager.wait_until_finished()


def restore_latest(manager: ocp.CheckpointManager, model: nnx.Module, opt_state):
    step = manager.latest_step()
    if step is None:
        return None, model, opt_state
    restored = manager.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
            opt_state=ocp.args.StandardRestore(opt_state),
        ),
    )
    nnx.update(model, restored.model)
    return step, model, restored.opt_state
