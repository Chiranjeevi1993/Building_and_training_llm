"""Memmap-backed loader over packed uint16 token shards. Slices contiguous
ctx_len blocks — no padding, deterministic epochs, scales to corpora far
larger than RAM."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PackedDataset:
    def __init__(self, bin_path: str | Path, ctx_len: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.ctx_len = ctx_len
        # Number of non-overlapping (x, y) blocks available.
        self.n_blocks = (len(self.data) - 1) // ctx_len

    def __len__(self) -> int:
        return self.n_blocks

    def get_batch(self, batch_size: int, rng: np.random.Generator):
        """Random-offset sampling (not epoch-aligned) — simpler and, for a
        streaming pretraining loop, statistically equivalent to sharded epochs."""
        ix = rng.integers(0, len(self.data) - self.ctx_len - 1, size=batch_size)
        x = np.stack([self.data[i : i + self.ctx_len].astype(np.int32) for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.ctx_len].astype(np.int32) for i in ix])
        return x, y
