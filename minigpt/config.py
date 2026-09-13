"""Frozen dataclass configs loaded from YAML. This is the ONLY place
hyperparameters are defined — never hardcode them elsewhere."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    maxlen: int = 512
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclasses.dataclass(frozen=True)
class DataConfig:
    source: str  # "local_file" or a HF dataset name
    file_path: str | None = None
    max_stories: int | None = None
    val_fraction: float = 0.005
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    num_epochs: int = 1
    max_steps: int | None = None
    peak_lr: float = 3e-4
    end_lr: float = 1e-5
    warmup_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_b2: float = 0.95
    compute_dtype: str = "bfloat16"  # "bfloat16" (A100) or "float16" (T4, needs loss scaling)
    log_every: int = 20
    eval_every: int = 200
    ckpt_every: int = 200
    ckpt_max_to_keep: int = 3


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    out_dir: str = "runs"

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return Config(
            name=raw.get("name", Path(path).stem),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            out_dir=raw.get("out_dir", "runs"),
        )
