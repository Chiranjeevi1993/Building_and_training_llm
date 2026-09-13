"""The gate that makes it safe to fire an expensive cloud job: a tiny CPU
train run must complete and the loss must actually decrease."""

import builtins
import json
from pathlib import Path

from minigpt.config import Config
from minigpt.data.prepare import prepare
from minigpt.train import train

REPO_ROOT = Path(__file__).parent.parent


def test_cpu_smoke_train_loss_decreases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "TinyStories-1000.txt").write_text(
        "Once upon a time there was a small dog. It liked to play in the sun.<|endoftext|>" * 40
    )

    base = Config.from_yaml(str(REPO_ROOT / "configs" / "dev_cpu.yaml"))
    data_cfg = type(base.data)(
        source="local_file", file_path="TinyStories-1000.txt", max_stories=None, val_fraction=0.1
    )
    train_overrides = {
        **base.train.__dict__,
        "max_steps": 20,
        "log_every": 5,
        "eval_every": 100,
        "ckpt_every": 100,
    }
    cfg = Config(
        name=base.name,
        model=base.model,
        data=data_cfg,
        train=type(base.train)(**train_overrides),
        out_dir="runs",
    )

    prepare(cfg)

    losses = []
    orig_print = print

    def capture(*args, **kwargs):
        orig_print(*args, **kwargs)
        if args and isinstance(args[0], str) and args[0].startswith("{"):
            try:
                rec = json.loads(args[0])
            except json.JSONDecodeError:
                return
            if "loss" in rec:
                losses.append(rec["loss"])

    monkeypatch.setattr(builtins, "print", capture)
    train(cfg, resume=False)

    assert len(losses) >= 2
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
