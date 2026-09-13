import numpy as np

from minigpt.config import Config
from minigpt.data.loader import PackedDataset
from minigpt.data.prepare import prepare


def test_prepare_roundtrip(tmp_path):
    src = tmp_path / "stories.txt"
    src.write_text("Once upon a time.<|endoftext|>The end.<|endoftext|>")

    cfg = Config.from_yaml("configs/dev_cpu.yaml")
    cfg = Config(
        name="test_roundtrip",
        model=cfg.model,
        data=type(cfg.data)(source="local_file", file_path=str(src), max_stories=None, val_fraction=0.5),
        train=cfg.train,
        out_dir=str(tmp_path / "runs"),
    )
    prepare(cfg)

    data_dir = tmp_path / "runs" / "test_roundtrip" / "data"
    assert (data_dir / "train.bin").exists()
    assert (data_dir / "val.bin").exists()

    train_tokens = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    val_tokens = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
    assert len(train_tokens) > 0
    assert len(val_tokens) > 0
    # The streaming temp file must not survive prepare().
    assert not (data_dir / "_all.bin").exists()


def test_prepare_unknown_source_raises(tmp_path):
    cfg = Config.from_yaml("configs/dev_cpu.yaml")
    cfg = Config(
        name="test_bad_source",
        model=cfg.model,
        data=type(cfg.data)(source="not_a_real_source", file_path=None, max_stories=None),
        train=cfg.train,
        out_dir=str(tmp_path / "runs"),
    )
    try:
        prepare(cfg)
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_packed_dataset_batch_shapes(tmp_path):
    bin_path = tmp_path / "toy.bin"
    np.arange(1000, dtype=np.uint16).tofile(bin_path)

    ds = PackedDataset(bin_path, ctx_len=32)
    rng = np.random.default_rng(0)
    x, y = ds.get_batch(batch_size=4, rng=rng)

    assert x.shape == (4, 32)
    assert y.shape == (4, 32)
    # y is x shifted by one position within the same block.
    for i in range(4):
        assert np.array_equal(y[i][:-1], x[i][1:])
