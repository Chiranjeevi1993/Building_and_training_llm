"""Tokenize a text corpus into flat uint16 token shards (train.bin / val.bin).
Replaces the old pad-every-story-to-maxlen pipeline in helper.py, which spent
most of the compute multiplying zeros. Packing removes padding entirely.

Streams tokens to disk in chunks rather than holding them all in memory, so
this scales from the 930 KB TinyStories sample up to multi-billion-token
corpora like FineWeb-Edu without needing a bigger machine."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken

from minigpt.config import Config

_tokenizer = tiktoken.get_encoding("gpt2")
_EOS_ID = _tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
_WRITE_CHUNK_TOKENS = 1_000_000  # flush to disk every ~1M tokens


def _iter_stories_local(file_path: Path, max_stories: int | None):
    """Stream stories from a local file, each ending in <|endoftext|>."""
    current: list[str] = []
    n_yielded = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "<|endoftext|>" in line:
                parts = line.split("<|endoftext|>")
                for part in parts[:-1]:
                    current.append(part)
                    text = "".join(current).strip()
                    if text:
                        yield text
                        n_yielded += 1
                        if max_stories and n_yielded >= max_stories:
                            return
                    current = []
                current = [parts[-1]] if parts[-1].strip() else []
            else:
                current.append(line)
        if current:
            text = "".join(current).strip()
            if text:
                yield text


def _iter_texts_hf(spec: str, max_stories: int | None):
    """Stream a HuggingFace dataset's text column, e.g.
    'HuggingFaceFW/fineweb-edu:sample-10BT'. Requires the `datasets` package
    (see requirements-gpu.txt) -- deliberately not a base dependency, since
    only the training image needs to stream large web-scale corpora."""
    from datasets import load_dataset

    if ":" in spec:
        name, config = spec.split(":", 1)
    else:
        name, config = spec, None

    ds = load_dataset(name, config, split="train", streaming=True)
    for i, example in enumerate(ds):
        if max_stories and i >= max_stories:
            return
        text = example.get("text")
        if text:
            yield text


def _tokenize_stream(texts, out_path: Path) -> tuple[int, int]:
    """Tokenize `texts` and stream uint16 token ids to `out_path` in chunks.
    Returns (n_documents, n_tokens)."""
    n_docs = 0
    n_tokens = 0
    buf: list[int] = []
    with open(out_path, "wb") as f:
        for text in texts:
            ids = _tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            if not ids or ids[-1] != _EOS_ID:
                ids.append(_EOS_ID)
            buf.extend(ids)
            n_docs += 1
            if len(buf) >= _WRITE_CHUNK_TOKENS:
                np.array(buf, dtype=np.uint16).tofile(f)
                n_tokens += len(buf)
                buf.clear()
        if buf:
            np.array(buf, dtype=np.uint16).tofile(f)
            n_tokens += len(buf)
    return n_docs, n_tokens


def prepare(cfg: Config) -> None:
    out_dir = Path(cfg.out_dir) / cfg.name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / "_all.bin"

    if cfg.data.source == "local_file":
        texts = _iter_stories_local(Path(cfg.data.file_path), cfg.data.max_stories)
    elif cfg.data.source == "hf_dataset":
        texts = _iter_texts_hf(cfg.data.file_path, cfg.data.max_stories)
    else:
        raise NotImplementedError(f"Unknown data.source={cfg.data.source!r}")

    n_docs, n_tokens = _tokenize_stream(texts, tmp_path)

    # Held-out val split is a contiguous tail slice, not shuffled — keeps eval
    # deterministic and avoids leaking adjacent-document context across the
    # split. Read back via memmap so the split itself stays O(1) memory too.
    all_tokens = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    n_val = max(1, int(n_tokens * cfg.data.val_fraction))
    all_tokens[: n_tokens - n_val].tofile(out_dir / "train.bin")
    all_tokens[n_tokens - n_val :].tofile(out_dir / "val.bin")
    del all_tokens
    tmp_path.unlink()

    print(
        f"Prepared {n_docs:,} documents -> {n_tokens:,} tokens "
        f"({n_tokens - n_val:,} train / {n_val:,} val) into {out_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    prepare(cfg)


if __name__ == "__main__":
    main()
