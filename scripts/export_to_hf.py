"""Push local/Blob checkpoints to Hugging Face Hub. Run this before any
teardown -- see CLAUDE.md and .plans/azure-migration.md Phase 4."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def export(run_dir: str, repo_id: str, token: str | None = None):
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (env var or --token) before exporting.")

    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        print(f"No checkpoints found at {ckpt_dir}, nothing to export.")
        return

    api = HfApi(token=token)
    create_repo(repo_id, token=token, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt_dir), repo_id=repo_id, path_in_repo="checkpoints")

    metrics_path = Path(run_dir) / "metrics.jsonl"
    if metrics_path.exists():
        api.upload_file(path_or_fileobj=str(metrics_path), path_in_repo="metrics.jsonl", repo_id=repo_id)

    print(f"Exported {ckpt_dir} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.environ.get("MINIGPT_RUN_DIR", "runs/gpt2_small_tinystories"))
    ap.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", "minigpt-tinystories"))
    ap.add_argument("--token", default=None)
    args = ap.parse_args()
    export(args.run_dir, args.repo_id, args.token)
