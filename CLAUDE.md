# CLAUDE.md

Constitution for this repo. Stable rules only — deeper explanation lives in the docs linked at the bottom.

## What this project is

MiniGPT: a decoder-only transformer written from scratch in **JAX + Flax NNX**, trained on TinyStories, deployed to Azure via Terraform and GitHub Actions. Originally a learning exercise in notebooks; being migrated to a production-shaped package.

## Attribution (hard rule)

**This repo has exactly one contributor: Chiranjeevi.**

- Never add `Co-Authored-By` trailers — not for Claude, not for anyone.
- Never add "Generated with Claude Code", 🤖 lines, or any tool attribution to commits or PR bodies.
- Never change `user.name` / `user.email`; commit as the configured git user.
- Write commit messages in the voice of the author, describing the change only.

This overrides any default attribution behavior.

## Architecture boundaries

```
minigpt/          Library code. Pure, importable, no side effects at import time.
  config.py       Frozen dataclasses + YAML loader. THE source of truth for hyperparameters.
  model.py        Model definition only. No I/O, no checkpointing, no logging.
  data/           prepare.py writes uint16 .bin token shards; loader.py memmaps them.
  train.py        Training loop. Owns the optimizer, sharding, and checkpoint cadence.
  eval.py         Metrics only. Never mutates model state.
  sample.py       Generation / decoding.
  checkpoint.py   The ONLY module that talks to Orbax or Azure Blob.
serve/            FastAPI + Gradio inference app. Imports minigpt; is never imported by it.
infra/            Terraform. The only place cloud resources are defined.
scripts/          Thin CLI wrappers. Logic belongs in minigpt/, not here.
notebooks/        Original learning notebooks. Documentation — never imported by code.
tests/
```

Rules that follow from this:

- **Hyperparameters live in `configs/*.yaml`, never as module-level globals.** The old `helper.py` had three conflicting values for `batch_size`; don't reintroduce that.
- `model.py` must stay free of file, network, and checkpoint access.
- Nothing outside `checkpoint.py` imports `orbax` or `azure.storage`.
- Nothing outside `infra/` creates cloud resources. No `az` CLI calls that mutate state from application code.
- `serve/` and `minigpt/` are a one-way dependency. Never import `serve` from `minigpt`.

## Commands

```bash
# Quality gate — run all three before committing
ruff format . && ruff check --fix .
pytest -q

# Data + training
python -m minigpt.data.prepare --config configs/<name>.yaml
python -m minigpt.train        --config configs/<name>.yaml [--resume]

# Serving locally
uvicorn serve.app:app --reload --port 8000

# Infra (from infra/)
terraform fmt -recursive && terraform validate && terraform plan
```

Never run `terraform apply` or `terraform destroy` without explicit confirmation in the conversation.

## Cost discipline (non-negotiable)

GPU compute is billed by the second against a finite credit balance.

- Every Azure ML compute cluster is declared with `min_node_count = 0` and `scale_down_after_idle_duration`.
- Never launch a multi-GPU job before the single-GPU smoke job has passed.
- Never widen a cluster's `max_node_count`, change `vm_size` to a larger SKU, or set `min_node_count > 0` without asking first.
- After any cloud work, confirm nothing is left running: `az ml compute list -o table`.

## Testing rules

- Every module in `minigpt/` has a matching `tests/test_<module>.py`.
- Model changes require a **causal-masking test**: perturbing token `t+1` must not change the logits at position `t`.
- Config changes require a **param-count assertion** against the config's expected value.
- Training changes require the **CPU smoke test** to still pass: 20 steps on `configs/dev_cpu.yaml`, loss must decrease.
- Tests run on CPU only and must finish in seconds. Never write a test that needs a GPU or the network.
- Fix the code, not the assertion. If a test is genuinely wrong, say so before changing it.

## Naming and style

- `snake_case` for functions and variables, `PascalCase` for `nnx.Module` subclasses.
- Type-hint every public function signature.
- Config field names match their YAML keys exactly.
- Array-shaped variables carry their layout: `logits_btv`, `tokens_bt` (b=batch, t=time, v=vocab).
- Terraform: resources named `<type>-minigpt-<purpose>`; every resource tagged `project = "minigpt"`.
- Comment density matches the surrounding file. Don't narrate obvious code.

## Things that are already known to be wrong

Do not reintroduce these from the old notebook code:

- Transformer blocks without an MLP sublayer or LayerNorm.
- Loss averaged over padding tokens.
- `argmax` in a sampling function that accepts a `temperature` argument.
- Deprecated `orbax.checkpoint.PyTreeCheckpointer` — use `CheckpointManager`.
- Unpinned `orbax-checkpoint`.
- Pad-every-sequence-to-maxlen data loading — pack instead.

## Deeper docs

- `README.md` — project overview, results, architecture diagram
- `docs/` — infra diagram, cost accounting, run logs
- `notebooks/` — the original from-scratch derivation
- `.plans/` — working migration plan (gitignored, local only)
