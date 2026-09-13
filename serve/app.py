"""FastAPI + Gradio inference app. Imports minigpt; must never be imported
by minigpt itself (one-way dependency, see CLAUDE.md)."""

from __future__ import annotations

import os
from pathlib import Path

import flax.nnx as nnx
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from minigpt.checkpoint import make_manager, restore_latest
from minigpt.config import Config
from minigpt.model import MiniGPT
from minigpt.sample import generate_story

CONFIG_PATH = os.environ.get("MINIGPT_CONFIG", "configs/gpt2_small_tinystories.yaml")
CKPT_DIR = os.environ.get("MINIGPT_CKPT_DIR")  # e.g. mounted Blob path or local runs/<name>/checkpoints

_cfg = Config.from_yaml(CONFIG_PATH)
_model = MiniGPT(_cfg.model, rngs=nnx.Rngs(0))

if CKPT_DIR and Path(CKPT_DIR).exists():
    manager = make_manager(CKPT_DIR)
    _, _model, _ = restore_latest(manager, _model, None)
    print(f"Loaded checkpoint from {CKPT_DIR}")
else:
    print("WARNING: no checkpoint loaded — serving randomly initialized weights")

app = FastAPI(title="MiniGPT")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_p: float = 0.95


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate")
def generate_endpoint(req: GenerateRequest):
    text = generate_story(
        _model,
        req.prompt,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
        top_p=req.top_p,
    )
    return {"text": text}


def _gradio_generate(prompt: str, temperature: float, max_new_tokens: int):
    return generate_story(_model, prompt, temperature=temperature, max_new_tokens=int(max_new_tokens))


demo = gr.Interface(
    fn=_gradio_generate,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time a big bear"),
        gr.Slider(0.0, 1.5, value=0.8, label="Temperature"),
        gr.Slider(1, 300, value=100, step=1, label="Max new tokens"),
    ],
    outputs=gr.Textbox(label="Generated story"),
    title="MiniGPT",
)
app = gr.mount_gradio_app(app, demo, path="/")
