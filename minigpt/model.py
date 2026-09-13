"""MiniGPT model definition. Pure JAX/Flax NNX — no I/O, no checkpointing,
no logging. Fixes vs. the original helper.py: adds the MLP sublayer,
pre-LayerNorm, weight tying, and a proper boolean causal mask."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from minigpt.config import ModelConfig


class MLP(nnx.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float, *, rngs: nnx.Rngs):
        hidden = int(embed_dim * mlp_ratio)
        self.fc_in = nnx.Linear(embed_dim, hidden, rngs=rngs)
        self.fc_out = nnx.Linear(hidden, embed_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, *, deterministic: bool):
        x = self.fc_in(x)
        x = nnx.gelu(x)
        x = self.fc_out(x)
        return self.dropout(x, deterministic=deterministic)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float, *, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            decode=False,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout, rngs=rngs)

    def __call__(self, x, mask=None, *, deterministic: bool = True):
        x = x + self.attention(self.ln1(x), mask=mask, deterministic=deterministic)
        x = x + self.mlp(self.ln2(x), deterministic=deterministic)
        return x


class TokenAndPositionEmbedding(nnx.Module):
    def __init__(self, maxlen: int, vocab_size: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.token_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.pos_emb = nnx.Embed(maxlen, embed_dim, rngs=rngs)

    def __call__(self, x):
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class MiniGPT(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.maxlen = cfg.maxlen
        self.embedding = TokenAndPositionEmbedding(cfg.maxlen, cfg.vocab_size, cfg.embed_dim, rngs=rngs)
        self.blocks = [
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, rngs=rngs)
            for _ in range(cfg.num_layers)
        ]
        self.ln_f = nnx.LayerNorm(cfg.embed_dim, rngs=rngs)
        if not cfg.tie_embeddings:
            self.output_layer = nnx.Linear(cfg.embed_dim, cfg.vocab_size, use_bias=False, rngs=rngs)
        else:
            self.output_layer = None

    def __call__(self, token_ids, *, deterministic: bool = True):
        mask = nnx.make_causal_mask(token_ids)

        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, mask=mask, deterministic=deterministic)
        x = self.ln_f(x)

        if self.cfg.tie_embeddings:
            # Weight-tied output head: reuse the token embedding matrix, transposed.
            logits = x @ self.embedding.token_emb.embedding.value.T
        else:
            logits = self.output_layer(x)
        return logits

    def num_params(self) -> int:
        state = nnx.state(self, nnx.Param)
        leaves = [leaf for leaf in state.flat_state()]
        return sum(v.value.size for _, v in leaves)
