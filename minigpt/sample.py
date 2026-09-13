"""Text generation / decoding. Temperature, top-k, top-p sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tiktoken

from minigpt.model import MiniGPT

_tokenizer = tiktoken.get_encoding("gpt2")
_EOS_ID = _tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]


def _top_k_top_p_filter(logits: jnp.ndarray, top_k: int, top_p: float) -> jnp.ndarray:
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        kth_value = jnp.sort(logits)[-top_k]
        logits = jnp.where(logits < kth_value, -jnp.inf, logits)
    if 0.0 < top_p < 1.0:
        sorted_logits = jnp.sort(logits)[::-1]
        sorted_probs = jax.nn.softmax(sorted_logits)
        cum_probs = jnp.cumsum(sorted_probs)
        cutoff_idx = jnp.searchsorted(cum_probs, top_p)
        cutoff_value = sorted_logits[cutoff_idx]
        logits = jnp.where(logits < cutoff_value, -jnp.inf, logits)
    return logits


def generate(
    model: MiniGPT,
    start_tokens: list[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
) -> str:
    """Greedy if temperature == 0, else stochastic sampling via categorical draw
    (NOT argmax — argmax under a temperature parameter was a bug in the original
    helper.generate_text, which silently made `temperature` a dead argument)."""
    tokens = list(start_tokens)
    key = jax.random.PRNGKey(seed)

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen :]
        actual_len = len(context)
        padded = context + [0] * (model.maxlen - actual_len)
        context_array = jnp.array(padded)[None, :]

        logits = model(context_array, deterministic=True)
        next_logits = logits[0, actual_len - 1, :]

        if temperature <= 0:
            next_token = int(jnp.argmax(next_logits))
        else:
            next_logits = next_logits / temperature
            next_logits = _top_k_top_p_filter(next_logits, top_k, top_p)
            key, subkey = jax.random.split(key)
            next_token = int(jax.random.categorical(subkey, next_logits))

        if next_token == _EOS_ID:
            break
        tokens.append(next_token)

    return _tokenizer.decode(tokens)


def generate_story(
    model: MiniGPT,
    prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 100,
    top_k: int = 0,
    top_p: float = 0.95,
    seed: int = 0,
) -> str:
    start_tokens = _tokenizer.encode(prompt)[: model.maxlen]
    return generate(
        model,
        start_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )
