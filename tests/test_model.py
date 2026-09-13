import flax.nnx as nnx
import jax.numpy as jnp

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT


def make_model(**overrides):
    cfg = ModelConfig(vocab_size=100, maxlen=16, embed_dim=32, num_heads=4, num_layers=2, **overrides)
    return MiniGPT(cfg, rngs=nnx.Rngs(0))


def test_output_shape():
    model = make_model()
    tokens = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (2, 16, 100)


def test_param_count_matches_config():
    model = make_model()
    # token_emb(100*32) + pos_emb(16*32) + 2 * (attn ~4*32*32 + 2*ln(64) + mlp(2*32*128)) + ln_f(64)
    # Weight-tied, so no separate output head params. Just assert it's in a sane
    # range and non-zero — an exact formula is brittle to flax internals.
    n = model.num_params()
    assert 10_000 < n < 60_000


def test_causal_mask_locality():
    """Perturbing token t+1 must not change logits at position t."""
    model = make_model()
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1]])
    tokens_perturbed = tokens.at[0, 10].set(99)

    logits = model(tokens)
    logits_perturbed = model(tokens_perturbed)

    # Positions before the perturbation must be identical.
    assert jnp.allclose(logits[0, :10], logits_perturbed[0, :10], atol=1e-5)
    # Position at/after the perturbation should generally differ.
    assert not jnp.allclose(logits[0, 10], logits_perturbed[0, 10], atol=1e-5)


def test_weight_tying():
    model = make_model(tie_embeddings=True)
    assert model.output_layer is None
    tokens = jnp.zeros((1, 16), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (1, 16, 100)


def test_untied_head_has_separate_params():
    model = make_model(tie_embeddings=False)
    assert model.output_layer is not None
