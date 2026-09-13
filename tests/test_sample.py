import flax.nnx as nnx

from minigpt.config import ModelConfig
from minigpt.model import MiniGPT
from minigpt.sample import generate


def make_tiny_model():
    cfg = ModelConfig(vocab_size=300, maxlen=16, embed_dim=32, num_heads=4, num_layers=2)
    return MiniGPT(cfg, rngs=nnx.Rngs(0))


def test_sampling_is_deterministic_under_fixed_seed():
    model = make_tiny_model()
    out1 = generate(model, [1, 2, 3], max_new_tokens=5, temperature=0.8, seed=42)
    out2 = generate(model, [1, 2, 3], max_new_tokens=5, temperature=0.8, seed=42)
    assert out1 == out2


def test_sampling_reacts_to_temperature():
    """Regression test for the original helper.generate_text bug: it called
    jnp.argmax on temperature-scaled logits, which is invariant to temperature
    for temperature > 0 -- making the argument dead code."""
    model = make_tiny_model()
    outputs = {generate(model, [1, 2, 3], max_new_tokens=8, temperature=1.5, seed=s) for s in range(5)}
    # Real sampling at a non-trivial temperature should not collapse every
    # seed onto an identical greedy output.
    assert len(outputs) > 1


def test_greedy_at_zero_temperature_is_deterministic_across_seeds():
    model = make_tiny_model()
    out1 = generate(model, [1, 2, 3], max_new_tokens=5, temperature=0.0, seed=1)
    out2 = generate(model, [1, 2, 3], max_new_tokens=5, temperature=0.0, seed=2)
    assert out1 == out2
