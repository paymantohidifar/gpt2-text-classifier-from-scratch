"""GPT-2 model configuration presets."""

from typing import TypedDict

URL_DIR: dict[str, str] = {
    "gpt2-small (124M)": "gpt2",
    "gpt2-medium (355M)": "gpt2-medium",
    "gpt2-large (774M)": "gpt2-large",
    "gpt2-xl (1558M)": "gpt2-xl",
}


class GPTConfig(TypedDict):
    """Shape of a fully-resolved GPT-2 model configuration dict."""

    vocab_size: int
    context_length: int
    drop_rate: float
    qkv_bias: bool
    emb_dim: int
    n_layers: int
    n_heads: int


BASE_CONFIG: dict[str, int | float | bool] = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_configs: dict[str, dict[str, int]] = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def get_model_config(model_name: str) -> GPTConfig:
    """Build a fresh, fully-resolved model config for the given model name.

    Args:
        model_name: A key of ``model_configs``, e.g. ``"gpt2-small (124M)"``.

    Returns:
        A new dict combining ``BASE_CONFIG`` with the model-size-specific
        overrides. Each call returns an independent dict, so callers may
        freely mutate the result without affecting ``BASE_CONFIG`` or other
        callers.

    Raises:
        KeyError: If ``model_name`` is not a recognized preset.
    """
    if model_name not in model_configs:
        raise KeyError(
            f"Unknown model_name {model_name!r}. Available options: "
            f"{sorted(model_configs)}"
        )
    return {**BASE_CONFIG, **model_configs[model_name]}  # type: ignore[typeddict-item]
