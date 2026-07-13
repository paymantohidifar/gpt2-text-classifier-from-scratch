import pandas as pd
import pytest
import torch

from gpt2_classifier.model import GPTModel


@pytest.fixture
def tiny_gpt_config() -> dict:
    """A tiny GPT config for fast unit tests (not a real GPT-2 size)."""
    return {
        "vocab_size": 100,
        "context_length": 16,
        "emb_dim": 8,
        "n_layers": 2,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


@pytest.fixture
def tiny_gpt_model(tiny_gpt_config) -> GPTModel:
    torch.manual_seed(0)
    return GPTModel(tiny_gpt_config)


@pytest.fixture
def tiny_gpt_config_real_vocab() -> dict:
    """Tiny GPT dims but the real GPT-2 vocab size, for tests that tokenize
    with the real tiktoken "gpt2" encoding (whose token ids go up to 50256)."""
    return {
        "vocab_size": 50257,
        "context_length": 32,
        "emb_dim": 8,
        "n_layers": 1,
        "n_heads": 2,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


@pytest.fixture
def tiny_gpt_model_real_vocab(tiny_gpt_config_real_vocab) -> GPTModel:
    torch.manual_seed(0)
    return GPTModel(tiny_gpt_config_real_vocab)


@pytest.fixture
def synthetic_labeled_dataframe() -> pd.DataFrame:
    """A small imbalanced text/label DataFrame for dataset-pipeline tests."""
    return pd.DataFrame(
        {
            "Text": [f"ham message {i}" for i in range(8)]
            + [f"spam message {i}" for i in range(2)],
            "Label": ["ham"] * 8 + ["spam"] * 2,
        }
    )
