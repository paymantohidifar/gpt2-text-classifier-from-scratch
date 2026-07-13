import pytest

from gpt2_classifier.config import BASE_CONFIG, get_model_config, model_configs


def test_get_model_config_merges_base_and_model_specific():
    cfg = get_model_config("gpt2-small (124M)")
    assert cfg["emb_dim"] == 768
    assert cfg["n_layers"] == 12
    assert cfg["n_heads"] == 12
    assert cfg["vocab_size"] == BASE_CONFIG["vocab_size"]
    assert cfg["context_length"] == BASE_CONFIG["context_length"]


def test_get_model_config_returns_fresh_dict_each_call():
    cfg1 = get_model_config("gpt2-small (124M)")
    cfg1["emb_dim"] = -1
    cfg2 = get_model_config("gpt2-small (124M)")
    assert cfg2["emb_dim"] == 768


def test_get_model_config_does_not_mutate_base_config():
    original_base_config = dict(BASE_CONFIG)
    get_model_config("gpt2-medium (355M)")["n_layers"] = 999
    assert BASE_CONFIG == original_base_config


def test_get_model_config_unknown_name_raises():
    with pytest.raises(KeyError):
        get_model_config("not-a-real-model")


def test_all_model_configs_produce_valid_config():
    for name in model_configs:
        cfg = get_model_config(name)
        assert cfg["emb_dim"] > 0
        assert cfg["n_layers"] > 0
        assert cfg["n_heads"] > 0
