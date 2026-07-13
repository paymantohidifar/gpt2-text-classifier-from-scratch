import pytest
import torch

from gpt2_classifier.model import GPTModel
from gpt2_classifier.weights import assign, load_weights_into_gpt


def test_assign_raises_on_shape_mismatch():
    left = torch.nn.Parameter(torch.zeros(4))
    right = torch.ones(3)
    with pytest.raises(ValueError):
        assign(left, right)


def test_assign_returns_parameter_wrapping_right_values():
    left = torch.nn.Parameter(torch.zeros(3))
    right = torch.tensor([1.0, 2.0, 3.0])
    result = assign(left, right)
    assert isinstance(result, torch.nn.Parameter)
    assert torch.equal(result, right)


def _fake_hf_state_dict(cfg: dict) -> dict[str, torch.Tensor]:
    emb_dim = cfg["emb_dim"]
    n_layers = cfg["n_layers"]
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]

    state_dict = {
        "wpe.weight": torch.randn(context_length, emb_dim),
        "wte.weight": torch.randn(vocab_size, emb_dim),
        "ln_f.weight": torch.randn(emb_dim),
        "ln_f.bias": torch.randn(emb_dim),
    }
    for b in range(n_layers):
        state_dict[f"h.{b}.attn.c_attn.weight"] = torch.randn(emb_dim, 3 * emb_dim)
        state_dict[f"h.{b}.attn.c_attn.bias"] = torch.randn(3 * emb_dim)
        state_dict[f"h.{b}.attn.c_proj.weight"] = torch.randn(emb_dim, emb_dim)
        state_dict[f"h.{b}.attn.c_proj.bias"] = torch.randn(emb_dim)
        state_dict[f"h.{b}.mlp.c_fc.weight"] = torch.randn(emb_dim, 4 * emb_dim)
        state_dict[f"h.{b}.mlp.c_fc.bias"] = torch.randn(4 * emb_dim)
        state_dict[f"h.{b}.mlp.c_proj.weight"] = torch.randn(4 * emb_dim, emb_dim)
        state_dict[f"h.{b}.mlp.c_proj.bias"] = torch.randn(emb_dim)
        state_dict[f"h.{b}.ln_1.weight"] = torch.randn(emb_dim)
        state_dict[f"h.{b}.ln_1.bias"] = torch.randn(emb_dim)
        state_dict[f"h.{b}.ln_2.weight"] = torch.randn(emb_dim)
        state_dict[f"h.{b}.ln_2.bias"] = torch.randn(emb_dim)
    return state_dict


def test_load_weights_into_gpt_maps_tensors_correctly(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    fake_params = _fake_hf_state_dict(tiny_gpt_config)

    load_weights_into_gpt(model, fake_params)

    assert torch.equal(model.pos_emb.weight, fake_params["wpe.weight"])
    assert torch.equal(model.tok_emb.weight, fake_params["wte.weight"])
    assert torch.equal(model.out_head.weight, fake_params["wte.weight"])
    assert torch.equal(model.final_norm.scale, fake_params["ln_f.weight"])

    emb_dim = tiny_gpt_config["emb_dim"]
    q_w, k_w, v_w = torch.chunk(fake_params["h.0.attn.c_attn.weight"], 3, axis=-1)
    assert torch.equal(model.trf_blocks[0].att.W_query.weight, q_w.T)
    assert torch.equal(model.trf_blocks[0].att.W_key.weight, k_w.T)
    assert torch.equal(model.trf_blocks[0].att.W_value.weight, v_w.T)
    assert model.trf_blocks[0].att.W_query.weight.shape == (emb_dim, emb_dim)


def test_load_weights_into_gpt_forward_pass_runs_after_loading(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    fake_params = _fake_hf_state_dict(tiny_gpt_config)
    load_weights_into_gpt(model, fake_params)

    input_ids = torch.randint(0, tiny_gpt_config["vocab_size"], (1, 4))
    logits = model(input_ids)
    assert logits.shape == (1, 4, tiny_gpt_config["vocab_size"])
