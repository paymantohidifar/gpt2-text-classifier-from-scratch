import torch
from torch import nn

from gpt2_classifier.model import GPTModel, LinearWithLoRA, LoRALayer


def test_gptmodel_forward_pass_shape(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, tiny_gpt_config["vocab_size"], (batch_size, seq_len))

    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, tiny_gpt_config["vocab_size"])


def test_gptmodel_forward_pass_is_deterministic_in_eval_mode(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    model.eval()
    input_ids = torch.randint(0, tiny_gpt_config["vocab_size"], (1, 4))

    with torch.no_grad():
        logits1 = model(input_ids)
        logits2 = model(input_ids)

    assert torch.allclose(logits1, logits2)


def test_gptmodel_respects_replaced_out_head(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    num_classes = 2
    model.out_head = torch.nn.Linear(tiny_gpt_config["emb_dim"], num_classes)

    input_ids = torch.randint(0, tiny_gpt_config["vocab_size"], (1, 3))
    logits = model(input_ids)

    assert logits.shape == (1, 3, num_classes)


def test_loralayer_output_shape_and_zero_init():
    layer = LoRALayer(in_dim=6, out_dim=4, rank=2, alpha=8)
    x = torch.randn(3, 6)

    out = layer(x)

    assert out.shape == (3, 4)
    # B is zero-initialized, so the LoRA delta starts at exactly zero.
    assert torch.equal(out, torch.zeros_like(out))


def test_loralayer_scales_by_alpha_over_rank():
    layer = LoRALayer(in_dim=4, out_dim=4, rank=2, alpha=8)
    layer.B.data = torch.randn(2, 4)
    x = torch.randn(2, 4)

    expected = (layer.alpha / layer.rank) * (x @ layer.A @ layer.B)

    assert torch.allclose(layer(x), expected)


def test_linear_with_lora_matches_base_linear_before_training():
    linear = nn.Linear(6, 4)
    wrapped = LinearWithLoRA(linear, rank=2, alpha=4)
    x = torch.randn(3, 6)

    # LoRA's B is zero-initialized, so the wrapped layer starts identical to
    # the base linear layer it wraps.
    assert torch.allclose(wrapped(x), linear(x))


def test_linear_with_lora_adds_trainable_params_on_top_of_frozen_base():
    linear = nn.Linear(6, 4)
    for param in linear.parameters():
        param.requires_grad = False
    wrapped = LinearWithLoRA(linear, rank=2, alpha=4)

    assert not wrapped.linear.weight.requires_grad
    assert wrapped.lora.A.requires_grad
    assert wrapped.lora.B.requires_grad
