import torch

from gpt2_classifier.model import GPTModel


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
