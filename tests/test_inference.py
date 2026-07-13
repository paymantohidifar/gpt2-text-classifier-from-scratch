import tiktoken
import torch

from gpt2_classifier.inference import classify_text, load_classifier, run_prediction
from gpt2_classifier.model import GPTModel


def _save_tiny_checkpoint(tmp_path, config, num_classes=2, label_names=None):
    torch.manual_seed(0)
    model = GPTModel(config)
    model.out_head = torch.nn.Linear(config["emb_dim"], num_classes)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "num_classes": num_classes,
            "label_names": label_names or {0: "ham", 1: "spam"},
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_load_classifier_reconstructs_model_and_metadata(tmp_path, tiny_gpt_config_real_vocab):
    checkpoint_path = _save_tiny_checkpoint(tmp_path, tiny_gpt_config_real_vocab)

    model, model_config, label_names = load_classifier(checkpoint_path, device=torch.device("cpu"))

    assert isinstance(model, GPTModel)
    assert model_config == tiny_gpt_config_real_vocab
    assert label_names == {0: "ham", 1: "spam"}
    assert not model.training  # eval mode
    assert model.out_head.out_features == 2


def test_classify_text_returns_a_known_label(tiny_gpt_model_real_vocab, tiny_gpt_config_real_vocab):
    tiny_gpt_model_real_vocab.out_head = torch.nn.Linear(tiny_gpt_config_real_vocab["emb_dim"], 2)
    tokenizer = tiktoken.get_encoding("gpt2")

    label = classify_text(
        "this is a test message",
        tiny_gpt_model_real_vocab,
        tokenizer,
        device=torch.device("cpu"),
        max_length=tiny_gpt_config_real_vocab["context_length"],
    )

    assert label in {"ham", "spam"}


def test_classify_text_handles_short_and_long_text(tiny_gpt_model_real_vocab, tiny_gpt_config_real_vocab):
    tiny_gpt_model_real_vocab.out_head = torch.nn.Linear(tiny_gpt_config_real_vocab["emb_dim"], 2)
    tokenizer = tiktoken.get_encoding("gpt2")
    max_length = tiny_gpt_config_real_vocab["context_length"]

    short_label = classify_text("hi", tiny_gpt_model_real_vocab, tokenizer, torch.device("cpu"), max_length)
    long_text = "word " * 200
    long_label = classify_text(long_text, tiny_gpt_model_real_vocab, tokenizer, torch.device("cpu"), max_length)

    assert short_label in {"ham", "spam"}
    assert long_label in {"ham", "spam"}


def test_classify_text_uses_custom_label_names(tiny_gpt_model_real_vocab, tiny_gpt_config_real_vocab):
    tiny_gpt_model_real_vocab.out_head = torch.nn.Linear(tiny_gpt_config_real_vocab["emb_dim"], 2)
    tokenizer = tiktoken.get_encoding("gpt2")

    label = classify_text(
        "custom labels test",
        tiny_gpt_model_real_vocab,
        tokenizer,
        torch.device("cpu"),
        tiny_gpt_config_real_vocab["context_length"],
        label_names={0: "not_phishing", 1: "phishing"},
    )

    assert label in {"not_phishing", "phishing"}


def test_run_prediction_end_to_end(tmp_path, tiny_gpt_config_real_vocab):
    checkpoint_path = _save_tiny_checkpoint(tmp_path, tiny_gpt_config_real_vocab)

    label = run_prediction(checkpoint_path, "free money now!!!")

    assert label in {"ham", "spam"}
