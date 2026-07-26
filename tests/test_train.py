import torch
from torch.utils.data import DataLoader, TensorDataset

from gpt2_classifier.model import GPTModel
from gpt2_classifier.train import _prepare_for_classification_finetuning, finetune_model, get_device


def _tiny_loader(vocab_size, num_examples=6, seq_len=4, batch_size=2):
    inputs = torch.randint(0, vocab_size, (num_examples, seq_len))
    labels = torch.randint(0, 2, (num_examples,))
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, drop_last=True)


def test_get_device_returns_cpu_when_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")


def test_prepare_for_classification_finetuning_freezes_all_but_last_block(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    device = torch.device("cpu")

    _prepare_for_classification_finetuning(model, tiny_gpt_config, num_classes=2, device=device)

    assert isinstance(model.out_head, torch.nn.Linear)
    assert model.out_head.out_features == 2
    assert all(p.requires_grad for p in model.trf_blocks[-1].parameters())
    assert all(p.requires_grad for p in model.final_norm.parameters())
    for block in model.trf_blocks[:-1]:
        assert all(not p.requires_grad for p in block.parameters())
    assert all(not p.requires_grad for p in model.tok_emb.parameters())
    assert all(not p.requires_grad for p in model.pos_emb.parameters())


def test_finetune_model_runs_and_saves_checkpoint(tiny_gpt_config, tmp_path):
    model = GPTModel(tiny_gpt_config)
    train_loader = _tiny_loader(tiny_gpt_config["vocab_size"])
    val_loader = _tiny_loader(tiny_gpt_config["vocab_size"])
    checkpoint_path = tmp_path / "classifier.pt"

    train_losses, val_losses, train_accs, val_accs, examples_seen = finetune_model(
        train_loader,
        val_loader,
        model,
        tiny_gpt_config,
        num_classes=2,
        num_epochs=1,
        eval_freq=1,
        eval_iter=1,
        plot_metrics=False,
        checkpoint_path=checkpoint_path,
    )

    assert len(train_accs) == 1
    assert len(val_accs) == 1
    assert len(train_losses) == len(val_losses)
    assert examples_seen > 0
    assert checkpoint_path.exists()

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint["num_classes"] == 2
    assert checkpoint["model_config"] == tiny_gpt_config
    assert "model_state_dict" in checkpoint


def test_finetune_model_skips_checkpoint_when_path_is_none(tiny_gpt_config):
    model = GPTModel(tiny_gpt_config)
    train_loader = _tiny_loader(tiny_gpt_config["vocab_size"])
    val_loader = _tiny_loader(tiny_gpt_config["vocab_size"])

    finetune_model(
        train_loader,
        val_loader,
        model,
        tiny_gpt_config,
        num_epochs=1,
        eval_freq=1,
        eval_iter=1,
        plot_metrics=False,
        checkpoint_path=None,
    )
    # No assertion needed beyond "did not raise" -- confirms the None path is safe.
