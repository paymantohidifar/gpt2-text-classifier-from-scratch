import torch
from torch.utils.data import DataLoader, TensorDataset

from gpt2_classifier.evaluate import (
    calc_accuracy_loader,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
)


def _classifier_model(tiny_gpt_model, num_classes=2):
    model = tiny_gpt_model
    model.out_head = torch.nn.Linear(model.tok_emb.embedding_dim, num_classes)
    return model


def _tiny_loader(tiny_gpt_config, num_examples=6, seq_len=4, batch_size=2):
    inputs = torch.randint(0, tiny_gpt_config["vocab_size"], (num_examples, seq_len))
    labels = torch.randint(0, 2, (num_examples,))
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)


def test_calc_loss_batch_returns_finite_scalar(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    device = torch.device("cpu")
    input_batch = torch.randint(0, tiny_gpt_config["vocab_size"], (2, 4))
    target_batch = torch.randint(0, 2, (2,))

    loss = calc_loss_batch(input_batch, target_batch, model, device)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_calc_loss_loader_averages_over_batches(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    loader = _tiny_loader(tiny_gpt_config)
    device = torch.device("cpu")

    avg_loss = calc_loss_loader(loader, model, device)

    assert isinstance(avg_loss, float)
    assert avg_loss == avg_loss  # not NaN


def test_calc_loss_loader_empty_returns_nan(tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 4, dtype=torch.long), torch.empty(0, dtype=torch.long)))

    result = calc_loss_loader(empty_loader, model, torch.device("cpu"))

    assert result != result  # NaN != NaN


def test_calc_accuracy_loader_in_valid_range(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    loader = _tiny_loader(tiny_gpt_config)

    accuracy = calc_accuracy_loader(loader, model, torch.device("cpu"))

    assert 0.0 <= accuracy <= 1.0


def test_evaluate_model_returns_train_and_val_loss(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    train_loader = _tiny_loader(tiny_gpt_config)
    val_loader = _tiny_loader(tiny_gpt_config)

    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, torch.device("cpu"), eval_iter=2)

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))
    assert model.training  # restored to train mode after evaluation
