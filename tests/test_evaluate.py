import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from gpt2_classifier.evaluate import (
    calc_accuracy_loader,
    calc_classification_metrics_loader,
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
    assert not math.isnan(avg_loss)


def test_calc_loss_loader_empty_returns_nan(tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 4, dtype=torch.long), torch.empty(0, dtype=torch.long)))

    result = calc_loss_loader(empty_loader, model, torch.device("cpu"))

    assert math.isnan(result)


def test_calc_accuracy_loader_in_valid_range(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    loader = _tiny_loader(tiny_gpt_config)

    accuracy = calc_accuracy_loader(loader, model, torch.device("cpu"))

    assert 0.0 <= accuracy <= 1.0


def test_calc_classification_metrics_loader_returns_all_fields(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    loader = _tiny_loader(tiny_gpt_config)

    metrics = calc_classification_metrics_loader(loader, model, torch.device("cpu"))

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    # roc_auc/pr_auc may be NaN if the small random loader has only one
    # class present in the evaluated targets; otherwise must be in [0, 1].
    assert metrics.roc_auc != metrics.roc_auc or 0.0 <= metrics.roc_auc <= 1.0
    assert metrics.pr_auc != metrics.pr_auc or 0.0 <= metrics.pr_auc <= 1.0


def test_calc_classification_metrics_loader_single_class_returns_nan_auc(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    inputs = torch.randint(0, tiny_gpt_config["vocab_size"], (6, 4))
    labels = torch.zeros(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    metrics = calc_classification_metrics_loader(loader, model, torch.device("cpu"))

    assert metrics.roc_auc != metrics.roc_auc  # NaN
    assert metrics.pr_auc != metrics.pr_auc  # NaN


def test_evaluate_model_returns_train_and_val_loss(tiny_gpt_config, tiny_gpt_model):
    model = _classifier_model(tiny_gpt_model)
    train_loader = _tiny_loader(tiny_gpt_config)
    val_loader = _tiny_loader(tiny_gpt_config)

    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, torch.device("cpu"), eval_iter=2)

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))
    assert model.training  # restored to train mode after evaluation
