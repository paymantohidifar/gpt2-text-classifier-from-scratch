"""Loss and accuracy computation for classification fine-tuning."""

import torch
from torch.utils.data import DataLoader


def calc_accuracy_loader(
    data_loader: DataLoader, model: torch.nn.Module, device: torch.device, num_batches: int | None = None
) -> float:
    """Compute classification accuracy over (a prefix of) a DataLoader.

    Args:
        data_loader: Yields ``(input_batch, target_batch)`` pairs.
        model: A classification model whose last-token logits are argmaxed.
        device: Device to run the forward pass on.
        num_batches: If given, only evaluate the first ``num_batches``
            batches instead of the whole loader.

    Returns:
        Fraction of correctly classified examples in ``[0, 1]``.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, device: torch.device
) -> torch.Tensor:
    """Compute cross-entropy loss for a single batch.

    Args:
        input_batch: Token id tensor, shape ``(batch, num_tokens)``.
        target_batch: Integer class labels, shape ``(batch,)``.
        model: A classification model.
        device: Device to run the forward pass on.

    Returns:
        Scalar loss tensor.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(
    data_loader: DataLoader, model: torch.nn.Module, device: torch.device, num_batches: int | None = None
) -> float:
    """Compute average cross-entropy loss over (a prefix of) a DataLoader.

    Args:
        data_loader: Yields ``(input_batch, target_batch)`` pairs.
        model: A classification model.
        device: Device to run the forward pass on.
        num_batches: If given, only evaluate the first ``num_batches``
            batches instead of the whole loader.

    Returns:
        Average loss, or ``nan`` if ``data_loader`` is empty.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Compute train/validation loss with the model temporarily in eval mode.

    Args:
        model: A classification model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Device to run the forward passes on.
        eval_iter: Number of batches to sample from each loader.

    Returns:
        A ``(train_loss, val_loss)`` tuple.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
