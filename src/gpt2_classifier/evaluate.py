"""Loss and accuracy computation for classification fine-tuning."""

from collections.abc import Iterator
from dataclasses import dataclass, fields

import torch
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification metrics computed over a full DataLoader pass.

    Assumes a binary classifier (2 output logits); ``precision``,
    ``roc_auc``, and ``pr_auc`` are computed for the positive class
    (index 1) using its softmax probability.

    Attributes:
        accuracy: Fraction of correctly classified examples in ``[0, 1]``.
        precision: Binary precision for the positive class.
        roc_auc: Area under the ROC curve. ``NaN`` if the evaluated subset
            contains only one class (undefined) or is empty.
        pr_auc: Area under the precision-recall curve (average precision).
            ``NaN`` under the same conditions as ``roc_auc``.
    """

    accuracy: float
    precision: float
    roc_auc: float
    pr_auc: float

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Yield (field_name, field_value) pairs for key-value unpacking."""
        for field in fields(self):
            yield field.name, getattr(self, field.name)


def calc_classification_metrics_loader(
    data_loader: DataLoader, model: torch.nn.Module, device: torch.device, num_batches: int | None = None
) -> ClassificationMetrics:
    """Compute accuracy, precision, ROC-AUC, and PR-AUC over a DataLoader.

    Args:
        data_loader: Yields ``(input_batch, target_batch)`` pairs.
        model: A classification model whose last-token logits are used.
        device: Device to run the forward pass on.
        num_batches: If given, only evaluate the first ``num_batches``
            batches instead of the whole loader.

    Returns:
        A :class:`ClassificationMetrics` with all fields ``NaN`` if
        ``data_loader`` (or the evaluated prefix) is empty.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_pos_probs: list[float] = []

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
            all_targets.extend(target_batch.cpu().tolist())
            all_preds.extend(predicted_labels.cpu().tolist())
            all_pos_probs.extend(probs[:, 1].cpu().tolist())
        else:
            break

    if num_examples == 0:
        return ClassificationMetrics(
            accuracy=float("nan"), precision=float("nan"), roc_auc=float("nan"), pr_auc=float("nan")
        )

    accuracy = correct_predictions / num_examples
    precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0.0)
    if len(set(all_targets)) < 2:
        # roc_auc/pr_auc are undefined when only one class is present.
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(all_targets, all_pos_probs)
        pr_auc = average_precision_score(all_targets, all_pos_probs)

    return ClassificationMetrics(accuracy=accuracy, precision=precision, roc_auc=roc_auc, pr_auc=pr_auc)


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
    return calc_classification_metrics_loader(data_loader, model, device, num_batches).accuracy


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
