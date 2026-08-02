"""Single-device classifier fine-tuning (the default training path).

Multi-GPU DDP training is available as an optional, secondary path in
:mod:`gpt2_classifier.train_ddp` -- ``finetune_model`` dispatches to it only
when explicitly requested (``use_ddp=True``), since it requires
``torchrun``/CUDA and is comparatively hard to exercise/verify outside a
multi-GPU machine.
"""

import time
from typing import Any

import torch

from gpt2_classifier import paths
from gpt2_classifier.evaluate import calc_classification_metrics_loader, calc_loss_batch, evaluate_model
from gpt2_classifier.logging_utils import RunLogger
from gpt2_classifier.model import GPTModel
from gpt2_classifier.utils import get_device


TrainingHistory = tuple[
    list[float],  # train_losses
    list[float],  # val_losses
    list[float],  # train_accs
    list[float],  # val_accs
    list[float],  # train_precisions
    list[float],  # val_precisions
    list[float],  # train_roc_aucs
    list[float],  # val_roc_aucs
    list[float],  # train_pr_aucs
    list[float],  # val_pr_aucs
    int,  # examples_seen
]


def train_classifier_simple(
    model: GPTModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    logger: RunLogger | None = None,
) -> TrainingHistory:
    """Fine-tune ``model`` for classification on a single device.

    Args:
        model: A classification model (LM head already swapped for a linear
            classification head).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer over ``model``'s trainable parameters.
        device: Device to train on.
        num_epochs: Number of epochs to train for.
        eval_freq: Evaluate loss every this many training steps.
        eval_iter: Number of batches to sample per evaluation.
        logger: Optional :class:`RunLogger` for wandb metric logging; a
            disabled/``None`` logger is a safe no-op.

    Returns:
        A ``(train_losses, val_losses, train_accs, val_accs, train_precisions,
        val_precisions, train_roc_aucs, val_roc_aucs, train_pr_aucs,
        val_pr_aucs, examples_seen)`` tuple recorded over the course of
        training.
    """
    if logger is None:
        logger = RunLogger(enabled=False)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    train_precisions, val_precisions = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_pr_aucs, val_pr_aucs = [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                logger.log(
                    {"train_loss": train_loss, "val_loss": val_loss, "examples_seen": examples_seen},
                    step=global_step,
                )
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        train_metrics = calc_classification_metrics_loader(train_loader, model, device, num_batches=eval_iter)
        val_metrics = calc_classification_metrics_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_metrics.accuracy * 100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_metrics.accuracy * 100:.2f}%")
        print(
            f"Train precision: {train_metrics.precision:.3f}, ROC-AUC: {train_metrics.roc_auc:.3f}, "
            f"PR-AUC: {train_metrics.pr_auc:.3f} | "
            f"Val precision: {val_metrics.precision:.3f}, ROC-AUC: {val_metrics.roc_auc:.3f}, "
            f"PR-AUC: {val_metrics.pr_auc:.3f}"
        )
        train_accs.append(train_metrics.accuracy)
        val_accs.append(val_metrics.accuracy)
        train_precisions.append(train_metrics.precision)
        val_precisions.append(val_metrics.precision)
        train_roc_aucs.append(train_metrics.roc_auc)
        val_roc_aucs.append(val_metrics.roc_auc)
        train_pr_aucs.append(train_metrics.pr_auc)
        val_pr_aucs.append(val_metrics.pr_auc)
        logger.log(
            {
                "train_accuracy": train_metrics.accuracy,
                "val_accuracy": val_metrics.accuracy,
                "train_precision": train_metrics.precision,
                "val_precision": val_metrics.precision,
                "train_roc_auc": train_metrics.roc_auc,
                "val_roc_auc": val_metrics.roc_auc,
                "train_pr_auc": train_metrics.pr_auc,
                "val_pr_auc": val_metrics.pr_auc,
                "epoch": epoch + 1,
            }
        )

    logger.finish()
    return (
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        train_precisions,
        val_precisions,
        train_roc_aucs,
        val_roc_aucs,
        train_pr_aucs,
        val_pr_aucs,
        examples_seen,
    )


def _prepare_for_classification_finetuning(
    model: GPTModel, model_config: dict[str, Any], num_classes: int, device: torch.device
) -> None:
    """Freeze the backbone, swap in a classification head, and unfreeze the top layers.

    Args:
        model: A language-model-headed :class:`GPTModel`.
        model_config: The model's config dict (needs ``emb_dim``).
        num_classes: Number of output classes for the new head.
        device: Device to move the model to.
    """
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=model_config["emb_dim"], out_features=num_classes)
    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True


def get_adam_param_groups(model: GPTModel, weight_decay: float = 0.1):
    """
    Filters trainable parameters and splits them into weight-decay 
    and no-weight-decay groups (excluding 1D tensors like biases/LayerNorm).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        # Skip non-trainable parameters completely
        if not param.requires_grad:
            continue

        # Separate 1D parameters (biases, norms) from 2D+ weight matrices
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim_groups


def finetune_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: GPTModel,
    model_config: dict[str, Any],
    num_classes: int = 2,
    lr: float = 5e-5,
    weight_decay: float = 0.1,
    optimize_adamw: bool = True,
    num_epochs: int = 5,
    eval_freq: int = 50,
    eval_iter: int = 5,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
    logger: RunLogger | None = None,
    checkpoint_name: str = "spam_classifier.pt",
    label_names: dict[int, str] | None = None,
) -> TrainingHistory:
    """Freeze the GPT-2 backbone, swap in a classifier head, and fine-tune it.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        model: A pretrained :class:`GPTModel` (LM head still intact).
        model_config: The model's config dict.
        num_classes: Number of output classes.
        lr: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        optimize_adamw: Exclude weight decay from 1D tensors
        num_epochs: Number of epochs to train for.
        eval_freq: Evaluate loss every this many training steps.
        eval_iter: Number of batches to sample per evaluation.
        use_ddp: If ``True``, dispatch to the multi-GPU DDP training loop in
            :mod:`gpt2_classifier.train_ddp` instead of the single-device
            loop. Requires ``torchrun`` and CUDA.
        rank: Process rank (only meaningful when ``use_ddp=True``).
        world_size: Number of DDP processes (only meaningful when
            ``use_ddp=True``).
        logger: Optional :class:`RunLogger` for wandb metric logging.
        checkpoint_name: Name of fine-tuned model's state dict
            (plus its config, for :mod:`gpt2_classifier.inference` to
            reconstruct the model). Pass ``None`` to skip saving.
        label_names: Mapping from integer class id to a human-readable label
            (e.g. ``{0: "ham", 1: "spam"}``), saved into the checkpoint so
            :mod:`gpt2_classifier.inference` can report class names rather
            than raw ids -- not hardcoded, since a dataset's labels aren't
            necessarily "ham"/"spam". Defaults to ``{0: "ham", 1: "spam"}``
            when not given.

    Returns:
        A ``(train_losses, val_losses, train_accs, val_accs, train_precisions,
        val_precisions, train_roc_aucs, val_roc_aucs, train_pr_aucs,
        val_pr_aucs, examples_seen)`` tuple recorded over the course of
        training.
    """
    device = get_device()
    _prepare_for_classification_finetuning(model, model_config, num_classes, device)

    torch.manual_seed(123)
    if optimize_adamw:
        optim_groups = get_adam_param_groups(model, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=lr)
        print("Excluded weight decay effect on 1D tensors (e.g. LayerNorm/biases) in AdamW optimizer.")
    else:
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay
        )

    start_time = time.time()
    if use_ddp:
        from gpt2_classifier.train_ddp import train_classifier_ddp

        history = train_classifier_ddp(
            rank, world_size, model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter
        )
    else:
        history = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, logger=logger
        )
    execution_time_minutes = (time.time() - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    if checkpoint_name is not None:
        checkpoint_path = paths.MODELS_DIR / checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "num_classes": num_classes,
                "label_names": label_names or {0: "ham", 1: "spam"},
            },
            checkpoint_path,
        )
        print(f"Saved fine-tuned checkpoint to {checkpoint_path}")

    return history
