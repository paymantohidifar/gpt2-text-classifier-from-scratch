"""Optional multi-GPU DDP training path.

This is a secondary path relative to :mod:`gpt2_classifier.train`'s
single-device loop -- it requires ``torchrun`` and CUDA, and is only
exercised when ``--ddp`` is passed to the ``train`` CLI subcommand. It is
not covered by unit tests (multi-process/CUDA cannot be meaningfully
exercised in a single-process CPU test suite); only an import/smoke check
applies.
"""

import os
import platform

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2_classifier.evaluate import calc_classification_metrics_loader, calc_loss_batch, evaluate_model
from gpt2_classifier.train import TrainingHistory


def ddp_setup(rank: int, world_size: int) -> None:
    """Initialize a distributed process group (one process per GPU).

    Args:
        rank: This process's unique id.
        world_size: Total number of processes in the group.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12345"

    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = "0"
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def train_classifier_ddp(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
) -> TrainingHistory:
    """Fine-tune ``model`` for classification across multiple GPUs via DDP.

    Requires ``train_loader`` to have been built with
    ``create_data_loaders(..., use_ddp=True)`` so its sampler is a
    ``DistributedSampler`` (this function calls
    ``train_loader.sampler.set_epoch(...)``).

    Works transparently with LoRA-adapted models: parameter freezing and
    LoRA-wrapping (see ``gpt2_classifier.train._prepare_for_classification_finetuning_with_lora``)
    happen in :func:`gpt2_classifier.train.finetune_model` before dispatch,
    so ``model`` may already have most parameters frozen by the time it
    reaches this function -- ``DDP`` only synchronizes gradients for
    parameters with ``requires_grad=True``, so frozen backbone weights are
    unaffected.

    Args:
        rank: This process's GPU/rank id.
        world_size: Total number of DDP processes (expected to be launched
            via ``torchrun``).
        model: A classification model with the head already swapped.
        train_loader: Training DataLoader with a ``DistributedSampler``.
        val_loader: Validation DataLoader.
        optimizer: Optimizer over ``model``'s trainable parameters.
        num_epochs: Number of epochs to train for.
        eval_freq: Evaluate loss every this many training steps.
        eval_iter: Number of batches to sample per evaluation.

    Returns:
        A ``(train_losses, val_losses, train_accs, val_accs, train_precisions,
        val_precisions, train_roc_aucs, val_roc_aucs, train_pr_aucs,
        val_pr_aucs, examples_seen)`` tuple recorded over the course of
        training.
    """
    ddp_setup(rank, world_size)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    train_precisions, val_precisions = [], []
    train_roc_aucs, val_roc_aucs = [], []
    train_pr_aucs, val_pr_aucs = [], []
    examples_seen, global_step = 0, -1

    model.to(rank)
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)

        model.train()

        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(rank), target_batch.to(rank)

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, rank)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, rank, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"[GPU{rank}] Epoch {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        try:
            train_metrics = calc_classification_metrics_loader(train_loader, model, rank, num_batches=eval_iter)
            val_metrics = calc_classification_metrics_loader(val_loader, model, rank, num_batches=eval_iter)
            print(f"[GPU{rank}] Training accuracy: {train_metrics.accuracy * 100:.2f}% | ", end="")
            print(f"[GPU{rank}] Validation accuracy: {val_metrics.accuracy * 100:.2f}%")
            train_accs.append(train_metrics.accuracy)
            val_accs.append(val_metrics.accuracy)
            train_precisions.append(train_metrics.precision)
            val_precisions.append(val_metrics.precision)
            train_roc_aucs.append(train_metrics.roc_auc)
            val_roc_aucs.append(val_metrics.roc_auc)
            train_pr_aucs.append(train_metrics.pr_auc)
            val_pr_aucs.append(val_metrics.pr_auc)
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"{e}\n\nThis path is designed for multi-GPU DDP training. Run it as:\n"
                "torchrun --nproc_per_node=<num_gpus> -m gpt2_classifier train --ddp ..."
            ) from e

    destroy_process_group()

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
