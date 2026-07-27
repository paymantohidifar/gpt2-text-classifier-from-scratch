"""Generic helpers: greedy text generation, tokenization glue, and plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken
import torch

from gpt2_classifier import paths
from gpt2_classifier.model import GPTModel


def get_device() -> torch.device:
    """Return the CUDA device if available, otherwise CPU.

    Returns:
        A ``torch.device``. Code that calls this never hardcodes "cuda" or
        "cpu" directly, so the same code works unchanged on a CPU-only
        development machine and on a CUDA-enabled machine (e.g. Colab).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_text_simple(
    model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    """Greedily decode ``max_new_tokens`` continuation tokens.

    Args:
        model: A language-model-headed :class:`GPTModel` (not a
            classification model).
        idx: Starting token ids, shape ``(batch, num_tokens)``.
        max_new_tokens: Number of tokens to generate.
        context_size: Maximum context window to feed the model (older
            tokens are cropped off).

    Returns:
        Token ids of shape ``(batch, num_tokens + max_new_tokens)``.
    """
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text: str, tokenizer: "tiktoken.Encoding") -> torch.Tensor:
    """Encode ``text`` into a batched token id tensor.

    Args:
        text: Raw input text.
        tokenizer: A ``tiktoken`` encoding.

    Returns:
        Token ids of shape ``(1, num_tokens)``.
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: "tiktoken.Encoding") -> str:
    """Decode a batched token id tensor back into text.

    Args:
        token_ids: Token ids of shape ``(1, num_tokens)``.
        tokenizer: A ``tiktoken`` encoding.

    Returns:
        The decoded string.
    """
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_response(text: str | None, model: GPTModel, max_new_tokens: int = 20) -> str:
    """Generate raw GPT-2 text continuation for a prompt (debug helper).

    This performs plain language-model generation, not classification --
    useful for sanity-checking that pretrained weights loaded correctly
    (see the ``generate`` CLI subcommand).

    Args:
        text: Prompt text. Defaults to a placeholder if ``None``.
        model: A language-model-headed :class:`GPTModel`.
        max_new_tokens: Number of tokens to generate.

    Returns:
        The prompt followed by the generated continuation.
    """
    tokenizer = tiktoken.get_encoding("gpt2")

    if text is None:
        text = "Every effort moves"

    context_size = model.pos_emb.weight.shape[0]
    input_ids = text_to_token_ids(text, tokenizer)
    response_ids = generate_text_simple(model, input_ids, max_new_tokens, context_size)
    response = token_ids_to_text(response_ids, tokenizer)

    return response


def _draw_metric(
    ax: plt.Axes,
    epochs_seen: torch.Tensor,
    examples_seen: torch.Tensor,
    train_values: list[float],
    val_values: list[float],
    label: str,
) -> None:
    """Draw train/validation curves for one metric onto ``ax``.

    Args:
        ax: Axes to draw onto.
        epochs_seen: X-axis values (epochs) for the primary axis.
        examples_seen: X-axis values (examples seen) for the secondary axis.
        train_values: Training metric values.
        val_values: Validation metric values.
        label: Metric name, used in the legend and axis label.
    """
    ax.plot(epochs_seen, train_values, label=f"Training {label}")
    ax.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel(label.capitalize())
    ax.legend()

    ax_top = ax.twiny()
    ax_top.plot(examples_seen, train_values, alpha=0)
    ax_top.set_xlabel("Examples seen")


def plot_values(
    epochs_seen: torch.Tensor,
    examples_seen: torch.Tensor,
    train_values: list[float],
    val_values: list[float],
    label: str = "loss",
    output_dir: Path = paths.MODELS_DIR / "metric-plots",
) -> Path:
    """Plot train/validation curves against both epochs and examples seen.

    Args:
        epochs_seen: X-axis values (epochs) for the primary axis.
        examples_seen: X-axis values (examples seen) for the secondary axis.
        train_values: Training metric values.
        val_values: Validation metric values.
        label: Metric name, used in the legend and output filename.
        output_dir: Directory to save the resulting PDF into.

    Returns:
        Path to the saved PDF.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    _draw_metric(ax, epochs_seen, examples_seen, train_values, val_values, label)

    fig.tight_layout()
    plt.show()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}-plot.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_results(
    num_epochs: int,
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    train_precisions: list[float],
    val_precisions: list[float],
    train_roc_aucs: list[float],
    val_roc_aucs: list[float],
    train_pr_aucs: list[float],
    val_pr_aucs: list[float],
    examples_seen: int,
    output_dir: Path = paths.PLOTS_DIR / "training-plots",
) -> Path:
    """Plot loss, accuracy, precision, ROC-AUC, and PR-AUC curves for a run.

    All five metrics are drawn as subplots of one combined figure, shown
    inline (e.g. in a Jupyter notebook) and saved once to disk.

    Args:
        num_epochs: Total number of epochs trained.
        train_losses: Training loss values recorded during training.
        val_losses: Validation loss values recorded during training.
        train_accs: Training accuracy values recorded per epoch.
        val_accs: Validation accuracy values recorded per epoch.
        train_precisions: Training precision values recorded per epoch.
        val_precisions: Validation precision values recorded per epoch.
        train_roc_aucs: Training ROC-AUC values recorded per epoch.
        val_roc_aucs: Validation ROC-AUC values recorded per epoch.
        train_pr_aucs: Training PR-AUC values recorded per epoch.
        val_pr_aucs: Validation PR-AUC values recorded per epoch.
        examples_seen: Total number of training examples seen.
        output_dir: Directory to save the resulting PDF into.

    Returns:
        Path to the saved combined-metrics PDF.
    """
    metrics = [
        ("loss", train_losses, val_losses),
        ("accuracy", train_accs, val_accs),
        ("precision", train_precisions, val_precisions),
        ("roc_auc", train_roc_aucs, val_roc_aucs),
        ("pr_auc", train_pr_aucs, val_pr_aucs),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, (label, train_values, val_values) in zip(axes, metrics):
        epochs_tensor = torch.linspace(0, num_epochs, len(train_values))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_values))
        _draw_metric(ax, epochs_tensor, examples_seen_tensor, train_values, val_values, label)
    axes[-1].axis("off")

    fig.tight_layout()
    plt.show()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.pdf"
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
