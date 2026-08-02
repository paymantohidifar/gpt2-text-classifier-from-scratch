"""Run a fine-tuned classifier checkpoint on arbitrary user-provided text."""

from pathlib import Path
from typing import Any

import tiktoken
import torch

from gpt2_classifier.model import GPTModel


def load_classifier(
    checkpoint_path: str | Path, device: torch.device | None = None
) -> tuple[GPTModel, dict[str, Any], dict[int, str]]:
    """Reconstruct a fine-tuned classification model from a checkpoint.

    Args:
        checkpoint_path: Path to a checkpoint saved by
            :func:`gpt2_classifier.train.finetune_model`.
        device: Device to load the model onto. Defaults to CUDA if
            available, else CPU.

    Returns:
        A ``(model, model_config, label_names)`` tuple, where ``model`` is
        in eval mode and already moved to ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]   
    num_classes = checkpoint["num_classes"]
    label_names = checkpoint.get("label_names", {0: "ham", 1: "spam"})

    model = GPTModel(model_config)
    model.out_head = torch.nn.Linear(model_config["emb_dim"], num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, model_config, label_names


def classify_text(
    text: str,
    model: GPTModel,
    tokenizer: "tiktoken.Encoding",
    device: torch.device,
    max_length: int,
    label_names: dict[int, str] | None = None,
    pad_token_id: int = 50256,
) -> str:
    """Classify a single piece of text with a fine-tuned model.

    Tokenizes/pads/truncates ``text`` the same way
    :class:`gpt2_classifier.data.TextClassificationDataset` does at training
    time, then argmaxes the last-token logits.

    Args:
        text: Raw input text to classify.
        model: A fine-tuned classification model (from
            :func:`load_classifier`).
        tokenizer: A ``tiktoken`` encoding.
        device: Device to run the forward pass on.
        max_length: Sequence length the model was trained with (pad/truncate
            to this length).
        label_names: Mapping from class id to label string. Defaults to
            ``{0: "ham", 1: "spam"}`` when not given.
        pad_token_id: Token id used for right-padding.

    Returns:
        The predicted label string.
    """
    if label_names is None:
        label_names = {0: "ham", 1: "spam"}

    encoded = tokenizer.encode(text)[:max_length]
    encoded = encoded + [pad_token_id] * (max_length - len(encoded))
    input_ids = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)[:, -1, :]
    predicted_id = int(torch.argmax(logits, dim=-1).item())

    return label_names.get(predicted_id, str(predicted_id))


def run_prediction(
    checkpoint_path: str | Path, text: str, max_length: int | None = None
) -> str:
    """End-to-end helper: load a checkpoint and classify one piece of text.

    Args:
        checkpoint_path: Path to a fine-tuned checkpoint.
        text: Raw input text to classify.
        max_length: Sequence length to pad/truncate to. Defaults to the
            model's ``context_length`` if not given.

    Returns:
        The predicted label string.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, label_names = load_classifier(checkpoint_path, device=device)
    tokenizer = tiktoken.get_encoding("gpt2")

    if max_length is None:
        max_length = model_config["context_length"]

    return classify_text(text, model, tokenizer, device, max_length, label_names=label_names)
