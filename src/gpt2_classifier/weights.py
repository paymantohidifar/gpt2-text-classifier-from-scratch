"""Loading pretrained GPT-2 weights from HuggingFace safetensors checkpoints.

Downloads the flat ``safetensors`` state dict HuggingFace publishes for the
``openai-community/gpt2*`` models and maps it into the from-scratch
:class:`gpt2_classifier.model.GPTModel`. HuggingFace's GPT-2 checkpoints use
``Conv1D``-style weight matrices (shape ``[in_features, out_features]``),
the transpose of PyTorch's ``nn.Linear`` convention, hence the ``.T`` calls
below.
"""

import os

import requests
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from gpt2_classifier.model import GPTModel


def assign(left: torch.nn.Parameter, right: torch.Tensor) -> torch.nn.Parameter:
    """Wrap ``right`` as a new Parameter, checking it matches ``left``'s shape.

    Args:
        left: The existing parameter being replaced (used only for its shape).
        right: The tensor of pretrained weights to assign.

    Returns:
        A new ``nn.Parameter`` wrapping ``right``.

    Raises:
        ValueError: If ``left`` and ``right`` have different shapes.
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.detach())


def load_weights_into_gpt(gpt: GPTModel, params: dict[str, torch.Tensor]) -> None:
    """Copy a flat HuggingFace GPT-2 safetensors state dict into ``gpt``.

    Args:
        gpt: A freshly constructed :class:`GPTModel` with a matching config
            (same ``emb_dim``/``n_layers``/``n_heads``/``context_length``).
        params: Flat state dict as returned by
            ``safetensors.torch.load_file`` on a HuggingFace
            ``openai-community/gpt2*`` checkpoint (keys like
            ``"wte.weight"``, ``"h.0.attn.c_attn.weight"``, etc.).
    """
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])

    for b in range(len(gpt.trf_blocks)):
        q_w, k_w, v_w = torch.chunk(params[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = torch.chunk(params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params[f"h.{b}.attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params[f"h.{b}.mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params[f"h.{b}.mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params[f"h.{b}.ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])


def download_and_load_gpt2(url: str, destination: str | os.PathLike) -> dict[str, torch.Tensor]:
    """Download (if needed) and load a HuggingFace GPT-2 safetensors checkpoint.

    Args:
        url: Direct download URL for the ``model.safetensors`` file.
        destination: Local path to save/read the checkpoint from. If a file
            already exists there with a matching size, the download is
            skipped and the existing file is loaded instead.

    Returns:
        The flat state dict loaded via ``safetensors.torch.load_file``.
    """
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    file_size = int(response.headers.get("content-length", 0))

    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"The model already exists and is up-to-date: {destination}")
            return load_file(destination)

    block_size = 1024
    progress_bar_description = url.split("/")[-1]
    with (
        tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar,
        open(destination, "wb") as file,
    ):
        for chunk in response.iter_content(block_size):
            progress_bar.update(len(chunk))
            file.write(chunk)

    return load_file(destination)
