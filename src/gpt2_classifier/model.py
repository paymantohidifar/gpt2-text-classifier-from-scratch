"""From-scratch GPT-2 architecture.

A standard pre-norm decoder-only transformer, built so that
pretrained GPT-2 weights can be loaded into it via
:mod:`gpt2_classifier.weights`.
"""

import torch
from torch import nn
import math

from gpt2_classifier.config import GPTConfig


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        """Initialize the attention layer.

        Args:
            d_in: Input embedding dimension.
            d_out: Output embedding dimension (must be divisible by
                ``num_heads``).
            context_length: Maximum sequence length, used to size the causal
                mask buffer.
            dropout: Dropout probability applied to attention weights.
            num_heads: Number of attention heads.
            qkv_bias: Whether the query/key/value projections have a bias
                term.
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal multi-head self-attention.

        Args:
            x: Input tensor of shape ``(batch, num_tokens, d_in)``.

        Returns:
            Tensor of shape ``(batch, num_tokens, d_out)``.
        """
        b, num_tokens, _d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class LayerNorm(nn.Module):
    """Manual layer normalization (kept distinct from ``nn.LayerNorm`` to
    match the parameter names expected by the pretrained-weight loader)."""

    def __init__(self, emb_dim: int) -> None:
        """Initialize learnable scale/shift parameters.

        Args:
            emb_dim: Size of the last dimension to normalize over.
        """
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``.

        Args:
            x: Input tensor.

        Returns:
            Normalized, scaled, and shifted tensor of the same shape.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Tanh approximation of the GELU activation, matching GPT-2's original."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GELU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor of the same shape with GELU applied elementwise.
        """
        return 0.5 * x * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Position-wise feed-forward network (4x expansion MLP)."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the feed-forward layers.

        Args:
            cfg: Model configuration dict; only ``emb_dim`` is used.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Input tensor of shape ``(batch, num_tokens, emb_dim)``.

        Returns:
            Tensor of the same shape.
        """
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + feed-forward with residuals."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize the block's attention, feed-forward, and norm layers.

        Args:
            cfg: Model configuration dict.
        """
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one transformer block.

        Args:
            x: Input tensor of shape ``(batch, num_tokens, emb_dim)``.

        Returns:
            Tensor of the same shape.
        """
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    """Decoder-only GPT-2 style transformer."""

    def __init__(self, cfg: GPTConfig) -> None:
        """Initialize embeddings, transformer blocks, and output head.

        Args:
            cfg: Model configuration dict (see
                :func:`gpt2_classifier.config.get_model_config`).
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            in_idx: Token id tensor of shape ``(batch, num_tokens)``.

        Returns:
            Logits tensor of shape ``(batch, num_tokens, vocab_size)`` (or
            ``(batch, num_tokens, num_classes)`` after ``out_head`` has been
            replaced for classification fine-tuning).
        """
        _batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(self, out_dim))
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        return (self.rank / self.alpha) * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


