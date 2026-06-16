"""DeepCard model.

A lightweight residual 1-D CNN tailored for structured echocardiographic
measurements, followed by multi-head self-attention and a multi-task output
module (8 multi-class severity heads + 9 binary heads). See manuscript section
"Deep learning model architecture".
"""
import torch
import torch.nn as nn

from config import MODEL, MULTICLASS_TASKS, BINARY_TASKS, NUM_SEVERITY


class ResidualBlock1D(nn.Module):
    """Conv-BN-ReLU-Dropout-Conv-BN with an identity skip connection.

    Batch normalization reduces internal covariate shift; the skip connection
    mitigates vanishing gradients and enables deeper stacking.
    """

    def __init__(self, channels, kernel_size=3, dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)          # skip connection


class DeepCard(nn.Module):
    """Residual 1-D CNN + multi-head self-attention + multi-task heads."""

    def __init__(self, cfg=MODEL):
        super().__init__()
        C = cfg["hidden_channels"]
        k = cfg["kernel_size"]

        # Initial transformation: lift the standardized 39-vector into a
        # higher-dimensional latent space.  (B, 1, 39) -> (B, C, 39)
        self.stem = nn.Sequential(
            nn.Conv1d(cfg["in_channels"], C, k, padding=k // 2),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
        )

        # Five sequential residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock1D(C, k, cfg["dropout"]) for _ in range(cfg["num_res_blocks"])]
        )

        # Multi-head self-attention over the 39 feature positions
        self.attn = nn.MultiheadAttention(
            embed_dim=C, num_heads=cfg["attn_heads"],
            dropout=cfg["dropout"], batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(C)

        # Task-specific output heads
        self.mc_heads = nn.ModuleDict({t: nn.Linear(C, NUM_SEVERITY) for t in MULTICLASS_TASKS})
        self.bin_heads = nn.ModuleDict({t: nn.Linear(C, 1) for t in BINARY_TASKS})

    def forward(self, x):
        """x: (B, 39) or (B, 1, 39).

        Returns (outputs, attention_weights) where outputs is a dict of raw
        logits per task (softmax/sigmoid are applied in the loss / at inference).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                    # (B, 1, 39)

        h = self.stem(x)                          # (B, C, 39)
        for blk in self.res_blocks:
            h = blk(h)                            # (B, C, 39)

        h = h.transpose(1, 2)                     # (B, 39, C): 39 tokens
        attn_out, attn_w = self.attn(h, h, h)     # self-attention
        h = self.attn_norm(h + attn_out)          # residual + layer-norm
        z = h.mean(dim=1)                         # global average pool -> (B, C)

        out = {}
        for t, head in self.mc_heads.items():
            out[t] = head(z)                      # (B, NUM_SEVERITY) logits
        for t, head in self.bin_heads.items():
            out[t] = head(z).squeeze(-1)          # (B,) logit
        return out, attn_w
