"""
Compression-aware fully connected layer.

A MaskedLinear behaves like nn.Linear but supports the two lossy stages of
Deep Compression:

* Pruning: a binary ``mask`` buffer is multiplied into the weight on every
  forward pass, so pruned connections contribute nothing and receive zero
  gradient while the network is fine-tuned.

* Weight sharing (trained quantization): after K-means clustering, the layer
  stops using its dense ``weight`` and instead stores a small ``codebook`` of
  shared values plus an ``indices`` tensor saying which codebook entry each
  weight uses. The effective weight is ``codebook[indices] * mask``. Because
  the lookup is differentiable with respect to the codebook, fine-tuning
  automatically gives each centroid the *sum* of the gradients of the
  weights in its cluster, which is exactly the centroid update from the
  Deep Compression paper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Buffers move with .to(device) and are saved in state_dict, but the
        # optimizer never touches them.
        self.register_buffer("mask", torch.ones(out_features, in_features))
        self.register_buffer("indices", None)
        self.register_parameter("codebook", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Same initialisation as nn.Linear (Kaiming-uniform).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------ #
    @property
    def is_quantized(self):
        return self.codebook is not None

    def effective_weight(self):
        w = self.codebook[self.indices] if self.is_quantized else self.weight
        return w * self.mask

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def set_mask(self, mask):
        """Install a new pruning mask and zero the pruned weights."""
        self.mask.copy_(mask.to(self.mask.dtype))
        self.weight.mul_(self.mask)

    @torch.no_grad()
    def apply_mask(self):
        """Re-zero pruned weights (called after each optimizer step)."""
        if not self.is_quantized:
            self.weight.mul_(self.mask)

    @torch.no_grad()
    def set_quantization(self, codebook, indices):
        """Switch the layer to codebook + index (weight-sharing) mode."""
        self.codebook = nn.Parameter(codebook.to(self.weight.device).float())
        self.indices = indices.to(self.weight.device).long()
        # The dense weight is no longer used by forward(); freeze it so it
        # is not handed to the optimizer.
        self.weight.requires_grad_(False)

    def num_weights(self):
        return self.weight.numel()

    def num_active(self):
        return int(self.mask.sum().item())

    def extra_repr(self):
        mode = f"quantized k={self.codebook.numel()}" if self.is_quantized else "dense"
        sparsity = 1 - self.num_active() / self.num_weights()
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"sparsity={sparsity:.1%}, {mode}")
