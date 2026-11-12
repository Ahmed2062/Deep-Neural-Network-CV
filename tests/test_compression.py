import os

import numpy as np
import torch

from compression.pruning import magnitude_prune, masked_layers, sparsity_schedule
from compression.quantization import kmeans_1d, quantize_model
from compression.serialization import load_compressed, save_compressed
from models.mlp import MnistMLP
from utils.metrics import sparsity


def test_pruning_hits_exact_sparsity_even_with_ties():
    torch.manual_seed(0)
    model = MnistMLP()
    for layer in masked_layers(model):            # force many tied magnitudes
        layer.weight.data = torch.round(layer.weight.data * 20) / 20
    for scope in ("global", "layer"):
        m = MnistMLP()
        m.load_state_dict(model.state_dict())
        magnitude_prune(m, 0.5, scope=scope)
        assert abs(sparsity(m) - 0.5) < 1e-4


def test_pruning_is_cumulative():
    model = MnistMLP()
    magnitude_prune(model, 0.5)
    before = [l.mask.clone() for l in masked_layers(model)]
    magnitude_prune(model, 0.8)
    for b, l in zip(before, masked_layers(model)):
        assert torch.all(l.mask <= b)             # nothing is revived


def test_schedule_ends_at_target():
    assert abs(sparsity_schedule(0.9, 3)[-1] - 0.9) < 1e-12


def test_kmeans_recovers_clusters():
    values = torch.cat([torch.full((100,), v) for v in (-1.0, 0.5, 2.0)])
    centroids, assign = kmeans_1d(values, 3)
    assert torch.allclose(centroids, torch.tensor([-1.0, 0.5, 2.0]))
    assert torch.equal(centroids[assign], values)


def test_save_load_is_lossless(tmp_path):
    torch.manual_seed(0)
    model = MnistMLP()
    magnitude_prune(model, 0.9)
    quantize_model(model, bits=4)
    path = os.path.join(tmp_path, "m.npz")
    save_compressed(model, path)
    restored = load_compressed(path, MnistMLP())
    x = torch.randn(8, 1, 28, 28)
    for a, b in zip(masked_layers(model), masked_layers(restored)):
        assert torch.equal(a.effective_weight(), b.effective_weight())
        assert torch.equal(a.bias, b.bias)
    assert torch.equal(model(x), restored(x))
