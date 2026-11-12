"""
On-disk format for a pruned + quantized model, and storage accounting.

For every MaskedLinear layer we store only what is needed to rebuild it:

    shape        2 x int32
    codebook     k x float32           shared weight values
    bias         out x float32         kept at full precision
    nnz          number of surviving weights
    idx stream   Huffman-coded cluster index of each surviving weight
    gap stream   Huffman-coded distance between consecutive surviving
                 weights (row-major order), i.e. a relative sparse index
    + the (symbol, length) tables for both Huffman codes

The file is written with np.savez (no zlib), so the size on disk reflects
our own encoding rather than a general-purpose compressor.

load_compressed() decodes the file into a fresh model, so accuracy can be
measured on what was actually stored rather than on the in-memory model.
"""

import math
import os

import numpy as np
import torch

from deep_compression import huffman
from deep_compression.pruning import masked_layers

FP32 = 32


def _layer_streams(layer):
    """Positions/gaps/cluster indices of the surviving weights of one layer."""
    mask = layer.mask.detach().cpu().numpy().ravel().astype(bool)
    positions = np.flatnonzero(mask)
    gaps = np.diff(positions, prepend=-1)          # all >= 1
    cluster_idx = layer.indices.detach().cpu().numpy().ravel()[positions]
    return positions, gaps, cluster_idx


def encode_model(model):
    """Return (arrays to save, per-layer metadata)."""
    arrays, meta = {}, []
    for i, layer in enumerate(masked_layers(model)):
        if not layer.is_quantized:
            raise RuntimeError("Quantize the model before serializing it.")
        _, gaps, cluster_idx = _layer_streams(layer)

        idx_bytes, idx_bits, idx_table = huffman.encode(cluster_idx)
        gap_bytes, gap_bits, gap_table = huffman.encode(gaps)

        p = f"layer{i}_"
        arrays[p + "shape"] = np.array(layer.weight.shape, dtype=np.int32)
        arrays[p + "codebook"] = layer.codebook.detach().cpu().numpy().astype(np.float32)
        arrays[p + "bias"] = layer.bias.detach().cpu().numpy().astype(np.float32)
        arrays[p + "nnz"] = np.array([cluster_idx.size], dtype=np.int64)
        for name, (data, nbits, table) in {
            "idx": (idx_bytes, idx_bits, idx_table),
            "gap": (gap_bytes, gap_bits, gap_table),
        }.items():
            arrays[p + name + "_data"] = data
            arrays[p + name + "_nbits"] = np.array([nbits], dtype=np.int64)
            arrays[p + name + "_symbols"] = table[:, 0].astype(np.int32)
            arrays[p + name + "_lengths"] = table[:, 1].astype(np.uint8)

        meta.append({
            "nnz": int(cluster_idx.size),
            "idx_bits": idx_bits, "gap_bits": gap_bits,
            "table_bits": huffman.table_bits(idx_table) + huffman.table_bits(gap_table),
            "max_gap": int(gaps.max()) if gaps.size else 0,
        })
    return arrays, meta


def save_compressed(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays, meta = encode_model(model)
    np.savez(path, **arrays)
    return meta


def _decode_layer(data, i):
    p = f"layer{i}_"
    shape = tuple(int(s) for s in data[p + "shape"])
    nnz = int(data[p + "nnz"][0])

    def stream(name):
        table = np.stack([data[p + name + "_symbols"].astype(np.int64),
                          data[p + name + "_lengths"].astype(np.int64)], axis=1)
        return huffman.decode(data[p + name + "_data"],
                              int(data[p + name + "_nbits"][0]), table, nnz)

    positions = np.cumsum(stream("gap")) - 1
    cluster_idx = stream("idx")

    weight = np.zeros(int(np.prod(shape)), dtype=np.float32)
    weight[positions] = data[p + "codebook"][cluster_idx]
    mask = np.zeros_like(weight)
    mask[positions] = 1.0
    return weight.reshape(shape), mask.reshape(shape), data[p + "bias"]


@torch.no_grad()
def load_compressed(path, model):
    """Decode ``path`` into ``model`` (a freshly constructed, dense model)."""
    data = np.load(path)
    for i, layer in enumerate(masked_layers(model)):
        weight, mask, bias = _decode_layer(data, i)
        layer.weight.copy_(torch.from_numpy(weight))
        layer.mask.copy_(torch.from_numpy(mask))
        layer.bias.copy_(torch.from_numpy(bias))
    return model


# ---------------------------------------------------------------------- #
# Storage accounting
# ---------------------------------------------------------------------- #
def storage_report(model, meta, bits):
    """
    Size in bits of the model at each stage, using explicit formats:

    dense_fp32   every weight and bias as float32 (the baseline)
    pruned       surviving weights as float32 + a fixed-width relative index
                 per weight (sparse storage is not free!) + float32 biases
    quantized    surviving weights as ``bits``-wide cluster indices + the
                 same relative index + float32 codebook + float32 biases
    huffman      both index streams Huffman-coded + code tables + codebook
                 + float32 biases (this is what save_compressed writes)
    """
    layers = masked_layers(model)
    n_weights = sum(l.num_weights() for l in layers)
    n_bias = sum(l.bias.numel() for l in layers)
    bias_bits = n_bias * FP32

    pruned = quantized = huff = bias_bits
    for layer, m in zip(layers, meta):
        rel_idx_bits = max(1, math.ceil(math.log2(m["max_gap"] + 1)))
        codebook_bits = layer.codebook.numel() * FP32
        pruned += m["nnz"] * (FP32 + rel_idx_bits)
        quantized += m["nnz"] * (bits + rel_idx_bits) + codebook_bits
        huff += m["idx_bits"] + m["gap_bits"] + m["table_bits"] + codebook_bits

    return {
        "dense_fp32": (n_weights + n_bias) * FP32,
        "pruned": pruned,
        "quantized": quantized,
        "huffman": huff,
    }
