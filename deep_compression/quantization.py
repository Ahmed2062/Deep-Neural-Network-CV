"""
Weight sharing via 1-D K-means (the "trained quantization" stage).

For each layer, the surviving (unpruned) weights are clustered into
k = 2**bits groups. Every weight is then replaced by its cluster centroid,
so the layer only needs to store:

* a codebook of k float32 centroids, and
* a ``bits``-wide index per surviving weight.

Centroids are initialised linearly between the min and max weight, as the
Deep Compression paper recommends: compared with random or density-based
initialisation, linear spacing keeps centroids near the rare large-magnitude
weights, which have the most influence on the output.

After clustering, the centroids are fine-tuned by gradient descent (see
MaskedLinear), which recovers most of the accuracy lost to clustering.
"""

import torch

from deep_compression.pruning import masked_layers


@torch.no_grad()
def kmeans_1d(values, k, iters=50, tol=1e-7):
    """
    Lloyd's algorithm in one dimension.

    With sorted centroids, the nearest centroid of each value is found with a
    binary search against the midpoints between neighbouring centroids
    (torch.bucketize), so each iteration is O(n log k) instead of O(n k).
    Returns (centroids [k], assignments [n]).
    """
    values = values.flatten().float()
    if values.numel() == 0:
        return torch.zeros(1, device=values.device), torch.zeros(0, dtype=torch.long,
                                                                  device=values.device)
    k = min(k, values.numel())
    centroids = torch.linspace(values.min().item(), values.max().item(), k,
                               device=values.device)
    for _ in range(iters):
        boundaries = (centroids[1:] + centroids[:-1]) / 2
        assign = torch.bucketize(values, boundaries)
        sums = torch.zeros(k, device=values.device).index_add_(0, assign, values)
        counts = torch.zeros(k, device=values.device).index_add_(
            0, assign, torch.ones_like(values))
        # Empty clusters keep their previous centroid.
        new = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)
        new, _ = torch.sort(new)
        shift = (new - centroids).abs().max().item()
        centroids = new
        if shift < tol:
            break
    boundaries = (centroids[1:] + centroids[:-1]) / 2
    return centroids, torch.bucketize(values, boundaries)


@torch.no_grad()
def quantize_model(model, bits):
    """Cluster every MaskedLinear layer's active weights into 2**bits values."""
    report = []
    for layer in masked_layers(model):
        w = layer.weight.data
        active = layer.mask.bool()
        centroids, assign = kmeans_1d(w[active], 2 ** bits)

        # Pruned positions get index 0; the mask zeroes them in forward()
        # and they are never written to the compressed file.
        indices = torch.zeros_like(w, dtype=torch.long)
        indices[active] = assign

        mse = ((centroids[assign] - w[active]) ** 2).mean().item()
        layer.set_quantization(centroids, indices)
        report.append({"clusters": centroids.numel(), "mse": mse})
    return report
