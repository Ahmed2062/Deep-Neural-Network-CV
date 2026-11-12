"""
Magnitude-based unstructured pruning.

Weights with the smallest absolute value are assumed to matter least and are
removed by setting their mask entry to 0. Pruning is cumulative: weights that
are already pruned have magnitude 0, so they are always among the smallest
and stay pruned when the target sparsity is raised.

The k smallest weights are selected by index (torch.topk) rather than by a
``> threshold`` comparison, so the achieved sparsity is exact even when many
weights share the same value (e.g. after quantization).
"""

import torch

from compression.linear import MaskedLinear


def masked_layers(model):
    return [m for m in model.modules() if isinstance(m, MaskedLinear)]


@torch.no_grad()
def _mask_smallest(magnitudes, k):
    mask = torch.ones_like(magnitudes)
    if k > 0:
        idx = torch.topk(magnitudes, k, largest=False).indices
        mask[idx] = 0.0
    return mask


@torch.no_grad()
def magnitude_prune(model, sparsity, scope="global", min_density=0.01):
    """
    Prune ``model`` to the given fraction of zero weights.

    scope="global": one threshold across all layers. Layers with many
        redundant weights (e.g. the 784-input layer, where border pixels of
        MNIST carry no information) end up sparser than small layers.
    scope="layer":  every layer is pruned to the same sparsity.

    min_density: with a global threshold a whole layer can be wiped out (for
        example in an untrained network, where layers with large fan-in start
        with the smallest weights). Any layer that would keep fewer than this
        fraction of its weights keeps its largest ``min_density`` instead, so
        the achieved sparsity can be marginally below the target.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    layers = masked_layers(model)

    if scope == "layer":
        for layer in layers:
            mags = (layer.weight * layer.mask).abs().flatten()
            k = int(round(sparsity * mags.numel()))
            layer.set_mask(_mask_smallest(mags, k).view_as(layer.weight))
    elif scope == "global":
        mags = torch.cat([(l.weight * l.mask).abs().flatten() for l in layers])
        k = int(round(sparsity * mags.numel()))
        flat_mask = _mask_smallest(mags, k)
        offset = 0
        for layer in layers:
            n = layer.weight.numel()
            mask = flat_mask[offset:offset + n]
            if mask.mean().item() < min_density:
                k_layer = n - max(1, int(round(min_density * n)))
                mask = _mask_smallest(mags[offset:offset + n], k_layer)
            layer.set_mask(mask.view_as(layer.weight))
            offset += n
    else:
        raise ValueError("scope must be 'global' or 'layer'")


def sparsity_schedule(target, steps):
    """
    Sparsity targets for iterative pruning, e.g. target=0.9, steps=3 ->
    [0.536, 0.785, 0.9]. Each step removes the same *fraction of the
    remaining* weights, which is gentler than jumping straight to the target.
    """
    keep = 1.0 - target
    return [1.0 - keep ** ((i + 1) / steps) for i in range(steps)]
