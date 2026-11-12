"""Model statistics used in the reports."""

from compression.linear import MaskedLinear
from compression.pruning import masked_layers


def sparsity(model):
    """Fraction of prunable weights that are pruned, in [0, 1]."""
    layers = masked_layers(model)
    total = sum(l.num_weights() for l in layers)
    active = sum(l.num_active() for l in layers)
    return 1.0 - active / total


def layer_stats(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            w = module.effective_weight()[module.mask.bool()]
            rows.append({
                "layer": name,
                "shape": tuple(module.weight.shape),
                "weights": module.num_weights(),
                "active": module.num_active(),
                "sparsity": 1 - module.num_active() / module.num_weights(),
                "unique_values": int(w.unique().numel()),
            })
    return rows


def count_parameters(model):
    return sum(p.numel() for n, p in model.named_parameters() if n.endswith(("weight", "bias")))


def to_mb(bits):
    return bits / 8 / 1024 / 1024
