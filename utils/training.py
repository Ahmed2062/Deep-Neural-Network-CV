"""Training and evaluation loops."""

import random

import numpy as np
import torch

from deep_compression.pruning import masked_layers


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, lr):
    """
    A fresh Adam over the parameters that currently require gradients.

    A new optimizer is created for every phase. Reusing the baseline optimizer
    would carry over Adam's momentum, which keeps nudging pruned weights even
    though their gradient is zero.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    layers = masked_layers(model)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        for layer in layers:          # keep pruned weights exactly zero
            layer.apply_mask()
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def fit(model, train_loader, test_loader, epochs, lr, device, tag, log=print):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, lr)
    acc = None
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        log(f"  [{tag}] epoch {epoch:>2}/{epochs}  loss {loss:.4f}  test acc {acc:.2f}%")
    return acc
