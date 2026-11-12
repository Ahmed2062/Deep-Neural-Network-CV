"""
Multi-layer perceptron for MNIST: 784 -> 256 -> 512 -> 256 -> 10.

Every fully connected layer is a MaskedLinear so it can be pruned and
quantized in place. 465,408 weights + 1,034 biases = 466,442 parameters.
"""

import torch.nn as nn

from deep_compression.linear import MaskedLinear


class MnistMLP(nn.Module):
    def __init__(self, hidden=(256, 512, 256), num_classes=10, in_features=28 * 28):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [MaskedLinear(prev, h), nn.ReLU()]
            prev = h
        layers.append(MaskedLinear(prev, num_classes))

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.flatten(x))
