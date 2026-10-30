"""Small inspectable CNN with explicit pooling switches for Zeiler-style projection."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class InspectableCNN(nn.Module):
    """Three convolutional layers; first two preserve MaxPool argmax switches.

    The input is 16x16; CIFAR-10 uses RGB, Digits uses one grayscale channel.
    All intermediate activations remain accessible for reproducible inspection.
    """

    def __init__(self, channels: int = 1, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 20, 3, padding=1)
        self.conv3 = nn.Conv2d(20, 28, 3, padding=1)
        self.fc = nn.Linear(28, classes)

    def forward_trace(self, x: Tensor) -> tuple[Tensor, dict]:
        a1 = F.relu(self.conv1(x))
        p1, s1 = F.max_pool2d(a1, 2, return_indices=True)
        a2 = F.relu(self.conv2(p1))
        p2, s2 = F.max_pool2d(a2, 2, return_indices=True)
        a3 = F.relu(self.conv3(p2))
        logits = self.fc(a3.mean(dim=(-2, -1)))
        return logits, {"features": [a1, a2, a3], "switches": [s1, s2]}

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_trace(x)[0]


def deconv_projection(model: InspectableCNN, trace: dict, layer: int, channel: int) -> Tensor:
    """Project one peak activation backward using DeconvNet-style ReLU + switches.

    This is a feature projection, NOT an inverse of the CNN or a class attribution.
    The target is a single spatial maximum in one feature map.
    """
    if layer not in (1, 2, 3):
        raise ValueError("layer must be 1, 2, or 3")
    features = trace["features"]
    target = features[layer - 1].detach()
    if channel < 0 or channel >= target.shape[1]:
        raise ValueError("channel out of range")
    signal = torch.zeros_like(target)
    position = target[0, channel].flatten().argmax().item()
    width = target.shape[-1]
    signal[0, channel, position // width, position % width] = target[0, channel, position // width, position % width]
    convolutions = (model.conv1, model.conv2, model.conv3)
    for depth in range(layer, 0, -1):
        if depth < layer:
            signal = F.max_unpool2d(
                signal,
                trace["switches"][depth - 1],
                kernel_size=2,
                output_size=features[depth - 1].shape,
            )
        signal = F.conv_transpose2d(F.relu(signal), convolutions[depth - 1].weight.detach(), padding=1)
    return signal.detach()