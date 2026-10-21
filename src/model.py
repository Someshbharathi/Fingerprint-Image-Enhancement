import torch
from torch import nn

class ResidualBlock(nn.Module):
    """ResNet-style residual block for grayscale image restoration."""
    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.body(x))


class ResNetEnhancer(nn.Module):
    """
    Residual CNN for fingerprint image enhancement.

    It predicts a correction to the input and adds it through a global skip
    connection. This is a ResNet-style restoration model, not a U-Net.
    """
    def __init__(self, in_channels=1, out_channels=1, features=64, num_blocks=8):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(features) for _ in range(num_blocks)]
        )
        self.refine = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        features = self.stem(x)
        features = self.residual_layers(features)
        correction = self.refine(features)
        # Predict an image correction rather than rebuilding every pixel
        # independently. Clamp output to valid grayscale range.
        return torch.clamp(x + correction, 0.0, 1.0)
