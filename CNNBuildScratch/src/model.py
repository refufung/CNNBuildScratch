"""CNNSE: 18-layer-style residual CNN with squeeze-excitation, trained from scratch.

Design (224x224 input, no pretrained weights):
  stem conv3x3(s2)-BN-ReLU + MaxPool(s2)          224 -> 56
  4 stages x 2 BasicBlocks, channels 64/128/256/512, stride-2 in each stage's
  first block                                     56 -> 28 -> 14 -> 7
  SE (reduction 16) in every block
  GAP -> Dropout -> Linear(num_classes)

~11M params, roughly the compute of the old SmallCNN at 224px.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-excitation with conv (1x1) bottleneck."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class BasicBlockSE(nn.Module):
    """Two 3x3 convs + SE + skip connection."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SE(out_ch, se_reduction)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity, inplace=True)


class CNNSE(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.3,
        channels: tuple[int, ...] = (64, 128, 256, 512),
        layers: tuple[int, ...] = (2, 2, 2, 2),
        se_reduction: int = 16,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        stages: list[nn.Module] = []
        in_ch = channels[0]
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(channels, layers)):
            for block_idx in range(n_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                stages.append(BasicBlockSE(in_ch, out_ch, stride=stride, se_reduction=se_reduction))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of each residual branch so blocks start as identity.
        for m in self.modules():
            if isinstance(m, BasicBlockSE):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = CNNSE(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    print("out:", net(x).shape)
    print(f"params: {count_params(net) / 1e6:.2f}M")
