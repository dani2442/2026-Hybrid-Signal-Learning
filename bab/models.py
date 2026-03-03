"""Neural network architectures for ball-and-beam pose estimation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


# ── DLCResNet50: exact replica of the DLC TF architecture ──

class BottleneckV1(nn.Module):
    """
    tf-slim bottleneck_v1 block.
    Differences from torchvision Bottleneck:
      - stride is on the LAST unit of each block (not first)
      - when stride>1 and channels don't change, shortcut = max_pool (no learned params)
    """
    def __init__(self, in_channels, bottleneck_channels, out_channels,
                 stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3,
                               stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        self.pool_shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        elif stride > 1:
            self.pool_shortcut = nn.MaxPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        if self.shortcut is not None:
            identity = self.shortcut(x)
        elif self.pool_shortcut is not None:
            identity = self.pool_shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


def _make_block(in_ch, bottleneck_ch, out_ch, num_units, last_stride=1, dilation=1):
    """
    Build a ResNet block matching tf-slim convention:
    stride applied to the LAST unit, all others stride=1.
    """
    layers = []
    layers.append(BottleneckV1(in_ch, bottleneck_ch, out_ch, stride=1, dilation=dilation))
    for _ in range(1, num_units - 1):
        layers.append(BottleneckV1(out_ch, bottleneck_ch, out_ch, stride=1, dilation=dilation))
    if num_units > 1:
        layers.append(BottleneckV1(out_ch, bottleneck_ch, out_ch, stride=last_stride, dilation=dilation))
    return nn.Sequential(*layers)


class DLCResNet50(nn.Module):
    """
    Exact replica of the DLC TensorFlow ResNet-50 with output_stride=16.

    Architecture:
      - Stem: 7x7 conv (stride=2) + BN + ReLU + MaxPool (stride=2)
      - Block1: 3 units [64->256],  last unit stride=2   -> H/8
      - Block2: 4 units [128->512], last unit stride=2   -> H/16
      - Block3: 6 units [256->1024], all stride=1         -> H/16
      - Block4: 3 units [512->2048], all stride=1, dil=2  -> H/16
      - Head: ConvTranspose2d (3x3, stride=2) -> H/8
    """
    def __init__(self, num_keypoints=2, location_refinement=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = _make_block(64, 64, 256, num_units=3, last_stride=2)
        self.block2 = _make_block(256, 128, 512, num_units=4, last_stride=2)
        self.block3 = _make_block(512, 256, 1024, num_units=6, last_stride=1)
        self.block4 = _make_block(1024, 512, 2048, num_units=3, last_stride=1, dilation=2)

        self.part_pred = nn.ConvTranspose2d(2048, num_keypoints, 3,
                                            stride=2, padding=1, output_padding=1)
        self.location_refinement = location_refinement
        if location_refinement:
            self.locref_pred = nn.ConvTranspose2d(2048, num_keypoints * 2, 3,
                                                  stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        out = {"part_pred": self.part_pred(x)}
        if self.location_refinement:
            out["locref_pred"] = self.locref_pred(x)
        return out


# ── PoseResNet50: simplified architecture for from-scratch training ──

class PoseResNet50(nn.Module):
    """
    Simplified ResNet-50 + 3-stage deconv head for from-scratch PyTorch training.
    NOT the same as DLC TF -- used only with train_vanilla / Part E.
    """
    def __init__(self, num_keypoints: int = 2, pretrained_backbone: bool = False):
        super().__init__()
        base = resnet50(pretrained=pretrained_backbone)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

        BN_MOMENTUM = 0.1
        deconv_layers = []
        in_channels = 2048
        for out_channels in [256, 256, 256]:
            deconv_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        self.deconv_head = nn.Sequential(*deconv_layers)
        self.heatmap_head = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        upsampled = self.deconv_head(features)
        return self.heatmap_head(upsampled)
