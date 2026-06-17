"""
Colab side-view landmark model: BasicBlock + SimpleHRNet.
Weights: model/best_hrnet_landmarks.pth
Output heatmaps: [B, 20, 96, 96] for input [B, 3, 384, 384].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 384
HEATMAP_SIZE = 96
NUM_LANDMARKS = 20


class BasicBlock(nn.Module):
    """Residual conv block (checkpoint keys: block.0, .1, .3, .4)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Module()
        self.block.add_module(
            "0", nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        )
        self.block.add_module("1", nn.BatchNorm2d(out_ch))
        self.block.add_module(
            "3", nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        )
        self.block.add_module("4", nn.BatchNorm2d(out_ch))
        self._use_residual = in_ch == out_ch and stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blk = self.block
        out = F.relu(getattr(blk, "1")(getattr(blk, "0")(x)), inplace=True)
        out = getattr(blk, "4")(getattr(blk, "3")(out))
        if self._use_residual:
            return F.relu(out + x, inplace=True)
        return F.relu(out, inplace=True)


class _TransitionBlock(nn.Module):
    """stage2.0: channel change with 1x1 skip (checkpoint: block.* + skip.*)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Module()
        self.block.add_module(
            "0", nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        )
        self.block.add_module("1", nn.BatchNorm2d(out_ch))
        self.block.add_module(
            "3", nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        )
        self.block.add_module("4", nn.BatchNorm2d(out_ch))
        self.skip = nn.Module()
        self.skip.add_module(
            "0", nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        )
        self.skip.add_module("1", nn.BatchNorm2d(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blk, sk = self.block, self.skip
        main = F.relu(getattr(blk, "1")(getattr(blk, "0")(x)), inplace=True)
        main = getattr(blk, "4")(getattr(blk, "3")(main))
        skip = getattr(sk, "1")(getattr(sk, "0")(x))
        return F.relu(main + skip, inplace=True)


class SimpleHRNet(nn.Module):
    """
    Side-profile HRNet-style network trained at 384x384.
    Forward returns heatmaps [B, num_keypoints, heatmap_size, heatmap_size].
    """

    def __init__(
        self,
        num_keypoints: int = NUM_LANDMARKS,
        heatmap_size: int = HEATMAP_SIZE,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.heatmap_size = heatmap_size

        self.stem = nn.Module()
        self.stem.add_module(
            "0", nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        )
        self.stem.add_module("1", nn.BatchNorm2d(32))
        self.stem.add_module(
            "3", nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        )
        self.stem.add_module("4", nn.BatchNorm2d(64))

        self.stage1 = nn.ModuleList(
            [BasicBlock(64, 64), BasicBlock(64, 64)]
        )
        self.stage2 = nn.ModuleList(
            [_TransitionBlock(64, 96), BasicBlock(96, 96)]
        )
        self.stage3 = nn.ModuleList(
            [BasicBlock(96, 96), BasicBlock(96, 96)]
        )

        self.head = nn.Module()
        self.head.add_module("1", nn.Conv2d(96, 64, 3, padding=1, bias=False))
        self.head.add_module("2", nn.BatchNorm2d(64))
        self.head.add_module("4", nn.Conv2d(64, num_keypoints, 1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem
        x = F.relu(getattr(stem, "1")(getattr(stem, "0")(x)), inplace=True)
        x = F.relu(getattr(stem, "4")(getattr(stem, "3")(x)), inplace=True)

        for stage in (self.stage1, self.stage2, self.stage3):
            for block in stage:
                x = block(x)

        head = self.head
        x = F.relu(getattr(head, "2")(getattr(head, "1")(x)), inplace=True)
        hm = getattr(head, "4")(x)

        return F.interpolate(
            hm,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False,
        )


# Backward-compatible alias used elsewhere in the project
SideCompactHRNet = SimpleHRNet
