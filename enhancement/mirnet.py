"""
Simplified MIRNet (Multi-Scale Residual Network) for low-light image enhancement.
Paper: "Learning Enriched Features for Real Image Restoration and Enhancement" (ECCV 2020)

Core idea: maintain multiple parallel resolution streams, exchange information
between them (Selective Kernel Feature Fusion), and use residual recursive
groups so fine spatial detail and rich contextual information are both preserved.

This is a compact educational implementation (fewer RRGs / channels than the
official repo) that is practical to train on a course-project-sized dataset/GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SelectiveKernelFusion(nn.Module):
    """Fuses features from multiple resolution streams using learned attention weights."""
    def __init__(self, channels, num_streams=3, reduction=8):
        super().__init__()
        self.num_streams = num_streams
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(channels // reduction, channels, 1) for _ in range(num_streams)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, streams):
        # streams: list of (B,C,H,W) tensors, all resized to the same H,W
        fused = sum(streams)
        z = self.fc_reduce(self.pool(fused))
        attn_vectors = [fc(z) for fc in self.fcs]                    # each (B,C,1,1)
        attn_stack = torch.stack(attn_vectors, dim=1)                 # (B,num_streams,C,1,1)
        attn_stack = self.softmax(attn_stack)
        out = sum(streams[i] * attn_stack[:, i] for i in range(self.num_streams))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        return x + res


class MultiScaleResidualBlock(nn.Module):
    """Three parallel resolution streams (1x, 1/2x, 1/4x) fused with SKFF."""
    def __init__(self, channels):
        super().__init__()
        self.stream_full = ResidualBlock(channels)
        self.stream_half = ResidualBlock(channels)
        self.stream_quarter = ResidualBlock(channels)

        self.down_half = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down_quarter = nn.Conv2d(channels, channels, 3, 2, 1)

        self.fusion = SelectiveKernelFusion(channels, num_streams=3)

    def forward(self, x):
        full = self.stream_full(x)

        half_in = self.down_half(x)
        half = self.stream_half(half_in)
        half_up = F.interpolate(half, size=x.shape[-2:], mode="bilinear", align_corners=False)

        quarter_in = self.down_quarter(half_in)
        quarter = self.stream_quarter(quarter_in)
        quarter_up = F.interpolate(quarter, size=x.shape[-2:], mode="bilinear", align_corners=False)

        fused = self.fusion([full, half_up, quarter_up])
        return x + fused


class RecursiveResidualGroup(nn.Module):
    def __init__(self, channels, num_msrb=2):
        super().__init__()
        self.blocks = nn.Sequential(*[MultiScaleResidualBlock(channels) for _ in range(num_msrb)])
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        res = self.blocks(x)
        res = self.conv(res)
        return x + res


class MIRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, channels=32, num_rrg=2, num_msrb=2):
        super().__init__()
        self.head = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.body = nn.Sequential(*[RecursiveResidualGroup(channels, num_msrb) for _ in range(num_rrg)])
        self.tail = nn.Conv2d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        return torch.sigmoid(x + out)  # residual learning: predict the correction, add to input


if __name__ == "__main__":
    model = MIRNet()
    dummy = torch.randn(1, 3, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)
    print("Params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
