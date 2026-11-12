from __future__ import annotations

import math

import torch
import torch.nn as nn


class DepthHead(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        b, n, c = patches.shape
        h = w = int(math.sqrt(n))
        if h * w != n:
            raise ValueError(f"Patch count {n} is not a square grid")
        x = patches.transpose(1, 2).reshape(b, c, h, w)
        return self.decode(x)


class DepthEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.GELU(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.net(depth)


def _out_dim(task: str, num_outputs: int, num_classes: int) -> int:
    if task == "regression":
        return num_outputs
    if task == "classification":
        return num_outputs * num_classes
    raise ValueError(f"Unknown task: {task}")


class MlpHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth_dim: int,
        task: str,
        num_outputs: int = 3,
        num_classes: int = 1,
        hidden: int = 256,
    ):
        super().__init__()
        self.task = task
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        out = _out_dim(task, num_outputs, num_classes)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + depth_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out),
        )

    def forward(self, cls: torch.Tensor, patches: torch.Tensor, depth_feat: torch.Tensor):
        del patches
        x = torch.cat([cls, depth_feat], dim=-1)
        out = self.net(x)
        if self.task == "classification":
            return out.view(-1, self.num_outputs, self.num_classes)
        return out


class TransformerHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth_dim: int,
        task: str,
        num_outputs: int = 3,
        num_classes: int = 1,
        hidden: int = 256,
        nhead: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.task = task
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.token_proj = nn.Linear(embed_dim, hidden)
        self.depth_token = nn.Linear(depth_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=hidden * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, _out_dim(task, num_outputs, num_classes))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor, depth_feat: torch.Tensor):
        cls_tok = self.token_proj(cls).unsqueeze(1)
        patch_tok = self.token_proj(patches)
        depth_tok = self.depth_token(depth_feat).unsqueeze(1)
        tokens = torch.cat([cls_tok, depth_tok, patch_tok], dim=1)
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens[:, 0])
        out = self.fc(pooled)
        if self.task == "classification":
            return out.view(-1, self.num_outputs, self.num_classes)
        return out


class CnnHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth_dim: int,
        task: str,
        num_outputs: int = 3,
        num_classes: int = 1,
        hidden: int = 128,
    ):
        super().__init__()
        self.task = task
        self.num_outputs = num_outputs
        self.num_classes = num_classes
        self.depth_map = nn.Sequential(
            nn.Linear(depth_dim, hidden),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim + 1, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 + hidden, 128),
            nn.GELU(),
            nn.Linear(128, _out_dim(task, num_outputs, num_classes)),
        )

    def forward(self, cls: torch.Tensor, patches: torch.Tensor, depth_feat: torch.Tensor):
        del cls
        b, n, c = patches.shape
        h = w = int(math.sqrt(n))
        fmap = patches.transpose(1, 2).reshape(b, c, h, w)
        depth_ch = depth_feat.mean(dim=-1, keepdim=True).view(b, 1, 1, 1).expand(b, 1, h, w)
        x = torch.cat([fmap, depth_ch], dim=1)
        x = self.conv(x)
        d = self.depth_map(depth_feat)
        out = self.fc(torch.cat([x.flatten(1), d], dim=-1))
        if self.task == "classification":
            return out.view(-1, self.num_outputs, self.num_classes)
        return out


def build_macros_head(
    name: str,
    embed_dim: int,
    depth_dim: int,
    task: str,
    num_outputs: int,
    num_classes: int,
) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MlpHead(embed_dim, depth_dim, task, num_outputs, num_classes)
    if name == "transformer":
        return TransformerHead(embed_dim, depth_dim, task, num_outputs, num_classes)
    if name == "cnn":
        return CnnHead(embed_dim, depth_dim, task, num_outputs, num_classes)
    raise ValueError(f"Unknown head: {name}")
