"""RGB-D DINOv3 regressor for Nutrition5k.

* Backbone: DINOv3 ViT (timm), fine-tuned end-to-end with layer-wise LR decay.
* Depth fusion: a second, zero-initialised patch embedding for [height, valid-mask]
  is added to the RGB patch tokens, so training starts exactly from the pretrained
  DINOv3 and learns how much geometry to inject at every patch.
* Geometry prior: explicit volume / area statistics of the height map (camera is
  fixed, so these are close to physical cm^3 / cm^2) are fed to the head.
* Head: [CLS | mean patch | attention-pooled patch | geometry] -> MLP trunk ->
  (a) 5 direct targets (kcal, mass, fat, carb, protein) in standardised units,
  (b) grams of each of the 256 Nutrition5k ingredients; totals derived from them
      through the dataset's per-gram nutrient table are exact for ground truth.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

HEIGHT_SCALE_CM = 5.0  # height map is divided by this before the depth patch-embed
INGR_SCALE_G = 20.0  # ingredient head predicts grams / 20


class RgbdPatchEmbed(nn.Module):
    def __init__(self, rgb_embed: nn.Module, embed_dim: int, patch: int):
        super().__init__()
        self.rgb = rgb_embed
        self.depth = nn.Conv2d(2, embed_dim, patch, patch)
        nn.init.zeros_(self.depth.weight)
        nn.init.zeros_(self.depth.bias)
        # attributes timm's Eva reads from patch_embed
        self.grid_size = rgb_embed.grid_size
        self.patch_size = rgb_embed.patch_size
        self.num_patches = rgb_embed.num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.rgb(x[:, :3])  # NHWC (dynamic_img_size)
        d = self.depth(x[:, 3:]).permute(0, 2, 3, 1)
        return tok + d


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):
        w = self.score(x).softmax(dim=1)
        return (w * x).sum(1)


def geometry_features(height_cm: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """height_cm, valid: [B,1,H,W] -> [B,6] log-scaled volume / area statistics."""
    h = height_cm.clamp(min=0) * valid
    frac = valid.mean((1, 2, 3)).clamp(min=1e-3)
    vol = h.mean((1, 2, 3)) / frac
    area1 = ((h > 1.0).float() * valid).mean((1, 2, 3))
    area3 = ((h > 3.0).float() * valid).mean((1, 2, 3))
    vol_hi = (h - 1.0).clamp(min=0).mean((1, 2, 3)) / frac
    hmax = torch.quantile(h.flatten(1), 0.98, dim=1)
    feats = torch.stack([vol, vol_hi, area1 * 10, area3 * 10, hmax, frac], 1)
    return torch.log1p(feats)


class NutritionNet(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_base_patch16_dinov3.lvd1689m",
        num_outputs: int = 5,
        drop_path: float = 0.1,
        head_dropout: float = 0.1,
        pretrained: bool = True,
        use_depth: bool = True,
        num_ingr: int = 0,
        head_hidden: int = 768,
    ):
        super().__init__()
        self.use_depth = use_depth
        self.vit = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
            drop_path_rate=drop_path,
        )
        dim = self.vit.num_features
        self.n_prefix = self.vit.num_prefix_tokens
        if use_depth:
            self.vit.patch_embed = RgbdPatchEmbed(
                self.vit.patch_embed, dim, self.vit.patch_embed.patch_size[0]
            )
        self.pool = AttnPool(dim)
        geo_dim = 6 if use_depth else 0
        self.geo = nn.Sequential(nn.Linear(geo_dim, 64), nn.GELU()) if use_depth else None
        in_dim = dim * 3 + (64 if use_depth else 0)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
        )
        self.head = nn.Linear(head_hidden, num_outputs)
        # ingredient head: grams of every ingredient; totals = grams @ per-gram nutrient table
        self.num_ingr = num_ingr
        if num_ingr:
            self.ingr = nn.Linear(head_hidden, num_ingr)
            nn.init.constant_(self.ingr.bias, -4.0)  # start near "absent"

    def set_grad_checkpointing(self, on: bool = True):
        self.vit.set_grad_checkpointing(on)

    def forward(self, rgb: torch.Tensor, height_cm: torch.Tensor | None = None, valid=None):
        if self.use_depth:
            d = torch.cat([height_cm / HEIGHT_SCALE_CM * valid, valid], 1)
            x = torch.cat([rgb, d], 1)
        else:
            x = rgb
        tok = self.vit.forward_features(x)
        cls, patches = tok[:, 0], tok[:, self.n_prefix:]
        feats = [cls, patches.mean(1), self.pool(patches)]
        if self.use_depth:
            feats.append(self.geo(geometry_features(height_cm.float(), valid.float()).to(cls.dtype)))
        z = self.trunk(torch.cat(feats, -1))
        direct = self.head(z)
        if not self.num_ingr:
            return direct, None
        grams = F.softplus(self.ingr(z).float()) * INGR_SCALE_G
        return direct, grams


def param_groups(model: NutritionNet, lr: float, head_lr: float, wd: float, layer_decay: float):
    """Layer-wise LR decay: blocks closer to the input get smaller LRs."""
    n_blocks = len(model.vit.blocks)
    groups: dict[tuple, dict] = {}

    def layer_id(name: str) -> int:
        if name.startswith("vit.blocks."):
            return int(name.split(".")[2]) + 1
        if name.startswith("vit.patch_embed.depth"):
            return n_blocks  # new layer: train fast
        if name.startswith(("vit.cls_token", "vit.reg_token", "vit.pos_embed", "vit.patch_embed")):
            return 0
        if name.startswith("vit."):
            return n_blocks + 1  # final norm
        return -1  # head

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        no_wd = p.ndim <= 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name
        if lid < 0:
            g_lr = head_lr
        else:
            g_lr = lr * layer_decay ** (n_blocks + 1 - lid)
        key = (g_lr, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": g_lr, "weight_decay": 0.0 if no_wd else wd}
        groups[key]["params"].append(p)
    for g in groups.values():
        g["base_lr"] = g["lr"]
    return list(groups.values())
