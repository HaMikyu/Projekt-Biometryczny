from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
NUM_REGISTER_TOKENS = 4


class FrozenDinov3(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        local_dir: str | None = None,
        local_files_only: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        import timm

        del local_files_only
        if local_dir:
            self.model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                dynamic_img_size=True,
            )
            blob = torch.load(local_dir, map_location="cpu", weights_only=False)
            state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
            self.model.load_state_dict(state, strict=False)
        else:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True,
            )

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.embed_dim = int(self.model.num_features)
        self.patch_size = int(self.model.patch_embed.patch_size[0])
        self.num_register_tokens = int(
            getattr(self.model, "num_prefix_tokens", 1 + NUM_REGISTER_TOKENS) - 1
        )
        if self.num_register_tokens < 0:
            self.num_register_tokens = NUM_REGISTER_TOKENS

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, pixel_values: torch.Tensor):
        with torch.no_grad():
            tokens = self.model.forward_features(pixel_values)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token sequence, got shape {tuple(tokens.shape)}")
        prefix = 1 + self.num_register_tokens
        cls = tokens[:, 0]
        patches = tokens[:, prefix:]
        return cls, patches


def resolve_local_dino_dir(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DINOv3 local path not found: {p}")
    return str(p)
