from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F

from baseline.dinov3_backbone import DEFAULT_MODEL_NAME, FrozenDinov3
from baseline.heads import DepthEncoder, build_macros_head


class MacrosLitModule(L.LightningModule):
    def __init__(
        self,
        head: str = "mlp",
        task: str = "regression",
        bin_size: float = 10.0,
        num_classes: int = 20,
        num_outputs: int = 3,
        depth_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dino_model_id: str = DEFAULT_MODEL_NAME,
        dino_local_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.bin_size = float(bin_size)
        self.num_classes = int(num_classes)
        self.num_outputs = int(num_outputs)
        self.lr = lr
        self.weight_decay = weight_decay

        self.backbone = FrozenDinov3(
            model_name=dino_model_id,
            local_dir=dino_local_dir,
            local_files_only=local_files_only,
        )
        self.depth_encoder = DepthEncoder(out_dim=depth_dim)
        self.head = build_macros_head(
            name=head,
            embed_dim=self.backbone.embed_dim,
            depth_dim=depth_dim,
            task=task,
            num_outputs=num_outputs,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor, depths: torch.Tensor):
        cls, patches = self.backbone(images)
        depth_feat = self.depth_encoder(depths)
        return self.head(cls, patches, depth_feat)

    def _values_to_bins(self, y: torch.Tensor) -> torch.Tensor:
        return (y / self.bin_size).long().clamp(0, self.num_classes - 1)

    def _bins_to_values(self, bins: torch.Tensor) -> torch.Tensor:
        return bins.float() * self.bin_size + self.bin_size * 0.5

    def _logits_to_values(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        centers = (
            torch.arange(self.num_classes, device=logits.device, dtype=probs.dtype)
            * self.bin_size
            + self.bin_size * 0.5
        )
        return (probs * centers.view(1, 1, -1)).sum(dim=-1)

    def _step(self, batch, stage: str):
        images, depths, targets, _ = batch
        targets = targets[:, : self.num_outputs]
        out = self(images, depths)

        if self.task == "regression":
            loss = F.smooth_l1_loss(out, targets)
            pred = out
        else:
            bins = self._values_to_bins(targets)
            loss = F.cross_entropy(
                out.reshape(-1, self.num_classes),
                bins.reshape(-1),
            )
            pred = self._logits_to_values(out)

        mae = (pred - targets).abs().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        for i, name in enumerate(["fat", "carb", "protein"][: self.num_outputs]):
            self.log(
                f"{stage}_mae_{name}",
                (pred[:, i] - targets[:, i]).abs().mean(),
                on_step=False,
                on_epoch=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        params = list(self.depth_encoder.parameters()) + list(self.head.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.trainer.max_epochs, 1)
        )
        return {"optimizer": opt, "lr_scheduler": sch}


def infer_num_classes(macro_max: float, bin_size: float) -> int:
    return max(1, int(math.ceil(macro_max / bin_size)) + 1)
