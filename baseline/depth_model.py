from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from baseline.dinov3_backbone import DEFAULT_MODEL_NAME, FrozenDinov3
from baseline.heads import DepthHead


class DepthLitModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dino_model_id: str = DEFAULT_MODEL_NAME,
        dino_local_dir: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = FrozenDinov3(
            model_name=dino_model_id,
            local_dir=dino_local_dir,
            local_files_only=local_files_only,
        )
        self.head = DepthHead(self.backbone.embed_dim)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, patches = self.backbone(images)
        return self.head(patches)

    def _step(self, batch, stage: str):
        images, depths, _ = batch
        pred = self(images)
        if pred.shape[-2:] != depths.shape[-2:]:
            pred = F.interpolate(
                pred, size=depths.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = F.smooth_l1_loss(pred, depths)
        mae = (pred - depths).abs().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.trainer.max_epochs, 1)
        )
        return {"optimizer": opt, "lr_scheduler": sch}
