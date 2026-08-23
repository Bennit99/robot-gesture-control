"""PyTorch Lightning module wrapping any architecture from the registry.

Handles:
- Loss computation (cross-entropy)
- Optimizer + LR scheduler setup
- Metric logging (accuracy, top-5 accuracy, F1, confusion matrix)
- W&B integration (via Lightning's WandbLogger)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

from torchmetrics import Accuracy, F1Score

from jester.training.architectures import build_model


class GestureClassifier(L.LightningModule):
    """Lightning wrapper for gesture classification.

    Parameters
    ----------
    arch          : architecture key (``lstm``, ``gru``, ``transformer``, ``conv1d``, ``stgcn``)
    input_dim     : per-frame feature dimension
    num_classes   : number of gesture classes
    seq_len       : fixed temporal sequence length
    lr            : initial learning rate
    weight_decay  : AdamW weight decay
    scheduler     : ``"cosine"`` or ``"plateau"`` or ``"none"``
    warmup_steps  : linear warm-up steps (for cosine scheduler)
    label_smoothing : cross-entropy label smoothing
    arch_kwargs   : extra kwargs forwarded to the model constructor
    """

    def __init__(
        self,
        arch: str = "lstm",
        input_dim: int = 63,
        num_classes: int = 27,
        seq_len: int = 40,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "cosine",
        warmup_steps: int = 500,
        label_smoothing: float = 0.0,
        **arch_kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(
            arch,
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=seq_len,
            **arch_kwargs,
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_top5 = Accuracy(
            task="multiclass", num_classes=num_classes, top_k=5
        )
        self.val_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    # -- Forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # -- Training step -----------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        preds = logits.argmax(dim=-1)

        self.train_acc(preds, batch["y"])
        self.log("train/loss", loss, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    # -- Validation step ---------------------------------------------------
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["x"])
        loss = self.loss_fn(logits, batch["y"])
        preds = logits.argmax(dim=-1)

        self.val_acc(preds, batch["y"])
        self.val_acc_top5(logits, batch["y"])
        self.val_f1(preds, batch["y"])

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_top5", self.val_acc_top5, on_step=False, on_epoch=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True)

    # -- Optimizer + scheduler ---------------------------------------------
    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler == "none":
            return {"optimizer": optimizer}

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,
                eta_min=1e-6,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        if self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }

        raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
