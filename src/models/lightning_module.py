"""
PyTorch Lightning Module for EEG Motor Imagery Classification.

This module encapsulates the complete training, validation, and testing lifecycle
for EEG classification models (Wav-KambaNet, EEGNet, ShallowConvNet, DeepConvNet).

Key Features:
-------------
- Model-agnostic wrapper accepting any PyTorch EEG model or model identifier
- AdamW optimizer with Cosine Annealing with Warm Restarts scheduler
- Label-smoothed cross-entropy loss (improves generalization on noisy EEG)
- Comprehensive evaluation metrics via torchmetrics:
    * Multiclass Accuracy
    * Cohen's Kappa Coefficient (standard MI BCI metric)
    * Macro-averaged F1 Score
    * Multiclass Confusion Matrix
- WandB experiment tracking integration
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from src.models.baselines import get_baseline_model
from src.models.wav_kambanet import WavKambaNet


class EEGClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for EEG Motor Imagery classification.

    Attributes:
        model: PyTorch neural network module (e.g. WavKambaNet, EEGNet).
        criterion: Loss function (CrossEntropy with label smoothing).
        train_acc, val_acc, test_acc: Multiclass accuracy metrics.
        train_kappa, val_kappa, test_kappa: Cohen's kappa metrics.
        train_f1, val_f1, test_f1: Macro F1 metrics.
        val_conf_mat, test_conf_mat: Confusion matrix accumulators.
    """

    def __init__(
        self,
        model: Optional[Union[nn.Module, str]] = "wav_kambanet",
        num_channels: int = 64,
        num_classes: int = 4,
        n_classes: Optional[int] = None,
        timepoints: int = 640,
        model_kwargs: Optional[Dict[str, Any]] = None,
        # Optimization
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        scheduler_type: str = "cosine_annealing_warm_restarts",
        T_0: int = 50,
        T_mult: int = 2,
        eta_min: float = 1e-5,
        # Loss
        label_smoothing: float = 0.1,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize EEGClassifier Lightning Module.

        Args:
            model: PyTorch nn.Module or string identifier ('wav_kambanet', 'eegnet', etc.).
            num_channels: Number of EEG electrode channels.
            num_classes: Number of motor imagery classes.
            timepoints: Number of temporal sample points.
            model_kwargs: Extra keyword arguments for model initialization.
            lr: Peak learning rate for AdamW optimizer.
            weight_decay: Weight decay coefficient.
            betas: AdamW beta parameters.
            scheduler_type: Learning rate scheduler type ('cosine_annealing_warm_restarts', 'cosine').
            T_0: Number of epochs for the first restart cycle.
            T_mult: Factor by which cycle length increases after each restart.
            eta_min: Minimum learning rate.
            label_smoothing: Cross-entropy label smoothing factor.
            class_names: Optional list of class names for confusion matrix visualization.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"] if isinstance(model, nn.Module) else [])

        if n_classes is not None:
            num_classes = n_classes
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.timepoints = timepoints
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.scheduler_type = scheduler_type
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.label_smoothing = label_smoothing

        if class_names is not None:
            self.class_names = class_names
        elif num_classes == 4:
            self.class_names = ["Left Fist", "Right Fist", "Both Fists", "Both Feet"]
        elif num_classes == 2:
            self.class_names = ["Left Fist", "Right Fist"]
        else:
            self.class_names = [f"Class {i}" for i in range(num_classes)]

        # --------------------------------------------------------------------
        # 1. Instantiate Model
        # --------------------------------------------------------------------
        extra_kwargs = model_kwargs or {}
        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, str):
            model_lower = model.lower()
            if model_lower in {"wav_kambanet", "wavkambanet"}:
                self.model = WavKambaNet(
                    num_channels=num_channels,
                    num_classes=num_classes,
                    timepoints=timepoints,
                    **extra_kwargs,
                )
            else:
                self.model = get_baseline_model(
                    model_name=model_lower,
                    n_channels=num_channels,
                    n_timepoints=timepoints,
                    n_classes=num_classes,
                    **extra_kwargs,
                )
        else:
            raise ValueError(f"Invalid model specification: {model}")

        # --------------------------------------------------------------------
        # 2. Loss Criterion
        # --------------------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # --------------------------------------------------------------------
        # 3. Torchmetrics
        # --------------------------------------------------------------------
        # Train metrics
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.train_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # Validation metrics
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_conf_mat = MulticlassConfusionMatrix(num_classes=num_classes)

        # Test metrics
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_conf_mat = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the underlying EEG classifier model."""
        return self.model(x)

    # ------------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------------
    def training_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run single training step on batch."""
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch["data"], batch["target"]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=-1)
        self.train_acc.update(preds, y)
        self.train_kappa.update(preds, y)
        self.train_f1.update(preds, y)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/kappa", self.train_kappa, on_step=False, on_epoch=True)
        self.log("train/f1_macro", self.train_f1, on_step=False, on_epoch=True)

        return loss

    # ------------------------------------------------------------------------
    # Validation Step
    # ------------------------------------------------------------------------
    def validation_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run single validation step on batch."""
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch["data"], batch["target"]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=-1)
        self.val_acc.update(preds, y)
        self.val_kappa.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_conf_mat.update(preds, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kappa", self.val_kappa, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute confusion matrix and optionally log to WandB at epoch end."""
        cm = self.val_conf_mat.compute().cpu().numpy()
        self._log_confusion_matrix(cm, split="val")
        self.val_conf_mat.reset()

    # ------------------------------------------------------------------------
    # Test Step
    # ------------------------------------------------------------------------
    def test_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run single test step on batch."""
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch["data"], batch["target"]

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=-1)
        self.test_acc.update(preds, y)
        self.test_kappa.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_conf_mat.update(preds, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/kappa", self.test_kappa, on_step=False, on_epoch=True)
        self.log("test/f1_macro", self.test_f1, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self) -> None:
        """Log test confusion matrix."""
        cm = self.test_conf_mat.compute().cpu().numpy()
        self._log_confusion_matrix(cm, split="test")
        self.test_conf_mat.reset()

    # ------------------------------------------------------------------------
    # Visualization & WandB Logger
    # ------------------------------------------------------------------------
    def _log_confusion_matrix(self, cm: np.ndarray, split: str = "val") -> None:
        """Generate and log a confusion matrix plot to experiment tracker."""
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(self.num_classes),
            yticks=np.arange(self.num_classes),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            title=f"{split.capitalize()} Confusion Matrix",
            ylabel="True Label",
            xlabel="Predicted Label",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

        # Text annotations inside confusion matrix
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(int(cm[i, j]), "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()

        # Check for WandB logger
        for logger in self.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                try:
                    import wandb
                    logger.experiment.log({f"{split}/confusion_matrix": wandb.Image(fig)})
                except Exception:
                    pass

        plt.close(fig)

    # ------------------------------------------------------------------------
    # Optimizers & Schedulers
    # ------------------------------------------------------------------------
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure AdamW optimizer and CosineAnnealingWarmRestarts scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.scheduler_type == "cosine_annealing_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val/loss",
                },
            }
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.T_0,
                eta_min=self.eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return {"optimizer": optimizer}
