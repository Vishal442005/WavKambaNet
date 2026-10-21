"""
Baseline Deep Learning Models for EEG Motor Imagery Classification.

This module provides factory functions and implementations for standard BCI baseline
architectures:
1. EEGNet (Lawhern et al., 2018): Compact CNN using temporal, depthwise spatial,
   and separable convolutions designed for BCI.
2. ShallowConvNet / ShallowFBCSPNet (Schirrmeister et al., 2017): Inspired by Filter Bank
   Common Spatial Pattern (FBCSP), uses large temporal kernels followed by squaring and log-pooling.
3. DeepConvNet (Schirrmeister et al., 2017): Deeper CNN architecture with multiple
   convolutional-maxpooling stages for hierarchical temporal-spatial feature learning.

Uses Braindecode implementations where available, with native PyTorch fallback implementations.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import braindecode.models as bdm
    HAS_BRAINDECODE = True
except ImportError:
    HAS_BRAINDECODE = False


# ============================================================================
# Native PyTorch Fallback Implementations
# ============================================================================

class NativeEEGNet(nn.Module):
    """
    Native PyTorch implementation of EEGNet-4,2 (Lawhern et al., 2018).

    Pipeline:
    1. 2D Temporal Conv (1 x kernel_length)
    2. Depthwise Spatial Conv (channels x 1)
    3. BatchNorm + ELU + AvgPool
    4. Separable Conv (Depthwise + Pointwise)
    5. BatchNorm + ELU + AvgPool
    6. Dense Classifier Head
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        n_timepoints: int = 640,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_timepoints = n_timepoints

        # Block 1: Temporal Conv
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Spatial Conv across electrodes
        self.depthwise = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable Conv
        self.separable = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F1 * D,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute output feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            feat = self._forward_features(dummy)
            feat_dim = feat.shape[1] * feat.shape[2] * feat.shape[3]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.depthwise(h)
        h = self.bn2(h)
        h = F.elu(h)
        h = self.pool1(h)
        h = self.drop1(h)

        h = self.separable(h)
        h = self.bn3(h)
        h = F.elu(h)
        h = self.pool2(h)
        h = self.drop2(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch, channels, timepoints).
        """
        if x.dim() == 3:
            # (B, C, T) -> (B, 1, C, T)
            x = x.unsqueeze(1)
        h = self._forward_features(x)
        h_flat = torch.flatten(h, start_dim=1)
        logits = self.classifier(h_flat)
        return logits


class NativeShallowConvNet(nn.Module):
    """
    Native PyTorch implementation of ShallowFBCSPNet (Schirrmeister et al., 2017).

    Pipeline:
    1. Temporal Conv (1 x 25)
    2. Spatial Conv (channels x 1)
    3. Squaring activation (computes power)
    4. Average Pooling (1 x 75, stride 15)
    5. Logarithmic activation (log power)
    6. Dropout and Dense classifier
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        n_timepoints: int = 640,
        n_filters_time: int = 40,
        filter_time_length: int = 25,
        n_filters_spat: int = 40,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_time = nn.Conv2d(
            1, n_filters_time, kernel_size=(1, filter_time_length), bias=True
        )
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat, kernel_size=(n_channels, 1), bias=False
        )
        self.bn = nn.BatchNorm2d(n_filters_spat)
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_time_length), stride=(1, pool_time_stride)
        )
        self.drop = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            h = self.conv_time(dummy)
            h = self.conv_spat(h)
            h = self.bn(h)
            h = torch.square(h)
            h = self.pool(h)
            h = torch.log(torch.clamp(h, min=1e-6))
            feat_dim = h.shape[1] * h.shape[2] * h.shape[3]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.conv_time(x)
        h = self.conv_spat(h)
        h = self.bn(h)
        h = torch.square(h)
        h = self.pool(h)
        h = torch.log(torch.clamp(h, min=1e-6))
        h = self.drop(h)
        h_flat = torch.flatten(h, start_dim=1)
        return self.classifier(h_flat)


class NativeDeepConvNet(nn.Module):
    """
    Native PyTorch implementation of DeepConvNet (Schirrmeister et al., 2017).

    Pipeline:
    1. Temporal Conv (1 x 10)
    2. Spatial Conv (channels x 1)
    3. MaxPool (1 x 3)
    4. 3x Convolutional Blocks: Conv (1 x 10) -> BatchNorm -> ELU -> MaxPool
    5. Classifier
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 4,
        n_timepoints: int = 640,
        n_filters_time: int = 25,
        filter_time_length: int = 10,
        n_filters_spat: int = 25,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_time, kernel_size=(1, filter_time_length), bias=True),
            nn.Conv2d(n_filters_time, n_filters_spat, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters_spat),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.Dropout(dropout),
        )

        filter_sizes = [50, 100, 200]
        in_ch = n_filters_spat
        self.blocks = nn.ModuleList()
        for out_ch in filter_sizes:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=(1, 10), bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ELU(),
                    nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
                    nn.Dropout(dropout),
                )
            )
            in_ch = out_ch

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            h = self.block1(dummy)
            for block in self.blocks:
                h = block(h)
            feat_dim = h.shape[1] * h.shape[2] * h.shape[3]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.block1(x)
        for block in self.blocks:
            h = block(h)
        h_flat = torch.flatten(h, start_dim=1)
        return self.classifier(h_flat)


# ============================================================================
# Factory Functions
# ============================================================================

def get_eegnet(
    n_channels: int = 64,
    n_timepoints: int = 640,
    n_classes: int = 4,
    prefer_braindecode: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Instantiate EEGNet baseline model.

    Args:
        n_channels: Number of EEG electrode channels.
        n_timepoints: Number of temporal sample points.
        n_classes: Number of target classification categories.
        prefer_braindecode: Whether to use Braindecode's implementation if available.
        **kwargs: Additional hyperparameters (F1, D, F2, kernel_length, dropout).

    Returns:
        Instantiated PyTorch EEGNet model.
    """
    if prefer_braindecode and HAS_BRAINDECODE and hasattr(bdm, "EEGNet"):
        try:
            return bdm.EEGNet(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=n_timepoints,
                **kwargs,
            )
        except Exception:
            pass

    return NativeEEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        n_timepoints=n_timepoints,
        **kwargs,
    )


def get_shallow_convnet(
    n_channels: int = 64,
    n_timepoints: int = 640,
    n_classes: int = 4,
    prefer_braindecode: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Instantiate ShallowConvNet (ShallowFBCSPNet) baseline model.

    Args:
        n_channels: Number of EEG electrode channels.
        n_timepoints: Number of temporal sample points.
        n_classes: Number of target classification categories.
        prefer_braindecode: Whether to use Braindecode's implementation if available.
        **kwargs: Additional hyperparameters.

    Returns:
        Instantiated PyTorch ShallowConvNet model.
    """
    if prefer_braindecode and HAS_BRAINDECODE and hasattr(bdm, "ShallowFBCSPNet"):
        try:
            return bdm.ShallowFBCSPNet(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=n_timepoints,
                **kwargs,
            )
        except Exception:
            pass

    return NativeShallowConvNet(
        n_channels=n_channels,
        n_classes=n_classes,
        n_timepoints=n_timepoints,
        **kwargs,
    )


def get_deep_convnet(
    n_channels: int = 64,
    n_timepoints: int = 640,
    n_classes: int = 4,
    prefer_braindecode: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Instantiate DeepConvNet (Deep4Net) baseline model.

    Args:
        n_channels: Number of EEG electrode channels.
        n_timepoints: Number of temporal sample points.
        n_classes: Number of target classification categories.
        prefer_braindecode: Whether to use Braindecode's implementation if available.
        **kwargs: Additional hyperparameters.

    Returns:
        Instantiated PyTorch DeepConvNet model.
    """
    if prefer_braindecode and HAS_BRAINDECODE and hasattr(bdm, "Deep4Net"):
        try:
            return bdm.Deep4Net(
                n_chans=n_channels,
                n_outputs=n_classes,
                n_times=n_timepoints,
                **kwargs,
            )
        except Exception:
            pass

    return NativeDeepConvNet(
        n_channels=n_channels,
        n_classes=n_classes,
        n_timepoints=n_timepoints,
        **kwargs,
    )


def get_baseline_model(
    model_name: str,
    n_channels: int = 64,
    n_timepoints: int = 640,
    n_classes: int = 4,
    **kwargs: Any,
) -> nn.Module:
    """
    Convenience factory to instantiate any baseline model by name.

    Supported names:
    - 'eegnet'
    - 'shallow_convnet', 'shallow_fbcspnet', 'shallow'
    - 'deep_convnet', 'deep4net', 'deep'
    """
    name = model_name.lower().replace("-", "_")
    if name in {"eegnet", "eegnetv4"}:
        return get_eegnet(n_channels, n_timepoints, n_classes, **kwargs)
    elif name in {"shallow_convnet", "shallow_fbcspnet", "shallow"}:
        return get_shallow_convnet(n_channels, n_timepoints, n_classes, **kwargs)
    elif name in {"deep_convnet", "deep4net", "deep"}:
        return get_deep_convnet(n_channels, n_timepoints, n_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown baseline model name '{model_name}'. "
            f"Supported names: eegnet, shallow_convnet, deep_convnet."
        )
