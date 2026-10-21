"""
Wav-KambaNet: Wavelet Kolmogorov-Arnold Mamba Network for EEG Motor Imagery.

Wav-KambaNet fuses four complementary inductive biases tailored specifically to
the neurophysiology of Motor Imagery (MI) Brain-Computer Interfaces:

1. Dynamic Spatial Graph Encoding:
   Electrodes are mapped to physical 3D scalp coordinates (10-10 system).
   Chebyshev spectral graph convolutions capture volume conduction and dynamic
   functional connectivity across sensorimotor cortices.

2. Wav-KAN Spectral Decomposition:
   Synapses parameterize continuous wavelet transforms (Morlet / Mexican Hat)
   with learnable scales initialized to resonance with sensorimotor rhythms:
   \\mu (8-12 Hz) and \\beta (16-30 Hz).

3. Bidirectional Mamba Sequence Modeling:
   Selective state space models run in dual directions with linear O(T) complexity:
   - Forward scan models Event-Related Desynchronization (ERD) onset.
   - Backward scan models Event-Related Synchronization (ERS) rebound.
   - Adaptive soft gating fuses directional contexts per time step.

4. KAN Classification Head:
   Non-linear Kolmogorov-Arnold decision boundary yielding interpretable
   class predictions for motor intent (e.g., Left Hand, Right Hand, Both Hands, Both Feet).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba_block import BidirectionalMamba, MambaBlock
from src.models.spatial_graph import SpatialGraphEncoder, get_electrode_positions
from src.models.wav_kan import BSplineKANLayer, WaveletKANLayer, WavKAN


class TemporalAttentionPool(nn.Module):
    """
    Learnable attention pooling over temporal sequence dimension.
    
    Computes a weighted average of sequence representations:
        a_t = Softmax(w^T \\tanh(W h_t + b))
        h_{pooled} = \\sum_{t=1}^T a_t h_t
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Sequence tensor of shape (batch, seq_len, d_model).

        Returns:
            Pooled feature tensor of shape (batch, d_model).
        """
        # x: (B, L, D) -> attn_scores: (B, L, 1)
        attn_scores = self.attn_net(x)
        weights = F.softmax(attn_scores, dim=1)
        # Weighted sum: (B, D)
        pooled = torch.sum(x * weights, dim=1)
        return pooled


class WavKambaNet(nn.Module):
    """
    Wav-KambaNet: Wavelet Kolmogorov-Arnold Mamba Network
    for EEG Motor Imagery Classification.

    Architecture Pipeline:
        Input: (B, C, T) EEG signal tensor
           │
           ▼
        [1. Dynamic Spatial Graph Encoder]
           │ ChebConv with 3D scalp coordinates & dynamic edge learning
           ▼ (B, d_spatial, T)
        [Temporal Downsampling (Optional)]
           │ Preserves frequency content while accelerating sequence modeling
           ▼ (B, T', d_spatial)
        [2. Wav-KAN Spectral Decomposition]
           │ Wavelet edge activations tuned to mu (8-12 Hz) & beta (16-30 Hz)
           ▼ (B, T', d_model)
        [3. Bidirectional Mamba Temporal Modeling]
           │ Forward scan (ERD) + Backward scan (ERS) with gated fusion
           ▼ (B, T', d_model)
        [Temporal Pooling]
           │ Learnable attention pooling over time
           ▼ (B, d_model)
        [4. KAN Classification Head]
           │ Interpretable wavelet / spline classification decision
           ▼ (B, num_classes) Logits
    """

    def __init__(
        self,
        num_channels: int = 64,
        num_classes: int = 4,
        timepoints: int = 640,
        sampling_rate: float = 160.0,
        # Spatial Graph parameters
        use_spatial_graph: bool = True,
        spatial_hidden_dim: int = 64,
        spatial_layers: int = 2,
        cheb_k: int = 3,
        learnable_edges: bool = True,
        spatial_dropout: float = 0.3,
        electrode_positions: Optional[np.ndarray] = None,
        ch_names: Optional[List[str]] = None,
        # Temporal downsampling
        downsample_factor: int = 2,
        # Wav-KAN Spectral parameters
        use_wav_kan: bool = True,
        wavelet_type: str = "morlet",
        num_wavelets: int = 16,
        scale_init: str = "eeg_rhythms",
        # Mamba Temporal parameters
        d_model: int = 64,
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        num_mamba_layers: int = 4,
        bidirectional: bool = True,
        use_wavkan_in_mamba: bool = False,
        mamba_dropout: float = 0.1,
        # Classifier Head parameters
        classifier_type: str = "kan",  # "kan" or "mlp"
        classifier_hidden_dims: Optional[List[int]] = None,
        classifier_dropout: float = 0.3,
    ) -> None:
        """
        Initialize WavKambaNet.

        Args:
            num_channels: Number of EEG electrode channels (e.g. 64 for PhysioNet).
            num_classes: Number of motor imagery classes (e.g. 2 or 4).
            timepoints: Number of temporal sample points per trial.
            sampling_rate: EEG sampling frequency in Hz.
            use_spatial_graph: Whether to use Chebyshev Spatial Graph Encoder.
            spatial_hidden_dim: Output dimension of spatial graph encoder.
            spatial_layers: Number of graph convolution layers.
            cheb_k: Chebyshev polynomial order.
            learnable_edges: Whether adjacency matrix has learnable adaptive weights.
            spatial_dropout: Dropout for spatial encoder.
            electrode_positions: Optional 3D scalp coordinates (num_channels, 3).
            ch_names: Optional electrode names list for MNE montage lookup.
            downsample_factor: Temporal downsampling factor (e.g., 2 or 4).
            use_wav_kan: Whether to use Wav-KAN for spectral decomposition.
            wavelet_type: Mother wavelet ('morlet', 'mexican_hat', 'shannon', 'bspline').
            num_wavelets: Number of wavelet basis functions per feature synapse.
            scale_init: Initialization method for wavelet scales ('eeg_rhythms').
            d_model: Latent feature dimension for Mamba blocks.
            d_state: SSM hidden state dimension.
            expand_factor: Mamba inner dimension expansion factor.
            d_conv: Local temporal convolution width.
            num_mamba_layers: Number of stacked Mamba sequence blocks.
            bidirectional: Whether to run Bidirectional Mamba (forward + backward).
            use_wavkan_in_mamba: Whether to replace linear layers inside Mamba with Wav-KAN.
            mamba_dropout: Dropout inside Mamba blocks.
            classifier_type: 'kan' or 'mlp' classification head.
            classifier_hidden_dims: Intermediate hidden dimensions for classifier.
            classifier_dropout: Classifier head dropout probability.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.timepoints = timepoints
        self.sampling_rate = sampling_rate
        self.use_spatial_graph = use_spatial_graph
        self.use_wav_kan = use_wav_kan
        self.d_model = d_model
        self.downsample_factor = downsample_factor

        # --------------------------------------------------------------------
        # 1. Spatial Processing Stage
        # --------------------------------------------------------------------
        if use_spatial_graph:
            self.spatial_encoder = SpatialGraphEncoder(
                num_channels=num_channels,
                hidden_dim=spatial_hidden_dim,
                num_layers=spatial_layers,
                cheb_k=cheb_k,
                electrode_positions=electrode_positions,
                ch_names=ch_names,
                learnable_edges=learnable_edges,
                dropout=spatial_dropout,
                pool_channels=True,  # Outputs (B, spatial_hidden_dim, T)
            )
        else:
            # Fallback: 1D convolution over channels
            self.spatial_encoder = nn.Sequential(
                nn.Conv1d(num_channels, spatial_hidden_dim, kernel_size=1),
                nn.BatchNorm1d(spatial_hidden_dim),
                nn.GELU(),
                nn.Dropout(spatial_dropout),
            )

        # Optional Temporal Downsampler
        if downsample_factor > 1:
            self.temporal_downsampler = nn.Sequential(
                nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor),
                nn.Dropout(spatial_dropout * 0.5),
            )
        else:
            self.temporal_downsampler = nn.Identity()

        # --------------------------------------------------------------------
        # 2. Wav-KAN Spectral Decomposition Stage
        # --------------------------------------------------------------------
        if use_wav_kan:
            self.spectral_encoder = WaveletKANLayer(
                in_features=spatial_hidden_dim,
                out_features=d_model,
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type,
                scale_init=scale_init,
                sampling_rate=sampling_rate / max(downsample_factor, 1),
                include_base_weight=True,
            )
        else:
            self.spectral_encoder = nn.Sequential(
                nn.Linear(spatial_hidden_dim, d_model),
                nn.GELU(),
            )

        # --------------------------------------------------------------------
        # 3. Bidirectional Mamba Temporal Modeling Stage
        # --------------------------------------------------------------------
        self.mamba_layers = nn.ModuleList()
        for _ in range(num_mamba_layers):
            if bidirectional:
                self.mamba_layers.append(
                    BidirectionalMamba(
                        d_model=d_model,
                        d_state=d_state,
                        expand_factor=expand_factor,
                        d_conv=d_conv,
                        use_wavkan_projection=use_wavkan_in_mamba,
                        num_wavelets=num_wavelets // 2 if use_wavkan_in_mamba else 8,
                        wavelet_type=wavelet_type,
                        dropout=mamba_dropout,
                    )
                )
            else:
                self.mamba_layers.append(
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        expand_factor=expand_factor,
                        d_conv=d_conv,
                        use_wavkan_projection=use_wavkan_in_mamba,
                        num_wavelets=num_wavelets // 2 if use_wavkan_in_mamba else 8,
                        wavelet_type=wavelet_type,
                        dropout=mamba_dropout,
                    )
                )

        self.final_norm = nn.LayerNorm(d_model)

        # Temporal Pooling
        self.temporal_pool = TemporalAttentionPool(d_model=d_model)

        # --------------------------------------------------------------------
        # 4. Classification Decision Head
        # --------------------------------------------------------------------
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [32]

        head_dims = [d_model] + classifier_hidden_dims + [num_classes]

        if classifier_type.lower() == "kan":
            self.classifier = WavKAN(
                layers_hidden=head_dims,
                num_wavelets=8,
                wavelet_type=wavelet_type,
                dropout=classifier_dropout,
                use_layernorm=True,
            )
        else:
            # Standard MLP head
            mlp_layers = []
            for i in range(len(head_dims) - 1):
                mlp_layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
                if i < len(head_dims) - 2:
                    mlp_layers.append(nn.LayerNorm(head_dims[i + 1]))
                    mlp_layers.append(nn.GELU())
                    mlp_layers.append(nn.Dropout(classifier_dropout))
            self.classifier = nn.Sequential(*mlp_layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent trial representation prior to final classification.

        Args:
            x: EEG tensor of shape (batch, channels, timepoints).

        Returns:
            Latent feature tensor of shape (batch, d_model).
        """
        # 1. Spatial Graph Encoding
        # x: (B, C, T) -> h_spatial: (B, spatial_hidden_dim, T)
        h_spatial = self.spatial_encoder(x)

        # 2. Temporal Downsampling: (B, spatial_hidden_dim, T')
        h_down = self.temporal_downsampler(h_spatial)

        # Transpose to (B, T', spatial_hidden_dim) for sequence/spectral operations
        h_seq = h_down.transpose(1, 2).contiguous()

        # 3. Wav-KAN Spectral Decomposition: (B, T', d_model)
        h_spectral = self.spectral_encoder(h_seq)

        # 4. Bidirectional Mamba Scanning: (B, T', d_model)
        h_temporal = h_spectral
        for mamba_block in self.mamba_layers:
            h_temporal = mamba_block(h_temporal)

        h_temporal = self.final_norm(h_temporal)

        # 5. Temporal Attention Pooling: (B, d_model)
        trial_features = self.temporal_pool(h_temporal)
        return trial_features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of WavKambaNet.

        Args:
            x: Input EEG tensor of shape (batch, channels, timepoints).
               E.g., (B, 64, 640) for 4.0s trial at 160 Hz.
            return_features: If True, returns (logits, latent_features).

        Returns:
            logits: Class logits tensor of shape (batch, num_classes).
            features (optional): Latent trial features of shape (batch, d_model).
        """
        features = self.extract_features(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    @classmethod
    def from_config(cls, cfg: Union[Dict[str, Any], Any]) -> "WavKambaNet":
        """
        Instantiate WavKambaNet directly from a configuration dictionary or Hydra/OmegaConf dict.

        Args:
            cfg: Configuration object containing model, preprocessing, and physionet keys.

        Returns:
            Configured WavKambaNet instance.
        """
        # Extract model config
        m_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else cfg.model
        sp_cfg = m_cfg.get("spatial", {})
        wav_cfg = m_cfg.get("wav_kan", {})
        mb_cfg = m_cfg.get("mamba", {})
        cl_cfg = m_cfg.get("classifier", {})

        return cls(
            num_channels=cfg.get("num_channels", 64),
            num_classes=cfg.get("num_classes", 4),
            timepoints=cfg.get("timepoints", 640),
            sampling_rate=cfg.get("preprocessing", {}).get("resample_freq", 160.0)
            if isinstance(cfg, dict)
            else 160.0,
            downsample_factor=m_cfg.get("downsample_factor", 1),
            # Spatial
            spatial_hidden_dim=sp_cfg.get("hidden_dim", 64),
            spatial_layers=sp_cfg.get("num_layers", 2),
            cheb_k=sp_cfg.get("cheb_k", 3),
            spatial_dropout=sp_cfg.get("dropout", 0.3),
            # Wav-KAN
            wavelet_type=wav_cfg.get("wavelet_type", "morlet"),
            num_wavelets=wav_cfg.get("num_wavelets", 16),
            # Mamba
            d_model=mb_cfg.get("d_model", 64),
            d_state=mb_cfg.get("d_state", 16),
            d_conv=mb_cfg.get("d_conv", 4),
            expand_factor=mb_cfg.get("expand_factor", 2),
            num_mamba_layers=mb_cfg.get("num_layers", 4),
            bidirectional=mb_cfg.get("bidirectional", True),
            use_wavkan_in_mamba=mb_cfg.get("use_wavkan_projection", False),
            mamba_dropout=mb_cfg.get("dropout", 0.1),
            # Classifier
            classifier_type=cl_cfg.get("type", "kan"),
            classifier_hidden_dims=cl_cfg.get("hidden_dims", [32]),
            classifier_dropout=cl_cfg.get("dropout", 0.3),
        )
