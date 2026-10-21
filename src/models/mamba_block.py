"""
Pure PyTorch Bidirectional Mamba Block for Temporal EEG Sequence Modeling.

This module implements the Selective State Space Model (Mamba) architecture
without requiring external CUDA/C++ dependencies (mamba-ssm), ensuring 100%
portability across Windows, Linux, CPU, and GPU environments.

Mathematical Formulation:
-------------------------
1. Continuous State Space Model:
   h'(t) = A h(t) + B(t) x(t)
   y(t)  = C(t) h(t) + D x(t)

2. Discretization with Input-Dependent Step Size \\Delta_t:
   In classical LTI SSMs (S4), A, B, C are fixed across time.
   In Mamba (Gu & Dao, 2023), B_t, C_t, and \\Delta_t are dynamic projections of x_t:
       B_t = \\text{Linear}_B(x_t) \\in \\mathbb{R}^{d_{state}}
       C_t = \\text{Linear}_C(x_t) \\in \\mathbb{R}^{d_{state}}
       \\Delta_t = \\text{Softplus}(\\text{Linear}_\\Delta(x_t) + \\text{bias}_\\Delta) \\in \\mathbb{R}^{d_{inner}}

   Discretization (Zero-Order Hold):
       \\bar{A}_t = \\exp(\\Delta_t \\cdot A)
       \\bar{B}_t = (\\Delta_t \\cdot A)^{-1} (\\exp(\\Delta_t \\cdot A) - I) \\cdot (\\Delta_t \\cdot B_t) \\approx \\Delta_t \\cdot B_t

3. Selective Recurrence:
       h_t = \\bar{A}_t \\odot h_{t-1} + \\bar{B}_t \\odot x_t
       y_t = \\sum_{n=1}^{d_{state}} C_{t, n} \\cdot h_{t, n} + D \\odot x_t

4. Bidirectional Temporal Modeling for EEG Motor Imagery:
   Motor imagery exhibits distinct temporal phases:
   - Event-Related Desynchronization (ERD): \\mu / \\beta power suppression during
     motor preparation and execution (tracked by the forward scan).
   - Event-Related Synchronization (ERS): \\beta rhythm rebound post-imagery
     (captured by the backward scan).
   BidirectionalMamba scans both forward (t=1..T) and backward (t=T..1), fusing the
   latent states via an adaptive gating mechanism:
       g_t = \\sigma(W_g [y^{fwd}_t ; y^{bwd}_t])
       y^{fused}_t = g_t \\odot y^{fwd}_t + (1 - g_t) \\odot y^{bwd}_t

5. Optional Wav-KAN Projections:
   Internal linear input/output projections can be substituted with WaveletKANLayer,
   enabling non-linear, wavelet-parameterized projections before and after state space transitions.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.wav_kan import WaveletKANLayer


# ============================================================================
# Selective Scan (Pure PyTorch Implementation)
# ============================================================================

def selective_scan_sequential(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure PyTorch sequential selective scan for input-dependent SSM.

    Args:
        u: Input sequence of shape (B, L, d_inner).
        delta: Input-dependent discretization parameter of shape (B, L, d_inner).
        A: State transition matrix parameter of shape (d_inner, d_state).
           Negative real part enforced: A <= 0.
        B: Input-dependent transition matrix of shape (B, L, d_state).
        C: Input-dependent observation matrix of shape (B, L, d_state).
        D: Skip connection parameter of shape (d_inner,).

    Returns:
        y: Output sequence tensor of shape (B, L, d_inner).
    """
    B_sz, L, d_inner = u.shape
    d_state = A.shape[1]

    # Pre-scale delta * u to avoid repeated computation
    delta_u = delta * u  # (B, L, d_inner)
    A_exp = A.unsqueeze(0)  # (1, d_inner, d_state)

    # Initialize hidden state: h_0 = 0 of shape (B, d_inner, d_state)
    h = torch.zeros(B_sz, d_inner, d_state, device=u.device, dtype=u.dtype)

    ys = []
    # Time loop over sequence length L with on-the-fly discretization to minimize VRAM footprint
    for t in range(L):
        # A_bar_t = exp(delta_t * A): (B, d_inner, d_state)
        A_bar_t = torch.exp(delta[:, t].unsqueeze(-1) * A_exp)
        # B_bar_u_t = (delta_t * u_t) * B_t: (B, d_inner, d_state)
        B_bar_u_t = delta_u[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)
        # Recurrence: h_t = A_bar_t * h_{t-1} + B_bar_u_t
        h = A_bar_t * h + B_bar_u_t
        # Output: y_t = sum_n (C_{t, n} * h_{t, n}) -> (B, d_inner)
        y_t = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)
        ys.append(y_t)

    # Stack over time dimension: (B, L, d_inner)
    y = torch.stack(ys, dim=1)

    # Add D * u skip connection
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(0)

    return y


# ============================================================================
# Single Direction Mamba Block
# ============================================================================

class MambaBlock(nn.Module):
    """
    Selective State Space Model (Mamba) block for sequence modeling.

    Features:
    - O(L) time and memory complexity
    - Input-dependent state-space parameters (selective gating)
    - Depthwise causal 1D convolution for local feature extraction
    - Optional Wav-KAN internal projections for adaptive wavelet transformations

    Args:
        d_model: Input and output feature dimension.
        d_state: SSM hidden state dimension (default: 16).
        expand_factor: Inner expansion factor (default: 2, inner_dim = expand_factor * d_model).
        d_conv: 1D convolution kernel width (default: 4).
        dt_rank: Rank of \\Delta projection ('auto' sets rank to ceil(d_model / 16)).
        dt_min: Minimum initial step size \\Delta.
        dt_max: Maximum initial step size \\Delta.
        use_wavkan_projection: If True, uses WaveletKANLayer for in/out projections.
        num_wavelets: Number of wavelet basis functions if using Wav-KAN projections.
        wavelet_type: 'morlet', 'mexican_hat', etc. for Wav-KAN projections.
        dropout: Dropout rate.
        bias: Whether to use bias in linear projections.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        use_wavkan_projection: bool = False,
        num_wavelets: int = 8,
        wavelet_type: str = "morlet",
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)
        self.d_conv = d_conv
        self.use_wavkan_projection = use_wavkan_projection

        # Determine dt projection rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        # --------------------------------------------------------------------
        # 1. Input Projections: maps d_model -> 2 * d_inner (x and gate z)
        # --------------------------------------------------------------------
        if use_wavkan_projection:
            self.in_proj = WaveletKANLayer(
                in_features=d_model,
                out_features=2 * self.d_inner,
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type,
            )
        else:
            self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        # --------------------------------------------------------------------
        # 2. Local 1D Depthwise Convolution
        # --------------------------------------------------------------------
        # Processes sequences along time dimension: (B, d_inner, L)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # --------------------------------------------------------------------
        # 3. Dynamic SSM Projections (B, C, \Delta)
        # --------------------------------------------------------------------
        # x_proj projects from d_inner to (dt_rank + 2 * d_state)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)

        # dt_proj projects from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias to span log(dt_min)..log(dt_max)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Uniform sampling in log-space for dt
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        # Inverse of softplus: inv_softplus(y) = log(exp(y) - 1)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # --------------------------------------------------------------------
        # 4. State Matrix A and Skip Parameter D
        # --------------------------------------------------------------------
        # Structured HiPPO initialization: A_{d, n} = -(n + 1)
        # Log(-A) is stored to ensure A is strictly negative (stable SSM)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Skip parameter D: initialized to ones
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # --------------------------------------------------------------------
        # 5. Output Projection
        # --------------------------------------------------------------------
        if use_wavkan_projection:
            self.out_proj = WaveletKANLayer(
                in_features=self.d_inner,
                out_features=d_model,
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type,
            )
        else:
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MambaBlock.

        Args:
            u: Input sequence tensor of shape (batch, seq_len, d_model).

        Returns:
            Output sequence tensor of shape (batch, seq_len, d_model).
        """
        # u: (B, L, D)
        B, L, D = u.shape

        # 1. Project input to branch x and gate z: (B, L, 2 * d_inner)
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 2. Local 1D Convolution over time
        # Conv1d expects (B, d_inner, L)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L]  # Causal trim to length L
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_act = F.silu(x_conv)

        # 3. Compute input-dependent SSM parameters: B, C, \Delta
        # x_dbl: (B, L, dt_rank + 2 * d_state)
        x_dbl = self.x_proj(x_act)
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Discretization step delta: (B, L, d_inner)
        delta = F.softplus(self.dt_proj(dt_raw))

        # Transition matrix A: enforce negative real part A = -exp(A_log)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # 4. Selective State Space Scan
        # y_ssm: (B, L, d_inner)
        y_ssm = selective_scan_sequential(
            u=x_act,
            delta=delta,
            A=A,
            B=B_ssm,
            C=C_ssm,
            D=self.D,
        )

        # 5. Gating with branch z
        y_gated = y_ssm * F.silu(z)

        # 6. Output projection
        out = self.out_proj(y_gated)
        out = self.dropout(out)
        return out


# ============================================================================
# Bidirectional Mamba Layer
# ============================================================================

class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba: executes forward + backward selective scans
    and adaptively fuses them via gated projection.

    Physiological Motivation:
    - Forward scan tracks ERD onset (suppression of mu/beta motor rhythms).
    - Backward scan tracks ERS rebound (burst of post-movement beta power).
    - Adaptive gating learns per-time-step weighting between forward and backward context.

    Input:
        (batch, seq_len, d_model)
    Output:
        (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        use_wavkan_projection: bool = False,
        num_wavelets: int = 8,
        wavelet_type: str = "morlet",
        fusion_type: str = "gated",
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize Bidirectional Mamba.

        Args:
            d_model: Feature dimension.
            d_state: SSM state dimension.
            expand_factor: Inner expansion factor.
            d_conv: 1D convolution kernel width.
            use_wavkan_projection: Whether to use Wav-KAN for internal projections.
            num_wavelets: Number of wavelets if use_wavkan_projection is True.
            wavelet_type: Wavelet basis type.
            fusion_type: 'gated' (adaptive soft gating) or 'concat_linear' (projection).
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion_type

        # Pre-normalization layer
        self.norm = nn.LayerNorm(d_model)

        # Forward Mamba scan
        self.mamba_fwd = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            use_wavkan_projection=use_wavkan_projection,
            num_wavelets=num_wavelets,
            wavelet_type=wavelet_type,
            dropout=dropout,
        )

        # Backward Mamba scan
        self.mamba_bwd = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            use_wavkan_projection=use_wavkan_projection,
            num_wavelets=num_wavelets,
            wavelet_type=wavelet_type,
            dropout=dropout,
        )

        # Fusion mechanism
        if fusion_type == "gated":
            # Learns gate g in [0, 1] per channel: y = g * fwd + (1 - g) * bwd
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid(),
            )
        else:
            # Linear projection of concatenated [fwd, bwd]
            self.proj = nn.Linear(2 * d_model, d_model)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of BidirectionalMamba with residual connection.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Residual shortcut
        residual = x

        # Pre-norm
        h = self.norm(x)

        # Forward scan (t = 1 .. L)
        y_fwd = self.mamba_fwd(h)

        # Backward scan: flip along time dimension L (dim 1), run mamba, flip back
        h_bwd = torch.flip(h, dims=[1])
        y_bwd_flipped = self.mamba_bwd(h_bwd)
        y_bwd = torch.flip(y_bwd_flipped, dims=[1])

        # Directional fusion
        concat_states = torch.cat([y_fwd, y_bwd], dim=-1)  # (B, L, 2 * d_model)

        if self.fusion_type == "gated":
            g = self.gate(concat_states)  # (B, L, d_model)
            fused = g * y_fwd + (1.0 - g) * y_bwd
        else:
            fused = self.proj(concat_states)

        fused = self.dropout(fused)

        # Residual connection
        return residual + fused
