"""
Wavelet Kolmogorov-Arnold Network (Wav-KAN) for EEG Signal Processing.

This module implements the Wavelet KAN architecture where edge activation functions
are parameterized by learnable wavelet transforms rather than standard fixed
activations (like ReLU/GELU) or B-splines.

Mathematical Formulation:
-------------------------
In Kolmogorov-Arnold Networks, a multivariate function is represented as:
    f(x_1, ..., x_n) = \\sum_{q=1}^{2n+1} \\Phi_q \\left( \\sum_{p=1}^n \\phi_{q,p}(x_p) \\right)

In Wav-KAN, each 1D edge activation \\phi_{j,i}(x_i) is expanded into a weighted
superposition of wavelet basis functions:
    \\phi_{j,i}(x) = w_{base, j, i} \\cdot b(x) + \\sum_{k=1}^K w_{j,i,k} \\cdot \\psi\\left(\\frac{x - b_{i,k}}{a_{i,k}}\\right)

where:
    - \\psi(u) is a mother wavelet (Morlet, Mexican Hat, Shannon, or B-spline)
    - a_{i,k} \\in \\mathbb{R}^+ is the learnable scale (dilation) parameter
    - b_{i,k} \\in \\mathbb{R} is the learnable shift (translation) parameter
    - w_{j,i,k} \\in \\mathbb{R} is the learnable expansion weight
    - b(x) is a base residual activation (typically SiLU) with learnable weight w_{base}

For EEG Motor Imagery analysis, scales a_{i,k} are initialized to cover the
physiological \\mu rhythm (8-12 Hz) and \\beta rhythm (16-30 Hz) bands, allowing
the network to perform adaptive, subject-specific time-frequency decomposition.
"""

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Wavelet Basis Functions
# ============================================================================

def morlet_wavelet(u: torch.Tensor, omega0: float = 5.0) -> torch.Tensor:
    """
    Real Morlet (Gabor) wavelet basis:
        \\psi(u) = \\cos(\\omega_0 u) \\cdot \\exp(-u^2 / 2)

    Args:
        u: Normalized input coordinate (x - b) / a.
        omega0: Central angular frequency of the mother wavelet (default: 5.0,
                satisfying the admissibility condition exp(-omega0^2/2) ~ 0).

    Returns:
        Evaluated Morlet wavelet tensor of the same shape as u.
    """
    return torch.cos(omega0 * u) * torch.exp(-0.5 * (u ** 2))


def mexican_hat_wavelet(u: torch.Tensor) -> torch.Tensor:
    """
    Mexican Hat (Ricker) wavelet basis:
        \\psi(u) = \\frac{2}{\\sqrt{3} \\pi^{1/4}} (1 - u^2) \\exp(-u^2 / 2)

    Second derivative of the Gaussian function, zero-mean, highly localized.

    Args:
        u: Normalized input coordinate (x - b) / a.

    Returns:
        Evaluated Mexican Hat wavelet tensor of same shape as u.
    """
    # 2 / (sqrt(3) * pi^(1/4)) approx 0.8673250705840776
    c = 2.0 / (math.sqrt(3.0) * (math.pi ** 0.25))
    return c * (1.0 - (u ** 2)) * torch.exp(-0.5 * (u ** 2))


def shannon_wavelet(u: torch.Tensor) -> torch.Tensor:
    """
    Shannon (bandpass sinc) wavelet basis:
        \\psi(u) = \\text{sinc}(u) \\cdot \\cos(3\\pi u)

    Provides ideal rectangular frequency bandpass characteristics.

    Args:
        u: Normalized input coordinate (x - b) / a.

    Returns:
        Evaluated Shannon wavelet tensor of same shape as u.
    """
    # torch.sinc computes sin(pi * x) / (pi * x)
    return torch.sinc(u) * torch.cos(3.0 * math.pi * u)


def bspline_basis(
    x: torch.Tensor,
    grid: torch.Tensor,
    spline_order: int = 3,
) -> torch.Tensor:
    """
    Evaluates uniform B-spline basis functions using Cox-de Boor recursion.

    Args:
        x: Input tensor of shape (..., in_features).
        grid: Knot grid tensor of shape (in_features, num_knots).
        spline_order: Polynomial spline order (default: 3 for cubic splines).

    Returns:
        Spline basis tensor of shape (..., in_features, num_bases).
    """
    # Reshape x to (..., in_features, 1)
    x_expanded = x.unsqueeze(-1)
    # Grid: (in_features, num_knots)
    # 0-th order B-spline: indicator function on [t_i, t_{i+1})
    grid_left = grid[:, :-1]
    grid_right = grid[:, 1:]
    
    # Shape: (..., in_features, num_knots - 1)
    bases = ((x_expanded >= grid_left) & (x_expanded < grid_right)).to(x.dtype)
    
    # Recursive Cox-de Boor formula:
    # B_{i, k}(x) = (x - t_i) / (t_{i+k} - t_i) * B_{i, k-1}(x)
    #             + (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1, k-1}(x)
    for k in range(1, spline_order + 1):
        left_denom = grid[:, k:-1] - grid[:, :-(k + 1)]
        right_denom = grid[:, (k + 1):] - grid[:, 1:-k]
        
        # Avoid division by zero on degenerate knots
        left_factor = torch.where(
            left_denom > 1e-7,
            (x_expanded - grid[:, :-(k + 1)]) / left_denom,
            torch.zeros_like(x_expanded),
        )
        right_factor = torch.where(
            right_denom > 1e-7,
            (grid[:, (k + 1):] - x_expanded) / right_denom,
            torch.zeros_like(x_expanded),
        )
        
        bases = left_factor * bases[..., :-1] + right_factor * bases[..., 1:]
        
    return bases


# ============================================================================
# Wavelet KAN Layer
# ============================================================================

class WaveletKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer with learnable wavelet edge activations.

    Instead of fixed activation functions on nodes (like ReLU in MLPs),
    KAN places learnable 1D functions on edges:
        \\phi_{j,i}(x) = w_{b, j, i} \\cdot b(x) + \\sum_k w_{j,i,k} \\cdot \\psi\\left(\\frac{x - b_{i,k}}{a_{i,k}}\\right)

    where \\psi is a wavelet basis function (Morlet, Mexican Hat, or Shannon),
    a_{i,k} is the learnable scale, b_{i,k} is the learnable shift.

    This makes each synapse an adaptive wavelet transform that can
    automatically discover subject-specific EEG rhythm frequencies (e.g., \\mu, \\beta).

    Attributes:
        in_features: Number of input dimensions.
        out_features: Number of output dimensions.
        num_wavelets: Number of wavelet basis functions per input dimension.
        wavelet_type: 'morlet', 'mexican_hat', 'shannon', or 'bspline'.
        scale: Learnable dilation parameter tensor of shape (in_features, num_wavelets).
        shift: Learnable translation parameter tensor of shape (in_features, num_wavelets).
        wavelet_weight: Edge weights tensor of shape (out_features, in_features, num_wavelets).
        base_weight: Residual linear/base weight tensor of shape (out_features, in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_wavelets: int = 16,
        wavelet_type: str = "morlet",
        learnable_scale: bool = True,
        learnable_shift: bool = True,
        scale_init: str = "eeg_rhythms",
        sampling_rate: float = 250.0,
        include_base_weight: bool = True,
        base_activation: Optional[Callable] = None,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the WaveletKANLayer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            num_wavelets: Number of wavelet basis functions (grid size) per edge.
            wavelet_type: Type of wavelet basis ('morlet', 'mexican_hat', 'shannon', 'bspline').
            learnable_scale: Whether dilation parameters a_{i,k} are learnable.
            learnable_shift: Whether translation parameters b_{i,k} are learnable.
            scale_init: Initialization scheme for scales ('eeg_rhythms', 'log_uniform', or 'standard').
            sampling_rate: Nominal EEG sampling rate in Hz (used when scale_init='eeg_rhythms').
            include_base_weight: Whether to include the base residual path w_b * b(x).
            base_activation: Residual base activation function (default: nn.SiLU()).
            dropout: Dropout probability applied to wavelet activations.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type.lower()
        self.learnable_scale = learnable_scale
        self.learnable_shift = learnable_shift
        self.scale_init = scale_init
        self.sampling_rate = sampling_rate
        self.include_base_weight = include_base_weight
        self.base_activation = base_activation if base_activation is not None else nn.SiLU()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Validate wavelet type
        valid_types = {"morlet", "mexican_hat", "shannon", "bspline"}
        if self.wavelet_type not in valid_types:
            raise ValueError(f"Unknown wavelet_type: {self.wavelet_type}. Expected one of {valid_types}.")

        # --------------------------------------------------------------------
        # Initialize Wavelet Scales (a_k) and Shifts (b_k)
        # --------------------------------------------------------------------
        if self.wavelet_type == "bspline":
            # For B-spline, construct knot vector
            # Spline order 3 -> num_wavelets bases requires num_wavelets + order + 1 knots
            spline_order = 3
            self.spline_order = spline_order
            num_knots = self.num_wavelets + spline_order + 1
            # Grid spaced over [-2, 2]
            knot_grid = torch.linspace(-2.0, 2.0, steps=num_knots).unsqueeze(0).repeat(in_features, 1)
            self.register_buffer("knot_grid", knot_grid)
            self.scale = None
            self.shift = None
        else:
            # Shifts b_k: uniformly span the dynamic range [-2.0, 2.0]
            # Shape: (in_features, num_wavelets)
            shifts = torch.linspace(-2.0, 2.0, steps=num_wavelets).unsqueeze(0).repeat(in_features, 1)
            if learnable_shift:
                self.shift = nn.Parameter(shifts)
            else:
                self.register_buffer("shift", shifts)

            # Scales a_k: initialize according to desired frequency coverage
            if self.scale_init == "eeg_rhythms":
                # Cover mu (8-12 Hz) and beta (16-30 Hz), extended to [4 Hz, 40 Hz]
                # In wavelet theory, f = f_0 * f_s / a where f_0 is the mother wavelet central freq.
                # For Morlet (omega0=5), f_0 = omega0 / (2*pi) ~ 0.796 Hz.
                # Scale a = (f_0 * f_s) / f
                f0 = 5.0 / (2.0 * math.pi) if self.wavelet_type == "morlet" else 0.25
                target_freqs = torch.logspace(
                    math.log10(4.0), math.log10(40.0), steps=num_wavelets
                )
                scales_init = (f0 * self.sampling_rate) / target_freqs
                # Normalize scales to standard neural activation range [0.5, 3.0]
                scales_init = scales_init / scales_init.mean() * 1.5
                scales = scales_init.unsqueeze(0).repeat(in_features, 1)
            elif self.scale_init == "log_uniform":
                scales = torch.logspace(-0.5, 0.8, steps=num_wavelets).unsqueeze(0).repeat(in_features, 1)
            else:
                # Standard initialization: uniform scale ~ 1.0
                scales = torch.ones(in_features, num_wavelets)

            if learnable_scale:
                self.scale = nn.Parameter(scales)
            else:
                self.register_buffer("scale", scales)

        # --------------------------------------------------------------------
        # Learnable Expansion Weights w_{j, i, k}
        # --------------------------------------------------------------------
        # Shape: (out_features, in_features, num_wavelets)
        self.wavelet_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.num_wavelets)
        )

        # Base residual weight: w_b * b(x)
        if self.include_base_weight:
            self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("base_weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with scaled normal distribution."""
        # Kaiming-style initialization scaled by number of wavelet basis functions
        fan_in = self.in_features * self.num_wavelets
        std = 1.0 / math.sqrt(fan_in)
        nn.init.normal_(self.wavelet_weight, mean=0.0, std=std)

        if self.base_weight is not None:
            nn.init.xavier_uniform_(self.base_weight)

    def _evaluate_wavelet(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate wavelet basis functions for given input.

        Args:
            x: Input tensor with shape (..., in_features).

        Returns:
            Wavelet activations tensor of shape (..., in_features, num_wavelets).
        """
        if self.wavelet_type == "bspline":
            return bspline_basis(x, self.knot_grid, self.spline_order)

        # Expand x: (..., in_features, 1)
        x_exp = x.unsqueeze(-1)
        
        # Avoid division by zero: clamp scale to positive values
        scale = torch.abs(self.scale) + 1e-5
        shift = self.shift

        # Compute normalized coordinate u = (x - b) / a
        u = (x_exp - shift) / scale

        # Dispatch to mother wavelet
        if self.wavelet_type == "morlet":
            return morlet_wavelet(u)
        elif self.wavelet_type == "mexican_hat":
            return mexican_hat_wavelet(u)
        elif self.wavelet_type == "shannon":
            return shannon_wavelet(u)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for WaveletKANLayer.

        Args:
            x: Input tensor of shape (..., in_features).
               Supports 2D (batch, in_features) or 3D (batch, seq_len, in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # Save batch shape for arbitrary leading dimensions
        original_shape = x.shape[:-1]
        in_dim = x.shape[-1]
        if in_dim != self.in_features:
            raise ValueError(
                f"Input last dimension {in_dim} does not match layer in_features {self.in_features}."
            )

        # Flatten leading dimensions to batch: (B_flat, in_features)
        x_flat = x.reshape(-1, self.in_features)

        # 1. Base residual activation: w_base * b(x)
        # x_flat: (B_flat, in_features) -> base_out: (B_flat, out_features)
        if self.base_weight is not None:
            base_act = self.base_activation(x_flat)
            base_out = F.linear(base_act, self.base_weight)
        else:
            base_out = 0.0

        # 2. Wavelet transform on edges:
        # psi_val: (B_flat, in_features, num_wavelets)
        psi_val = self._evaluate_wavelet(x_flat)
        psi_val = self.dropout(psi_val)

        # 3. Weighted contraction over synapses:
        # psi_val: (B_flat, in_features, num_wavelets)
        # wavelet_weight: (out_features, in_features, num_wavelets)
        # out: (B_flat, out_features)
        wavelet_out = torch.einsum("bik, oik -> bo", psi_val, self.wavelet_weight)

        # 4. Total output = base + wavelet
        out = base_out + wavelet_out

        # Restore original leading dimensions
        return out.reshape(*original_shape, self.out_features)

    def extra_repr(self) -> str:
        """String representation of layer hyperparameters."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"num_wavelets={self.num_wavelets}, wavelet_type={self.wavelet_type}, "
            f"learnable_scale={self.learnable_scale}, learnable_shift={self.learnable_shift}"
        )


# ============================================================================
# B-Spline KAN Fallback Layer
# ============================================================================

class BSplineKANLayer(nn.Module):
    """
    Standard B-spline KAN layer as a simpler baseline/fallback.

    Evaluates cubic B-spline basis functions over a uniform grid with learnable
    spline coefficients and a residual SiLU path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-2.0, 2.0),
        base_activation: Optional[Callable] = None,
    ) -> None:
        """
        Initialize standard B-Spline KAN layer.

        Args:
            in_features: Input dimensionality.
            out_features: Output dimensionality.
            grid_size: Number of spline grid intervals (produces grid_size + spline_order basis functions).
            spline_order: Spline polynomial degree (default: 3 for cubic splines).
            grid_range: Domain interval [min_val, max_val] for the knot grid.
            base_activation: Residual activation (default: nn.SiLU()).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation if base_activation is not None else nn.SiLU()

        # Total number of basis functions
        self.num_bases = grid_size + spline_order

        # Knot vector calculation with clamped extensions
        # Grid step
        h = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            steps=grid_size + 2 * spline_order + 1,
        )
        # Register grid as buffer (shape: in_features, num_knots)
        self.register_buffer("knot_grid", knots.unsqueeze(0).repeat(in_features, 1))

        # Learnable parameters
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.num_bases)
        )
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights."""
        fan_in = self.in_features * self.num_bases
        nn.init.normal_(self.spline_weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for BSplineKANLayer.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Base path
        base_out = F.linear(self.base_activation(x_flat), self.base_weight)

        # Spline bases: (B_flat, in_features, num_bases)
        bases = bspline_basis(x_flat, self.knot_grid, self.spline_order)

        # Spline path contraction
        spline_out = torch.einsum("bik, oik -> bo", bases, self.spline_weight)

        out = base_out + spline_out
        return out.reshape(*original_shape, self.out_features)


# ============================================================================
# Multi-Layer Wav-KAN Sequential Network
# ============================================================================

class WavKAN(nn.Module):
    """
    Multi-layer Wavelet Kolmogorov-Arnold Network.

    Stacks multiple WaveletKANLayers with optional LayerNorm, residual
    connections, and dropout for use as a classifier head or feature extractor.
    """

    def __init__(
        self,
        layers_hidden: list[int],
        num_wavelets: int = 16,
        wavelet_type: str = "morlet",
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ) -> None:
        """
        Args:
            layers_hidden: List of layer dimensions, e.g. [in_dim, hidden_dim, out_dim].
            num_wavelets: Number of wavelet basis functions per edge.
            wavelet_type: 'morlet', 'mexican_hat', 'shannon', or 'bspline'.
            dropout: Dropout probability between layers.
            use_layernorm: Whether to apply LayerNorm between layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layernorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        for i in range(len(layers_hidden) - 1):
            in_dim = layers_hidden[i]
            out_dim = layers_hidden[i + 1]
            self.layers.append(
                WaveletKANLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_wavelets=num_wavelets,
                    wavelet_type=wavelet_type,
                    dropout=dropout,
                )
            )
            if use_layernorm and i < len(layers_hidden) - 2:
                self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequential Wav-KAN layers."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None and i < len(self.norms):
                x = self.norms[i](x)
                x = self.dropout(x)
        return x
