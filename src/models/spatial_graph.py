"""
Dynamic Spatial Graph Encoder for EEG Channel Topology.

This module models the 3D physical and functional spatial relationships between
EEG electrodes using Chebyshev spectral graph convolution.

Mathematical Formulation:
-------------------------
Let G = (V, E, W) be an undirected graph where V = {v_1, ..., v_C} represents
the C scalp electrodes, and W \\in \\mathbb{R}^{C \\times C} is the spatial adjacency matrix.

1. Scalp Coordinates and Distance:
   Electrode positions p_i = (x_i, y_i, z_i) \\in \\mathbb{R}^3 are obtained from
   the international standard 10-10 / 10-20 system (via MNE).
   Euclidean distance: d_{ij} = ||p_i - p_j||_2.

2. Static Adjacency Matrix (Gaussian RBF Kernel):
   W^{fixed}_{ij} = \\exp\\left( -\\frac{d_{ij}^2}{2\\sigma^2} \\right) \\cdot \\mathbb{I}(d_{ij} \\le \\tau)
   where \\sigma is the distance scale and \\tau is the spatial neighborhood threshold.

3. Dynamic / Learnable Adjacency:
   Scalp conduction is not strictly isotropic and functional connectivity varies
   dynamically during motor imagery. We define the dynamic adjacency:
   A_{dyn} = \\alpha \\cdot A_{fixed} + (1 - \\alpha) \\cdot \\text{Sigmoid}(M_{learnable})
   where \\alpha \\in [0, 1] is a learnable gating parameter and M_{learnable}
   is a symmetric learnable edge parameter matrix.

4. Normalized Laplacian and Chebyshev Polynomials:
   L = I - D^{-1/2} A D^{-1/2}
   Rescaled Laplacian: \\tilde{L} = \\frac{2}{\\lambda_{max}} L - I \\approx L - I = -D^{-1/2} A D^{-1/2}
   Recursive Chebyshev polynomials (Defferrard et al., 2016):
   T_0(\\tilde{L}) = I
   T_1(\\tilde{L}) = \\tilde{L}
   T_k(\\tilde{L}) = 2 \\tilde{L} T_{k-1}(\\tilde{L}) - T_{k-2}(\\tilde{L}), \\quad k \\ge 2

5. Chebyshev Graph Convolution:
   y = \\sum_{k=0}^{K-1} T_k(\\tilde{L}) X \\Theta_k
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


# ============================================================================
# Electrode Positions and Adjacency Matrix Utilities
# ============================================================================

def clean_channel_name(ch_name: str) -> str:
    """
    Standardize EEG channel name by removing dots, spaces, and converting to upper case.
    
    Examples:
        'Fc5.' -> 'FC5'
        'c3..' -> 'C3'
        'Fpz' -> 'FPZ'
    """
    return ch_name.strip(" .-_").upper()


def get_electrode_positions(
    ch_names: List[str],
    montage_name: str = "standard_1005",
) -> np.ndarray:
    """
    Extract 3D Cartesian coordinates (x, y, z in meters) for the given electrode names
    using MNE standard montages.

    Args:
        ch_names: List of channel names (e.g. ['FC5.', 'FC3.', ..., 'Oz.']).
        montage_name: MNE standard montage name ('standard_1005', 'standard_1020').

    Returns:
        positions: Array of shape (len(ch_names), 3) with (x, y, z) coordinates.
    """
    if not HAS_MNE:
        raise ImportError("MNE-Python is required for get_electrode_positions. Please install mne.")

    montage = mne.channels.make_standard_montage(montage_name)
    montage_pos = montage.get_positions()["ch_pos"]
    # Build uppercase lookup table
    pos_lookup = {k.upper(): v for k, v in montage_pos.items()}

    positions = []
    missing_channels = []

    for idx, ch in enumerate(ch_names):
        cleaned = clean_channel_name(ch)
        if cleaned in pos_lookup:
            positions.append(pos_lookup[cleaned])
        else:
            missing_channels.append((idx, ch, cleaned))
            # Fallback: estimate position on unit sphere or default coordinate
            positions.append(np.array([0.0, 0.0, 0.0], dtype=np.float64))

    positions_arr = np.array(positions, dtype=np.float32)

    if missing_channels:
        # Interpolate or fill missing coordinates uniformly
        for idx, orig, cln in missing_channels:
            theta = 2.0 * np.pi * idx / len(ch_names)
            positions_arr[idx] = np.array([0.085 * np.cos(theta), 0.085 * np.sin(theta), 0.0], dtype=np.float32)

    return positions_arr


def build_eeg_adjacency_matrix(
    positions: np.ndarray,
    sigma: Optional[float] = None,
    threshold: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    add_self_loops: bool = False,
) -> torch.Tensor:
    """
    Construct spatial adjacency matrix from 3D electrode coordinates.

    Uses Gaussian RBF kernel over Euclidean distance:
        W_{ij} = \\exp(-d_{ij}^2 / (2\\sigma^2))

    Args:
        positions: 3D coordinates array of shape (num_channels, 3).
        sigma: Gaussian bandwidth parameter in meters. If None, set to median distance.
        threshold: Distance cutoff in meters. Edges with distance > threshold are pruned.
        k_neighbors: If specified, only retain top-k nearest neighbors per channel.
        add_self_loops: Whether to include self-loops (diagonal = 1).

    Returns:
        adj: Symmetric adjacency tensor of shape (num_channels, num_channels).
    """
    n_nodes = positions.shape[0]
    pos = torch.from_numpy(positions).float()

    # Pairwise Euclidean distances: (C, C)
    dist_sq = torch.cdist(pos, pos, p=2.0) ** 2
    dist = torch.sqrt(dist_sq + 1e-12)

    # Automatic sigma calculation if not provided
    if sigma is None:
        mask = ~torch.eye(n_nodes, dtype=torch.bool)
        sigma = float(torch.median(dist[mask]))
        if sigma <= 0.0:
            sigma = 0.05  # Default 5 cm

    # Gaussian kernel
    adj = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))

    # Apply distance threshold
    if threshold is not None:
        adj = torch.where(dist <= threshold, adj, torch.zeros_like(adj))

    # Apply k-NN sparsification if requested
    if k_neighbors is not None and k_neighbors < n_nodes:
        # For each node, keep only top k nearest neighbors
        topk_vals, topk_indices = torch.topk(adj, k=k_neighbors + 1, dim=-1)
        knn_mask = torch.zeros_like(adj, dtype=torch.bool)
        knn_mask.scatter_(1, topk_indices, True)
        # Symmetrize k-NN mask
        knn_mask = knn_mask | knn_mask.t()
        adj = torch.where(knn_mask, adj, torch.zeros_like(adj))

    # Remove self-loops or set diagonal
    if not add_self_loops:
        adj.fill_diagonal_(0.0)
    else:
        adj.fill_diagonal_(1.0)

    # Ensure exact symmetry
    adj = 0.5 * (adj + adj.t())
    return adj


def compute_chebyshev_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled Chebyshev Laplacian \\tilde{L} from adjacency matrix.

    \\tilde{L} = 2 / \\lambda_{max} * L - I \\approx L - I = -D^{-1/2} A D^{-1/2}

    Args:
        adj: Adjacency matrix of shape (C, C) or (B, C, C).

    Returns:
        cheb_laplacian: Scaled Laplacian tensor of shape (C, C) or (B, C, C).
    """
    # Degree matrix D: d_i = sum_j A_{ij}
    degree = torch.sum(adj, dim=-1)  # (..., C)
    d_inv_sqrt = torch.pow(degree + 1e-6, -0.5)

    if adj.dim() == 2:
        d_mat = torch.diag(d_inv_sqrt)
        norm_adj = d_mat @ adj @ d_mat
        eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
        # L = I - norm_adj, rescaled: \tilde{L} = (2/2)*L - I = -norm_adj
        return -norm_adj
    else:
        # Batched computation: (B, C, C)
        d_mat = torch.diag_embed(d_inv_sqrt)
        norm_adj = d_mat @ adj @ d_mat
        return -norm_adj


# ============================================================================
# Chebyshev Spectral Graph Convolution Layer
# ============================================================================

class ChebConv(nn.Module):
    """
    Chebyshev spectral graph convolution layer for EEG channel processing.

    Evaluates Chebyshev polynomial expansion up to order K:
        y = \\sum_{k=0}^{K-1} T_k(\\tilde{L}) X \\Theta_k + \\text{bias}

    Supports dynamic/learnable edge weights and batched operations.

    Args:
        in_channels: Number of input features per electrode node.
        out_channels: Number of output features per electrode node.
        K: Chebyshev polynomial order (typically K=3).
        bias: Whether to add a learnable bias vector.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        # Learnable Chebyshev polynomial weight filters \\Theta_k for k in 0..K-1
        # Shape: (K, in_channels, out_channels)
        self.weight = nn.Parameter(torch.empty(K, in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize Chebyshev weights using Xavier uniform initialization."""
        for k in range(self.K):
            nn.init.xavier_uniform_(self.weight[k])
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for ChebConv.

        Args:
            x: Node feature tensor.
               Shape: (B, num_nodes, in_channels) or (B, num_nodes, in_channels, T).
            laplacian: Scaled Chebyshev Laplacian tensor \\tilde{L}.
               Shape: (num_nodes, num_nodes) or (B, num_nodes, num_nodes).

        Returns:
            Output feature tensor of shape (B, num_nodes, out_channels)
            or (B, num_nodes, out_channels, T).
        """
        has_time_dim = (x.dim() == 4)
        if has_time_dim:
            # Shape: (B, C, in_channels, T) -> combine in_channels & T for linear projection
            # We can treat each time step or permute: (B, C, in_channels, T)
            B, C, F_in, T = x.shape
            # Reshape x to (B, C, F_in * T)
            x_reshaped = x.reshape(B, C, F_in * T)
        else:
            # Shape: (B, C, F_in)
            B, C, F_in = x.shape
            x_reshaped = x

        # --------------------------------------------------------------------
        # Recursive Chebyshev Polynomial Evaluation:
        # T_0(L) X = X
        # T_1(L) X = L X
        # T_k(L) X = 2 L T_{k-1}(L) X - T_{k-2}(L) X
        # --------------------------------------------------------------------
        # T_list stores [T_0 X, T_1 X, ..., T_{K-1} X]
        # Each element shape: (B, C, F_in * T) or (B, C, F_in)
        T_list = []
        
        # T_0 X = X
        T_0 = x_reshaped
        T_list.append(T_0)

        if self.K > 1:
            # T_1 X = \tilde{L} X
            if laplacian.dim() == 2:
                # laplacian: (C, C), x: (B, C, feat) -> einsum 'ij, bjf -> bif'
                T_1 = torch.einsum("ij, bjf -> bif", laplacian, x_reshaped)
            else:
                # laplacian: (B, C, C)
                T_1 = torch.bmm(laplacian, x_reshaped)
            T_list.append(T_1)

        for k in range(2, self.K):
            if laplacian.dim() == 2:
                L_prev = torch.einsum("ij, bjf -> bif", laplacian, T_list[k - 1])
            else:
                L_prev = torch.bmm(laplacian, T_list[k - 1])
            T_k = 2.0 * L_prev - T_list[k - 2]
            T_list.append(T_k)

        # --------------------------------------------------------------------
        # Linear Projections and Accumulation:
        # y = sum_{k=0}^{K-1} (T_k X) \Theta_k
        # --------------------------------------------------------------------
        out = 0.0
        for k in range(self.K):
            T_k = T_list[k]
            if has_time_dim:
                # T_k: (B, C, F_in * T) -> reshape to (B, C, F_in, T)
                T_k_unflat = T_k.reshape(B, C, F_in, T)
                # Permute for linear transform: (B, C, T, F_in) @ \Theta_k -> (B, C, T, F_out)
                proj = torch.einsum("bcft, fo -> bcot", T_k_unflat, self.weight[k])
            else:
                # T_k: (B, C, F_in) @ \Theta_k -> (B, C, F_out)
                proj = torch.einsum("bcf, fo -> bco", T_k, self.weight[k])
            out = out + proj

        if self.bias is not None:
            if has_time_dim:
                out = out + self.bias.view(1, 1, self.out_channels, 1)
            else:
                out = out + self.bias.view(1, 1, self.out_channels)

        return out


# ============================================================================
# Dynamic Spatial Graph Encoder
# ============================================================================

class SpatialGraphEncoder(nn.Module):
    """
    Dynamic Spatial Graph Encoder for EEG Motor Imagery.

    Encodes EEG channel relationships using graph convolutions across electrodes.
    Combines physical scalp 3D coordinates (fixed geometric prior) with a
    learnable adaptive functional adjacency matrix.

    Input:
        (batch, channels, timepoints) EEG tensor
    Output:
        (batch, hidden_dim, timepoints) spatially filtered representation
        or (batch, channels, hidden_dim, timepoints).

    Attributes:
        fixed_adj: Fixed distance-based adjacency from 10-10 scalp positions.
        learnable_adj: Learnable parameter matrix for adaptive functional connectivity.
        alpha: Gating parameter balancing geometric prior vs data-driven connectivity.
        layers: Sequential Chebyshev graph convolution layers.
    """

    def __init__(
        self,
        num_channels: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        cheb_k: int = 3,
        electrode_positions: Optional[np.ndarray] = None,
        ch_names: Optional[List[str]] = None,
        learnable_edges: bool = True,
        dropout: float = 0.3,
        pool_channels: bool = True,
    ) -> None:
        """
        Initialize the Dynamic Spatial Graph Encoder.

        Args:
            num_channels: Number of EEG electrode channels.
            hidden_dim: Hidden feature dimension for spatial filtering.
            num_layers: Number of stacked Chebyshev graph convolution layers.
            cheb_k: Order of Chebyshev polynomials (default: 3).
            electrode_positions: Optional (num_channels, 3) 3D coordinate array.
            ch_names: Optional list of channel names to derive 3D positions via MNE.
            learnable_edges: Whether to learn dynamic edge connectivity.
            dropout: Spatial dropout probability.
            pool_channels: If True, aggregates across channels to output (B, hidden_dim, T).
        """
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.learnable_edges = learnable_edges
        self.pool_channels = pool_channels

        # --------------------------------------------------------------------
        # 1. Scalp Geometry & Adjacency Initialization
        # --------------------------------------------------------------------
        if electrode_positions is None:
            if ch_names is not None and HAS_MNE:
                electrode_positions = get_electrode_positions(ch_names)
            else:
                # Default synthetic spherical positions for num_channels
                indices = np.arange(0, num_channels, dtype=float) + 0.5
                phi = np.arccos(1 - 2 * indices / num_channels)
                theta = np.pi * (1 + 5 ** 0.5) * indices
                x = 0.085 * np.cos(theta) * np.sin(phi)
                y = 0.085 * np.sin(theta) * np.sin(phi)
                z = 0.085 * np.cos(phi)
                electrode_positions = np.stack([x, y, z], axis=1)

        fixed_adj = build_eeg_adjacency_matrix(
            electrode_positions,
            sigma=None,
            threshold=0.10,  # 10 cm cutoff
            add_self_loops=False,
        )
        self.register_buffer("fixed_adj", fixed_adj)

        # Dynamic learnable adjacency parameter
        if learnable_edges:
            # Initialize with small symmetric random noise around 0.5
            init_edge = torch.randn(num_channels, num_channels) * 0.02
            init_edge = 0.5 * (init_edge + init_edge.t())
            self.learnable_adj_param = nn.Parameter(init_edge)
            # Learnable balance gate (initialized to ~0.7 favor towards geometric prior)
            self.alpha = nn.Parameter(torch.tensor(0.7))
        else:
            self.register_parameter("learnable_adj_param", None)
            self.register_parameter("alpha", None)

        # --------------------------------------------------------------------
        # 2. Graph Convolution Layers
        # --------------------------------------------------------------------
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer maps 1 input channel (raw signal amplitude) to hidden_dim
        in_dim = 1
        for i in range(num_layers):
            out_dim = hidden_dim
            self.gcn_layers.append(
                ChebConv(in_channels=in_dim, out_channels=out_dim, K=cheb_k)
            )
            self.batch_norms.append(nn.BatchNorm2d(out_dim))
            self.dropouts.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
            in_dim = out_dim

        # Channel pooling / aggregation projection
        if pool_channels:
            # Learnable spatial attention weights across channels
            self.channel_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # Final 1x1 spatial projection layer
            self.spatial_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def get_current_adjacency(self) -> torch.Tensor:
        """
        Compute effective dynamic adjacency matrix by blending geometric prior
        and learnable parameters.
        """
        if not self.learnable_edges:
            return self.fixed_adj

        # Clamp alpha to [0, 1]
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        # Symmetrize learnable matrix
        learn_adj = 0.5 * (self.learnable_adj_param + self.learnable_adj_param.t())
        learn_adj = torch.sigmoid(learn_adj)
        # Out-of-place zero diagonal to avoid autograd in-place modification
        diag_mask = ~torch.eye(self.num_channels, dtype=torch.bool, device=learn_adj.device)
        learn_adj = learn_adj * diag_mask

        # Blend
        adj = alpha * self.fixed_adj + (1.0 - alpha) * learn_adj
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for SpatialGraphEncoder.

        Args:
            x: Raw EEG tensor of shape (batch, num_channels, timepoints).

        Returns:
            If pool_channels=True:
                Output tensor of shape (batch, hidden_dim, timepoints).
            If pool_channels=False:
                Output tensor of shape (batch, num_channels, hidden_dim, timepoints).
        """
        # Shape: (B, C, T)
        B, C, T = x.shape

        # Expand channel feature dimension: (B, C, 1, T)
        # Here node feature is 1D scalar at each time step
        h = x.unsqueeze(2)  # (B, C, 1, T)

        # Dynamic scaled Laplacian
        adj = self.get_current_adjacency()
        laplacian = compute_chebyshev_laplacian(adj)

        for i in range(self.num_layers):
            # ChebConv forward: (B, C, F_in, T) -> (B, C, F_out, T)
            h = self.gcn_layers[i](h, laplacian)
            
            # BatchNorm expects (B, Channels, Height, Width) -> permute to (B, F_out, C, T)
            h = h.permute(0, 2, 1, 3).contiguous()
            h = self.batch_norms[i](h)
            h = F.gelu(h)
            h = self.dropouts[i](h)
            # Permute back to (B, C, F_out, T)
            h = h.permute(0, 2, 1, 3).contiguous()

        if not self.pool_channels:
            return h

        # --------------------------------------------------------------------
        # Channel Aggregation / Spatial Attention Pooling:
        # --------------------------------------------------------------------
        # h: (B, C, hidden_dim, T) -> permute to (B, T, C, hidden_dim)
        h_perm = h.permute(0, 3, 1, 2).contiguous()
        # Compute spatial attention scores per electrode: (B, T, C, 1)
        attn_scores = self.channel_attention(h_perm)
        attn_weights = F.softmax(attn_scores, dim=2)  # Softmax over electrodes C
        
        # Weighted sum over electrodes: (B, T, hidden_dim)
        h_pooled = torch.sum(h_perm * attn_weights, dim=2)
        
        # Transpose to standard temporal format: (B, hidden_dim, T)
        out = h_pooled.transpose(1, 2).contiguous()
        out = self.spatial_proj(out)
        return out
