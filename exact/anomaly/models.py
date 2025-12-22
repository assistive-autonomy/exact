"""
STG-NF: Spatio-Temporal Graph Normalizing Flows model.

Lightweight implementation using standard PyTorch components.
"""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from exact.anomaly.graph import Graph


def split_feature(tensor: torch.Tensor, mode: str = "split") -> tuple[torch.Tensor, torch.Tensor]:
    """Split tensor along channel dimension."""
    C = tensor.size(1)
    if mode == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif mode == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]
    else:
        raise ValueError(f"Unknown split mode: {mode}")


def gaussian_log_prob(mean: torch.Tensor, log_std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute log probability under Gaussian distribution."""
    c = math.log(2 * math.pi)
    return -0.5 * (log_std * 2.0 + ((x - mean) ** 2) / (torch.exp(log_std * 2.0) + 1e-6) + c)


def gaussian_likelihood(mean: torch.Tensor, log_std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sum log probabilities over spatial dimensions."""
    log_p = gaussian_log_prob(mean, log_std, x)
    return torch.sum(log_p, dim=[1, 2, 3])


def gaussian_sample(mean: torch.Tensor, log_std: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Sample from Gaussian distribution."""
    return torch.normal(mean, torch.exp(log_std) * temperature)


class ActNorm2d(nn.Module):
    """
    Activation Normalization layer.
    
    Initializes scale and bias with first minibatch to have zero mean and unit variance.
    """
    
    def __init__(self, num_features: int, scale: float = 1.0):
        super().__init__()
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.initialized = False
    
    def initialize(self, x: torch.Tensor):
        """Data-dependent initialization."""
        if not self.training:
            raise ValueError("ActNorm not initialized but in eval mode")
        
        with torch.no_grad():
            bias = -torch.mean(x, dim=[0, 2, 3], keepdim=True)
            var = torch.mean((x + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6))
            
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.initialized = True
    
    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False):
        if not self.initialized:
            self.initialize(x)
        
        b, c, h, w = x.shape
        
        if reverse:
            x = x * torch.exp(-self.logs) - self.bias
            if logdet is not None:
                logdet = logdet - torch.sum(self.logs) * h * w
        else:
            x = (x + self.bias) * torch.exp(self.logs)
            if logdet is not None:
                logdet = logdet + torch.sum(self.logs) * h * w
        
        return x, logdet
    
    def set_initialized(self):
        """Mark as initialized (for loading checkpoints)."""
        self.initialized = True


class InvertibleConv1x1(nn.Module):
    """Invertible 1x1 convolution using LU decomposition for efficiency."""
    
    def __init__(self, num_channels: int, lu_decomposed: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.lu_decomposed = lu_decomposed
        
        # Initialize with random orthogonal matrix
        W = torch.linalg.qr(torch.randn(num_channels, num_channels))[0]
        
        if lu_decomposed:
            # LU decomposition for efficient computation
            P, L, U = torch.linalg.lu(W)
            s = torch.diag(U)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            U = torch.triu(U, diagonal=1)
            
            self.register_buffer("P", P)
            self.register_buffer("sign_s", sign_s)
            self.L = nn.Parameter(L)
            self.log_s = nn.Parameter(log_s)
            self.U = nn.Parameter(U)
        else:
            self.W = nn.Parameter(W)
    
    def _get_weight(self):
        if self.lu_decomposed:
            L = torch.tril(self.L, diagonal=-1) + torch.eye(self.num_channels, device=self.L.device)
            U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_s * torch.exp(self.log_s))
            return self.P @ L @ U
        else:
            return self.W
    
    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False):
        b, c, h, w = x.shape
        
        if reverse:
            W = self._get_weight()
            W_inv = torch.linalg.inv(W)
            x = F.conv2d(x, W_inv.unsqueeze(-1).unsqueeze(-1))
            if logdet is not None:
                if self.lu_decomposed:
                    logdet = logdet - torch.sum(self.log_s) * h * w
                else:
                    logdet = logdet - torch.slogdet(W)[1] * h * w
        else:
            W = self._get_weight()
            x = F.conv2d(x, W.unsqueeze(-1).unsqueeze(-1))
            if logdet is not None:
                if self.lu_decomposed:
                    logdet = logdet + torch.sum(self.log_s) * h * w
                else:
                    logdet = logdet + torch.slogdet(W)[1] * h * w
        
        return x, logdet


class GraphConv(nn.Module):
    """
    Graph convolution layer for skeleton data.
    
    Applies spatial graph convolution followed by temporal convolution.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,  # spatial kernel size (number of adjacency matrices)
        t_kernel_size: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_kernel_size // 2, 0),
            stride=(stride, 1),
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, V)
            A: Adjacency matrix (K, V, V)
        """
        assert A.size(0) == self.kernel_size
        
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', x, A)
        
        return x.contiguous()


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolution block.
    
    Graph convolution followed by temporal convolution with residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],  # (temporal, spatial)
        stride: int = 1,
        residual: bool = True,
    ):
        super().__init__()
        t_kernel, s_kernel = kernel_size
        
        self.gcn = GraphConv(in_channels, out_channels, s_kernel, t_kernel_size=1)
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), (stride, 1), (t_kernel // 2, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class AffineCoupling(nn.Module):
    """
    Affine coupling layer with ST-GCN for the coupling network.
    
    Splits input channels and applies learned affine transformation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        A: torch.Tensor,
        temporal_kernel_size: int = 9,
        first: bool = False,
    ):
        super().__init__()
        self.register_buffer("A", A)
        
        spatial_kernel = A.size(0)
        kernel_size = (temporal_kernel_size, spatial_kernel)
        
        # ST-GCN blocks for computing scale and shift
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels // 2, hidden_channels, kernel_size, residual=not first),
            STGCNBlock(hidden_channels, in_channels, kernel_size, residual=not first),
        ])
    
    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False):
        # Ensure 4D
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x1, x2 = split_feature(x, "split")
        
        # Compute transformation parameters
        h = x1.clone()
        for block in self.blocks:
            h = block(h, self.A)
        
        shift, log_scale = split_feature(h, "cross")
        
        # Ensure matching dimensions
        if len(log_scale.shape) == 3:
            log_scale = log_scale.unsqueeze(1)
        if len(shift.shape) == 3:
            shift = shift.unsqueeze(1)
        
        # Sigmoid scaling for numerical stability
        scale = torch.sigmoid(log_scale + 2.) + 1e-6
        
        if reverse:
            x2 = (x2 - shift) / scale
            if logdet is not None:
                logdet = logdet - torch.sum(torch.log(scale), dim=[1, 2, 3])
        else:
            x2 = x2 * scale + shift
            if logdet is not None:
                logdet = logdet + torch.sum(torch.log(scale), dim=[1, 2, 3])
        
        return torch.cat([x1, x2], dim=1), logdet


class SqueezeLayer(nn.Module):
    """Squeeze layer that trades spatial resolution for channels."""
    
    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor
    
    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False):
        if reverse:
            return self._unsqueeze(x), logdet
        else:
            return self._squeeze(x), logdet
    
    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v = x.size()
        assert t % self.factor == 0
        x = x.view(b, c, t // self.factor, self.factor, v)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(b, c * self.factor, t // self.factor, v)
        return x
    
    def _unsqueeze(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v = x.size()
        assert c % self.factor == 0
        x = x.view(b, c // self.factor, self.factor, t, v)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(b, c // self.factor, t * self.factor, v)
        return x


class Permute2d(nn.Module):
    """Channel permutation layer."""
    
    def __init__(self, num_channels: int, shuffle: bool = True):
        super().__init__()
        indices = torch.arange(num_channels)
        if shuffle:
            indices = indices[torch.randperm(num_channels)]
        else:
            indices = indices.flip(0)
        
        indices_inv = torch.zeros_like(indices)
        for i in range(num_channels):
            indices_inv[indices[i]] = i
        
        self.register_buffer("indices", indices)
        self.register_buffer("indices_inv", indices_inv)
    
    def forward(self, x: torch.Tensor, reverse: bool = False):
        if reverse:
            return x[:, self.indices_inv, ...]
        else:
            return x[:, self.indices, ...]


class FlowStep(nn.Module):
    """
    Single flow step: ActNorm -> Permutation -> Affine Coupling
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        A: torch.Tensor,
        temporal_kernel_size: int = 9,
        permutation: Literal["invconv", "shuffle", "reverse"] = "invconv",
        lu_decomposed: bool = True,
        first: bool = False,
    ):
        super().__init__()
        
        # 1. ActNorm
        self.actnorm = ActNorm2d(in_channels)
        
        # 2. Permutation
        if permutation == "invconv":
            self.permute = InvertibleConv1x1(in_channels, lu_decomposed)
        elif permutation == "shuffle":
            perm = Permute2d(in_channels, shuffle=True)
            self.permute = lambda x, logdet, rev: (perm(x, rev), logdet)
        else:
            perm = Permute2d(in_channels, shuffle=False)
            self.permute = lambda x, logdet, rev: (perm(x, rev), logdet)
        
        # 3. Affine coupling
        self.coupling = AffineCoupling(
            in_channels, hidden_channels, A, temporal_kernel_size, first
        )
    
    def forward(self, x: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False):
        if reverse:
            x, logdet = self.coupling(x, logdet, reverse=True)
            x, logdet = self.permute(x, logdet, True)
            x, logdet = self.actnorm(x, logdet, reverse=True)
        else:
            x, logdet = self.actnorm(x, logdet, reverse=False)
            x, logdet = self.permute(x, logdet, False)
            x, logdet = self.coupling(x, logdet, reverse=False)
        
        return x, logdet


class FlowNet(nn.Module):
    """
    Multi-scale normalizing flow network.
    
    Composed of multiple levels, each with K flow steps.
    """
    
    def __init__(
        self,
        pose_shape: tuple[int, int, int],  # (C, T, V)
        hidden_channels: int = 64,
        K: int = 8,  # Flow steps per level
        L: int = 1,  # Number of levels
        graph: Optional[Graph] = None,
        temporal_kernel_size: Optional[int] = None,
        permutation: str = "invconv",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.K = K
        self.L = L
        
        C, T, V = pose_shape
        
        # Build graph if not provided
        if graph is None:
            graph = Graph(layout="smpl", strategy="uniform", max_hop=1)
        
        A = torch.from_numpy(graph.A).float()
        
        if temporal_kernel_size is None:
            temporal_kernel_size = max(T // 2, 3) | 1  # Ensure odd
        
        self.layers = nn.ModuleList()
        self.output_shapes = []
        
        for level in range(L):
            # Squeeze (except first level)
            if level > 0:
                C, T = C * 2, T // 2
                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, T, V])
            
            # K flow steps
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        A=A,
                        temporal_kernel_size=temporal_kernel_size,
                        permutation=permutation,
                        first=(k == 0),
                    )
                )
                self.output_shapes.append([-1, C, T, V])
    
    def forward(self, x: torch.Tensor, logdet: float = 0.0, reverse: bool = False):
        if reverse:
            return self._decode(x)
        else:
            return self._encode(x, logdet)
    
    def _encode(self, z: torch.Tensor, logdet: float = 0.0):
        logdet = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet
    
    def _decode(self, z: torch.Tensor):
        for layer in reversed(self.layers):
            z, _ = layer(z, logdet=None, reverse=True)
        return z


class STG_NF(nn.Module):
    """
    STG-NF: Spatio-Temporal Graph Normalizing Flows for anomaly detection.
    
    Args:
        pose_shape: Input shape (channels, time, vertices)
        hidden_channels: Hidden dimension for ST-GCN
        K: Number of flow steps per level
        L: Number of multi-scale levels
        graph: Graph object for skeleton structure
        learn_prior: Whether to learn the prior distribution
        R: Prior mean offset for normal/abnormal separation
        temporal_kernel_size: Temporal convolution kernel size
        permutation: Permutation type ("invconv", "shuffle", "reverse")
        device: Computing device
    """
    
    def __init__(
        self,
        pose_shape: tuple[int, int, int],
        hidden_channels: int = 64,
        K: int = 8,
        L: int = 1,
        graph: Optional[Graph] = None,
        learn_prior: bool = False,
        R: float = 3.0,
        temporal_kernel_size: Optional[int] = None,
        permutation: str = "invconv",
        device: str = "cuda",
    ):
        super().__init__()
        self.R = R
        self.learn_prior = learn_prior
        self.device = device
        
        # Build flow network
        self.flow = FlowNet(
            pose_shape=pose_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            graph=graph,
            temporal_kernel_size=temporal_kernel_size,
            permutation=permutation,
            device=device,
        )
        
        # Get output shape from flow
        out_shape = self.flow.output_shapes[-1]
        C, T, V = out_shape[1], out_shape[2], out_shape[3]
        
        # Learnable prior (optional)
        if learn_prior:
            self.prior_net = nn.Conv2d(C * 2, C * 2, kernel_size=1)
            nn.init.zeros_(self.prior_net.weight)
            nn.init.zeros_(self.prior_net.bias)
        
        # Prior buffers
        self.register_buffer("prior_h", torch.zeros(1, C * 2, T, V))
        self.register_buffer(
            "prior_h_normal",
            torch.cat([
                torch.ones(C, T, V) * R,
                torch.zeros(C, T, V),
            ], dim=0)
        )
        self.register_buffer(
            "prior_h_abnormal",
            torch.cat([
                torch.ones(C, T, V) * (-R),
                torch.zeros(C, T, V),
            ], dim=0)
        )
    
    def prior(self, x: Optional[torch.Tensor], label: Optional[torch.Tensor] = None):
        """Compute prior distribution parameters."""
        if x is not None:
            h = self.prior_h.repeat(x.shape[0], 1, 1, 1)
            if label is not None:
                # Handle both binary and multi-class labels
                if label.dim() == 1:
                    # Binary labels: 1 = normal, -1 = abnormal
                    normal_mask = (label == 1)
                    abnormal_mask = (label == -1)
                else:
                    # Multi-class labels: use any activity as "normal"
                    # (activity presence > 0.5 means normal)
                    normal_mask = label.sum(dim=-1) > 0.5
                    abnormal_mask = ~normal_mask
                
                if normal_mask.any():
                    h[normal_mask] = self.prior_h_normal
                if abnormal_mask.any():
                    h[abnormal_mask] = self.prior_h_abnormal
        else:
            h = self.prior_h_normal.unsqueeze(0).repeat(32, 1, 1, 1)
        
        if self.learn_prior:
            h = self.prior_net(h)
        
        return split_feature(h, "split")
    
    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        reverse: bool = False,
        label: Optional[torch.Tensor] = None,
    ):
        if reverse:
            return self._reverse(z, temperature)
        else:
            return self._forward(x, label)
    
    def _forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        """Forward pass: compute negative log-likelihood."""
        b, c, t, v = x.shape
        
        # Flow transformation
        z, logdet = self.flow(x, reverse=False)
        
        # Prior likelihood
        mean, logs = self.prior(x, label)
        prior_ll = gaussian_likelihood(mean, logs, z)
        
        # Total log-likelihood -> NLL in bits per dimension
        nll = -(logdet + prior_ll) / (math.log(2.0) * c * t * v)
        
        return z, nll
    
    def _reverse(self, z: Optional[torch.Tensor], temperature: float = 1.0):
        """Reverse pass: sample from model."""
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(None)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, reverse=True)
        return x
    
    def set_actnorm_initialized(self):
        """Mark all ActNorm layers as initialized (for checkpoint loading)."""
        for module in self.modules():
            if isinstance(module, ActNorm2d):
                module.set_initialized()
