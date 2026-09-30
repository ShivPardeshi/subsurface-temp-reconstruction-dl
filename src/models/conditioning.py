"""Conditioning Modules (Spatial Concatenation and AdaGN / FiLM Injection).

Implements:
1. Spatial conditioning fusion (u_cond + 6 static/region channels -> 70 channels).
2. AdaGN MLP projecting non-spatial scalars into scale and shift parameters for GroupNorm.
3. AdaGroupNorm layer applying affine modulation.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaGroupNorm(nn.Module):
    """Adaptive Group Normalization (AdaGN / FiLM).

    Normalizes input feature map with GroupNorm and modulates with scale (gamma)
    and shift (beta) computed from conditioning vector:
        out = (1 + gamma) * norm(x) + beta
    """

    def __init__(self, num_groups: int, num_channels: int, cond_dim: int):
        super().__init__()
        self.num_groups = min(num_groups, num_channels)
        self.num_channels = num_channels
        self.gn = nn.GroupNorm(self.num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, 2 * num_channels)

        # Initialize projection with small weights so conditioning is immediately active
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """Apply modulated group normalization.

        Args:
            x: Feature map tensor of shape (B, C, H, W).
            cond_emb: Conditioning embedding tensor of shape (B, cond_dim).

        Returns:
            Modulated feature map of shape (B, C, H, W).
        """
        normed = self.gn(x)
        scale_shift = self.proj(cond_emb)  # (B, 2*C)
        gamma, beta = torch.split(scale_shift, self.num_channels, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + gamma) * normed + beta


class NonSpatialConditioningMLP(nn.Module):
    """Shared 2-layer MLP for projecting non-spatial scalar conditions.

    Conditioning vector components:
    - timestep embedding (sinusoidal or learned)
    - log-normalized depth id (e.g. log(depth + 1))
    - climatology scalar / mean value at depth
    - ONI index
    - IOD / DMI index
    - day-of-year sin & cos
    - optional summary statistic (mean/std) of previous depth clean output
    """

    def __init__(self, in_dim: int = 16, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond_vec: torch.Tensor) -> torch.Tensor:
        """Project scalar conditions to shared conditioning embedding.

        Args:
            cond_vec: Tensor of shape (B, in_dim).

        Returns:
            Tensor of shape (B, out_dim).
        """
        return self.net(cond_vec)


class SpatialConditioningFusion(nn.Module):
    """Combines u_cond (64 channels) with 6 static/region channels and optional SSHA channels."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        u_cond: torch.Tensor,
        static_features: torch.Tensor,
        ssha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate u_cond with static features and optional SSHA features.

        Args:
            u_cond: Context tensor of shape (B, 64, H, W).
            static_features: Static spatial channels of shape (B, 6, H, W).
            ssha: Optional SSHA tensor of shape (B, 1, H, W).

        Returns:
            Combined tensor of shape (B, 70, H, W) or (B, 72, H, W) if SSHA provided.
        """
        feats = [u_cond, static_features]
        if ssha is not None:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=ssha.dtype, device=ssha.device).view(1, 1, 3, 3) / 8.0
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=ssha.dtype, device=ssha.device).view(1, 1, 3, 3) / 8.0
            gx = F.conv2d(ssha, sobel_x, padding=1)
            gy = F.conv2d(ssha, sobel_y, padding=1)
            ssha_grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
            feats.extend([ssha, ssha_grad_mag])
        return torch.cat(feats, dim=1)
