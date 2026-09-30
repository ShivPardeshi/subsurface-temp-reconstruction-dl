"""Auxiliary Target Prediction Heads (Stage 5 Architecture).

Implements 3 MLP prediction heads operating on global-average-pooled u_cond:
1. Mixed Layer Depth (MLD) head.
2. Bay of Bengal Barrier Layer Thickness (BLT) head.
3. Arabian Sea Subsurface Salinity Maximum (depth + strength) head.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class AuxiliaryHeads(nn.Module):
    """Auxiliary physical heads predicting MLD, BLT, and Salinity Maximum parameters."""

    def __init__(self, in_features: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.in_features = in_features

        # Global average pool over spatial dimensions (B, 64, H, W) -> (B, 64)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 1. Mixed Layer Depth Head (scalar depth in meters)
        self.mld_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 2. Bay of Bengal Barrier Layer Thickness Head (scalar thickness in meters)
        self.blt_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 3. Arabian Sea Salinity Maximum Head (2 values: depth in meters + strength anomaly in PSU)
        self.sal_max_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, u_cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute predictions for all 3 auxiliary physical targets.

        Args:
            u_cond: Spatiotemporal context tensor (B, 64, H, W).

        Returns:
            Dictionary containing:
            - 'mld': (B, 1) predicted MLD in meters
            - 'blt': (B, 1) predicted BLT in meters
            - 'sal_max_depth': (B, 1) predicted Salinity Max depth in meters
            - 'sal_max_strength': (B, 1) predicted Salinity Max strength anomaly
        """
        pooled = self.gap(u_cond).flatten(1)  # (B, 64)

        pred_mld = self.mld_head(pooled)
        pred_blt = self.blt_head(pooled)
        pred_sal_max = self.sal_max_head(pooled)  # (B, 2)

        return {
            "mld": pred_mld,
            "blt": pred_blt,
            "sal_max_depth": pred_sal_max[:, 0:1],
            "sal_max_strength": pred_sal_max[:, 1:2],
        }
