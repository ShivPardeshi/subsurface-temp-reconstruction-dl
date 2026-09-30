"""U-Net Denoiser Backbone for Diffusion-based Subsurface Temperature Reconstruction.

4 resolution stages (32 -> 64 -> 128 -> 256) with GroupNorm + AdaGN + SiLU at every block,
skip connections, and symmetric decoder predicting noise epsilon_theta.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conditioning import AdaGroupNorm, NonSpatialConditioningMLP


class ResBlock(nn.Module):
    """Residual convolution block with AdaGroupNorm modulation."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaGroupNorm(num_groups, out_channels, cond_dim)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaGroupNorm(num_groups, out_channels, cond_dim)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h, cond_emb)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h, cond_emb)
        h = self.act2(h)

        return h + self.skip(x)


class UNetDenoiser(nn.Module):
    """4-stage U-Net Denoiser with AdaGN conditioning injection."""

    def __init__(
        self,
        in_channels: int = 72,  # 1 (x_tau) + 1 (prev_depth_clean) + 70 (spatial_cond)
        out_channels: int = 1,
        stage_channels: Tuple[int, ...] = (32, 64, 128, 256),
        cond_in_dim: int = 16,
        cond_emb_dim: int = 128,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stage_channels = stage_channels
        self.cond_emb_dim = cond_emb_dim

        # Shared non-spatial conditioning MLP
        self.cond_mlp = NonSpatialConditioningMLP(
            in_dim=cond_in_dim, hidden_dim=cond_emb_dim, out_dim=cond_emb_dim
        )

        # Initial projection
        c0, c1, c2, c3 = stage_channels
        self.init_conv = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)

        # Encoder stages
        self.enc1 = ResBlock(c0, c0, cond_emb_dim, num_groups)
        self.down1 = nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResBlock(c1, c1, cond_emb_dim, num_groups)
        self.down2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)

        self.enc3 = ResBlock(c2, c2, cond_emb_dim, num_groups)
        self.down3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bot1 = ResBlock(c3, c3, cond_emb_dim, num_groups)
        self.bot2 = ResBlock(c3, c3, cond_emb_dim, num_groups)

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(c2 + c2, c2, cond_emb_dim, num_groups)

        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(c1 + c1, c1, cond_emb_dim, num_groups)

        self.up1 = nn.ConvTranspose2d(c1, c0, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(c0 + c0, c0, cond_emb_dim, num_groups)

        # Final projection to predicted noise
        self.final_norm = nn.GroupNorm(min(num_groups, c0), c0)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(c0, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        spatial_cond: torch.Tensor,
        non_spatial_cond: torch.Tensor,
        prev_depth_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting noise epsilon_theta.

        Args:
            x_noisy: Noisy anomaly field (B, 1, H, W).
            spatial_cond: Spatial conditioning (B, 70, H, W).
            non_spatial_cond: Scalar conditions (B, cond_in_dim).
            prev_depth_clean: Clean output from previous depth (B, 1, H, W), or zeros.

        Returns:
            eps_pred: Predicted noise tensor of shape (B, 1, H, W).
        """
        b, _, h, w = x_noisy.shape

        if prev_depth_clean is None:
            prev_depth_clean = torch.zeros_like(x_noisy)

        # Non-spatial conditioning embedding
        cond_emb = self.cond_mlp(non_spatial_cond)

        # Full spatial input stack: (B, 1 + 1 + 70, H, W) = (B, 72, H, W)
        x_in = torch.cat([x_noisy, prev_depth_clean, spatial_cond], dim=1)

        # Initial conv
        h0 = self.init_conv(x_in)

        # Encoder
        e1 = self.enc1(h0, cond_emb)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, cond_emb)
        d2 = self.down2(e2)

        e3 = self.enc3(d2, cond_emb)
        d3 = self.down3(e3)

        # Bottleneck
        b1 = self.bot1(d3, cond_emb)
        b2 = self.bot2(b1, cond_emb)

        # Decoder with skip connections (matching spatial dimensions if odd)
        u3 = self.up3(b2)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = F.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        c3 = torch.cat([u3, e3], dim=1)
        d_out3 = self.dec3(c3, cond_emb)

        u2 = self.up2(d_out3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        c2 = torch.cat([u2, e2], dim=1)
        d_out2 = self.dec2(c2, cond_emb)

        u1 = self.up1(d_out2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        c1 = torch.cat([u1, e1], dim=1)
        d_out1 = self.dec1(c1, cond_emb)

        # Output projection
        out = self.final_norm(d_out1)
        out = self.final_act(out)
        eps_pred = self.final_conv(out)

        return eps_pred
