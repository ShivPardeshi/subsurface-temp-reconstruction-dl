"""Depth Cascade Sampling Orchestrator.

Enforces strict sequential shallow-to-deep sampling across all 15 canonical depths:
0 -> 5 -> 10 -> 20 -> 30 -> 50 -> 75 -> 100 -> 125 -> 150 -> 200 -> 300 -> 500 -> 700 -> 1000m.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
from pathlib import Path
import json
import torch
import torch.nn as nn

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.sampling.ddim_sampler import DDIMSampler
from src.utils.grid import CANONICAL_DEPTHS


class DepthCascadeSampler:
    """Orchestrates sequential 15-depth cascade sampling."""

    def __init__(
        self,
        context_encoder: ContextEncoder,
        unet_denoiser: UNetDenoiser,
        ddim_sampler: DDIMSampler,
        depths: List[float] = CANONICAL_DEPTHS,
        rescale_output: bool = True,
        depth_scales_path: str = "data/processed/anomaly_depth_scales.json",
        aux_heads: Optional[nn.Module] = None,
    ):
        self.context_encoder = context_encoder
        self.unet_denoiser = unet_denoiser
        self.ddim_sampler = ddim_sampler
        self.depths = depths
        self.rescale_output = rescale_output
        self.aux_heads = aux_heads

        scales_file = Path(depth_scales_path)
        if scales_file.exists():
            with open(scales_file, "r") as f:
                scales_data = json.load(f)
            self.anomaly_stds = scales_data.get("anomaly_stds", [1.0] * len(depths))
            self.clim_mean_temps = scales_data.get("clim_mean_temps", [20.0] * len(depths))
        else:
            self.anomaly_stds = [1.0] * len(depths)
            self.clim_mean_temps = [20.0] * len(depths)

    @torch.no_grad()
    def sample_full_profile(
        self,
        x_seq: torch.Tensor,
        static_features: torch.Tensor,
        scalar_conditions: torch.Tensor,
        climatology_fields: Optional[torch.Tensor] = None,
        use_cascade: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Generate a complete 15-depth 3D temperature profile.

        Args:
            x_seq: (B, 7, 25, H, W) 7-day spatiotemporal surface inputs.
            static_features: (B, 6, H, W) static fields (landmask, bathymetry, 4 region masks).
            scalar_conditions: (B, S) base scalar conditions (ONI, IOD, DOY_sin, DOY_cos).
            climatology_fields: Optional (B, 15, H, W) climatology to compute absolute temperatures.
            use_cascade: bool, if False, disables sequential shallow-to-deep feedback (Stage D ablation).

        Returns:
            Dictionary containing:
            - 'anomalies': (B, 15, H, W) predicted temperature anomaly profile
            - 'temperatures': (B, 15, H, W) predicted absolute temperature (if climatology provided)
        """
        b, _, _, h, w = x_seq.shape
        device = x_seq.device

        # Stage 1: Encode 7-day spatiotemporal context -> u_cond (B, 64, H, W)
        u_cond = self.context_encoder(x_seq)

        # Direct SSHA Injection (Bottleneck 11)
        ssha = x_seq[:, -1, 2:3]
        if hasattr(self.unet_denoiser, "in_channels") and self.unet_denoiser.in_channels == 74:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=ssha.dtype, device=device).view(1, 1, 3, 3) / 8.0
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=ssha.dtype, device=device).view(1, 1, 3, 3) / 8.0
            gx = torch.nn.functional.conv2d(ssha, sobel_x, padding=1)
            gy = torch.nn.functional.conv2d(ssha, sobel_y, padding=1)
            ssha_grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
            spatial_cond = torch.cat([u_cond, static_features, ssha, ssha_grad_mag], dim=1)
        else:
            spatial_cond = torch.cat([u_cond, static_features], dim=1)

        # Auxiliary Physical Predictions (Bottleneck 14)
        if self.aux_heads is not None:
            aux_preds = self.aux_heads(u_cond)
            mld_cond = aux_preds["mld"] / 50.0
            blt_cond = aux_preds["blt"] / 20.0
            sal_cond = aux_preds["sal_max_depth"] / 100.0
        else:
            mld_cond = torch.zeros((b, 1), device=device)
            blt_cond = torch.zeros((b, 1), device=device)
            sal_cond = torch.zeros((b, 1), device=device)

        predicted_anomalies = []
        prev_clean_sample: Optional[torch.Tensor] = None

        # Stage 4: Sequential shallow-to-deep depth cascade
        for depth_idx, depth in enumerate(self.depths):
            # Formulate non-spatial condition vector for this depth
            log_depth = math.log(depth + 1.0) / math.log(1001.0)  # normalized to [0, 1]
            depth_tensor = torch.full((b, 1), log_depth, device=device, dtype=torch.float32)

            # Vertical lapse rate and layer thickness (Bottleneck 12)
            if depth_idx > 0:
                dz = float(depth - self.depths[depth_idx - 1])
                c_prev = float(self.clim_mean_temps[depth_idx - 1])
                c_curr = float(self.clim_mean_temps[depth_idx])
                lapse_val = (c_prev - c_curr) / dz
            else:
                dz = 5.0
                lapse_val = 0.0
            lapse_tensor = torch.full((b, 1), lapse_val / 0.10, device=device, dtype=torch.float32)
            dz_tensor = torch.full((b, 1), dz / 100.0, device=device, dtype=torch.float32)

            # Scalar summary of previous depth (mean and std, or zeros if surface/no cascade)
            if use_cascade and prev_clean_sample is not None:
                prev_mean = prev_clean_sample.mean(dim=(-2, -1), keepdim=True).view(b, 1)
                prev_std = prev_clean_sample.std(dim=(-2, -1), keepdim=True).view(b, 1)
                prev_sample_input = prev_clean_sample
            else:
                prev_mean = torch.zeros((b, 1), device=device)
                prev_std = torch.zeros((b, 1), device=device)
                prev_sample_input = None

            clim_val = torch.full((b, 1), float(self.clim_mean_temps[depth_idx]) / 30.0, device=device, dtype=torch.float32)

            # Check conditioning dimension of unet (total in_features including t_norm (+1))
            cond_in_dim = getattr(self.unet_denoiser.cond_mlp.net[0], "in_features", 14)
            if cond_in_dim == 14:
                # 4 (scalar) + 1 (depth) + 1 (clim) + 1 (lapse) + 1 (dz) + 1 (mean) + 1 (std) + 3 (aux) = 13 (before t_norm)
                non_spatial_cond_base = torch.cat(
                    [scalar_conditions, depth_tensor, clim_val, lapse_tensor, dz_tensor, prev_mean, prev_std, mld_cond, blt_cond, sal_cond], dim=1
                )
            elif cond_in_dim == 9:
                # 4 (scalar) + 1 (depth) + 1 (clim) + 1 (mean) + 1 (std) = 8 (before t_norm)
                non_spatial_cond_base = torch.cat(
                    [scalar_conditions, depth_tensor, clim_val, prev_mean, prev_std], dim=1
                )
            elif cond_in_dim == 8:
                # 4 (scalar) + 1 (depth) + 1 (mean) + 1 (std) = 7 (before t_norm)
                non_spatial_cond_base = torch.cat(
                    [scalar_conditions, depth_tensor, prev_mean, prev_std], dim=1
                )
            else:
                non_spatial_cond_base = torch.cat(
                    [scalar_conditions, depth_tensor], dim=1
                )

            # Sample this depth level via DDIM
            clean_sample = self.ddim_sampler.sample_single_depth(
                unet=self.unet_denoiser,
                spatial_cond=spatial_cond,
                non_spatial_cond_base=non_spatial_cond_base,
                prev_depth_clean=prev_sample_input,
                shape=(b, 1, h, w),
                device=device,
            )

            # If standardized training was used, rescale clean_sample to physical degrees C for final output
            if self.rescale_output:
                std_factor = float(self.anomaly_stds[depth_idx])
                predicted_anomalies.append(clean_sample * std_factor)
            else:
                predicted_anomalies.append(clean_sample)

            # Pass clean sample forward to next depth in cascade if enabled (in standardized space)
            if use_cascade:
                prev_clean_sample = clean_sample

        # Stack into (B, 15, H, W)
        stacked_anomalies = torch.cat(predicted_anomalies, dim=1)

        result = {"anomalies": stacked_anomalies}

        if climatology_fields is not None:
            result["temperatures"] = stacked_anomalies + climatology_fields

        return result
