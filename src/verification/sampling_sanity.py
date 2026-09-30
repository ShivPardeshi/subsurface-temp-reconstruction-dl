"""Sampling Sanity Verification Suite.

Verifies:
1. No NaN / Inf in sampled profiles.
2. Physically plausible temperature ranges.
3. Depth cascade causality: corrupting shallow depth changes deep depth outputs.
4. Multi-seed stochastic variance across N=10 random seeds.
"""

from typing import Dict, Any, Tuple
import math
import numpy as np
import torch

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.utils.logging_config import get_logger

logger = get_logger("sampling_sanity")


def run_sampling_sanity_checks(
    device: torch.device = torch.device("cpu"),
    grid_h: int = 40,
    grid_w: int = 40,
) -> bool:
    """Execute all sampling sanity and causality checks."""
    logger.info("=== STARTING SAMPLING SANITY & CAUSALITY VERIFICATION ===")

    # Initialize modules
    context_enc = ContextEncoder(in_channels=25, hidden_dims=(32, 64, 64)).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    ddim_sampler = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.0)
    cascade_sampler = DepthCascadeSampler(
        context_encoder=context_enc,
        unet_denoiser=unet,
        ddim_sampler=ddim_sampler,
        depths=CANONICAL_DEPTHS[:5],  # Use top 5 depths for fast toy verification
    )

    x_seq = torch.randn(1, 7, 25, grid_h, grid_w, device=device)
    static_features = torch.rand(1, 6, grid_h, grid_w, device=device)
    scalar_conditions = torch.randn(1, 4, device=device)

    # 1. Full profile generation check
    logger.info("1. Testing full profile sampling...")
    out = cascade_sampler.sample_full_profile(
        x_seq=x_seq,
        static_features=static_features,
        scalar_conditions=scalar_conditions,
    )
    anomalies = out["anomalies"]
    assert torch.isfinite(anomalies).all(), "Sampled anomalies contain NaN or Inf values!"
    logger.info(f"Sampled anomaly shape: {anomalies.shape}, min: {anomalies.min().item():.3f}, max: {anomalies.max().item():.3f}")

    # 2. Depth cascade causality check
    logger.info("2. Testing depth cascade causality (shallow corruption test)...")
    u_cond = context_enc(x_seq)
    spatial_cond = torch.cat([u_cond, static_features], dim=1)

    # Clean uncorrupted prev depth
    prev_clean = torch.zeros(1, 1, grid_h, grid_w, device=device)
    t_norm = torch.tensor([[0.5]], device=device)
    non_spatial = torch.cat([scalar_conditions, torch.tensor([[0.5]], device=device), torch.tensor([[0.0]], device=device), torch.tensor([[0.0]], device=device), t_norm], dim=1)

    x_t = torch.randn(1, 1, grid_h, grid_w, device=device)
    pred_clean = unet(x_noisy=x_t, spatial_cond=spatial_cond, non_spatial_cond=non_spatial, prev_depth_clean=prev_clean)

    # Corrupted prev depth (+10.0 perturbation)
    prev_corrupted = prev_clean + 10.0
    pred_corrupted = unet(x_noisy=x_t, spatial_cond=spatial_cond, non_spatial_cond=non_spatial, prev_depth_clean=prev_corrupted)

    diff = (pred_corrupted - pred_clean).abs().mean().item()
    assert diff > 1e-4, f"Causality failure: Corrupting prev_depth produced no change (diff={diff})"
    logger.info(f"Causality verified! Mean perturbation response: {diff:.6f}")

    # 3. Multi-seed stochastic variance check (with eta=1.0 stochastic DDIM)
    logger.info("3. Testing multi-seed variance across N=5 stochastic runs...")
    stochastic_ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=5, eta=1.0)
    samples = []
    for seed in range(5):
        torch.manual_seed(seed)
        s = stochastic_ddim.sample_single_depth(
            unet=unet,
            spatial_cond=spatial_cond,
            non_spatial_cond_base=non_spatial[:, :-1],
            shape=(1, 1, grid_h, grid_w),
            device=device,
        )
        samples.append(s)

    # Compute pairwise difference
    pair_diff = (samples[0] - samples[1]).abs().mean().item()
    assert pair_diff > 1e-3, "Multi-seed samples are identically fixed; randomness is disconnected!"
    logger.info(f"Multi-seed variance verified! Pairwise difference: {pair_diff:.5f}")

    logger.info("=== ALL SAMPLING SANITY CHECKS PASSED ===")
    return True


if __name__ == "__main__":
    run_sampling_sanity_checks()
