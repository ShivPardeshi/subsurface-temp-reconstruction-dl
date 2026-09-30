"""Unit tests for depth cascade order and profile generation."""

import pytest
import torch
from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS


def test_depth_cascade_full_profile_shape():
    context_enc = ContextEncoder(in_channels=25, hidden_dims=(32, 64, 64))
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8)
    diffusion = GaussianDiffusion(timesteps=100)
    ddim_sampler = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=5)

    cascade = DepthCascadeSampler(
        context_encoder=context_enc,
        unet_denoiser=unet,
        ddim_sampler=ddim_sampler,
        depths=CANONICAL_DEPTHS,  # 15 canonical depths
    )

    x_seq = torch.randn(1, 7, 25, 20, 20)
    static_features = torch.rand(1, 6, 20, 20)
    scalar_conditions = torch.randn(1, 4)

    out = cascade.sample_full_profile(
        x_seq=x_seq,
        static_features=static_features,
        scalar_conditions=scalar_conditions,
    )

    assert "anomalies" in out
    assert out["anomalies"].shape == (1, 15, 20, 20)
    assert torch.isfinite(out["anomalies"]).all()
