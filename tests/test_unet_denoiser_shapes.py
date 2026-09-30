"""Unit tests for UNetDenoiser shapes and skip connections."""

import pytest
import torch
from src.models.unet_denoiser import UNetDenoiser


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("h,w", [(40, 40), (112, 240)])
def test_unet_denoiser_shapes(batch_size, h, w):
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8)

    x_noisy = torch.randn(batch_size, 1, h, w)
    spatial_cond = torch.randn(batch_size, 70, h, w)
    non_spatial_cond = torch.randn(batch_size, 8)
    prev_depth_clean = torch.randn(batch_size, 1, h, w)

    out = unet(
        x_noisy=x_noisy,
        spatial_cond=spatial_cond,
        non_spatial_cond=non_spatial_cond,
        prev_depth_clean=prev_depth_clean,
    )

    assert out.shape == (batch_size, 1, h, w)
    assert torch.isfinite(out).all()
