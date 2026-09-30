"""Unit tests for conditioning module and AdaGN injection responsiveness."""

import pytest
import torch
from src.models.conditioning import SpatialConditioningFusion, AdaGroupNorm, NonSpatialConditioningMLP
from src.models.unet_denoiser import UNetDenoiser


def test_spatial_conditioning_fusion_shape():
    fusion = SpatialConditioningFusion()
    u_cond = torch.randn(2, 64, 40, 40)
    static_feats = torch.randn(2, 6, 40, 40)
    out = fusion(u_cond, static_feats)

    assert out.shape == (2, 70, 40, 40)


def test_adagn_injection_sensitivity():
    """Verify that changing non-spatial conditioning produces a measurably different output."""
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8)
    unet.eval()

    x_noisy = torch.randn(1, 1, 40, 40)
    spatial_cond = torch.randn(1, 70, 40, 40)
    prev_clean = torch.zeros(1, 1, 40, 40)

    cond_a = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    cond_b = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])

    with torch.no_grad():
        out_a = unet(x_noisy, spatial_cond, cond_a, prev_clean)
        out_b = unet(x_noisy, spatial_cond, cond_b, prev_clean)

    diff = (out_a - out_b).abs().mean().item()
    assert diff > 1e-4, f"Conditioning injection failed: Output unchanged despite different conditions (diff={diff})"
