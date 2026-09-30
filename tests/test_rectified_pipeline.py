"""Smoke test for rectified pure diffusion pipeline with all 8 bottlenecks resolved."""

import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.losses import OceanEmbedLoss
from src.sampling.ddim_sampler import DDIMSampler
from src.models.diffusion import GaussianDiffusion
from src.models.unet_denoiser import UNetDenoiser
from src.models.context_encoder import ContextEncoder
from src.models.auxiliary_heads import AuxiliaryHeads
from src.sampling.depth_cascade import DepthCascadeSampler
from src.utils.grid import CANONICAL_DEPTHS

def test_pipeline():
    print("Testing Loss with Min-SNR and Stratification...")
    device = torch.device("cpu")
    loss_fn = OceanEmbedLoss(skill_focused=True, use_min_snr=True, min_snr_gamma=5.0)

    b, h, w = 2, 28, 60
    eps_pred = torch.randn(b, 1, h, w)
    eps_true = torch.randn(b, 1, h, w)
    ocean_mask = torch.ones(b, 1, h, w)
    as_mask = torch.zeros(b, 1, h, w)
    bob_mask = torch.zeros(b, 1, h, w)
    aux_preds = {
        "mld": torch.randn(b, 1),
        "blt": torch.randn(b, 1),
        "sal_max_depth": torch.randn(b, 1),
        "sal_max_strength": torch.randn(b, 1),
    }
    aux_trues = torch.randn(b, 4, h, w)
    x0_hat = torch.randn(b, 1, h, w)
    x0_true = torch.randn(b, 1, h, w)
    alpha_bar = torch.tensor([0.5, 0.5])
    prev_x0_true = torch.randn(b, 1, h, w)

    out = loss_fn(
        eps_pred=eps_pred,
        eps_true=eps_true,
        depth=100.0,
        ocean_mask=ocean_mask,
        arabian_sea_mask=as_mask,
        bob_mask=bob_mask,
        aux_preds=aux_preds,
        aux_trues=aux_trues,
        x0_hat=x0_hat,
        x0_true=x0_true,
        alpha_bar_t=alpha_bar,
        t=torch.tensor([500, 500]),
        prev_x0_true=prev_x0_true,
        prev_depth=75.0,
        clim_temp=25.0,
        prev_clim_temp=27.0,
        depth_std=1.0,
        prev_depth_std=1.0,
    )
    print("Loss output keys:", list(out.keys()))
    assert "loss_total" in out
    assert out["loss_total"].item() > 0
    print("Loss check passed successfully!")

    print("Testing DDIMSampler with quadratic schedule...")
    diffusion = GaussianDiffusion(timesteps=1000)
    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=15, schedule_type="quadratic")
    print(f"Generated {len(ddim.ddim_timesteps)} quadratic timesteps: {ddim.ddim_timesteps}")
    assert len(ddim.ddim_timesteps) > 0

    print("Testing UNetDenoiser with cond_in_dim=9...")
    unet = UNetDenoiser(
        in_channels=72,
        stage_channels=(16, 32, 64, 128),
        cond_in_dim=9,
    )
    x_noisy = torch.randn(b, 1, h, w)
    spatial_cond = torch.randn(b, 70, h, w)
    non_spatial_cond = torch.randn(b, 9)
    prev_clean = torch.randn(b, 1, h, w)

    pred = unet(
        x_noisy=x_noisy,
        spatial_cond=spatial_cond,
        non_spatial_cond=non_spatial_cond,
        prev_depth_clean=prev_clean,
    )
    assert pred.shape == (b, 1, h, w)
    print(f"UNet output shape: {pred.shape} - PASSED!")

    print("All smoke tests passed!")

if __name__ == "__main__":
    test_pipeline()
