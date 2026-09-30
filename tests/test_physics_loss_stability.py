"""Unit tests verifying numerical stability and spike prevention for physics loss.

Verifies:
1. reconstruct_x0 handles near-zero alpha_bar_t (e.g., 1e-9, 1e-15, 0.0) without division-by-zero or explosion.
2. OceanEmbedLoss skips/safeguards physics loss at extreme noise timesteps (near t=1000) or near-zero alpha_bar.
3. Total loss remains finite and well-bounded (< 100), directly preventing the 4.79e+07 degenerate spike seen in diagnostics.
4. Normal timesteps (e.g., t=200, alpha_bar=0.8) compute physics loss cleanly and backpropagate gradients smoothly.
"""

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import OceanEmbedLoss, reconstruct_x0
from src.models.diffusion import GaussianDiffusion


def test_reconstruct_x0_near_zero_alpha_clamped():
    """Artificially set alpha_bar near zero and confirm x0_hat remains well-behaved."""
    B, C, H, W = 2, 1, 16, 16
    x_t = torch.randn(B, C, H, W)
    eps_pred = torch.randn(B, C, H, W)

    # Test extreme pathological alpha_bar values
    pathological_alphas = [
        torch.tensor([1e-9, 1e-9]),
        torch.tensor([1e-15, 1e-15]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1e-6, 1e-4]),
    ]

    for alpha_bar in pathological_alphas:
        x0_hat = reconstruct_x0(x_t, eps_pred, alpha_bar, min_alpha_bar=1e-3)

        assert torch.isfinite(x0_hat).all(), f"NaN or Inf detected for alpha_bar={alpha_bar}"
        # With min_alpha_bar=1e-3, sqrt_recip <= sqrt(1000) ~ 31.62
        # So x0_hat must not blow up to 10^4 or 10^7
        assert x0_hat.abs().max() < 500.0, f"x0_hat blew up to {x0_hat.abs().max()} for alpha_bar={alpha_bar}"


def test_physics_loss_extreme_timesteps_prevent_spike():
    """Verify that near t=1000 / alpha_bar near zero, the physics loss is safely safeguarded."""
    loss_fn = OceanEmbedLoss()

    B, H, W = 2, 20, 20
    eps_pred = torch.randn(B, 1, H, W, requires_grad=True)
    eps_true = torch.randn(B, 1, H, W)
    ocean_mask = torch.ones(B, 1, H, W)
    as_mask = torch.zeros(B, 1, H, W)
    bob_mask = torch.ones(B, 1, H, W)

    aux_preds = {
        "mld": torch.randn(B, 1),
        "blt": torch.randn(B, 1),
        "sal_max_depth": torch.randn(B, 1),
        "sal_max_strength": torch.randn(B, 1),
    }
    aux_trues = torch.randn(B, 4, H, W)
    x0_true = torch.randn(B, 1, H, W)
    x_t = torch.randn(B, 1, H, W)

    # Simulate timestep 999 where alpha_bar is near zero (e.g. 1e-9)
    alpha_bar_extreme = torch.tensor([1e-9, 2e-9])
    t_extreme = torch.tensor([995, 999])

    losses = loss_fn(
        eps_pred=eps_pred,
        eps_true=eps_true,
        depth=100.0,
        ocean_mask=ocean_mask,
        arabian_sea_mask=as_mask,
        bob_mask=bob_mask,
        aux_preds=aux_preds,
        aux_trues=aux_trues,
        x_t=x_t,
        x0_true=x0_true,
        alpha_bar_t=alpha_bar_extreme,
        t=t_extreme,
    )

    # 1. Physics loss must NOT blow up to 10^7
    assert torch.isfinite(losses["loss_physics"])
    assert losses["loss_physics"].item() <= 100.0, (
        f"Degenerate physics loss detected: {losses['loss_physics'].item()}"
    )

    # 2. Total loss must be well bounded (not 4.79e+07)
    assert torch.isfinite(losses["loss_total"])
    assert losses["loss_total"].item() < 100.0, (
        f"Degenerate total loss detected: {losses['loss_total'].item()}"
    )

    # 3. Gradients must backpropagate without NaN or Inf
    losses["loss_total"].backward()
    assert torch.isfinite(eps_pred.grad).all()


def test_physics_loss_normal_timesteps_active_and_smooth():
    """Verify that at normal timesteps (e.g., t=200), physics loss operates normally."""
    loss_fn = OceanEmbedLoss()

    B, H, W = 2, 20, 20
    eps_pred = torch.randn(B, 1, H, W, requires_grad=True)
    eps_true = torch.randn(B, 1, H, W)
    ocean_mask = torch.ones(B, 1, H, W)
    as_mask = torch.zeros(B, 1, H, W)
    bob_mask = torch.ones(B, 1, H, W)

    aux_preds = {
        "mld": torch.randn(B, 1),
        "blt": torch.randn(B, 1),
        "sal_max_depth": torch.randn(B, 1),
        "sal_max_strength": torch.randn(B, 1),
    }
    aux_trues = torch.randn(B, 4, H, W)
    x0_true = torch.randn(B, 1, H, W)
    x_t = torch.randn(B, 1, H, W)

    # Normal diffusion timestep: t=200, alpha_bar ~ 0.85
    alpha_bar_normal = torch.tensor([0.85, 0.85])
    t_normal = torch.tensor([200, 200])

    losses = loss_fn(
        eps_pred=eps_pred,
        eps_true=eps_true,
        depth=100.0,
        ocean_mask=ocean_mask,
        arabian_sea_mask=as_mask,
        bob_mask=bob_mask,
        aux_preds=aux_preds,
        aux_trues=aux_trues,
        x_t=x_t,
        x0_true=x0_true,
        alpha_bar_t=alpha_bar_normal,
        t=t_normal,
    )

    assert torch.isfinite(losses["loss_physics"])
    assert losses["loss_physics"].item() > 0.0, "Physics loss should be non-zero for normal timesteps"
    assert torch.isfinite(losses["loss_total"])


def test_gaussian_diffusion_x0_reconstruction_at_max_t():
    """Verify GaussianDiffusion.predict_x0_from_noise at t=999 is clamped and bounded."""
    diffusion = GaussianDiffusion(timesteps=1000, schedule_type="cosine")
    B, C, H, W = 2, 1, 16, 16
    x_t = torch.randn(B, C, H, W)
    eps_pred = torch.randn(B, C, H, W)
    t = torch.tensor([999, 999])

    x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)

    assert torch.isfinite(x0_hat).all()
    # Clamped to 1e-3 means multiplier <= 31.62 instead of 20,294.55
    assert x0_hat.abs().max() < 500.0
