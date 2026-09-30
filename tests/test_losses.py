"""Unit tests for loss functions and depth/region weighting alpha(d, region)."""

import pytest
import torch
from src.training.losses import compute_depth_region_weight, OceanEmbedLoss


def test_depth_region_weight_values():
    as_mask = torch.zeros(1, 1, 10, 10)
    as_mask[:, :, :, :5] = 1.0  # Left half is Arabian Sea

    # 1. Shallow depth (10m) -> 1.0 everywhere
    w_10m = compute_depth_region_weight(10.0, as_mask)
    assert torch.allclose(w_10m, torch.tensor(1.0))

    # 2. Thermocline depth (20m, 30m, 100m) -> 1.5 everywhere
    w_20m = compute_depth_region_weight(20.0, as_mask)
    assert torch.allclose(w_20m, torch.tensor(1.5))
    w_30m = compute_depth_region_weight(30.0, as_mask)
    assert torch.allclose(w_30m, torch.tensor(1.5))
    w_100m = compute_depth_region_weight(100.0, as_mask)
    assert torch.allclose(w_100m, torch.tensor(1.5))

    # 3. ASHSW depth (250m) -> 1.95 in Arabian Sea, 1.0 elsewhere
    w_250m = compute_depth_region_weight(250.0, as_mask)
    assert torch.allclose(w_250m[:, :, :, :5], torch.tensor(1.95))
    assert torch.allclose(w_250m[:, :, :, 5:], torch.tensor(1.0))


def test_skill_focused_depth_region_weight_values():
    as_mask = torch.zeros(1, 1, 10, 10)
    as_mask[:, :, :, :5] = 1.0

    # Surface 10m -> 1.35 everywhere
    w_10m = compute_depth_region_weight(10.0, as_mask, skill_focused=True)
    assert torch.allclose(w_10m, torch.tensor(1.35))

    # Thermocline 100m -> 1.95 in AS, 1.50 elsewhere
    w_100m = compute_depth_region_weight(100.0, as_mask, skill_focused=True)
    assert torch.allclose(w_100m[:, :, :, :5], torch.tensor(1.95))
    assert torch.allclose(w_100m[:, :, :, 5:], torch.tensor(1.50))

    # Deep 500m -> 1.30 everywhere
    w_500m = compute_depth_region_weight(500.0, as_mask, skill_focused=True)
    assert torch.allclose(w_500m, torch.tensor(1.30))


def test_oceanembed_loss_backward():
    loss_fn = OceanEmbedLoss()

    eps_pred = torch.randn(2, 1, 20, 20, requires_grad=True)
    eps_true = torch.randn(2, 1, 20, 20)
    ocean_mask = torch.ones(2, 1, 20, 20)
    as_mask = torch.zeros(2, 1, 20, 20)
    bob_mask = torch.ones(2, 1, 20, 20)

    aux_preds = {
        "mld": torch.randn(2, 1, requires_grad=True),
        "blt": torch.randn(2, 1, requires_grad=True),
        "sal_max_depth": torch.randn(2, 1, requires_grad=True),
        "sal_max_strength": torch.randn(2, 1, requires_grad=True),
    }
    aux_trues = torch.randn(2, 4, 20, 20)

    losses = loss_fn(
        eps_pred=eps_pred,
        eps_true=eps_true,
        depth=100.0,
        ocean_mask=ocean_mask,
        arabian_sea_mask=as_mask,
        bob_mask=bob_mask,
        aux_preds=aux_preds,
        aux_trues=aux_trues,
    )

    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()

    assert eps_pred.grad is not None
    assert loss_fn.w1.grad is not None
