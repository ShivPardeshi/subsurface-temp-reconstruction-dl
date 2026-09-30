"""Unit test verifying Phase 7 dampened scaling pipeline and loss exponent."""

import sys
import os
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import yaml

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.training.losses import OceanEmbedLoss
from src.training.dataset import OceanEmbedDataset
from src.utils.grid import CANONICAL_DEPTHS


def test_phase7():
    print("Testing Phase 7 Pipeline...")
    device = torch.device("cpu")

    # 1. Verify dampened scales JSON
    scales_path = "data/processed/anomaly_depth_scales_dampened_p05.json"
    assert os.path.exists(scales_path), f"File missing: {scales_path}"
    with open(scales_path, "r") as f:
        scales = json.load(f)
    assert len(scales["anomaly_stds"]) == 15
    assert scales.get("exponent_p") == 0.5
    print(f"Verified dampened scales: 125m scale={scales['anomaly_stds'][8]:.4f}, 500m scale={scales['anomaly_stds'][12]:.4f}")

    # 2. Test dataset loading with dampened scales
    dataset = OceanEmbedDataset(
        inputs_zarr_path="data/processed/phase2_dataset/oceanembed_training_inputs.zarr",
        anomaly_targets_zarr_path="data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr",
        aux_targets_zarr_path="data/processed/phase2_dataset/oceanembed_auxiliary_targets.zarr",
        scalar_csv_path="data/processed/phase2_dataset/scalar_conditioning.csv",
        depth_scales_path=scales_path,
        sequence_length=7,
        standardize_targets=True,
    )
    sample = dataset[10]
    print("Dataset sample fetched cleanly. Target shape:", sample["anomaly_target"].shape)
    assert not torch.isnan(sample["anomaly_target"]).any()

    # 3. Test loss with variance_exponent=1.50
    loss_fn = OceanEmbedLoss(skill_focused=True, use_min_snr=True, min_snr_gamma=5.0, variance_exponent=1.50)
    eps_pred = torch.randn(2, 1, 16, 16, requires_grad=True)
    eps_true = torch.randn(2, 1, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    aux_preds = {
        "mld": torch.randn(2, 1),
        "blt": torch.randn(2, 1),
        "sal_max_depth": torch.randn(2, 1),
        "sal_max_strength": torch.randn(2, 1),
    }
    aux_trues = torch.randn(2, 4, 16, 16)

    loss_dict = loss_fn(
        eps_pred=eps_pred,
        eps_true=eps_true,
        depth=125.0,
        ocean_mask=mask,
        arabian_sea_mask=mask,
        bob_mask=mask,
        aux_preds=aux_preds,
        aux_trues=aux_trues,
        depth_std=1.058,  # dampened scale at 125m
    )
    loss_total = loss_dict["loss_total"]
    loss_total.backward()
    assert eps_pred.grad is not None and torch.isfinite(eps_pred.grad).all()
    print("Loss and backward pass verified. Total loss:", loss_total.item())

    # 4. Test DepthCascadeSampler with dampened scales
    encoder = ContextEncoder(in_channels=25, hidden_dims=[32, 64, 64]).to(device)
    unet = UNetDenoiser(in_channels=74, stage_channels=[32, 64, 128, 256], cond_in_dim=14).to(device)
    diff = GaussianDiffusion(timesteps=1000, schedule_type="cosine").to(device)
    ddim = DDIMSampler(diff, num_ddim_timesteps=5, eta=0.0)
    cascade = DepthCascadeSampler(
        context_encoder=encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS[:3],
        depth_scales_path=scales_path,
        rescale_output=True,
    )
    out = cascade.sample_full_profile(
        x_seq=sample["x_seq"].unsqueeze(0),
        static_features=sample["static_features"].unsqueeze(0),
        scalar_conditions=sample["scalar_cond"].unsqueeze(0),
    )
    assert out["anomalies"].shape == (1, 3, 112, 240)
    assert not torch.isnan(out["anomalies"]).any()
    print("Cascade sampler verified with dampened scales. Shape:", out["anomalies"].shape)

    print("\nALL PHASE 7 PIPELINE TESTS PASSED CLEANLY!")


if __name__ == "__main__":
    test_phase7()
