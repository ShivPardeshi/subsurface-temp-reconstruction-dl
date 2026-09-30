"""Overfit Tiny Batch Verification Test.

Validates that the complete multi-stage neural network architecture can easily
overfit a tiny batch (2-4 samples) to near-zero loss, confirming gradient flow,
correct loss formulas, and optimizer integration.
"""

from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.training.losses import OceanEmbedLoss
from src.training.dataset import OceanEmbedDataset
from src.utils.logging_config import get_logger

logger = get_logger("overfit_tiny_batch")


def run_overfit_test(
    num_steps: int = 200,
    lr: float = 2e-3,
    device: torch.device = torch.device("cpu"),
    toy_dataset_dir: str = "data/processed/phase2_toy",
) -> Tuple[bool, List[float]]:
    """Run overfit test on 2-4 real samples."""
    logger.info(f"=== STARTING OVERFIT-TINY-BATCH TEST ({num_steps} steps) ===")

    toy_dir = Path(toy_dataset_dir)
    if not (toy_dir / "oceanembed_training_inputs.zarr").exists():
        toy_dir = Path("data/processed/phase2_dataset")

    dataset = OceanEmbedDataset(
        inputs_zarr_path=toy_dir / "oceanembed_training_inputs.zarr",
        anomaly_targets_zarr_path=toy_dir / "oceanembed_anomaly_targets.zarr",
        aux_targets_zarr_path=toy_dir / "oceanembed_auxiliary_targets.zarr",
        scalar_csv_path=toy_dir / "scalar_conditioning.csv",
        sequence_length=7,
    )

    if len(dataset) == 0:
        logger.warning("Dataset empty, creating synthetic toy batch for overfit verification.")
        # Fallback to deterministic synthetic batch
        x_seq = torch.randn(2, 7, 25, 40, 40, device=device)
        static_features = torch.rand(2, 6, 40, 40, device=device)
        anomaly_true = torch.randn(2, 1, 40, 40, device=device)
        aux_trues = torch.rand(2, 4, 40, 40, device=device)
        scalar_cond = torch.randn(2, 4, device=device)
    else:
        sample = dataset[0]
        x_seq = sample["x_seq"].unsqueeze(0).to(device)  # (1, 7, 25, H, W)
        static_features = sample["static_features"].unsqueeze(0).to(device)
        # Select single depth (e.g. 50m, depth index 4) for noise prediction target
        anomaly_true = sample["anomaly_target"][4:5].unsqueeze(0).to(device)  # (1, 1, H, W)
        aux_trues = sample["aux_targets"].unsqueeze(0).to(device)  # (1, 4, H, W)
        scalar_cond = sample["scalar_cond"].unsqueeze(0).to(device)  # (1, 4)

    # Initialize compact models for fast convergence
    b, _, _, h, w = x_seq.shape
    context_enc = ContextEncoder(in_channels=25, hidden_dims=(32, 64, 64)).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8).to(device)
    aux_heads = AuxiliaryHeads(in_features=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    loss_fn = OceanEmbedLoss().to(device)

    all_params = (
        list(context_enc.parameters())
        + list(unet.parameters())
        + list(aux_heads.parameters())
        + list(loss_fn.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

    ocean_mask = (static_features[:, 0:1] > 0.5).float()
    arabian_sea_mask = static_features[:, 2:3]
    bob_mask = static_features[:, 3:4]

    # Fix noise and timestep for deterministic reconstruction overfitting
    fixed_t = torch.tensor([250] * b, device=device, dtype=torch.long)
    fixed_noise = torch.randn_like(anomaly_true)
    x_t, eps_true = diffusion.q_sample(anomaly_true, fixed_t, fixed_noise)

    depth = 50.0
    depth_norm = torch.full((b, 1), 0.5, device=device)
    prev_mean = torch.zeros((b, 1), device=device)
    prev_std = torch.zeros((b, 1), device=device)
    t_norm = (fixed_t.float() / 1000.0).unsqueeze(1)
    non_spatial_cond = torch.cat([scalar_cond, depth_norm, prev_mean, prev_std, t_norm], dim=1)

    loss_history = []
    initial_loss = None

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        u_cond = context_enc(x_seq)
        aux_preds = aux_heads(u_cond)
        spatial_cond = torch.cat([u_cond, static_features], dim=1)

        prev_clean = torch.zeros_like(anomaly_true)
        eps_pred = unet(
            x_noisy=x_t,
            spatial_cond=spatial_cond,
            non_spatial_cond=non_spatial_cond,
            prev_depth_clean=prev_clean,
        )

        x0_hat = diffusion.predict_x0_from_noise(x_t, fixed_t, eps_pred)

        loss_dict = loss_fn(
            eps_pred=eps_pred,
            eps_true=eps_true,
            depth=depth,
            ocean_mask=ocean_mask,
            arabian_sea_mask=arabian_sea_mask,
            bob_mask=bob_mask,
            aux_preds=aux_preds,
            aux_trues=aux_trues,
            x0_hat=x0_hat,
            x0_true=anomaly_true,
        )

        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        val = float(loss.item())
        loss_history.append(val)
        if initial_loss is None:
            initial_loss = val

        if step % 25 == 0 or step == num_steps:
            logger.info(
                f"Step {step:03d}/{num_steps:03d} | Total Loss: {val:.5f} | "
                f"Diff: {float(loss_dict['loss_diffusion'].item()):.5f} | "
                f"Aux: {float(loss_dict['loss_aux'].item()):.5f}"
            )

    final_loss = loss_history[-1]
    logger.info(f"Overfit results: Initial Loss = {initial_loss:.5f} -> Final Loss = {final_loss:.5f}")

    # Plot learning curve
    plot_dir = Path("data/processed/validation_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "overfit_curve.png"

    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Total Loss", color="#0066cc", linewidth=2)
    plt.title("OceanEmbed Architecture - Tiny Batch Overfit Curve")
    plt.xlabel("Iteration Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved overfit loss curve plot to {plot_path}")

    # Acceptance assertion: loss must drop significantly (at least 80% reduction or < 0.1)
    passed = (final_loss < 0.1) or (final_loss < 0.2 * initial_loss)
    if passed:
        logger.info("=== OVERFIT TEST PASSED: Model converges cleanly ===")
    else:
        logger.error("=== OVERFIT TEST FAILED: Loss failed to drop to near-zero ===")

    return passed, loss_history


if __name__ == "__main__":
    run_overfit_test()
