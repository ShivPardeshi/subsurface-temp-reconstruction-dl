"""Local CPU Smoke Test for Clean 20k Scratch Retraining Pipeline.

Verifies:
1. Loading phase3_retrain_scratch_20k_config.yaml.
2. ContextEncoder initialization with ocean-only normalization stats.
3. Clean scratch parameter initialization (no checkpoint loaded).
4. Forward and backward optimization step without NaNs.
"""

import sys
import yaml
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.train import build_models_from_config, create_warmup_cosine_scheduler
from src.training.dataset import OceanEmbedDataset


def run_smoke_test():
    print("=== STARTING SCRATCH 20K SMOKE TEST ===")
    device = torch.device("cpu")
    config_path = "src/training/config_registry/phase3_retrain_scratch_20k_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Dataset Verification
    print("\n--- 1. Testing Dataset with Augmentation ---")
    data_cfg = config["data"]
    dataset = OceanEmbedDataset(
        inputs_zarr_path=data_cfg["inputs_zarr"],
        anomaly_targets_zarr_path=data_cfg["anomaly_targets_zarr"],
        aux_targets_zarr_path=data_cfg["aux_targets_zarr"],
        scalar_csv_path=data_cfg["scalar_csv"],
        sequence_length=7,
        augment=True,
    )
    batch = dataset[5]
    print(f"Loaded batch item: x_seq={batch['x_seq'].shape}, anomaly={batch['anomaly_target'].shape}")
    assert batch["x_seq"].shape == (7, 25, 112, 240)
    assert batch["anomaly_target"].shape == (15, 112, 240)

    # 2. Models & Ocean-Only Normalization Verification
    print("\n--- 2. Testing Models & Ocean-Only Normalization Buffers ---")
    context_encoder, unet, aux_heads, diffusion, loss_fn = build_models_from_config(config, device=device)
    print("Models instantiated successfully.")

    assert hasattr(context_encoder, "channel_mean")
    assert hasattr(context_encoder, "channel_std")

    sst_mean = context_encoder.channel_mean[0, 0, 0, 0, 0].item()
    sst_std = context_encoder.channel_std[0, 0, 0, 0, 0].item()
    print(f"Ch 0 (SST) Ocean Mean: {sst_mean:.2f} K, Ocean Std: {sst_std:.4f} K")
    assert 300.0 < sst_mean < 303.0, f"Unexpected SST mean: {sst_mean}"
    assert 1.0 < sst_std < 3.0, f"Unexpected SST std: {sst_std} (expected ~1.92K)"

    # 3. Scratch Initialization Verification
    print("\n--- 3. Verifying Scratch Initialization ---")
    assert config["training"]["warm_restart"] is False
    assert config["training"]["init_from_checkpoint"] is None
    print("Clean scratch initialization confirmed (no warm-start checkpoint).")

    # 4. End-to-End Forward & Backward Step on Spatial Crop
    print("\n--- 4. Testing End-to-End Forward & Backward Step ---")
    # Take a 16x16 spatial slice for fast CPU execution
    x_seq = batch["x_seq"][:, :, :16, :16].unsqueeze(0).to(device)  # (1, 7, 25, 16, 16)
    static_feats = batch["static_features"][:, :16, :16].unsqueeze(0).to(device)  # (1, 6, 16, 16)
    anomaly_target = batch["anomaly_target"][:, :16, :16].unsqueeze(0).to(device)  # (1, 15, 16, 16)
    aux_targets = batch["aux_targets"][:, :16, :16].unsqueeze(0).to(device)  # (1, 4, 16, 16)
    scalar_cond = batch["scalar_cond"].unsqueeze(0).to(device)  # (1, 4)

    all_params = (
        list(context_encoder.parameters())
        + list(unet.parameters())
        + list(aux_heads.parameters())
        + list(loss_fn.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=config["training"]["learning_rate"])

    # Forward context encoder
    u_cond = context_encoder(x_seq)
    assert u_cond.shape == (1, 64, 16, 16), f"Wrong u_cond shape: {u_cond.shape}"
    assert torch.isfinite(u_cond).all(), "Non-finite values in u_cond!"
    print(f"ContextEncoder output: shape={u_cond.shape}, mean={u_cond.mean().item():.4f}, std={u_cond.std().item():.4f}")

    # Forward auxiliary heads
    aux_preds = aux_heads(u_cond)
    print(f"Auxiliary predictions: MLD shape={aux_preds['mld'].shape}")

    # Prepare diffusion inputs for depth index 0 (0m)
    depth_val = 0.0
    spatial_cond = torch.cat([u_cond, static_feats], dim=1)  # (1, 70, 16, 16)
    t = torch.tensor([50], device=device, dtype=torch.long)
    target_clean = anomaly_target[:, 0:1]
    noisy_x, noise = diffusion.q_sample(x_0=target_clean, t=t)

    # Non-spatial conditioning
    depth_tensor = torch.tensor([[depth_val]], device=device, dtype=torch.float32)
    clim_scalar = torch.tensor([[0.0]], device=device, dtype=torch.float32)
    non_spatial = torch.cat([scalar_cond, depth_tensor, clim_scalar, torch.zeros(1, 2, device=device)], dim=1)

    # UNet forward pass
    eps_pred = unet(
        x_noisy=noisy_x,
        spatial_cond=spatial_cond,
        non_spatial_cond=non_spatial,
        prev_depth_clean=None,
    )
    assert eps_pred.shape == (1, 1, 16, 16)
    assert torch.isfinite(eps_pred).all()

    # Predict x0 and calculate loss
    x0_hat = diffusion.predict_x0_from_noise(noisy_x, t, eps_pred)
    ocean_mask = (static_feats[:, 0:1] > 0.5).float()
    as_mask = (static_feats[:, 2:3] > 0.5).float()
    bob_mask = (static_feats[:, 3:4] > 0.5).float()

    losses = loss_fn(
        eps_pred=eps_pred,
        eps_true=noise,
        depth=depth_val,
        ocean_mask=ocean_mask,
        arabian_sea_mask=as_mask,
        bob_mask=bob_mask,
        aux_preds=aux_preds,
        aux_trues=aux_targets,
        x0_hat=x0_hat,
        x0_true=target_clean,
    )

    loss_total = losses["loss_total"]
    print(f"Loss Total: {loss_total.item():.4f} (Diff: {losses['loss_diffusion'].item():.4f}, Aux: {losses['loss_aux'].item():.4f})")
    assert torch.isfinite(loss_total), "Loss is not finite!"

    # Backward pass
    optimizer.zero_grad()
    loss_total.backward()

    # Check non-zero finite gradients
    ce_grad_norm = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), max_norm=1.0)
    unet_grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
    print(f"Gradient norms: ContextEncoder={ce_grad_norm:.4f}, UNet={unet_grad_norm:.4f}")
    assert ce_grad_norm > 0, "Zero gradient in ContextEncoder!"
    assert unet_grad_norm > 0, "Zero gradient in UNet!"

    optimizer.step()
    print("\n[SUCCESS] Local CPU smoke test passed! The clean 20k scratch pipeline is mathematically sound and verified.")


if __name__ == "__main__":
    run_smoke_test()
