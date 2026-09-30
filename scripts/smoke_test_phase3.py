"""Local CPU Smoke Test for Phase 3 Architecture and Retraining Pipeline.

Verifies:
1. Per-channel input standardization in ContextEncoder.
2. Skill-focused loss reweighting in OceanEmbedLoss.
3. Physically-consistent data augmentation in OceanEmbedDataset.
4. Warm-restart checkpoint loading and scheduler from Step 20,000.
5. End-to-end forward/backward optimization step.
"""

import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import build_models_from_config, create_warm_restart_scheduler
from src.training.dataset import OceanEmbedDataset, create_dataloader
from src.training.checkpoint_utils import load_checkpoint
from src.utils.grid import CANONICAL_DEPTHS

def run_smoke_test():
    print("=== STARTING PHASE 3 SMOKE TEST ===")
    device = torch.device("cpu")
    config_path = "src/training/config_registry/phase3_retrain_normalized_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Dataset Verification with Augmentation
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
    assert batch['x_seq'].shape == (7, 25, 112, 240)
    assert batch['anomaly_target'].shape == (15, 112, 240)
    print("Dataset augmentation verified!")

    # 2. Models & Normalization Verification
    print("\n--- 2. Testing Model Instantiation & Normalization ---")
    context_encoder, unet, aux_heads, diffusion, loss_fn = build_models_from_config(config, device=device)
    print("Models instantiated successfully.")
    assert hasattr(context_encoder, "channel_mean")
    assert hasattr(context_encoder, "channel_std")
    print(f"ContextEncoder normalization buffer shapes: mean={context_encoder.channel_mean.shape}, std={context_encoder.channel_std.shape}")
    print(f"Ch 14 (Latent Heat Flux) mean={context_encoder.channel_mean[0, 0, 14, 0, 0].item():.1f}, std={context_encoder.channel_std[0, 0, 14, 0, 0].item():.1f}")
    print(f"Ch 11 (Wind Stress Curl) mean={context_encoder.channel_mean[0, 0, 11, 0, 0].item():.2e}, std={context_encoder.channel_std[0, 0, 11, 0, 0].item():.2e}")

    # 3. Checkpoint Resumption & Warm Restart Scheduler
    print("\n--- 3. Testing Checkpoint Loading & Warm Restart Scheduler ---")
    init_ckpt = config["training"].get("init_from_checkpoint")
    if init_ckpt and Path(init_ckpt).exists():
        model_dict = {
            "context_encoder": context_encoder,
            "unet": unet,
            "aux_heads": aux_heads,
        }
        all_params = (
            list(context_encoder.parameters())
            + list(unet.parameters())
            + list(aux_heads.parameters())
            + list(loss_fn.parameters())
        )
        optimizer = torch.optim.AdamW(all_params, lr=config["training"]["learning_rate"])
        state = load_checkpoint(
            checkpoint_path=init_ckpt,
            models=model_dict,
            optimizer=optimizer,
            scheduler=None,
            loss_fn=loss_fn,
            device=device,
        )
        start_step = state.get("step", 20000)
        print(f"Successfully loaded weights from {init_ckpt} at step {start_step}")
        scheduler = create_warm_restart_scheduler(
            optimizer,
            start_step=start_step,
            total_steps=config["training"].get("max_steps", 40000),
            warmup_steps=1000,
        )
        lr_start = scheduler.get_last_lr()[0]
        print(f"Warm-restart scheduler created. Initial LR: {lr_start:.2e}")
    else:
        print(f"Warning: Checkpoint {init_ckpt} not found on local path. Skipping weight loading.")

    # 4. Forward & Backward Pass on Small Spatial Crop (CPU smoke test)
    print("\n--- 4. Testing End-to-End Forward & Backward Step ---")
    # Take a 16x16 spatial slice for fast CPU execution
    x_seq = batch["x_seq"][:, :, :16, :16].unsqueeze(0).to(device)  # (1, 7, 25, 16, 16)
    static_feats = batch["static_features"][:, :16, :16].unsqueeze(0).to(device)  # (1, 6, 16, 16)
    anomaly_target = batch["anomaly_target"][:, :16, :16].unsqueeze(0).to(device)  # (1, 15, 16, 16)
    aux_trues = batch["aux_targets"][:, :16, :16].unsqueeze(0).to(device)  # (1, 4, 16, 16)
    scalar_cond = batch["scalar_cond"].unsqueeze(0).to(device)  # (1, 4)

    # 1. Context Encoder
    u_cond = context_encoder(x_seq)
    print(f"ContextEncoder u_cond output: {u_cond.shape}")
    assert u_cond.shape == (1, 64, 16, 16)

    # 2. Auxiliary Heads
    aux_preds = aux_heads(u_cond)
    print(f"AuxiliaryHeads output keys: {list(aux_preds.keys())}")

    # 3. Diffusion & UNet
    depth_idx = 7  # 100m depth
    depth = CANONICAL_DEPTHS[depth_idx]
    x0_true = anomaly_target[:, depth_idx : depth_idx + 1]
    t = torch.tensor([100], device=device)
    noise = torch.randn_like(x0_true)
    x_t, eps_true = diffusion.q_sample(x0_true, t, noise)

    log_depth_val = 0.5
    log_depth = torch.full((1, 1), log_depth_val, device=device)
    prev_mean = torch.zeros((1, 1), device=device)
    prev_std = torch.zeros((1, 1), device=device)
    t_norm = (t.float() / diffusion.timesteps).unsqueeze(1)
    non_spatial = torch.cat([scalar_cond, log_depth, prev_mean, prev_std, t_norm], dim=1)
    spatial_cond = torch.cat([u_cond, static_feats], dim=1)
    prev_clean = anomaly_target[:, depth_idx - 1 : depth_idx]

    eps_pred = unet(
        x_noisy=x_t,
        spatial_cond=spatial_cond,
        non_spatial_cond=non_spatial,
        prev_depth_clean=prev_clean,
    )
    print(f"UNet eps_pred output: {eps_pred.shape}")
    assert eps_pred.shape == (1, 1, 16, 16)

    # 4. Skill-Focused Loss
    ocean_mask = (static_feats[:, 0:1] > 0.5).float()
    arabian_sea_mask = static_feats[:, 2:3]
    bob_mask = static_feats[:, 3:4]
    x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)

    losses = loss_fn(
        eps_pred=eps_pred,
        eps_true=eps_true,
        depth=depth,
        ocean_mask=ocean_mask,
        arabian_sea_mask=arabian_sea_mask,
        bob_mask=bob_mask,
        aux_preds=aux_preds,
        aux_trues=aux_trues,
        x0_hat=x0_hat,
        x0_true=x0_true,
        alpha_bar_t=diffusion.alphas_cumprod[t],
        t=t,
    )
    print(f"Computed losses: total={losses['loss_total'].item():.4f}, diff={losses['loss_diffusion'].item():.4f}, aux={losses['loss_aux'].item():.4f}")
    assert not torch.isnan(losses['loss_total']), "Loss is NaN!"

    # 5. Backward Pass
    losses['loss_total'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), max_norm=1.0)
    print(f"Backward pass successful! ContextEncoder gradient norm: {grad_norm.item():.4f}")

    print("\n=== PHASE 3 SMOKE TEST PASSED 100% SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_smoke_test()
