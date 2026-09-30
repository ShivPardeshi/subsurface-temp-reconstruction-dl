"""Modular, Interruption-Resilient Training Loop Orchestrator for OceanEmbed.

Features:
1. YAML Configuration registry parsing (Baseline and Ablations).
2. Auto-resume from checkpoint by default (Spot/Preemptible instance resilience).
3. Ablation flags (region conditioning toggle, depth cascade toggle, regional loss weighting toggle).
4. Automatic mixed precision (bf16 / fp16 / fp32).
5. Validation loop with Best-Checkpoint selection based on validation RMSE.
6. Telemetry monitoring and GCP compute budget tracking.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import os
import time
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.sampling.ddim_sampler import DDIMSampler
from src.sampling.depth_cascade import DepthCascadeSampler
from src.training.losses import OceanEmbedLoss
from src.training.dataset import OceanEmbedDataset, create_dataloader
from src.training.checkpoint_utils import save_checkpoint, load_checkpoint
from src.training.monitoring import TrainingMonitor
from src.training.budget_tracker import BudgetTracker
from src.utils.grid import CANONICAL_DEPTHS
from src.utils.logging_config import get_logger

logger = get_logger("train")


def build_models_from_config(config: Dict[str, Any], device: torch.device) -> Tuple[
    ContextEncoder, UNetDenoiser, AuxiliaryHeads, GaussianDiffusion, OceanEmbedLoss
]:
    """Instantiate model components according to configuration."""
    arch = config.get("architecture", {})
    train_cfg = config.get("training", {})
    in_channels = arch.get("in_channels", 25)
    convlstm_hidden = tuple(arch.get("convlstm_hidden", [32, 64, 64]))
    unet_stages = tuple(arch.get("unet_stages", [32, 64, 128, 256]))
    unet_in_channels = arch.get("unet_in_channels", 74)
    non_spatial_dim = arch.get("non_spatial_cond_dim", 14)
    timesteps = arch.get("diffusion_timesteps", 1000)
    normalize_inputs = arch.get("normalize_inputs", True)
    norm_stats_path = arch.get("norm_stats_path", "data/processed/channel_normalization_stats.json")
    skill_focused = train_cfg.get("skill_focused_loss", True)
    inverse_variance = train_cfg.get("inverse_variance_loss", False)
    use_min_snr = train_cfg.get("use_min_snr", True)
    min_snr_gamma = float(train_cfg.get("min_snr_gamma", 5.0))

    context_encoder = ContextEncoder(
        in_channels=in_channels,
        hidden_dims=convlstm_hidden,
        normalize_inputs=normalize_inputs,
        norm_stats_path=norm_stats_path,
    ).to(device)
    unet = UNetDenoiser(
        in_channels=unet_in_channels,  # 74: 1 (x_t) + 1 (x_prev) + 72 (spatial_cond with SSHA & grad)
        stage_channels=unet_stages,
        cond_in_dim=non_spatial_dim,
    ).to(device)
    aux_heads = AuxiliaryHeads(in_features=convlstm_hidden[-1]).to(device)
    diffusion = GaussianDiffusion(timesteps=timesteps, schedule_type=arch.get("noise_schedule", "cosine")).to(device)
    vexp_raw = train_cfg.get("variance_weight_exponent", 1.50)
    try:
        variance_exponent = float(vexp_raw)
    except (ValueError, TypeError):
        variance_exponent = str(vexp_raw)
    loss_fn = OceanEmbedLoss(
        skill_focused=skill_focused,
        inverse_variance=inverse_variance,
        use_min_snr=use_min_snr,
        min_snr_gamma=min_snr_gamma,
        variance_exponent=variance_exponent,
    ).to(device)

    return context_encoder, unet, aux_heads, diffusion, loss_fn


def create_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def create_warm_restart_scheduler(
    optimizer: optim.Optimizer,
    start_step: int,
    total_steps: int,
    warmup_steps: int = 1000,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """Warm restart scheduler for extending training past 20,000 steps (Item 3.1)."""
    active_steps = max(1, total_steps - start_step)
    warmup = min(warmup_steps, max(1, active_steps // 4))

    def lr_lambda(step: int) -> float:
        if step < start_step:
            return 1.0
        rel_step = step - start_step
        if rel_step < warmup:
            return float(rel_step + 1) / float(max(1, warmup))
        progress = float(rel_step - warmup) / float(max(1, active_steps - warmup))
        cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def evaluate_validation_rmse(
    context_encoder: nn.Module,
    unet: nn.Module,
    diffusion: GaussianDiffusion,
    val_loader: Any,
    device: torch.device,
    config: Dict[str, Any],
    max_val_batches: int = 5,
    aux_heads: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate RMSE on validation sequences across all 15 canonical depths (Bottleneck 2)."""
    context_encoder.eval()
    unet.eval()

    ddim = DDIMSampler(diffusion=diffusion, num_ddim_timesteps=10, eta=0.0, schedule_type="quadratic")
    scales_path = config.get("data", {}).get("depth_scales_path", "data/processed/anomaly_depth_scales.json")
    cascade = DepthCascadeSampler(
        context_encoder=context_encoder,
        unet_denoiser=unet,
        ddim_sampler=ddim,
        depths=CANONICAL_DEPTHS,  # Full 15 depths down to 1000m
        depth_scales_path=scales_path,
        aux_heads=aux_heads,
    )

    all_sq_errs = []
    surf_sq_errs = []
    thermo_sq_errs = []
    deep_sq_errs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break
            x_seq = batch["x_seq"].to(device)
            static_feats = batch["static_features"].to(device)
            anomaly_true_std = batch["anomaly_target"].to(device)  # standardized
            scalar_cond = batch["scalar_cond"].to(device)
            anomaly_stds = batch["anomaly_stds"].to(device)

            if not config.get("ablations", {}).get("region_conditioning_enabled", True):
                static_feats[:, 2:6] = 0.0

            use_cascade = config.get("ablations", {}).get("depth_cascade_enabled", True)
            out = cascade.sample_full_profile(
                x_seq=x_seq,
                static_features=static_feats,
                scalar_conditions=scalar_cond,
                use_cascade=use_cascade,
            )
            pred = out["anomalies"]  # Physical anomalies in °C

            # Unstandardize ground truth to physical °C
            if anomaly_stds.dim() == 1:
                stds_4d = anomaly_stds.view(1, 15, 1, 1)
            elif anomaly_stds.dim() == 2:
                stds_4d = anomaly_stds.view(-1, 15, 1, 1)
            else:
                stds_4d = anomaly_stds
            true_phys = anomaly_true_std * stds_4d

            ocean_mask = (static_feats[:, 0:1] > 0.5).expand_as(pred)
            denom = torch.clamp(ocean_mask.sum(dim=(-2, -1), keepdim=True), min=1.0)
            sq_diff = ((pred - true_phys) ** 2) * ocean_mask

            # Per-depth MSE for each sample: (B, 15)
            per_depth_mse = (sq_diff.sum(dim=(-2, -1)) / denom.squeeze(-1)).mean(dim=0)  # (15,)

            all_sq_errs.append(per_depth_mse.mean().item())
            surf_sq_errs.append(per_depth_mse[:5].mean().item())        # 0-30m
            thermo_sq_errs.append(per_depth_mse[5:11].mean().item())    # 50-200m
            deep_sq_errs.append(per_depth_mse[11:].mean().item())       # 250-1000m

    def _rmse(err_list):
        return float(math.sqrt(sum(err_list) / len(err_list))) if err_list else 1.0

    return {
        "val_rmse": _rmse(all_sq_errs),
        "val_surface_rmse": _rmse(surf_sq_errs),
        "val_thermo_rmse": _rmse(thermo_sq_errs),
        "val_deep_rmse": _rmse(deep_sq_errs),
    }


def train_model(
    config_path: str,
    override_epochs: Optional[int] = None,
    override_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train OceanEmbed according to configuration with auto-resume."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp = config.get("experiment", {})
    run_name = exp.get("name", "oceanembed_training")
    run_dir = Path(exp.get("run_dir", f"checkpoints/{run_name}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== STARTING TRAINING: {run_name} ON {device} ===")

    train_cfg = config.get("training", {})
    ablations = config.get("ablations", {})

    # 1. Dataset & Loaders
    data_cfg = config.get("data", {})
    augment_train = train_cfg.get("augmentation_enabled", True)
    scales_path = data_cfg.get("depth_scales_path", "data/processed/anomaly_depth_scales.json")
    train_base_dataset = OceanEmbedDataset(
        inputs_zarr_path=data_cfg.get("inputs_zarr"),
        anomaly_targets_zarr_path=data_cfg.get("anomaly_targets_zarr"),
        aux_targets_zarr_path=data_cfg.get("aux_targets_zarr"),
        scalar_csv_path=data_cfg.get("scalar_csv"),
        depth_scales_path=scales_path,
        sequence_length=data_cfg.get("sequence_length", 7),
        augment=augment_train,
    )
    val_base_dataset = OceanEmbedDataset(
        inputs_zarr_path=data_cfg.get("inputs_zarr"),
        anomaly_targets_zarr_path=data_cfg.get("anomaly_targets_zarr"),
        aux_targets_zarr_path=data_cfg.get("aux_targets_zarr"),
        scalar_csv_path=data_cfg.get("scalar_csv"),
        depth_scales_path=scales_path,
        sequence_length=data_cfg.get("sequence_length", 7),
        augment=False,
    )
    logger.info(f"Dataset loaded with {len(train_base_dataset)} sequence samples (Augmentation: {augment_train}).")

    if len(train_base_dataset) > 100:
        # Full year 365-day dataset: ~359 sequences
        # Train: indices 0 to 236 (~Jan 1 to Aug 31, 237 sequences)
        # Val: indices 237 to 297 (~Sep 1 to Oct 31, 61 sequences)
        # Test: indices 298 to len(dataset)-1 (~Nov 1 to Dec 31, 61 sequences)
        train_end = min(237, int(len(train_base_dataset) * 0.66))
        val_end = min(298, int(len(train_base_dataset) * 0.83))
        train_dataset = torch.utils.data.Subset(train_base_dataset, range(0, train_end))
        val_dataset = torch.utils.data.Subset(val_base_dataset, range(train_end, val_end))
        logger.info(f"Split dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(train_base_dataset) - val_end}")
    else:
        train_dataset = train_base_dataset
        val_dataset = val_base_dataset
        logger.info(f"Using full dataset ({len(train_base_dataset)} samples) for train/val.")

    batch_size = train_cfg.get("batch_size", 4)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    # 2. Build Models, Optimizer, and Schedulers
    (
        context_encoder,
        unet,
        aux_heads,
        diffusion,
        loss_fn,
    ) = build_models_from_config(config, device=device)

    all_params = (
        list(context_encoder.parameters())
        + list(unet.parameters())
        + list(aux_heads.parameters())
        + list(loss_fn.parameters())
    )
    lr = float(train_cfg.get("learning_rate", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    epochs = override_epochs or train_cfg.get("epochs", 50)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    if override_steps is not None:
        max_steps = override_steps
        epochs = max(epochs, (override_steps + len(train_loader) - 1) // max(1, len(train_loader)))
    else:
        max_steps = epochs * len(train_loader)

    warmup_steps = min(train_cfg.get("lr_warmup_steps", 500), max_steps // 4)
    scheduler = create_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=max_steps)

    # 3. Telemetry & Monitoring
    monitor = TrainingMonitor(log_dir=str(run_dir / "logs"))
    tracker = BudgetTracker(run_name=run_name, log_dir=str(run_dir / "logs"))

    # 4. Auto-Resume Check & Warm Restart Support (Item 3.1)
    last_ckpt_path = run_dir / "last_checkpoint.pt"
    best_ckpt_path = run_dir / "best_checkpoint.pt"
    start_step = 0
    start_epoch = 0
    best_val_rmse = float("inf")

    model_dict = {
        "context_encoder": context_encoder,
        "unet": unet,
        "aux_heads": aux_heads,
    }

    warm_restart = train_cfg.get("warm_restart", False)
    init_from_checkpoint = train_cfg.get("init_from_checkpoint", None)

    ckpt_to_load = None
    if resume and last_ckpt_path.exists():
        ckpt_to_load = last_ckpt_path
    elif init_from_checkpoint is not None and Path(init_from_checkpoint).exists():
        ckpt_to_load = Path(init_from_checkpoint)
        logger.info(f"Explicit initialization checkpoint specified: {ckpt_to_load}")

    if ckpt_to_load is not None:
        logger.info(f"Loading checkpoint from {ckpt_to_load}...")
        state = load_checkpoint(
            checkpoint_path=ckpt_to_load,
            models=model_dict,
            optimizer=optimizer if resume else None,
            scheduler=None if (warm_restart or not resume) else scheduler,
            loss_fn=loss_fn,
            device=device,
        )
        if resume:
            start_step = state.get("step", 0)
            start_epoch = state.get("epoch", 0)
            best_val_rmse = state.get("metrics", {}).get("best_val_rmse", float("inf"))
            logger.info(f"Resumed from Step {start_step}, Epoch {start_epoch} (Best Val RMSE: {best_val_rmse:.4f})")
        else:
            start_step = 0
            start_epoch = 0
            best_val_rmse = float("inf")
            logger.info(f"Model weights initialized from {ckpt_to_load}. Fine-tuning starts from Step 0 to {max_steps}.")

        if warm_restart and resume:
            warmup_steps = train_cfg.get("lr_warmup_steps", 1000)
            scheduler = create_warm_restart_scheduler(
                optimizer,
                start_step=start_step,
                total_steps=max_steps,
                warmup_steps=warmup_steps,
                min_lr_ratio=train_cfg.get("min_lr_ratio", 0.05),
            )
            logger.info(f"Warm-restart scheduler active from Step {start_step} to {max_steps} (warmup={warmup_steps})")
    elif not resume:
        logger.info(f"Fresh training requested (resume=False). Initializing from step 0.")

    # Precision settings
    prec = train_cfg.get("precision", "bf16")
    use_amp = device.type == "cuda" and prec in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16

    # Training Loop
    global_step = start_step
    logger.info(f"Beginning training loop: {global_step}/{max_steps} steps (Target Epochs: {epochs})")

    for epoch in range(start_epoch, epochs):
        if global_step >= max_steps:
            break

        context_encoder.train()
        unet.train()
        aux_heads.train()
        loss_fn.train()

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            t0 = time.time()
            x_seq = batch["x_seq"].to(device)
            static_features = batch["static_features"].to(device)
            anomaly_target = batch["anomaly_target"].to(device)  # standardized
            aux_trues = batch["aux_targets"].to(device)
            scalar_cond = batch["scalar_cond"].to(device)
            clim_mean_temps = batch["clim_mean_temps"].to(device)
            anomaly_stds = batch["anomaly_stds"].to(device)

            # Ablation 1: Region conditioning toggle
            if not ablations.get("region_conditioning_enabled", True):
                static_features[:, 2:6] = 0.0

            b, _, _, h, w = x_seq.shape
            ocean_mask = (static_features[:, 0:1] > 0.5).float()
            arabian_sea_mask = static_features[:, 2:3]
            bob_mask = static_features[:, 3:4]

            # Ablation 2: Regional loss boost toggle
            if not ablations.get("regional_loss_boost_enabled", True):
                arabian_sea_mask = torch.zeros_like(arabian_sea_mask)

            # Compute shared context and auxiliary features once per batch
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    u_cond = context_encoder(x_seq)
                    aux_preds = aux_heads(u_cond)
            else:
                u_cond = context_encoder(x_seq)
                aux_preds = aux_heads(u_cond)

            # Direct SSHA & Gradient Magnitude Injection (Bottleneck 11)
            ssha = x_seq[:, -1, 2:3]
            if unet.in_channels == 74:
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=ssha.dtype, device=device).view(1, 1, 3, 3) / 8.0
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=ssha.dtype, device=device).view(1, 1, 3, 3) / 8.0
                gx = F.conv2d(ssha, sobel_x, padding=1)
                gy = F.conv2d(ssha, sobel_y, padding=1)
                ssha_grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
                spatial_cond = torch.cat([u_cond, static_features, ssha, ssha_grad_mag], dim=1)
            else:
                spatial_cond = torch.cat([u_cond, static_features], dim=1)

            # Tri-Stratum Balanced Column Sampling (Bottleneck 10)
            # Stratum 1: Surface [0, 5, 10, 20, 30m] (indices 0..4)
            # Stratum 2: Thermocline Core [50, 75, 100, 125, 150m] (indices 5..9)
            # Stratum 3: Deep Stratified Abyss [200, 300, 500, 700, 1000m] (indices 10..14)
            if torch.rand(1).item() < 0.40:
                # 40% consecutive depth pair [k-1, k] to train vertical gradient continuity
                k_curr = torch.randint(1, len(CANONICAL_DEPTHS), (1,)).item()
                column_depth_indices = [k_curr - 1, k_curr]
            else:
                # 60% tri-stratum balanced mini-column: 1 surface, 1 thermocline, 1 deep
                k_surf = torch.randint(0, 5, (1,)).item()
                k_thermo = torch.randint(5, 10, (1,)).item()
                k_deep = torch.randint(10, len(CANONICAL_DEPTHS), (1,)).item()
                column_depth_indices = [k_surf, k_thermo, k_deep]

            batch_losses = []
            total_step_loss = 0.0
            for depth_idx in column_depth_indices:
                depth = CANONICAL_DEPTHS[depth_idx]
                x0_true = anomaly_target[:, depth_idx : depth_idx + 1]

                # Sample diffusion timestep & noise
                t = torch.randint(0, diffusion.timesteps, (b,), device=device)
                noise = torch.randn_like(x0_true)
                x_t, eps_true = diffusion.q_sample(x0_true, t, noise)

                # Realistic depth-adaptive cascade conditioning with jitter & dropout (Bottlenecks 3 & 15)
                prev_clean = torch.zeros_like(x0_true)
                prev_mean = torch.zeros((b, 1), device=device)
                prev_std = torch.zeros((b, 1), device=device)

                if ablations.get("depth_cascade_enabled", True) and depth_idx > 0:
                    clean_prev_target = anomaly_target[:, depth_idx - 1 : depth_idx]
                    # 20% cascade conditioning dropout
                    if torch.rand(1).item() > 0.20:
                        std_ratio = float(anomaly_stds[0, depth_idx] if anomaly_stds.dim() > 1 else anomaly_stds[depth_idx]) / 0.5408
                        jitter_scale = min(0.75, 0.30 + 0.35 * std_ratio)
                        jitter = torch.randn_like(clean_prev_target) * jitter_scale
                        prev_clean = clean_prev_target + jitter
                        prev_mean = clean_prev_target.mean(dim=(-2, -1), keepdim=True).view(b, 1)
                        prev_std = clean_prev_target.std(dim=(-2, -1), keepdim=True).view(b, 1)

                # Vertical lapse rate Gamma_clim(d) and layer thickness (Bottleneck 12)
                if depth_idx > 0:
                    dz = float(depth - CANONICAL_DEPTHS[depth_idx - 1])
                    c_prev_val = float(clim_mean_temps[0, depth_idx - 1] if clim_mean_temps.dim() > 1 else clim_mean_temps[depth_idx - 1])
                    c_curr_val = float(clim_mean_temps[0, depth_idx] if clim_mean_temps.dim() > 1 else clim_mean_temps[depth_idx])
                    lapse_val = (c_prev_val - c_curr_val) / dz  # °C / m
                else:
                    dz = 5.0
                    lapse_val = 0.0
                lapse_tensor = torch.full((b, 1), lapse_val / 0.10, device=device)  # normalized by 0.10 °C/m
                dz_tensor = torch.full((b, 1), dz / 100.0, device=device)  # normalized by 100m

                # Conditioning vectors: log-depth + climatology temperature (Bottleneck 4, 12, 14)
                log_depth_val = math.log(depth + 1.0) / math.log(1001.0)
                log_depth = torch.full((b, 1), log_depth_val, device=device)
                if clim_mean_temps.dim() > 1:
                    clim_val = clim_mean_temps[:, depth_idx : depth_idx + 1] / 30.0
                else:
                    clim_val = torch.full((b, 1), float(clim_mean_temps[depth_idx]) / 30.0, device=device)
                t_norm = (t.float() / diffusion.timesteps).unsqueeze(1)

                if unet.cond_mlp.net[0].in_features == 14:
                    mld_cond = aux_preds["mld"].detach() / 50.0
                    blt_cond = aux_preds["blt"].detach() / 20.0
                    sal_cond = aux_preds["sal_max_depth"].detach() / 100.0
                    non_spatial = torch.cat([scalar_cond, log_depth, clim_val, lapse_tensor, dz_tensor, prev_mean, prev_std, mld_cond, blt_cond, sal_cond, t_norm], dim=1)
                else:
                    non_spatial = torch.cat([scalar_cond, log_depth, clim_val, prev_mean, prev_std, t_norm], dim=1)

                # Stratification metadata for physics loss (Bottleneck 7)
                prev_x0_val = None
                prev_d_val = None
                s_curr = None
                s_prev = None
                c_curr = None
                c_prev = None
                if depth_idx > 0:
                    prev_x0_val = anomaly_target[:, depth_idx - 1 : depth_idx]
                    prev_d_val = CANONICAL_DEPTHS[depth_idx - 1]
                    s_curr = float(anomaly_stds[0, depth_idx] if anomaly_stds.dim() > 1 else anomaly_stds[depth_idx])
                    s_prev = float(anomaly_stds[0, depth_idx - 1] if anomaly_stds.dim() > 1 else anomaly_stds[depth_idx - 1])
                    c_curr = float(clim_mean_temps[0, depth_idx] if clim_mean_temps.dim() > 1 else clim_mean_temps[depth_idx])
                    c_prev = float(clim_mean_temps[0, depth_idx - 1] if clim_mean_temps.dim() > 1 else clim_mean_temps[depth_idx - 1])

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        eps_pred = unet(
                            x_noisy=x_t,
                            spatial_cond=spatial_cond,
                            non_spatial_cond=non_spatial,
                            prev_depth_clean=prev_clean,
                        )
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
                            prev_x0_true=prev_x0_val,
                            prev_depth=prev_d_val,
                            clim_temp=c_curr,
                            prev_clim_temp=c_prev,
                            depth_std=s_curr,
                            prev_depth_std=s_prev,
                        )
                        loss_step = losses["loss_total"] / (grad_accum_steps * len(column_depth_indices))
                else:
                    eps_pred = unet(
                        x_noisy=x_t,
                        spatial_cond=spatial_cond,
                        non_spatial_cond=non_spatial,
                        prev_depth_clean=prev_clean,
                    )
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
                        prev_x0_true=prev_x0_val,
                        prev_depth=prev_d_val,
                        clim_temp=c_curr,
                        prev_clim_temp=c_prev,
                        depth_std=s_curr,
                        prev_depth_std=s_prev,
                    )
                    loss_step = losses["loss_total"] / (grad_accum_steps * len(column_depth_indices))

                total_step_loss = total_step_loss + loss_step
                batch_losses.append(losses)

            total_step_loss.backward()

            # Average losses across the sampled mini-column depths
            avg_losses = {
                k: sum(l[k] for l in batch_losses) / len(batch_losses)
                for k in batch_losses[0]
            }

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if train_cfg.get("max_grad_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, train_cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            step_sec = time.time() - t0
            tracker.record_step(step_sec)
            global_step += 1

            current_lr = scheduler.get_last_lr()[0]
            monitor.log_step(global_step, avg_losses, current_lr)

            if global_step % 20 == 0:
                logger.info(
                    f"Step {global_step:05d}/{max_steps:05d} (Epoch {epoch+1:02d}/{epochs}) | "
                    f"Total Loss: {float(avg_losses['loss_total'].item()):.4f} | "
                    f"Diff: {float(avg_losses['loss_diffusion'].item()):.4f} | "
                    f"Aux: {float(avg_losses['loss_aux'].item()):.4f} | "
                    f"w1={float(avg_losses['w1'].item()):.2f}, w2={float(avg_losses['w2'].item()):.2f} | "
                    f"{tracker.get_summary()['avg_step_ms']:.1f} ms/step"
                )

            # Checkpointing
            ckpt_interval = train_cfg.get("checkpoint_interval_steps", 500)
            if global_step % ckpt_interval == 0:
                save_checkpoint(
                    checkpoint_dir=run_dir,
                    filename="last_checkpoint.pt",
                    models=model_dict,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epoch=epoch,
                    step=global_step,
                    metrics={"best_val_rmse": best_val_rmse},
                )
                tracker.save_report()

        # Validation & Checkpoint Selection (gated by val_interval_steps or end of training)
        val_interval = train_cfg.get("val_interval_steps", 50)
        is_val_step = (global_step % val_interval == 0) or (global_step >= max_steps)
        if is_val_step:
            logger.info(f"Running validation evaluation across all 15 depths at step {global_step} (Epoch {epoch+1})...")
            val_metrics = evaluate_validation_rmse(
                context_encoder, unet, diffusion, val_loader, device, config, aux_heads=aux_heads
            )
            val_rmse = val_metrics["val_rmse"]
            monitor.log_validation(global_step, val_rmse)
            logger.info(
                f"Step {global_step} / Epoch {epoch+1} Complete | "
                f"Validation RMSE: {val_rmse:.4f} °C (Surface: {val_metrics['val_surface_rmse']:.4f}, "
                f"Thermo: {val_metrics['val_thermo_rmse']:.4f}, Deep: {val_metrics['val_deep_rmse']:.4f})"
            )

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                logger.info(f"New Best Validation RMSE: {best_val_rmse:.4f} °C -> Saving best_checkpoint.pt")
                save_checkpoint(
                    checkpoint_dir=run_dir,
                    filename="best_checkpoint.pt",
                    models=model_dict,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    epoch=epoch,
                    step=global_step,
                    metrics={"best_val_rmse": best_val_rmse},
                )

    # Final Save
    save_checkpoint(
        checkpoint_dir=run_dir,
        filename="last_checkpoint.pt",
        models=model_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epoch=epochs,
        step=global_step,
        metrics={"best_val_rmse": best_val_rmse},
    )
    tracker.save_report()
    logger.info(f"=== TRAINING COMPLETE: {run_name} (Best Val RMSE: {best_val_rmse:.4f}) ===")

    return {
        "run_name": run_name,
        "total_steps": global_step,
        "best_val_rmse": best_val_rmse,
        "budget": tracker.get_summary(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed Phase 4 Training Orchestrator")
    parser.add_argument("--config", type=str, default="src/training/config_registry/baseline_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    train_model(config_path=args.config, override_epochs=args.epochs, override_steps=args.steps)
