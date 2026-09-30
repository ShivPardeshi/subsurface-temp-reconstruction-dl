"""Shape and Gradient Stability Verification Suite.

Validates end-to-end forward pass, backward pass, tensor dimensions,
and gradient finite checks across all model stages.
"""

from typing import Dict, Any
import torch

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.training.losses import OceanEmbedLoss
from src.utils.logging_config import get_logger

logger = get_logger("shape_checks")


def run_full_pipeline_shape_check(
    batch_size: int = 2,
    grid_h: int = 112,
    grid_w: int = 240,
    seq_len: int = 7,
    device: torch.device = torch.device("cpu"),
) -> bool:
    """Execute end-to-end forward and backward passes with dummy inputs."""
    logger.info(f"Running pipeline shape check on device={device}, shape=({batch_size}, {seq_len}, 25, {grid_h}, {grid_w})")

    # Instantiate modules
    context_enc = ContextEncoder(in_channels=25).to(device)
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8).to(device)
    aux_heads = AuxiliaryHeads(in_features=64).to(device)
    diffusion = GaussianDiffusion(timesteps=1000).to(device)
    loss_fn = OceanEmbedLoss().to(device)

    # 1. Create dummy inputs matching Phase 2 outputs
    x_seq = torch.randn(batch_size, seq_len, 25, grid_h, grid_w, device=device)
    static_features = torch.rand(batch_size, 6, grid_h, grid_w, device=device)
    ocean_mask = (static_features[:, 0:1] > 0.5).float()
    arabian_sea_mask = static_features[:, 2:3]
    bob_mask = static_features[:, 3:4]

    anomaly_true = torch.randn(batch_size, 1, grid_h, grid_w, device=device)
    aux_trues = torch.rand(batch_size, 4, grid_h, grid_w, device=device)
    scalar_cond = torch.randn(batch_size, 4, device=device)  # sin, cos, oni, iod

    # 2. Stage 1: Context Encoder
    u_cond = context_enc(x_seq)
    assert u_cond.shape == (batch_size, 64, grid_h, grid_w), f"Unexpected u_cond shape: {u_cond.shape}"

    # 3. Stage 5: Auxiliary Heads
    aux_preds = aux_heads(u_cond)
    assert aux_preds["mld"].shape == (batch_size, 1), f"Unexpected mld shape: {aux_preds['mld'].shape}"
    assert aux_preds["blt"].shape == (batch_size, 1), f"Unexpected blt shape: {aux_preds['blt'].shape}"
    assert aux_preds["sal_max_depth"].shape == (batch_size, 1), f"Unexpected sal_max_depth shape"

    # 4. Spatial conditioning fusion: 64 + 6 = 70 channels
    spatial_cond = torch.cat([u_cond, static_features], dim=1)
    assert spatial_cond.shape == (batch_size, 70, grid_h, grid_w), f"Unexpected spatial_cond shape"

    # 5. Diffusion forward process
    t = torch.randint(0, 1000, (batch_size,), device=device)
    noise = torch.randn_like(anomaly_true)
    x_t, eps_true = diffusion.q_sample(anomaly_true, t, noise)
    assert x_t.shape == (batch_size, 1, grid_h, grid_w)

    # Non-spatial condition vector (8 dims: 4 scalar + log_depth + prev_mean + prev_std + t_norm)
    depth = 75.0
    depth_norm = torch.full((batch_size, 1), 0.5, device=device)
    prev_mean = torch.zeros((batch_size, 1), device=device)
    prev_std = torch.zeros((batch_size, 1), device=device)
    t_norm = (t.float() / 1000.0).unsqueeze(1)
    non_spatial_cond = torch.cat([scalar_cond, depth_norm, prev_mean, prev_std, t_norm], dim=1)
    assert non_spatial_cond.shape == (batch_size, 8)

    # 6. U-Net Denoiser forward
    prev_clean = torch.zeros_like(anomaly_true)
    eps_pred = unet(
        x_noisy=x_t,
        spatial_cond=spatial_cond,
        non_spatial_cond=non_spatial_cond,
        prev_depth_clean=prev_clean,
    )
    assert eps_pred.shape == (batch_size, 1, grid_h, grid_w), f"Unexpected eps_pred shape: {eps_pred.shape}"

    # 7. Clean estimate reconstruction
    x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)
    assert x0_hat.shape == (batch_size, 1, grid_h, grid_w)

    # 8. Loss computation
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
        x0_true=anomaly_true,
    )

    loss_total = losses["loss_total"]
    assert torch.isfinite(loss_total), "loss_total is not finite (NaN or Inf)"

    # 9. Backward pass and gradient check
    loss_total.backward()

    for name, param in list(context_enc.named_parameters()) + list(unet.named_parameters()) + list(aux_heads.named_parameters()):
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    logger.info("All shape and gradient stability checks passed successfully!")
    return True


if __name__ == "__main__":
    run_full_pipeline_shape_check()
