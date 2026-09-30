"""Loss Functions with Adaptive Homoscedastic Multi-Task Weighting.

Implements:
1. alpha(d, region) depth-and-region weighting function.
2. L_diffusion: depth-and-region weighted noise prediction MSE.
3. L_aux: region-masked auxiliary target loss across MLD, BLT, and Salinity Maximum.
4. L_physics: thermocline consistency loss on predicted clean data estimate x0_hat.
5. OceanEmbedLoss: Adaptive homoscedastic multi-task loss with learnable parameters w1, w2, w3.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_depth_region_weight(
    depth: float,
    arabian_sea_mask: torch.Tensor,
    skill_focused: bool = False,
    inverse_variance: bool = False,
) -> torch.Tensor:
    """Compute depth-and-region weighting factor alpha(d, region).

    Args:
        depth: Depth in meters (float).
        arabian_sea_mask: Tensor of shape (B, 1, H, W) with AS membership [0, 1].
        skill_focused: If True, uses skill-targeted depth schedule.
        inverse_variance: If True, uses Model V2 inverse-variance depth schedule.

    Returns:
        Weight tensor of shape (B, 1, H, W).
    """
    weights = torch.ones_like(arabian_sea_mask)
    as_cond = (arabian_sea_mask > 0.5).float()

    if inverse_variance:
        if depth < 20.0:
            weights = weights * 1.20
        elif 20.0 <= depth < 50.0:
            weights = weights * 1.40
        elif 50.0 <= depth <= 150.0:
            weights = weights * (1.60 + 0.35 * as_cond)
        elif 150.0 < depth <= 300.0:
            weights = weights * (1.50 + 0.45 * as_cond)
        elif depth == 500.0:
            weights = weights * 1.80
        elif depth == 700.0:
            weights = weights * 2.00
        elif depth >= 1000.0:
            weights = weights * 2.20
        else:
            weights = weights * 1.50
    elif skill_focused:
        if depth < 20.0:
            weights = weights * 1.35
        elif 20.0 <= depth <= 200.0:
            if 50.0 <= depth <= 150.0:
                weights = weights * (1.50 + 0.45 * as_cond)
            else:
                weights = weights * 1.50
        elif 200.0 < depth <= 300.0:
            weights = weights * (1.20 + 0.75 * as_cond)
        elif depth >= 500.0:
            weights = weights * 1.30
    else:
        # Legacy schedule
        if 20.0 <= depth <= 200.0:
            weights = weights * 1.5
        elif 200.0 < depth <= 300.0:
            weights = 1.0 + (1.5 * 1.3 - 1.0) * as_cond

    return weights


def compute_sobel_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute horizontal (x, y) spatial gradients using Sobel kernels."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return grad_x, grad_y


def sobel_gradient_loss(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sobel horizontal gradient consistency loss (Bottleneck 13)."""
    gx_pred, gy_pred = compute_sobel_gradients(pred)
    gx_true, gy_true = compute_sobel_gradients(true)
    diff_sq = ((gx_pred - gx_true) ** 2 + (gy_pred - gy_true) ** 2) * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff_sq.sum() / denom


def reconstruct_x0(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    min_alpha_bar: float = 1e-3,
) -> torch.Tensor:
    """Reconstruct clean data estimate x0_hat with numerical clamping on alpha_bar_t."""
    if not isinstance(alpha_bar_t, torch.Tensor):
        alpha_bar_t = torch.tensor(alpha_bar_t, device=x_t.device, dtype=x_t.dtype)
    else:
        alpha_bar_t = alpha_bar_t.to(device=x_t.device, dtype=x_t.dtype)

    safe_alpha = torch.clamp(alpha_bar_t, min=min_alpha_bar)
    sqrt_alpha = torch.sqrt(safe_alpha)
    sqrt_one_minus = torch.sqrt(torch.clamp(1.0 - safe_alpha, min=0.0))

    while sqrt_alpha.dim() < x_t.dim():
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

    return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha


class OceanEmbedLoss(nn.Module):
    """Multi-task loss module with Pinn-Ocean style homoscedastic uncertainty weighting."""

    def __init__(
        self,
        skill_focused: bool = True,
        inverse_variance: bool = False,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
        variance_exponent: float = 1.50,
    ):
        super().__init__()
        self.skill_focused = skill_focused
        self.inverse_variance = inverse_variance
        self.use_min_snr = use_min_snr
        self.min_snr_gamma = min_snr_gamma
        self.variance_exponent = variance_exponent
        # Learnable log-variance weights w1, w2, w3 initialized to 0.0
        self.w1 = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # diffusion
        self.w2 = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # auxiliary
        self.w3 = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # physics

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        depth: float,
        ocean_mask: torch.Tensor,
        arabian_sea_mask: torch.Tensor,
        bob_mask: torch.Tensor,
        aux_preds: Dict[str, torch.Tensor],
        aux_trues: torch.Tensor,  # (B, 4, H, W)
        x0_hat: Optional[torch.Tensor] = None,
        x0_true: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        alpha_bar_t: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        min_alpha_bar: float = 1e-3,
        max_physics_timestep: int = 900,
        prev_x0_true: Optional[torch.Tensor] = None,
        prev_depth: Optional[float] = None,
        clim_temp: Optional[float] = None,
        prev_clim_temp: Optional[float] = None,
        depth_std: Optional[float] = None,
        prev_depth_std: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute total weighted loss and individual component losses."""
        # 1. Diffusion Loss with alpha(d, region) weighting, physical variance weighting (Bottleneck 9), and Min-SNR-gamma weighting (Bottleneck 5)
        alpha_weights = compute_depth_region_weight(
            depth, arabian_sea_mask, skill_focused=self.skill_focused, inverse_variance=self.inverse_variance
        )
        if depth_std is not None:
            # Physical variance weighting: w_var(d) = (sigma_d / bar_sigma)^beta, bar_sigma ~ 0.5408
            vexp = getattr(self, "variance_exponent", 1.50)
            if vexp == "zone_adaptive" or str(vexp).lower() == "zone_adaptive":
                if depth <= 30.0:
                    b_val = 1.00  # Surface: moderate weighting
                elif depth <= 200.0:
                    b_val = 1.50  # Thermocline: IDENTICAL to Phase 7
                elif depth <= 300.0:
                    b_val = 1.00  # Transition bridge
                else:
                    b_val = 0.50  # Deep: mild scaling, prevents deep ocean attenuation
            else:
                try:
                    b_val = float(vexp)
                except (ValueError, TypeError):
                    b_val = 1.50
            w_var = (float(depth_std) / 0.5408) ** b_val
            alpha_weights = alpha_weights * w_var

        pixel_weights = alpha_weights * ocean_mask
        sq_err = (eps_pred - eps_true) ** 2

        if self.use_min_snr and alpha_bar_t is not None:
            if not isinstance(alpha_bar_t, torch.Tensor):
                ab = torch.tensor(alpha_bar_t, device=eps_pred.device, dtype=eps_pred.dtype)
            else:
                ab = alpha_bar_t.to(device=eps_pred.device, dtype=eps_pred.dtype)
            snr = ab / torch.clamp(1.0 - ab, min=1e-5)
            # Min-SNR-gamma weight = min(SNR, gamma) / SNR
            snr_weight = torch.clamp(snr, max=self.min_snr_gamma) / torch.clamp(snr, min=1e-5)
            while snr_weight.dim() < sq_err.dim():
                snr_weight = snr_weight.unsqueeze(-1)
            sq_err = sq_err * snr_weight

        denom = torch.clamp(pixel_weights.sum(), min=1.0)
        loss_diffusion = (sq_err * pixel_weights).sum() / denom

        # 2. Auxiliary Loss with region masking & O(1) loss-scale normalization (Fix A2)
        # aux_trues: channel 0=mld, 1=blt, 2=sal_depth, 3=sal_strength
        # Compute spatial mean of ground truth over valid region for scalar target comparison
        ocean_m = (ocean_mask.squeeze(1) > 0.5)
        bob_m = (bob_mask.squeeze(1) > 0.5) & ocean_m
        as_m = (arabian_sea_mask.squeeze(1) > 0.5) & ocean_m

        loss_aux = torch.tensor(0.0, device=eps_pred.device, dtype=eps_pred.dtype)

        # MLD loss (global ocean, normalized by 50m reference)
        if ocean_m.any():
            ocean_denom = torch.clamp(ocean_m.unsqueeze(1).float().sum(dim=(-2, -1)), min=1.0)
            true_mld_scalar = (aux_trues[:, 0:1] * ocean_m.unsqueeze(1).float()).sum(dim=(-2, -1)) / ocean_denom
            loss_aux = loss_aux + F.mse_loss(aux_preds["mld"] / 50.0, true_mld_scalar / 50.0)

        # BLT loss (Bay of Bengal only, normalized by 20m reference)
        if bob_m.any():
            bob_denom = torch.clamp(bob_m.unsqueeze(1).float().sum(dim=(-2, -1)), min=1.0)
            true_blt_scalar = (aux_trues[:, 1:2] * bob_m.unsqueeze(1).float()).sum(dim=(-2, -1)) / bob_denom
            loss_aux = loss_aux + F.mse_loss(aux_preds["blt"] / 20.0, true_blt_scalar / 20.0)

        # Salinity Maximum (Arabian Sea only, depth norm 100m, strength norm 0.1 PSU)
        if as_m.any():
            as_denom = torch.clamp(as_m.unsqueeze(1).float().sum(dim=(-2, -1)), min=1.0)
            true_sal_depth_scalar = (aux_trues[:, 2:3] * as_m.unsqueeze(1).float()).sum(dim=(-2, -1)) / as_denom
            true_sal_str_scalar = (aux_trues[:, 3:4] * as_m.unsqueeze(1).float()).sum(dim=(-2, -1)) / as_denom
            loss_aux = loss_aux + 0.5 * (
                F.mse_loss(aux_preds["sal_max_depth"] / 100.0, true_sal_depth_scalar / 100.0)
                + F.mse_loss(aux_preds["sal_max_strength"] / 0.1, true_sal_str_scalar / 0.1)
            )

        # 3. Physics Loss (Thermocline consistency & Vertical Stratification Loss - Bottleneck 7, Sobel Gradient - Bottleneck 13)
        # Skip or downweight physics loss at very high noise timesteps (near t=1000) or near-zero alpha_bar
        skip_physics = False
        if t is not None:
            if isinstance(t, torch.Tensor) and (t >= max_physics_timestep).all():
                skip_physics = True
            elif isinstance(t, (int, float)) and t >= max_physics_timestep:
                skip_physics = True

        if alpha_bar_t is not None:
            if isinstance(alpha_bar_t, torch.Tensor) and (alpha_bar_t < min_alpha_bar).all():
                skip_physics = True
            elif isinstance(alpha_bar_t, (int, float)) and alpha_bar_t < min_alpha_bar:
                skip_physics = True

        if not skip_physics and x0_hat is None and x_t is not None and alpha_bar_t is not None:
            x0_hat = reconstruct_x0(x_t, eps_pred, alpha_bar_t, min_alpha_bar=min_alpha_bar)

        if not skip_physics and x0_hat is not None and x0_true is not None:
            # Gradient consistency penalty: penalize difference between predicted x0_hat and ground truth x0
            diff_sq = torch.clamp((x0_hat - x0_true) ** 2, max=100.0) * ocean_mask
            loss_physics = diff_sq.sum() / denom

            # Sobel horizontal gradient consistency (Bottleneck 13)
            loss_sobel = sobel_gradient_loss(x0_hat, x0_true, ocean_mask)

            # Vertical Stratification Gradient Loss (Bottleneck 7)
            loss_strat = torch.tensor(0.0, device=eps_pred.device, dtype=eps_pred.dtype)
            loss_inv = torch.tensor(0.0, device=eps_pred.device, dtype=eps_pred.dtype)
            if (
                prev_x0_true is not None
                and prev_depth is not None
                and depth > prev_depth
            ):
                dz = float(depth - prev_depth)
                s_curr = depth_std if depth_std is not None else 1.0
                s_prev = prev_depth_std if prev_depth_std is not None else 1.0
                c_curr = clim_temp if clim_temp is not None else 0.0
                c_prev = prev_clim_temp if prev_clim_temp is not None else 0.0

                t_pred_curr = x0_hat * s_curr + c_curr
                t_true_curr = x0_true * s_curr + c_curr
                t_prev = prev_x0_true * s_prev + c_prev

                grad_pred = (t_pred_curr - t_prev) / dz
                grad_true = (t_true_curr - t_prev) / dz

                strat_err = torch.clamp((grad_pred - grad_true) ** 2, max=25.0) * ocean_mask
                loss_strat = strat_err.sum() / denom

                # Inversion penalty: static stability dT/dz <= 0.5 °C / m (allows slight barrier layer, penalizes unphysical warm pools at depth)
                inversion = torch.clamp(F.relu(t_pred_curr - t_prev - 0.5) ** 2, max=25.0) * ocean_mask
                loss_inv = inversion.sum() / denom

            loss_physics = loss_physics + 0.5 * loss_strat + 0.25 * loss_inv + 0.5 * loss_sobel
        else:
            loss_physics = torch.tensor(0.0, device=eps_pred.device, dtype=eps_pred.dtype)

        # 4. Total Multi-Task Loss with Homoscedastic Uncertainty Weighting
        # L_total = exp(-w1)*L_diff + w1 + exp(-w2)*L_aux + w2 + exp(-w3)*L_phys + w3
        loss_total = (
            torch.exp(-self.w1) * loss_diffusion + self.w1
            + torch.exp(-self.w2) * loss_aux + self.w2
            + torch.exp(-self.w3) * loss_physics + self.w3
        ).squeeze()

        return {
            "loss_total": loss_total,
            "loss_diffusion": loss_diffusion,
            "loss_aux": loss_aux,
            "loss_physics": loss_physics,
            "w1": self.w1.detach(),
            "w2": self.w2.detach(),
            "w3": self.w3.detach(),
        }
