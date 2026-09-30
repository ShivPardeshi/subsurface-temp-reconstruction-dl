"""Fast Denoising Diffusion Implicit Models (DDIM) Sampler.

Implements deterministic / stochastic accelerated reverse diffusion sampling
(Song et al., 2020) with customizable step counts.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from src.models.diffusion import GaussianDiffusion


class DDIMSampler:
    """DDIM Sampler supporting sub-sequence reverse steps (e.g. 20-50 steps from 1000)."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        num_ddim_timesteps: int = 25,
        eta: float = 0.0,  # 0.0 = deterministic DDIM (optimal conditional mean); >0 = stochastic DDIM
        schedule_type: str = "quadratic",  # "quadratic", "cosine", or "linear"
    ):
        self.diffusion = diffusion
        self.num_ddim_timesteps = min(num_ddim_timesteps, diffusion.timesteps)
        self.eta = eta
        self.schedule_type = schedule_type

        # Generate sub-sequence of timesteps
        max_t = int(diffusion.timesteps * 0.90)  # Cap at 900 to avoid alpha_bar -> 0 divergence
        if schedule_type == "quadratic":
            # Quadratic schedule allocates more steps to intermediate and low-noise regimes (Bottleneck 8)
            steps = np.linspace(0, np.sqrt(max_t), self.num_ddim_timesteps) ** 2
            timesteps = np.unique(np.round(steps).astype(np.int64))
            if timesteps[0] != 0:
                timesteps = np.insert(timesteps, 0, 0)
            self.ddim_timesteps = timesteps
        elif schedule_type == "cosine":
            steps = (1.0 - np.cos(np.linspace(0, np.pi / 2, self.num_ddim_timesteps))) * max_t
            timesteps = np.unique(np.round(steps).astype(np.int64))
            if timesteps[0] != 0:
                timesteps = np.insert(timesteps, 0, 0)
            self.ddim_timesteps = timesteps
        else:
            # Linear spacing
            c = max(1, max_t // self.num_ddim_timesteps)
            self.ddim_timesteps = np.asarray(list(range(0, max_t, c)))

        # Previous timesteps in sub-sequence
        self.ddim_timesteps_prev = np.append(np.array([0]), self.ddim_timesteps[:-1])

    @torch.no_grad()
    def sample_single_depth(
        self,
        unet: nn.Module,
        spatial_cond: torch.Tensor,
        non_spatial_cond_base: torch.Tensor,
        prev_depth_clean: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample a clean 2D anomaly field x_0 for a single depth level using DDIM.

        Args:
            unet: UNetDenoiser model.
            spatial_cond: (B, 70, H, W) spatial conditioning tensor.
            non_spatial_cond_base: (B, cond_dim - 1) non-spatial conditions without timestep.
            prev_depth_clean: (B, 1, H, W) clean output from previous depth level (or zeros).
            shape: (B, 1, H, W) output shape.
            device: Target torch device.

        Returns:
            x_0_clean: Sampled anomaly tensor (B, 1, H, W).
        """
        if shape is None:
            b, _, h, w = spatial_cond.shape
            shape = (b, 1, h, w)
        else:
            b, _, h, w = shape

        if device is None:
            device = spatial_cond.device

        # Start from standard Gaussian noise
        x_t = torch.randn(shape, device=device)

        # Reverse time loop (from highest timestep to 0)
        time_pairs = list(zip(reversed(self.ddim_timesteps), reversed(self.ddim_timesteps_prev)))

        for t_curr_int, t_prev_int in time_pairs:
            t_curr = torch.full((b,), t_curr_int, device=device, dtype=torch.long)
            t_prev = torch.full((b,), t_prev_int, device=device, dtype=torch.long)

            # Build non_spatial_cond vector including normalized timestep
            t_norm = (t_curr.float() / self.diffusion.timesteps).unsqueeze(1)
            full_non_spatial_cond = torch.cat([non_spatial_cond_base, t_norm], dim=1)

            # Predict noise
            eps_pred = unet(
                x_noisy=x_t,
                spatial_cond=spatial_cond,
                non_spatial_cond=full_non_spatial_cond,
                prev_depth_clean=prev_depth_clean,
            )

            # Coefficients
            alpha_bar_curr = self.diffusion.alphas_cumprod[t_curr].view(-1, 1, 1, 1)
            alpha_bar_prev = self.diffusion.alphas_cumprod[t_prev].view(-1, 1, 1, 1)

            # Estimate x_0 with physical clamping for standardized anomalies ([-5.0, 5.0])
            safe_alpha = torch.clamp(alpha_bar_curr, min=1e-4)
            x_0_hat = (x_t - torch.sqrt(1.0 - safe_alpha) * eps_pred) / torch.sqrt(safe_alpha)
            x_0_hat = torch.clamp(x_0_hat, min=-5.0, max=5.0)

            # DDIM step equation
            current_eta = self.eta if eta is None else eta
            if t_prev_int == 0 and t_curr_int == self.ddim_timesteps[0]:
                x_t = x_0_hat
            else:
                sigma_t = (
                    current_eta
                    * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr))
                    * torch.sqrt(1.0 - alpha_bar_curr / alpha_bar_prev)
                )
                dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * eps_pred
                noise = torch.randn_like(x_t) if current_eta > 0 else torch.zeros_like(x_t)
                x_t = torch.sqrt(alpha_bar_prev) * x_0_hat + dir_xt + sigma_t * noise

        return x_t
