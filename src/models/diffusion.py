"""Diffusion Process Utilities and Schedules (DDPM / DDIM).

Implements:
1. Cosine and Linear noise schedules.
2. Forward diffusion sampling q_sample(x0, t, noise).
3. Clean data reconstruction utility predict_x0_from_noise(xt, t, eps_pred).
4. Sinusoidal timestep embeddings.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn


def get_cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule as proposed by Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999)


def get_linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (B, embedding_dim)."""
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad if odd
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion engine managing forward process, schedules, and x0 recovery."""

    def __init__(
        self,
        timesteps: int = 1000,
        schedule_type: str = "cosine",
    ):
        super().__init__()
        self.timesteps = timesteps

        if schedule_type == "cosine":
            betas = get_cosine_beta_schedule(timesteps)
        else:
            betas = get_linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        # Forward process coefficients
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())

        # x0 reconstruction coefficients (safely clamped to min 1e-3 to prevent numerical divergence near t=T)
        safe_alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-3)
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / safe_alphas_cumprod).float())
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(torch.clamp(1.0 / safe_alphas_cumprod - 1.0, min=0.0)).float()
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse data x_0 to timestep t: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps.

        Args:
            x_0: Clean ground truth tensor (B, C, H, W).
            t: Integer timestep indices (B,).
            noise: Optional pre-sampled Gaussian noise (B, C, H, W).

        Returns:
            Tuple (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Analytical x0 reconstruction: x0_hat = (x_t - sqrt(1 - alpha_bar_t)*eps) / sqrt(alpha_bar_t).

        Args:
            x_t: Noisy tensor at timestep t (B, C, H, W).
            t: Timestep indices (B,).
            eps_pred: Model predicted noise (B, C, H, W).

        Returns:
            x0_hat: Reconstructed clean data estimate (B, C, H, W).
        """
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)

        x0_hat = sqrt_recip * x_t - sqrt_recipm1 * eps_pred
        return x0_hat
