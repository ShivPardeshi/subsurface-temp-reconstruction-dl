"""ConvLSTM Spatiotemporal Context Encoder.

Encodes a 7-day sequence of 25-channel oceanographic surface and static fields
into a compact 64-channel spatial context representation u_cond.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single-layer 2D Convolutional LSTM cell."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.padding = kernel_size // 2

        # 4 gates: input, forget, cell candidate, output
        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single time step.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).
            state: Optional tuple (h, c) of tensors of shape (B, hidden_channels, H, W).

        Returns:
            Tuple (h_next, c_next).
        """
        b, _, height, width = x.shape
        if state is None:
            h_cur = torch.zeros(b, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
            c_cur = torch.zeros(b, self.hidden_channels, height, width, device=x.device, dtype=x.dtype)
        else:
            h_cur, c_cur = state

        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i_gate, f_gate, c_gate, o_gate = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(c_gate)
        o = torch.sigmoid(o_gate)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


import json
from pathlib import Path


class ContextEncoder(nn.Module):
    """3-layer ConvLSTM stack encoding (B, T, C_in, H, W) -> (B, 64, H, W).
    
    Includes per-channel z-score standardization to equalize dynamic ranges across
    25 oceanographic channels (Item 1.3 & Phase 3).
    """

    def __init__(
        self,
        in_channels: int = 25,
        hidden_dims: Tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 3,
        normalize_inputs: bool = True,
        norm_stats_path: Optional[str] = "data/processed/channel_normalization_stats_ocean_only.json",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = tuple(hidden_dims)
        self.num_layers = len(hidden_dims)
        self.normalize_inputs = normalize_inputs

        # Fallback to old path if ocean_only not yet created
        stats_path = None
        if norm_stats_path is not None:
            p = Path(norm_stats_path)
            if p.exists():
                stats_path = p
            elif Path("data/processed/channel_normalization_stats.json").exists():
                stats_path = Path("data/processed/channel_normalization_stats.json")

        # Persistent normalization buffers: shape (1, 1, C, 1, 1) for broadcasting over (B, T, C, H, W)
        mean_init = torch.zeros(1, 1, in_channels, 1, 1)
        std_init = torch.ones(1, 1, in_channels, 1, 1)

        if stats_path is not None and stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                means = stats.get("means", [])
                stds = stats.get("stds", [])
                if len(means) == in_channels and len(stds) == in_channels:
                    mean_init = torch.tensor(means, dtype=torch.float32).view(1, 1, in_channels, 1, 1)
                    std_init = torch.tensor(stds, dtype=torch.float32).view(1, 1, in_channels, 1, 1)
                    # Guard against near-zero std (using 1e-12 to avoid corrupting small geophysical scales like wind curl ~1e-7)
                    std_init = torch.clamp(std_init, min=1e-12)
            except Exception as e:
                print(f"[ContextEncoder] Warning loading norm stats from {stats_path}: {e}")

        self.register_buffer("channel_mean", mean_init)
        self.register_buffer("channel_std", std_init)

        layers = []
        for i in range(self.num_layers):
            cur_in = in_channels if i == 0 else hidden_dims[i - 1]
            cur_out = hidden_dims[i]
            layers.append(ConvLSTMCell(cur_in, cur_out, kernel_size=kernel_size))
        self.layers = nn.ModuleList(layers)

    def set_normalization_stats(self, means: torch.Tensor, stds: torch.Tensor) -> None:
        """Explicitly set channel mean and standard deviation vectors."""
        self.channel_mean.copy_(means.view(1, 1, self.in_channels, 1, 1))
        self.channel_std.copy_(torch.clamp(stds.view(1, 1, self.in_channels, 1, 1), min=1e-12))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Process spatiotemporal input sequence.

        Args:
            x_seq: Tensor of shape (B, T, C, H, W).

        Returns:
            u_cond: Final hidden state from top layer of shape (B, hidden_dims[-1], H, W).
        """
        b, t, c, h, w = x_seq.shape

        # Per-channel z-score standardization
        if self.normalize_inputs:
            # Extract ocean mask from raw Channel 20 (land_ocean_mask: 1 for ocean, 0 for land)
            is_ocean = (x_seq[:, :, 20:21] > 0.5) if c >= 21 else None
            x_seq = (x_seq - self.channel_mean) / self.channel_std
            # For physical ocean channels (0 to 18), zero out land cells so they are exactly 0.0 (neutral) in normalized space
            if is_ocean is not None and c >= 21:
                ocean_part = torch.where(is_ocean, x_seq[:, :, :19], torch.zeros_like(x_seq[:, :, :19]))
                x_seq = torch.cat([ocean_part, x_seq[:, :, 19:]], dim=2)

        states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self.num_layers

        for time_idx in range(t):
            current_input = x_seq[:, time_idx]  # (B, C, H, W)
            for layer_idx, cell in enumerate(self.layers):
                h, c_state = cell(current_input, states[layer_idx])
                states[layer_idx] = (h, c_state)
                current_input = h

        # Return final hidden state of the top layer
        final_h, _ = states[-1]
        return final_h

