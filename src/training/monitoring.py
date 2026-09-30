"""Training Monitoring, Telemetry and Diagnostic Previews.

Tracks:
1. Individual multi-task loss components (L_diffusion, L_aux, L_physics).
2. Learnable homoscedastic uncertainty weights (w1, w2, w3).
3. Periodic full-depth profile sampling previews against ground truth.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.sampling.depth_cascade import DepthCascadeSampler
from src.sampling.ddim_sampler import DDIMSampler
from src.utils.grid import CANONICAL_DEPTHS
from src.utils.logging_config import get_logger

logger = get_logger("monitoring")


class TrainingMonitor:
    """Manages telemetry logging and periodic physical validation previews."""

    def __init__(self, log_dir: str = "logs/training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir = self.log_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.log_dir / "training_history.json"
        self.history: Dict[str, List[float]] = {
            "step": [],
            "loss_total": [],
            "loss_diffusion": [],
            "loss_aux": [],
            "loss_physics": [],
            "w1": [],
            "w2": [],
            "w3": [],
            "val_rmse": [],
            "learning_rate": [],
        }

    def log_step(
        self,
        step: int,
        loss_dict: Dict[str, Any],
        lr: float,
    ) -> None:
        """Record step-level losses and parameters."""
        def _to_float(v: Any) -> float:
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            return float(v)

        self.history["step"].append(step)
        self.history["loss_total"].append(_to_float(loss_dict.get("loss_total", 0.0)))
        self.history["loss_diffusion"].append(_to_float(loss_dict.get("loss_diffusion", 0.0)))
        self.history["loss_aux"].append(_to_float(loss_dict.get("loss_aux", 0.0)))
        self.history["loss_physics"].append(_to_float(loss_dict.get("loss_physics", 0.0)))
        self.history["w1"].append(_to_float(loss_dict.get("w1", 0.0)))
        self.history["w2"].append(_to_float(loss_dict.get("w2", 0.0)))
        self.history["w3"].append(_to_float(loss_dict.get("w3", 0.0)))
        self.history["learning_rate"].append(lr)

        # Periodically dump history to JSON
        if step % 50 == 0:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

    def log_validation(self, step: int, val_rmse: float) -> None:
        """Record validation RMSE metric."""
        self.history["val_rmse"].append({"step": step, "rmse": float(val_rmse)})

    def generate_preview_plot(
        self,
        step: int,
        cascade_sampler: DepthCascadeSampler,
        val_sample: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Path:
        """Run DDIM cascade sampling on a held-out sample and plot T(z) vs Truth."""
        x_seq = val_sample["x_seq"].unsqueeze(0).to(device)
        static_feats = val_sample["static_features"].unsqueeze(0).to(device)
        anomaly_true = val_sample["anomaly_target"]  # (15, H, W)
        scalar_cond = val_sample["scalar_cond"].unsqueeze(0).to(device)

        out = cascade_sampler.sample_full_profile(
            x_seq=x_seq,
            static_features=static_feats,
            scalar_conditions=scalar_cond,
        )
        anomaly_pred = out["anomalies"].squeeze(0).cpu().numpy()  # (15, H, W)
        anomaly_true_np = anomaly_true.numpy()

        # Find an ocean pixel (where static_feats landmask == 1)
        ocean_mask = static_feats[0, 0].cpu().numpy() > 0.5
        y_idxs, x_idxs = np.where(ocean_mask)
        if len(y_idxs) > 0:
            center_idx = len(y_idxs) // 2
            cy, cx = y_idxs[center_idx], x_idxs[center_idx]
        else:
            cy, cx = anomaly_pred.shape[1] // 2, anomaly_pred.shape[2] // 2

        pred_profile = anomaly_pred[:, cy, cx]
        true_profile = anomaly_true_np[:, cy, cx]
        depths = CANONICAL_DEPTHS

        plot_path = self.preview_dir / f"profile_preview_step_{step:06d}.png"
        plt.figure(figsize=(6, 8))
        plt.plot(true_profile, depths, "o-", label="Ground Truth Anomaly", color="#059669", linewidth=2)
        plt.plot(pred_profile, depths, "s--", label="Model DDIM Reconstruction", color="#2563eb", linewidth=2)
        plt.gca().invert_yaxis()
        plt.xlabel("Temperature Anomaly (°C)")
        plt.ylabel("Depth (m)")
        plt.title(f"Subsurface Profile Preview @ Step {step}\n(Lat/Lon Ocean Point [{cy}, {cx}])")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()

        logger.info(f"Generated validation profile preview at step {step} -> {plot_path}")
        return plot_path
