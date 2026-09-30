"""Checkpointing Utilities for OceanEmbed Training.

Supports atomic saving, resuming, and spot-instance preemption recovery.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.logging_config import get_logger

logger = get_logger("checkpoint_utils")


def save_checkpoint(
    checkpoint_dir: Union[str, Path],
    filename: str,
    models: Dict[str, nn.Module],
    optimizer: optim.Optimizer,
    scheduler: Optional[Any] = None,
    loss_fn: Optional[nn.Module] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Atomically save training state to checkpoint file."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_dir / f"{filename}.tmp"
    final_path = checkpoint_dir / filename

    state = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "models": {k: v.state_dict() for k, v in models.items()},
        "optimizer": optimizer.state_dict(),
    }

    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if loss_fn is not None:
        state["loss_fn"] = loss_fn.state_dict()

    torch.save(state, temp_path)
    os.replace(temp_path, final_path)
    logger.info(f"Saved atomic checkpoint to {final_path}")
    return final_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    models: Dict[str, nn.Module],
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model and optimizer states from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device or "cpu")

    for k, v in models.items():
        if k in state["models"]:
            v.load_state_dict(state["models"][k], strict=False)
            logger.info(f"Loaded weights for model component: {k} (strict=False)")

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    if loss_fn is not None and "loss_fn" in state:
        loss_fn.load_state_dict(state["loss_fn"])

    logger.info(f"Restored checkpoint from epoch {state.get('epoch', 0)}, step {state.get('step', 0)}")
    return state
