"""Unit tests for training interruption and checkpoint resumption."""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch
import yaml

from src.training.train import train_model
from src.training.checkpoint_utils import load_checkpoint, save_checkpoint
from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads


def test_checkpoint_atomic_save_and_load(tmp_path):
    """Verify save_checkpoint and load_checkpoint correctly roundtrip weights and optimizer states."""
    context_enc = ContextEncoder(in_channels=25, hidden_dims=(32, 64, 64))
    unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8)
    aux_heads = AuxiliaryHeads(in_features=64)

    models = {
        "context_encoder": context_enc,
        "unet": unet,
        "aux_heads": aux_heads,
    }
    optimizer = torch.optim.AdamW(
        list(context_enc.parameters()) + list(unet.parameters()) + list(aux_heads.parameters()),
        lr=1e-3,
    )

    ckpt_file = save_checkpoint(
        checkpoint_dir=tmp_path,
        filename="test_ckpt.pt",
        models=models,
        optimizer=optimizer,
        epoch=2,
        step=50,
        metrics={"best_val_rmse": 0.42},
    )
    assert ckpt_file.exists()

    # Create fresh models and load
    new_context = ContextEncoder(in_channels=25, hidden_dims=(32, 64, 64))
    new_unet = UNetDenoiser(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8)
    new_aux = AuxiliaryHeads(in_features=64)
    new_opt = torch.optim.AdamW(
        list(new_context.parameters()) + list(new_unet.parameters()) + list(new_aux.parameters()),
        lr=1e-3,
    )
    new_models = {
        "context_encoder": new_context,
        "unet": new_unet,
        "aux_heads": new_aux,
    }

    state = load_checkpoint(
        checkpoint_path=ckpt_file,
        models=new_models,
        optimizer=new_opt,
    )

    assert state["step"] == 50
    assert state["epoch"] == 2
    assert state["metrics"]["best_val_rmse"] == 0.42

    # Assert parameter equality
    for p1, p2 in zip(context_enc.parameters(), new_context.parameters()):
        assert torch.equal(p1, p2)


def test_interruption_resume_continuity(tmp_path):
    """Simulates training interruption: runs 5 steps, saves checkpoint, resumes, and runs 5 more steps."""
    # Write temporary config
    config_dict = {
        "experiment": {
            "name": "test_resume",
            "run_dir": str(tmp_path / "resume_run"),
        },
        "data": {
            "inputs_zarr": "data/processed/phase2_toy/oceanembed_training_inputs.zarr",
            "anomaly_targets_zarr": "data/processed/phase2_toy/oceanembed_anomaly_targets.zarr",
            "aux_targets_zarr": "data/processed/phase2_toy/oceanembed_auxiliary_targets.zarr",
            "scalar_csv": "data/processed/phase2_toy/scalar_conditioning.csv",
            "sequence_length": 7,
        },
        "architecture": {
            "in_channels": 25,
            "convlstm_hidden": [32, 64, 64],
            "unet_stages": [32, 64, 128, 256],
            "spatial_cond_dim": 70,
            "non_spatial_cond_dim": 9,
            "diffusion_timesteps": 100,
        },
        "ablations": {
            "region_conditioning_enabled": True,
            "depth_cascade_enabled": True,
            "regional_loss_boost_enabled": True,
        },
        "training": {
            "batch_size": 1,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "checkpoint_interval_steps": 2,
            "epochs": 1,
            "precision": "fp32",
        },
    }

    cfg_file = tmp_path / "test_cfg.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(config_dict, f)

    # Phase 1: Train for 4 steps and save checkpoint
    res1 = train_model(config_path=str(cfg_file), override_steps=4, device=torch.device("cpu"))
    assert res1["total_steps"] == 4
    last_ckpt = Path(config_dict["experiment"]["run_dir"]) / "last_checkpoint.pt"
    assert last_ckpt.exists()

    # Phase 2: Resume training to 8 steps
    res2 = train_model(config_path=str(cfg_file), override_steps=8, device=torch.device("cpu"))
    assert res2["total_steps"] == 8
