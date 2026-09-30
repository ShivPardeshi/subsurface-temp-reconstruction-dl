"""Stage A0: Laptop Real-Training Capability Test Diagnostic.

Measures whether real training of the Phase 3 OceanEmbed architecture
fits in 4GB VRAM on the RTX 3050 and quantifies throughput and wall-clock projections.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import json
import time
import psutil
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

from src.models.context_encoder import ContextEncoder
from src.models.unet_denoiser import UNetDenoiser
from src.models.auxiliary_heads import AuxiliaryHeads
from src.models.diffusion import GaussianDiffusion
from src.training.losses import OceanEmbedLoss
from src.training.dataset import OceanEmbedDataset
from src.utils.grid import CANONICAL_DEPTHS
from src.utils.logging_config import get_logger

logger = get_logger("laptop_capability_test")


def get_gpu_temperature() -> Optional[float]:
    """Attempt to query GPU temperature via nvidia-smi."""
    try:
        import subprocess
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return None


class CheckpointedUNetDenoiser(UNetDenoiser):
    """UNetDenoiser subclass enabling PyTorch gradient checkpointing on deeper stages."""

    def forward(
        self,
        x_noisy: torch.Tensor,
        spatial_cond: torch.Tensor,
        non_spatial_cond: torch.Tensor,
        prev_depth_clean: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, h, w = x_noisy.shape
        if prev_depth_clean is None:
            prev_depth_clean = torch.zeros_like(x_noisy)

        cond_emb = self.cond_mlp(non_spatial_cond)
        x_in = torch.cat([x_noisy, prev_depth_clean, spatial_cond], dim=1)
        h0 = self.init_conv(x_in)

        # Stage 1 & 2 standard
        e1 = self.enc1(h0, cond_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, cond_emb)
        d2 = self.down2(e2)

        # Checkpoint deeper stages 3, bottleneck, and decoders to save VRAM
        def custom_enc3(inp, emb):
            return self.enc3(inp, emb)

        e3 = checkpoint(custom_enc3, d2, cond_emb, use_reentrant=False)
        d3 = self.down3(e3)

        def custom_bottleneck(inp, emb):
            return self.bot2(self.bot1(inp, emb), emb)

        b_out = checkpoint(custom_bottleneck, d3, cond_emb, use_reentrant=False)

        u3 = self.up3(b_out)
        if u3.shape[-2:] != e3.shape[-2:]:
            u3 = nn.functional.interpolate(u3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        c3 = torch.cat([u3, e3], dim=1)
        d_out3 = self.dec3(c3, cond_emb)

        u2 = self.up2(d_out3)
        if u2.shape[-2:] != e2.shape[-2:]:
            u2 = nn.functional.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        c2 = torch.cat([u2, e2], dim=1)
        d_out2 = self.dec2(c2, cond_emb)

        u1 = self.up1(d_out2)
        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = nn.functional.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        c1 = torch.cat([u1, e1], dim=1)
        d_out1 = self.dec1(c1, cond_emb)

        out = self.final_norm(d_out1)
        out = self.final_act(out)
        eps_pred = self.final_conv(out)
        return eps_pred


def run_laptop_capability_test() -> Dict[str, Any]:
    print("================================================================================")
    print("      OceanEmbed (PS26066) — Stage A0: Laptop Capability Test Diagnostic       ")
    print("================================================================================")

    # Step 1: Environment Report
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    total_vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0.0
    )
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_model = os.environ.get("PROCESSOR_IDENTIFIER", "Ryzen 7 6800H / Multi-core CPU")

    env_report = {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu": gpu_name,
        "vram_total_gb": round(total_vram_gb, 2),
        "ram_total_gb": round(total_ram_gb, 2),
        "cpu": cpu_model,
    }

    print("\n--- STEP 1: Environment Report ---")
    print(f"PyTorch Version:  {env_report['torch_version']}")
    print(f"CUDA Available:   {env_report['cuda_available']}")
    print(f"GPU Hardware:     {env_report['gpu']} ({env_report['vram_total_gb']} GB VRAM)")
    print(f"System RAM:       {env_report['ram_total_gb']} GB")
    print(f"CPU:              {env_report['cpu']}")

    device = torch.device("cuda:0" if cuda_available else "cpu")

    # Step 2: Load real dataset
    print("\n--- STEP 2: Loading Real Batch from Phase 2 Preprocessed Store ---")
    data_dir = Path("data/processed/phase2_dataset")
    if not (data_dir / "oceanembed_training_inputs.zarr").exists():
        data_dir = Path("data/processed/phase2_toy")

    dataset = OceanEmbedDataset(
        inputs_zarr_path=data_dir / "oceanembed_training_inputs.zarr",
        anomaly_targets_zarr_path=data_dir / "oceanembed_anomaly_targets.zarr",
        aux_targets_zarr_path=data_dir / "oceanembed_auxiliary_targets.zarr",
        scalar_csv_path=data_dir / "scalar_conditioning.csv",
        sequence_length=7,
    )
    print(f"Loaded dataset with {len(dataset)} valid temporal sequence windows.")

    # Step 3: Instantiate Model Architecture
    def build_models(use_checkpointing: bool = False):
        context_enc = ContextEncoder(in_channels=25).to(device)
        unet_cls = CheckpointedUNetDenoiser if use_checkpointing else UNetDenoiser
        unet = unet_cls(in_channels=72, stage_channels=(32, 64, 128, 256), cond_in_dim=8).to(device)
        aux_heads = AuxiliaryHeads(in_features=64).to(device)
        diffusion = GaussianDiffusion(timesteps=1000).to(device)
        loss_fn = OceanEmbedLoss().to(device)
        all_params = (
            list(context_enc.parameters())
            + list(unet.parameters())
            + list(aux_heads.parameters())
            + list(loss_fn.parameters())
        )
        optimizer = optim.AdamW(all_params, lr=1e-3)
        return context_enc, unet, aux_heads, diffusion, loss_fn, optimizer

    # Step 4: Progressive Memory-Fit Test
    print("\n--- STEP 4: Progressive Memory-Fit Test (Batch Size = 1) ---")
    step4_results = []
    configs = [
        ("fp32", False, False),
        ("mixed_precision", True, False),
        ("mixed_precision_plus_checkpointing", True, True),
    ]

    successful_config = None
    successful_models = None

    for config_name, use_amp, use_chkpt in configs:
        if not cuda_available:
            print(f"Skipping GPU memory test for {config_name} (CUDA unavailable).")
            step4_results.append({"config": config_name, "success": False, "peak_vram_gb": None})
            continue

        print(f"Testing configuration: {config_name} ...", end=" ", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            context_enc, unet, aux_heads, diffusion, loss_fn, optimizer = build_models(use_chkpt)
            sample = dataset[0]
            x_seq = sample["x_seq"].unsqueeze(0).to(device)
            static_feats = sample["static_features"].unsqueeze(0).to(device)
            anomaly_tgt = sample["anomaly_target"][4:5].unsqueeze(0).to(device)
            aux_tgt = sample["aux_targets"].unsqueeze(0).to(device)
            scalar_c = sample["scalar_cond"].unsqueeze(0).to(device)

            ocean_m = (static_feats[:, 0:1] > 0.5).float()
            as_m = static_feats[:, 3:4]
            bob_m = static_feats[:, 2:3]

            t = torch.randint(0, 1000, (1,), device=device)
            noise = torch.randn_like(anomaly_tgt)
            x_t, eps_true = diffusion.q_sample(anomaly_tgt, t, noise)

            depth_norm = torch.tensor([[0.5]], device=device)
            t_norm = (t.float() / 1000.0).unsqueeze(1)
            prev_m = torch.zeros((1, 1), device=device)
            prev_s = torch.zeros((1, 1), device=device)
            non_spatial = torch.cat([scalar_c, depth_norm, prev_m, prev_s, t_norm], dim=1)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    u_cond = context_enc(x_seq)
                    aux_preds = aux_heads(u_cond)
                    spatial_cond = torch.cat([u_cond, static_feats], dim=1)
                    eps_pred = unet(x_t, spatial_cond, non_spatial, torch.zeros_like(x_t))
                    x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)
                    losses = loss_fn(
                        eps_pred, eps_true, 50.0, ocean_m, as_m, bob_m,
                        aux_preds, aux_tgt, x0_hat, anomaly_tgt
                    )
                    loss = losses["loss_total"]
            else:
                u_cond = context_enc(x_seq)
                aux_preds = aux_heads(u_cond)
                spatial_cond = torch.cat([u_cond, static_feats], dim=1)
                eps_pred = unet(x_t, spatial_cond, non_spatial, torch.zeros_like(x_t))
                x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)
                losses = loss_fn(
                    eps_pred, eps_true, 50.0, ocean_m, as_m, bob_m,
                    aux_preds, aux_tgt, x0_hat, anomaly_tgt
                )
                loss = losses["loss_total"]

            loss.backward()
            optimizer.step()

            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"SUCCESS (Peak VRAM: {peak_vram:.2f} GB)")
            step4_results.append({"config": config_name, "success": True, "peak_vram_gb": round(peak_vram, 3)})

            if successful_config is None:
                successful_config = (config_name, use_amp, use_chkpt)
                successful_models = (context_enc, unet, aux_heads, diffusion, loss_fn, optimizer)
                break  # Stop at first success per specification
        except RuntimeError as e:
            print(f"FAILED ({e})")
            step4_results.append({"config": config_name, "success": False, "peak_vram_gb": None})

    # Step 5: Throughput Measurement (50 consecutive steps)
    throughput_data = {
        "steps_per_second": 0.0,
        "peak_sustained_vram_gb": 0.0,
        "gpu_temp_trend_note": "N/A",
    }
    projections = {}

    if successful_config is not None and cuda_available:
        config_name, use_amp, use_chkpt = successful_config
        context_enc, unet, aux_heads, diffusion, loss_fn, optimizer = successful_models

        print(f"\n--- STEP 5: Measuring Real Throughput (50 steps on {config_name}) ---")
        num_steps = 50
        warmup_steps = 5
        timings = []
        temps = []

        initial_temp = get_gpu_temperature()
        if initial_temp:
            print(f"Initial GPU Temperature: {initial_temp:.1f} °C")

        for step in range(1, num_steps + 1):
            sample_idx = step % len(dataset)
            sample = dataset[sample_idx]
            x_seq = sample["x_seq"].unsqueeze(0).to(device)
            static_feats = sample["static_features"].unsqueeze(0).to(device)
            d_idx = step % 15
            depth = CANONICAL_DEPTHS[d_idx]
            anomaly_tgt = sample["anomaly_target"][d_idx : d_idx + 1].unsqueeze(0).to(device)
            aux_tgt = sample["aux_targets"].unsqueeze(0).to(device)
            scalar_c = sample["scalar_cond"].unsqueeze(0).to(device)

            ocean_m = (static_feats[:, 0:1] > 0.5).float()
            as_m = static_feats[:, 3:4]
            bob_m = static_feats[:, 2:3]

            t = torch.randint(0, 1000, (1,), device=device)
            noise = torch.randn_like(anomaly_tgt)
            x_t, eps_true = diffusion.q_sample(anomaly_tgt, t, noise)

            depth_norm = torch.tensor([[depth / 1000.0]], device=device)
            t_norm = (t.float() / 1000.0).unsqueeze(1)
            prev_m = torch.zeros((1, 1), device=device)
            prev_s = torch.zeros((1, 1), device=device)
            non_spatial = torch.cat([scalar_c, depth_norm, prev_m, prev_s, t_norm], dim=1)

            torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    u_cond = context_enc(x_seq)
                    aux_preds = aux_heads(u_cond)
                    spatial_cond = torch.cat([u_cond, static_feats], dim=1)
                    eps_pred = unet(x_t, spatial_cond, non_spatial, torch.zeros_like(x_t))
                    x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)
                    losses = loss_fn(
                        eps_pred, eps_true, depth, ocean_m, as_m, bob_m,
                        aux_preds, aux_tgt, x0_hat, anomaly_tgt
                    )
                    loss = losses["loss_total"]
            else:
                u_cond = context_enc(x_seq)
                aux_preds = aux_heads(u_cond)
                spatial_cond = torch.cat([u_cond, static_feats], dim=1)
                eps_pred = unet(x_t, spatial_cond, non_spatial, torch.zeros_like(x_t))
                x0_hat = diffusion.predict_x0_from_noise(x_t, t, eps_pred)
                losses = loss_fn(
                    eps_pred, eps_true, depth, ocean_m, as_m, bob_m,
                    aux_preds, aux_tgt, x0_hat, anomaly_tgt
                )
                loss = losses["loss_total"]

            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            t1 = time.time()

            if step > warmup_steps:
                timings.append(t1 - t0)

            if step % 10 == 0:
                cur_temp = get_gpu_temperature()
                if cur_temp:
                    temps.append(cur_temp)
                step_ms = (t1 - t0) * 1000
                print(f"Step {step:02d}/50 | Time: {step_ms:.1f} ms | Temp: {cur_temp or 'N/A'} °C")

        avg_step_time = sum(timings) / len(timings) if timings else 1.0
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        peak_sustained_vram = torch.cuda.max_memory_allocated() / (1024**3)

        temp_note = "Stable"
        if initial_temp and temps:
            delta_temp = temps[-1] - initial_temp
            temp_note = f"Initial: {initial_temp:.1f}°C -> Final: {temps[-1]:.1f}°C (Delta: +{delta_temp:.1f}°C)"

        throughput_data = {
            "steps_per_second": round(steps_per_sec, 2),
            "step_time_ms": round(avg_step_time * 1000, 1),
            "peak_sustained_vram_gb": round(peak_sustained_vram, 3),
            "gpu_temp_trend_note": temp_note,
        }

        # Step 6: Extrapolations
        for target_steps in [1000, 5000, 20000, 50000]:
            total_sec = target_steps * avg_step_time
            hours = total_sec / 3600.0
            projections[f"{target_steps}_steps"] = round(hours, 2)

    # Step 7: Structured Decision Recommendation
    if successful_config is None:
        recommendation = "GCP REQUIRED — does not fit in available VRAM"
    else:
        req_chkpt = successful_config[2]
        proj_5k = projections.get("5000_steps", 999.0)
        proj_20k = projections.get("20000_steps", 999.0)

        if req_chkpt and proj_5k > 48.0:
            recommendation = (
                "GCP RECOMMENDED — fits, but pilot-scale training would take "
                "more than 2 days of continuous laptop operation"
            )
        elif proj_20k < 48.0:
            recommendation = "LAPTOP VIABLE — proceed with laptop-only Phase 4"
        else:
            recommendation = (
                "BORDERLINE — review the specific numbers manually before deciding; "
                "consider a hybrid approach (laptop for smaller ablation runs, GCP only for the full baseline run)"
            )

    # Compile Final JSON Output
    results = {
        "environment": env_report,
        "step4_results": step4_results,
        "throughput": throughput_data,
        "projections_hours": projections,
        "recommendation": recommendation,
    }

    out_dir = Path("diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "laptop_capability_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================================================================================")
    print("                           FINAL SUMMARY & DECISION                             ")
    print("================================================================================")
    print(f"Successful Configuration:   {successful_config[0] if successful_config else 'None'}")
    print(f"Measured Throughput:        {throughput_data.get('steps_per_second', 0.0)} steps/sec ({throughput_data.get('step_time_ms', 0.0)} ms/step)")
    print(f"Peak Sustained VRAM:        {throughput_data.get('peak_sustained_vram_gb', 0.0)} GB / {env_report['vram_total_gb']} GB")
    print(f"GPU Thermal Trend:          {throughput_data.get('gpu_temp_trend_note')}")
    print("\nProjected Training Time:")
    for k, v in projections.items():
        print(f"  • {k.replace('_', ' '):<15}: {v:.2f} hours ({v/24.0:.2f} days)")
    print("--------------------------------------------------------------------------------")
    print(f"DECISION RECOMMENDATION:\n>>> {recommendation} <<<")
    print("================================================================================")
    print(f"Full results saved to: {out_file}\n")

    return results


if __name__ == "__main__":
    run_laptop_capability_test()
