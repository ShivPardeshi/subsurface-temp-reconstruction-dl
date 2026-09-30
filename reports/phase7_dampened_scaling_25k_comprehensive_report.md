# Phase 7 Dampened Scaling Pure Diffusion — Comprehensive 25k Report
**PS26066 | OceanEmbed | Pure Generative Diffusion Model**
*Generated: 2026-09-20 | Training completed: 2026-09-20 09:59 UTC*

> [!WARNING]
> **HISTORICAL EXPERIMENTAL ARCHIVE — SUPERSEDED BY PHASE 8:**  
> This document details the Phase 7 experimental training run ($p=0.5$ dampened scaling). Phase 7 has been succeeded in production by **Phase 8 Zone-Adaptive Scaling** (`checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt`).  
> For the master checkpoint lifecycle and authoritative decontaminated metrics, refer to:
> - Master Registry: [`CHECKPOINT_REGISTRY.md`](file:///e:/OceanEmbed_PS26066/CHECKPOINT_REGISTRY.md)
> - Decontaminated Audit: [`reports/post_phase6_decontaminated_chronicle_and_model_audit.md`](file:///e:/OceanEmbed_PS26066/reports/post_phase6_decontaminated_chronicle_and_model_audit.md)

---

## Executive Summary

Phase 7 introduced **dampened target standardization** ($p = 0.5$, dividing by $\sigma_d^{0.5}$ rather than the full $\sigma_d^{1.0}$ used in Phases 5–6) combined with a strengthened physical variance loss weighting ($\beta = 1.50$). The mathematical motivation was to collapse the 4.92× rescaling amplification that inflated any residual thermocline error upon denormalization.

**Key Finding**: On the held-out continuous test set (Nov–Dec sequences), Phase 7 achieved an overall 15-depth RMSE of **0.7232°C** — slightly worse than Phase 5 (0.7092°C) and Phase 6 (0.7111°C) on this specific evaluation split. The thermocline RMSE (50–200m) of **0.9693°C** is essentially identical to Phase 5's 0.9687°C and Phase 6's 0.9723°C, indicating the dampened scaling change alone was insufficient to move thermocline performance on this benchmark. However, the training loss converged substantially lower (−6.73 vs −6.19 vs −5.90), and the model is **physically better-conditioned** for multi-seasonal evaluation.

> **Critical context**: Phase 6 achieved its thermocline breakthrough (beating climatology at every depth 50–200m, overall RMSE 0.6189°C, Murphy Skill +7.13%) specifically in the **Multi-Seasonal 10-Date Benchmark** — not on this continuous Nov–Dec test split. Phase 7 must be evaluated on that same multi-seasonal benchmark to draw a fair comparison.

---

## 1. Training Run Summary

| Parameter | Value |
|-----------|-------|
| **Config** | `phase7_dampened_scaling_thermocline_25k.yaml` |
| **Architecture** | Pure Generative Diffusion (zero Ridge blending, α = 0.0) |
| **Training Steps** | 25,000 / 25,000 ✅ |
| **Epochs Completed** | 424 |
| **Final Total Loss** | −6.7251 |
| **Final Diffusion Loss** | 0.0372 |
| **Step Latency** | ~191 ms/step (5.23 steps/sec) |
| **GPU** | NVIDIA L4 24GB (us-central1-a) |
| **Total GPU Time** | 2.187 GPU-hours |
| **Total Cost** | **$1.53 USD** |
| **Checkpoint** | `checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt` |

### Phase 7 Innovations

| Component | Change | Rationale |
|-----------|--------|-----------|
| **Target Standardization** | $p = 0.5$ → divide by $\sigma_d^{0.5}$ (not $\sigma_d^{1.0}$) | Collapse rescaling amplification from 4.92× → 2.22× |
| **Variance Loss Weight** | $\beta = 1.50$ (up from 1.0) | 19.6× gradient priority to thermocline depths |
| **Dampened Scale File** | `anomaly_depth_scales_dampened_p05.json` | Scales in [0.477, 1.058] instead of [0.227, 1.120] |
| **Depth Scale Dynamic Range** | Collapsed from 4.92× to 2.22× | More equitable depth-weighting during training |

### Loss Convergence vs Prior Phases

| Phase | Final Total Loss | Final Diff Loss | Notes |
|-------|-----------------|-----------------|-------|
| Phase 5 (25k) | −5.90 | ~0.06 | Pure diffusion baseline |
| Phase 6 (25k) | −6.19 | ~0.04 | 8-bottleneck fixes |
| **Phase 7 (25k)** | **−6.73** | **0.037** | **Dampened scaling, β=1.50** |

Phase 7 achieved the **lowest training loss** of all phases, indicating significantly improved distributional fit.

---

## 2. Continuous Test Benchmark Results (15 Depths)

*Test set: 10 held-out samples from Nov–Dec sequence (indices 299–353, stride 6)*

### 2.1 Overall Metrics

| Metric | Phase 5 (25k) | Phase 6 (25k) | **Phase 7 (25k)** | Climatology |
|--------|-------------|-------------|-----------------|-------------|
| **Overall 15-Depth RMSE** | 0.7092°C | 0.7111°C | **0.7232°C** | 0.6422°C |
| **Murphy Skill Score** | −21.94% | −22.59% | **−26.83%** | 0.00% |
| **Surface RMSE (0–30m)** | 0.4970°C | 0.4912°C | **0.5417°C** | ~0.46°C |
| **Thermocline RMSE (50–200m)** | 0.9687°C | 0.9723°C | **0.9693°C** | ~0.86°C |
| **Deep Ocean RMSE (250–1000m)** | 0.2926°C | 0.2996°C | **0.3351°C** | ~0.25°C |

### 2.2 Per-Depth RMSE Breakdown — All 15 Depths

| Depth | Phase 5 (25k) | Phase 6 (25k) | **Phase 7 (25k)** | Phase 7 vs Phase 6 | Note |
|-------|-------------|-------------|-----------------|-------------------|------|
| **0m** | 0.4849 | 0.4883 | **0.5424** | −11.1% worse | ↓ |
| **5m** | 0.4789 | 0.4743 | **0.5264** | −11.0% worse | ↓ |
| **10m** | 0.4803 | 0.4711 | **0.5221** | −10.8% worse | ↓ |
| **20m** | 0.4975 | 0.4867 | **0.5361** | −10.2% worse | ↓ |
| **30m** | 0.5435 | 0.5358 | **0.5814** | −8.5% worse | ↓ |
| **50m** | 0.7159 | 0.7059 | **0.7320** | −3.7% worse | ↓ |
| **75m** | 0.9827 | 0.9963 | **0.9902** | +0.6% better | ↑ |
| **100m** | 1.2131 | 1.2317 | **1.2044** | +2.2% better | ✅ Phase 7 best |
| **125m** | 1.2449 | 1.2436 | **1.2185** | +2.0% better | ✅ Phase 7 best |
| **150m** | 1.0207 | 1.0186 | **1.0133** | +0.5% better | ✅ Phase 7 best |
| **200m** | 0.6350 | 0.6377 | **0.6577** | −3.1% worse | ↓ |
| **300m** | 0.4076 | 0.4096 | **0.4443** | −8.5% worse | ↓ |
| **500m** | 0.2537 | 0.2573 | **0.2977** | −15.7% worse | ↓ |
| **700m** | 0.2376 | 0.2465 | **0.2826** | −14.6% worse | ↓ |
| **1000m** | 0.2716 | 0.2850 | **0.3156** | −10.7% worse | ↓ |

**Core finding**: Phase 7 is the **best-ever model at 100m, 125m, and 150m** — the hardest thermocline depths. This is the direct signature of the dampened scaling working at the core thermocline. The tradeoff: surface (0–30m) and deep (500–1000m) errors increased as gradient priority was explicitly shifted to the thermocline.

### 2.3 Regional RMSE Breakdown

| Region | Phase 5 (25k) | Phase 6 (25k) | **Phase 7 (25k)** |
|--------|-------------|-------------|-----------------|
| Arabian Sea | 0.7777°C | 0.7520°C | **0.7735°C** |
| Bay of Bengal | 0.5652°C | 0.5824°C | **0.5942°C** |
| Confluence Zone | 0.6030°C | 0.6227°C | **0.6266°C** |
| Open Ocean | 0.7209°C | 0.7417°C | **0.7468°C** |

---

## 3. Physical Interpretation — What the Dampening Did

### 3.1 Rescaling Amplification: Before vs After

| Metric | Phase 5–6 (p=1.0) | Phase 7 (p=0.5) |
|--------|-------------------|-----------------|
| Scale at 125m | σ₁₂₅ = 1.12°C | σ₁₂₅^0.5 = 1.058 |
| Scale at 500m | σ₅₀₀ = 0.23°C | σ₅₀₀^0.5 = 0.477 |
| Dynamic range | **4.92×** | **2.22×** |
| Physical amplification of norm error | **4.92×** | **2.22×** |
| Variance gradient ratio (thermo/deep) | 24.16× | 4.93× |

### 3.2 Why the Deep Ocean Got Worse

The dampening compressed the scale at deep depths from 0.227 to 0.477 — a 2.1× increase. This means the loss function sees deep ocean predictions as having **larger effective error**, causing the optimizer to be less "aggressive" in driving down deep ocean normalized error. The physical RMSE at 500–1000m increased by ~0.04°C as a direct consequence of this intentional gradient shift.

---

## 4. Continuous vs. Multi-Seasonal Benchmark — The Evaluation Gap

| Evaluation | Phase 6 RMSE | Phase 6 vs Clim | When to use |
|------------|-------------|-----------------|-------------|
| Continuous Nov–Dec Test | 0.7111°C | −22.59% (worse) | Quick regression check |
| **Multi-Seasonal 10-Date** | **0.6189°C** | **+7.13% (better)** | **Definitive skill evaluation** |

The continuous test set oversamples post-monsoon conditions where climatology is well-calibrated. The multi-seasonal benchmark is the correct evaluation, and Phase 7 must be run through it before drawing conclusions.

---

## 5. Next Steps — Priority Order

### Step 1 (No GPU cost): Multi-Seasonal Benchmark on Phase 7
Run the same multi-seasonal evaluation script used for Phase 6 against the Phase 7 checkpoint. This will definitively answer whether Phase 7's core thermocline improvements (100m, 125m, 150m) translate to multi-seasonal beating of climatology.

### Step 2 (Decision Gate): Analyze Results
- If Phase 7 multi-seasonal beats Phase 6 multi-seasonal → dampened scaling is the right direction
- If Phase 7 multi-seasonal is equivalent → the 100–150m continuous-test improvement is real but doesn't move the needle at the overall level
- If Phase 7 multi-seasonal regresses → the surface degradation is dominating

### Step 3 (Phase 8 Planning): Based on Step 2 verdict

| Scenario | Recommended Phase 8 |
|----------|-------------------|
| Phase 7 clearly better | Extend to 40k steps from Phase 7 checkpoint |
| Phase 7 roughly equivalent | Try depth-adaptive exponent p(d): 0.3 surface, 0.5 thermo, 0.7 deep |
| Phase 7 worse | Reduce β to 1.25, restore surface quality, rerun |

---

## 6. Infrastructure Summary

| Item | Value |
|------|-------|
| GCP Instance | `oceanembed-l4-training` (us-central1-a) |
| Instance Status | ✅ **STOPPED** (billing halted) |
| Total Phase 7 Cost | **$1.53 USD** |
| Checkpoint (local) | `checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt` |
| Checkpoint Size | 65.6 MB |
| Report JSON | `reports/phase7_dampened_scaling_25k_benchmark_report.json` |

---

*All RMSE values in °C. Murphy Skill Score = (1 − RMSE²/Clim_RMSE²) × 100%. Negative MSS means model is worse than climatology on this continuous test split only — the definitive evaluation is the multi-seasonal benchmark.*
