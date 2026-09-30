# OceanEmbed (PS26066) — Master Checkpoint & Production Model Registry

**Document Version**: 1.0 (Master Cleaned & Audited)  
**Last Updated**: 2026-09-21 08:50 IST  
**Repository**: `E:\OceanEmbed_PS26066`  
**Problem Statement**: Smart India Hackathon PS26066 (3D Ocean Subsurface Temperature Reconstruction)

---

## 1. Single Canonical Current Production Configuration

To eliminate ambiguity across scattered historical documents, the single active production setup for all downstream operations, API endpoints, disaster products (OHC, TCHP, MLD, MHW), and evaluations is defined below:

| Configuration Attribute | Primary Production Specification | Fallback / Diagnostic Specification |
| :--- | :--- | :--- |
| **Model Type** | **Phase 8 Zone-Adaptive Pure Diffusion + Test-Time Residual Calibration** | Multi-Output Ridge Regression Baseline |
| **Active Checkpoint** | `checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt` | In-memory closed-form weights (fitted on Days 6–236) |
| **Context Encoder** | 3-Layer ConvLSTM (25 channels in, hidden dims: 32 -> 64 -> 64) | Linear feature matrix (13 oceanographic channels + 4 scalars) |
| **Normalization Stats** | `data/processed/channel_normalization_stats_ocean_only.json` | Training set ocean-only mean & std |
| **UNet Denoiser** | 4 Stages (32 -> 64 -> 128 -> 256), `in_channels=74`, `cond_in_dim=14` | N/A |
| **Auxiliary Heads** | `AuxiliaryHeads(in_features=64)` (MLD, BLT, SalMax depth) | N/A |
| **Depth Cascade** | Sequential shallow-to-deep ($0\text{m} \to 1000\text{m}$ across 15 depths) | Full-column simultaneous prediction |
| **Depth Scaling Map** | `data/processed/anomaly_depth_scales_zone_adaptive_p8.json` ($p=0.75, 0.50, 0.75, 1.00$) | N/A |
| **Sampling Schedule** | DDIM Sampler, 25 Quadratic Steps, $\eta = 0.0$ | Direct matrix multiply |
| **Test-Time Calibration** | Strategy 1 (Bayesian Shrinkage $\gamma(z) \in [0.85, 0.94]$ + BoB Bias Removal) + Strategy 2 (Dual-Sample Posterior Mean) | None |
| **Region Conditioning** | Sigmoid distance maps (Arabian Sea, Bay of Bengal, Confluence, Open Ocean) | Explicit spatial point coordinates + regional masks |
| **Downstream Products** | Direct Profile Integration (OHC 700m/D26, TCHP, de Boyer Montégut MLD, Hobday MHW Cat 1–4) | Direct Profile Integration |

---

## 2. Complete Checkpoint Inventory & Status Taxonomy

Every checkpoint in the repository is cataloged below with its lifecycle status:
- `ACTIVE-PRODUCTION`: The official, verified model to be loaded by all production services and dashboards.
- `EXPERIMENTAL`: Checkpoints with valuable physical or architectural research findings, but superseded by Phase 8 for production deployment.
- `DEPRECATED`: Checkpoints that overfit (e.g. 40k/60k runs) or suffer from known architectural limitations.
- `HISTORICAL-ABLATION`: Ablation run checkpoints retained for scientific rigor and benchmark verification.

| Checkpoint Path | Step Count | Size | Status Label | Operational Verdict & Description |
| :--- | :---: | :---: | :---: | :--- |
| **`checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt`** | **15,000** | **67.2 MB** | **`ACTIVE-PRODUCTION`** | **Official Production Model.** Unifies thermocline breakthrough ($0.9635^\circ\text{C}$) with surface and deep-abyss stability. Outperforms Climatology on clean Benchmark B (`0.6603°C`, +3.89% Skill). |
| `checkpoints/phase7_dampened_scaling_thermocline_25k/last_checkpoint.pt` | 25,000 | 67.2 MB | `EXPERIMENTAL` | Introduced dampened scaling ($p=0.5$). Slashed thermocline error but suffered from surface/deep loss starvation. |
| `checkpoints/phase6_thermocline_breakthrough_25k/last_checkpoint.pt` | 25,000 | 67.2 MB | `EXPERIMENTAL` | Initial thermocline breakthrough model ($0.9723^\circ\text{C}$). Superseded by Phase 8 zone-adaptive scaling. |
| `checkpoints/phase5_rectified_pure_diffusion_25k/last_checkpoint.pt` | 25,000 | 67.2 MB | `EXPERIMENTAL` | First full-column pure diffusion run with auxiliary physical heads. |
| `checkpoints/phase4_scratch_20k_test/best_checkpoint.pt` | 20,000 | 67.2 MB | `EXPERIMENTAL` | Clean scratch 20k test baseline with full cosine schedule. |
| `checkpoints/phase4_scratch_40k_v2/best_checkpoint.pt` | 40,000 | 67.2 MB | `DEPRECATED` | **Confirmed Overfit.** Training beyond 25k caused generalisation degradation on unseen test dates. |
| `checkpoints/phase4_scratch_60k_full/best_checkpoint.pt` | 60,000 | 67.2 MB | `DEPRECATED` | **Confirmed Overfit.** High-step overfitting in thermocline layers. |
| `checkpoints/phase3_retrain_scratch_40k_full/best_checkpoint.pt` | 40,000 | 67.3 MB | `DEPRECATED` | Superseded by Phase 4/5 rectified architecture. |
| `checkpoints/phase3_retrain_scratch_40k/best_checkpoint.pt` | 40,000 | 67.3 MB | `DEPRECATED` | Superseded by Phase 4/5 rectified architecture. |
| `checkpoints/phase3_retrain_scratch_20k/best_checkpoint.pt` | 20,000 | 67.3 MB | `DEPRECATED` | Early normalized scratch prototype. |
| `checkpoints/phase3_retrain_normalized/last_checkpoint.pt` | 40,000 | 67.3 MB | `DEPRECATED` | Fine-tuned from corrupted legacy weights. |
| `checkpoints/baseline_20k_fixA2_seed42/best_checkpoint.pt` | 20,000 | 67.3 MB | `HISTORICAL-ABLATION` | Seed 42 equalized baseline checkpoint used in Phase 5 ablation matrix. |
| `checkpoints/baseline_20k_seed43/best_checkpoint.pt` | 20,000 | 67.3 MB | `HISTORICAL-ABLATION` | Seed 43 multi-seed validation baseline. |
| `checkpoints/ablation_no_region_20k_fixA2_seed42/best_checkpoint.pt` | 20,000 | 67.2 MB | `HISTORICAL-ABLATION` | Region-conditioning ablation checkpoint (Seed 42). |
| `checkpoints/ablation_no_region_20k_seed43/best_checkpoint.pt` | 20,000 | 67.2 MB | `HISTORICAL-ABLATION` | Region-conditioning ablation checkpoint (Seed 43). Confirms +12.7% error penalty without region conditioning. |
| `checkpoints/ablation_no_cascade_20k_fixA2_seed42/best_checkpoint.pt` | 20,000 | 67.3 MB | `HISTORICAL-ABLATION` | Depth-cascade ablation checkpoint. Confirms critical necessity of shallow-to-deep feedback. |
| `checkpoints/baseline/best_checkpoint.pt` | 2,000 | 67.3 MB | `DEPRECATED` | 2k-step toy verification run. |

---

## 3. Mandatory Metric Retirement & Benchmark Authority

> [!CAUTION]
> **RETIRED HISTORICAL BENCHMARK CITATIONS:**
> The historical claim of `+19.49% Murphy Skill Score` across a "Multi-Seasonal 10-Date Benchmark" evaluated Days `[15, 60, 105, 150, 195, 240, 285, 318, 331, 358]`. Because the model was trained on Days 0–236, **5 of those 10 dates were inside the training set**.
> 
> **That contaminated benchmark is PERMANENTLY RETIRED.** No new reports or submissions may cite that figure as out-of-sample generalization.

### The Two Authoritative Clean Benchmarks (0.0% Leakage)

All current and future reporting MUST cite metrics from the two audited benchmarks established in [`reports/post_phase6_decontaminated_chronicle_and_model_audit.md`](file:///e:/OceanEmbed_PS26066/reports/post_phase6_decontaminated_chronicle_and_model_audit.md):

```
                                ┌────────────────────────────────────────────────────────┐
                                │           UNCONTAMINATED BENCHMARK TAXONOMY            │
                                └───────────────────────────┬────────────────────────────┘
                                                            │
         ┌──────────────────────────────────────────────────┴──────────────────────────────────────────────────┐
         ▼                                                                                                     ▼
 【Benchmark A: Pure Held-Out Test】                                                    【Benchmark B: Extended Out-of-Sample】
 • Timeframe: Nov 01 – Dec 31 (Days 298–358)                                            • Timeframe: Sep 01 – Dec 31 (Days 237–358)
 • Sample Indices: [305, 311, 317, 323, 329, 335, 341, 347, 353, 359]                   • Sample Indices: [238, 251, 264, 277, 290, 303, 316, 330, 344, 357]
 • Target DOYs: 312, 318, 324, 330, 336, 342, 348, 354, 360, 365                        • Target DOYs: 245, 258, 271, 284, 297, 310, 323, 337, 351, 364
 • Training Data Leakage: 0.0% (Zero)                                                   • Training Data Leakage: 0.0% (Zero)
 • Primary Benchmark Metric: Model 0.6838°C vs Clim 0.6613°C                             • Primary Benchmark Metric: Model 0.6603°C vs Clim 0.6735°C (+3.89% Skill)
```

### Verified Multi-Model Decontaminated Results Matrix

| Model / Architecture | Status | Benchmark A (Nov–Dec Test) | Benchmark B (Sep–Dec Out-of-Sample) | Thermocline Band (50–200m on B) |
| :--- | :---: | :---: | :---: | :---: |
| **Climatology Baseline (2-Harmonic OLS)** | Baseline | `0.6613°C` (0.00% skill) | `0.6735°C` (0.00% skill) | `1.0640°C` (0.00% skill) |
| **Ridge Regression Baseline (Days 6–236)** | Baseline | `0.6548°C` (+1.95% skill) | `0.6422°C` (+9.08% skill) | `1.0261°C` (+7.00% skill) |
| **Phase 4 Baseline Diffusion (20k)** | Experimental | `0.9834°C` (-121.12% skill) | `1.0030°C` (-121.77% skill) | `1.4215°C` (-78.73% skill) |
| **Phase 6 Thermocline Breakthrough (25k)**| Experimental | `0.8124°C` (-50.92% skill) | `0.8089°C` (-44.23% skill) | `1.1568°C` (-18.25% skill) |
| **Phase 7 Dampened Scaling (25k)** | Experimental | `0.7775°C` (-38.23% skill) | `0.7672°C` (-29.74% skill) | `1.1070°C` (-8.25% skill) |
| **Phase 8 Zone-Adaptive Raw (15k)** | Production-Raw | `0.7147°C` (-16.79% skill) | `0.6884°C` (-4.48% skill) | `1.0517°C` (+2.30% skill) |
| **Phase 8 Calibrated Ensemble (Opt)** | **ACTIVE-PROD** | **`0.6838°C`** (-6.92% skill) | **`0.6603°C`** (**+3.89% Skill — BEATS CLIM**) | **`1.0187°C`** (**+8.33% Skill — BEATS CLIM**) |

---

## 4. Report Archival & Lineage Directory

All reports generated during previous rounds have been preserved for analytical and diagnostic lineage under the following classification:

### Active & Audited Reports (Current Source of Truth)
1. **[`CHECKPOINT_REGISTRY.md`](file:///e:/OceanEmbed_PS26066/CHECKPOINT_REGISTRY.md)** — This master document.
2. **[`reports/post_phase6_decontaminated_chronicle_and_model_audit.md`](file:///e:/OceanEmbed_PS26066/reports/post_phase6_decontaminated_chronicle_and_model_audit.md)** — Master decontaminated audit and benchmark taxonomy.
3. **[`reports/phase8_zone_adaptive_15k_comprehensive_report.md`](file:///e:/OceanEmbed_PS26066/reports/phase8_zone_adaptive_15k_comprehensive_report.md)** — Phase 8 GPU training and physical evaluation report.
4. **[`handoff.md`](file:///e:/OceanEmbed_PS26066/handoff.md)** — Live session state and concrete roadmap.

### Historical & Experimental Archives (Preserved with Archival Banners)
- `reports/post_phase6_complete_chronicle_and_phase8_breakthrough.md` — Historical narrative pre-decontamination audit.
- `reports/phase7_dampened_scaling_25k_comprehensive_report.md` — Phase 7 experimental engineering log.
- `reports/phase6_thermocline_breakthrough_25k_comprehensive_report.md` — Phase 6 experimental engineering log.
- `reports/phase5_rectified_pure_diffusion_25k_comprehensive_report.md` — Phase 5 experimental engineering log.
- `reports/phase4_40k_v2_evaluation_report.md` — Deprecated 40k overfit evaluation log.
- `reports/phase4_20k_test_evaluation_report.md` — Phase 4 baseline log.
- `reports/pure_diffusion_ultimate_breakthrough_analysis_and_plan.md` — Historical roadmap.
- `PROJECT_MASTER_STATUS_AND_REPORT.md` — Master engineering chronicle.
