# OceanEmbed (PS26066) — Comprehensive Project Handoff Document

> **Project**: OceanEmbed (Smart India Hackathon - Problem Statement **PS26066**)  
> **Theme**: Disaster Management / Oceanography  
> **Domain**: North Indian Ocean ($2.0^\circ\text{N}–30.0^\circ\text{N}, 45.0^\circ\text{E}–105.0^\circ\text{E}$ at $0.25^\circ$ resolution, $112 \times 240$ spatial grid)  
> **Target**: 3D Subsurface Ocean Temperature Reconstruction across 15 canonical depths ($0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{m}$).

---

## 1. Executive Summary & Problem Understanding

Reconstructing subsurface ocean thermal structures from 2D satellite surface observations is critical for tropical cyclone intensity forecasting, marine heatwave detection, and disaster management. Existing single-depth regression models fail to capture the physics of vertical thermal gradients, the Arabian Sea High Salinity Watermass (ASHSW), and Bay of Bengal barrier layers.

**OceanEmbed** addresses this with a **5-Stage Deep Learning Architecture**:
1. **Stage 1 (Context Encoder)**: 3-layer Spatiotemporal ConvLSTM encoding 7-day multi-satellite surface history `(B, T=7, C=25, H=112, W=240)`.
2. **Stage 2 (Conditioning Module)**: Spatial fusion (70 channels) + AdaGroupNorm / FiLM multi-stage non-spatial parameter modulation (log-depth, ONI, IOD/DMI, day-of-year cyclic embeddings, previous depth summary).
3. **Stage 3 (U-Net Denoiser)**: 4-stage noise prediction backbone (`32 -> 64 -> 128 -> 256`) with skip connections.
4. **Stage 4 (Diffusion Mechanics & Depth Cascade)**: Cosine DDPM forward process + accelerated DDIM reverse sampling with strict shallow-to-deep sequential cascade (`0 -> 5 -> 10 -> ... -> 1000m`).
5. **Stage 5 (Auxiliary Heads & Adaptive Losses)**: Multi-task heads for Mixed Layer Depth (MLD), Bay of Bengal Barrier Layer Thickness (BLT), and Arabian Sea Salinity Maximum, trained via Pinn-Ocean style adaptive homoscedastic uncertainty loss ($w_1, w_2, w_3$) and depth/region weighting $\alpha(d, \text{region})$.

---

## 2. Completed Phases & Implemented Codebase

### **Phase 1: Data Ingestion & Harmonization Pipeline (Completed & Passed)**
- **Raw Datasets Processed & Cataloged** in `data/raw/`:
  - `glorys/global_phy_subset.nc` (12.15 GB, 36 levels)
  - `argo/incois_argo_mnt_VAM...nc` (1.61 MB)
  - `sst_sss_ssh_currents_winds/` (OSTIA SST, SMAP/SMOS SSS, DUACS SSH, OSCAR currents, CCMP winds)
  - `river_discharge/`, `precipitation/`, `heat_flux/`, `wind_curl/`, `chlorophyll/`, `bathymetry/` (GEBCO 2026), `climate_indices/` (ONI, IOD), and `ibtracs/`.
- **Core Harmonization Engine (`src/data/harmonize/`)**:
  - `regrid.py`: Bilinear interpolation engine for 2D/3D grids.
  - `landmask.py`: High-performance GEBCO landmask and distance transform.
  - `region_masks.py`: Soft distance-blended 4-region maps (BoB, AS, EqIO, SouthNIO).
  - `build_datacube.py`: 25-channel daily data cube assembler.

### **Phase 2: Feature Engineering, Climatology & Auxiliary Targets (Completed & Passed)**
- **Derived Physical Features (`src/features/`)**:
  - `geostrophic.py`: Geostrophic currents with $\tanh(|\phi|/2.0)$ equatorial tapering.
  - `ageostrophic.py`: Ageostrophic residual currents ($u_a, v_a$).
  - `wind_derived.py`: Wind stress $(\tau_x, \tau_y)$, wind stress curl ($\nabla \times \vec{\tau}$), and mixing energy proxy ($E_{\text{mix}} = |\vec{v}|^3$).
  - `moisture_flux.py`: Evaporation ($E$) and net freshwater flux ($E - P$ in $\text{mm/day}$).
  - `river_plume.py`: Ganga-Brahmaputra-Meghna delta distance-decay proxy ($Q \exp(-d/L_d)$).
- **Harmonic Climatology Engine (`src/climatology/`)**:
  - `fit_climatology.py`: 2-harmonic OLS seasonal cycle with **strict anti-leakage assertion** (training period only).
  - `compute_anomaly.py`: 15-depth anomaly target computation ($\Delta T = T_{\text{actual}} - \bar{T}$).
- **Auxiliary Physical Targets (`src/auxiliary_targets/`)**:
  - `mixed_layer_depth.py`: Linear vertical interpolation MLD.
  - `barrier_layer_thickness.py`: $\text{BLT} = \text{ILD} - \text{MLD}_{\rho}$ masked to Bay of Bengal.
  - `salinity_maximum.py`: Subsurface salinity max depth ($100–400\text{m}$) and strength anomaly masked to Arabian Sea.
- **Assembled Outputs in `data/processed/phase2_dataset/`**:
  - `oceanembed_training_inputs.zarr` (25-channel daily inputs, shape `(7, 25, 112, 240)`)
  - `oceanembed_anomaly_targets.zarr` (15-depth anomaly targets, shape `(7, 15, 112, 240)`)
  - `oceanembed_auxiliary_targets.zarr` (4 auxiliary targets, shape `(7, 4, 112, 240)`)
  - `climatology_coefficients.nc` (5 harmonic parameters per cell and depth)
  - `scalar_conditioning.csv` (DOY cyclic embeddings, ONI, and IOD indices)

### **Phase 3 Track A: Model Architecture & Local Verification (Completed & Passed)**
- **Model Components (`src/models/`)**:
  - `context_encoder.py`: 3-layer ConvLSTM stack (`32 -> 64 -> 64`).
  - `conditioning.py`: Spatial fusion (70 channels) + AdaGroupNorm MLP injection.
  - `unet_denoiser.py`: 4-stage U-Net denoiser backbone with skip connections.
  - `diffusion.py`: Cosine schedule forward process and analytical $\hat{x}_0$ reconstruction (`predict_x0_from_noise`).
  - `auxiliary_heads.py`: 3 MLP auxiliary heads.
- **Sampling & Cascade (`src/sampling/`)**:
  - `ddim_sampler.py`: Accelerated DDIM reverse sampling.
  - `depth_cascade.py`: `sample_full_profile()` enforcing strict sequential cascade.
- **Training & Losses (`src/training/`)**:
  - `losses.py`: Adaptive loss $L_{\text{total}} = \sum e^{-w_i} L_i + w_i$ with depth/region weighting $\alpha(d, \text{region})$.
  - `dataset.py`: PyTorch Dataset/DataLoader.
  - `checkpoint_utils.py`: Atomic save/resume utilities.
  - `train.py`: Modular training loop orchestrator.
- **Verification Suite & Diagnostics**:
  - `tests/`: **All 37/37 Unit Tests Passing**.
  - `scripts/run_toy_forward_pass.py`: Forward/backward pass passed with finite gradients.
  - `scripts/run_overfit_test.py`: Overfit tiny batch passed (Loss dropped from **4809.46** to **-0.48122**, diffusion loss to **0.00005**).
  - `src/verification/sampling_sanity.py`: Sampling sanity, causality check, and multi-seed variance passed.
  - `scripts/diagnostics/laptop_capability_test.py`: Stage A0 empirical measurement diagnostic for RTX 3050.

---

## 3. Current State & GCP vs. Laptop Situation

### What happened with GCP:
- GCP command returned: `Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.`
- In GCP, new accounts have `GPUS_ALL_REGIONS = 0.0` by default and require requesting a global quota increase in the GCP Console (IAM & Admin $\to$ Quotas).
- **Data size consideration**: Uploading 60 GB of raw data from `data/raw/` is unnecessary; the model only consumes the preprocessed Zarr data (`data/processed/`), which is much smaller.

### Immediate Alternative & Decision (Stage A0 Diagnostic):
We created [`scripts/diagnostics/laptop_capability_test.py`](file:///E:/OceanEmbed_PS26066/scripts/diagnostics/laptop_capability_test.py) implementing the **Stage A0 Capability Test Spec** ([`references/plan/PS26066_StageA0_Laptop_Capability_Test.md`](file:///E:/OceanEmbed_PS26066/references/plan/PS26066_StageA0_Laptop_Capability_Test.md)).

You can run this diagnostic directly on your laptop right now:
```bash
python scripts/diagnostics/laptop_capability_test.py
```
This script will:
1. Test progressive memory configurations (FP32 $\to$ AMP FP16 $\to$ Gradient Checkpointing) at Batch Size = 1.
2. Measure exact steps/sec and peak VRAM across 50 real training steps on your RTX 3050.
3. Compute exact extrapolated wall-clock times (1k, 5k, 20k, 50k steps).
4. Output a clear automated recommendation (`LAPTOP VIABLE`, `GCP RECOMMENDED`, or `GCP REQUIRED`) saved to `diagnostics/laptop_capability_test_results.json`.

---

## 4. Most Important Files to Read First in the New Chat

When opening a new chat, the agent should read these key reference files first:

1. [`references/plan/PS26066_StageA0_Laptop_Capability_Test.md`](file:///E:/OceanEmbed_PS26066/references/plan/PS26066_StageA0_Laptop_Capability_Test.md) — Specification for the laptop capability diagnostic and decision logic.
2. [`references/plan/PS26066_Phase3_Plan.md`](file:///E:/OceanEmbed_PS26066/references/plan/PS26066_Phase3_Plan.md) — Model architecture, ConvLSTM, U-Net denoiser, loss functions, and GCP specs.
3. [`references/plan/PS26066_Phase2_Plan (1).md`](file:///E:/OceanEmbed_PS26066/references/plan/PS26066_Phase2_Plan%20%281%29.md) — Feature engineering, climatology formulas, and auxiliary targets.
4. [`references/plan/PS26066_Phase_Overview_and_Phase1_Plan (1).md`](file:///E:/OceanEmbed_PS26066/references/plan/PS26066_Phase_Overview_and_Phase1_Plan%20%281%29.md) — Master project roadmap, dataset catalog, and Phase 1 specifications.
5. [`walkthrough.md`](file:///C:/Users/123ta/.gemini/antigravity-ide/brain/0f4640c2-a12e-4eb6-a1d9-0b14d6d41a48/walkthrough.md) — Comprehensive technical walkthrough with all verification results and test outputs.
6. [`README.md`](file:///E:/OceanEmbed_PS26066/README.md) — Environment setup, channel table, and execution instructions.

---

## 5. Next Steps Moving Ahead

1. **Run the Diagnostic**:
   ```bash
   python scripts/diagnostics/laptop_capability_test.py
   ```
2. **Review the Output**:
   - If `LAPTOP VIABLE`: Proceed directly to Phase 4 full training locally on your RTX 3050 (AMP FP16 + Gradient Accumulation).
   - If `GCP REQUIRED / RECOMMENDED`: Request the `GPUS_ALL_REGIONS` quota increase in the GCP console, then launch the VM with the provided `scripts/gcp/setup_vm.sh`.
3. **Execute Phase 4 (Full Model Training & Validation)**:
   - Run multi-year training loop with checkpointing.
   - Evaluate against GLORYS ground truth and Argo profile benchmarks ($0–1000\text{m}$).
   - Compute RMSE, MAE, thermocline depth error, and MLD/BLT auxiliary metrics.
