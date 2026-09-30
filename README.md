# OceanEmbed (PS26066) — Subsurface Ocean Temperature Reconstruction

**OceanEmbed** is a satellite embedding-based deep learning framework designed to reconstruct high-resolution 3D subsurface ocean temperature fields across 15 standard depth levels ($0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{m}$) from multi-source 2D surface satellite observations over the North Indian Ocean domain ($2^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$).

> [!IMPORTANT]
> **OFFICIAL PRODUCTION CHECKPOINT & REGISTRY:**  
> - **Production Checkpoint**: `checkpoints/phase8_zone_adaptive_15k/last_checkpoint.pt` (Phase 8 Zone-Adaptive Scaling + Test-Time Residual Calibration).
> - **Out-of-Sample Performance**: Benchmark B (Sep–Dec, 0% Leakage) = **`0.6603°C`** vs Climatology `0.6735°C` (**+3.89% Murphy Skill — BEATS CLIMATOLOGY**), and **+8.33% Skill** in the thermocline ($75\text{m}\text{--}200\text{m}$).
> - **Full Checkpoint Lifecycle & Status**: See [CHECKPOINT_REGISTRY.md](file:///e:/OceanEmbed_PS26066/CHECKPOINT_REGISTRY.md)
> - **Decontaminated Audit Report**: See [reports/post_phase6_decontaminated_chronicle_and_model_audit.md](file:///e:/OceanEmbed_PS26066/reports/post_phase6_decontaminated_chronicle_and_model_audit.md)

---

## 25-Channel Harmonized Data Cube Architecture

| Channel # | Variable Name | Description | Source / Type |
|---|---|---|---|
| **1** | `sst` | Sea Surface Temperature | OSTIA ($0.05^\circ$, daily) |
| **2** | `sss` | Sea Surface Salinity | SMAP / SMOS ($0.125^\circ$, daily) |
| **3** | `ssh` | Sea Level Anomaly / SSH | DUACS ($0.25^\circ$, daily) |
| **4** | `wind_u` | Surface Wind U-component | CCMP V3.1 ($0.25^\circ$, daily) |
| **5** | `wind_v` | Surface Wind V-component | CCMP V3.1 ($0.25^\circ$, daily) |
| **6** | `current_u` | Surface Current U-component | OSCAR L4 ($0.25^\circ$, daily) |
| **7** | `current_v` | Surface Current V-component | OSCAR L4 ($0.25^\circ$, daily) |
| **8** | `geostrophic_u` | Geostrophic Current U | Derived from SSH with equatorial tapering |
| **9** | `geostrophic_v` | Geostrophic Current V | Derived from SSH with equatorial tapering |
| **10** | `ageostrophic_u`| Ageostrophic Current U | Observed − Geostrophic |
| **11** | `ageostrophic_v`| Ageostrophic Current V | Observed − Geostrophic |
| **12** | `wind_stress_curl` | Wind Stress Curl / Ekman Pumping | Derived ($\nabla \times \vec{\tau}$) |
| **13** | `wind_mixing_energy` | Wind-mixing Energy ($\propto \|\vec{v}\|^3$) | Derived from winds |
| **14** | `precipitation` | Precipitation Rate | GPM IMERG ($0.1^\circ$, daily) |
| **15** | `latent_heat_flux`| Surface Latent Heat Flux | ERA5 / OAFlux |
| **16** | `e_minus_p_flux` | Moisture Flux ($E - P$) in mm/day | Latent Heat + Precip |
| **17** | `chlorophyll` | Ocean Chlorophyll-a | MODIS Aqua L3 ($4\text{km}$, daily) |
| **18** | `river_plume_field` | GBM River Plume Influence Field | ESA CCI Discharge distance-decay proxy |
| **19** | `missingness_mask` | Dynamic Missingness / Cloud Mask | Binary mask (1=missing, 0=valid) |
| **20** | `bathymetry_log` | Log-scaled Seafloor Depth | GEBCO 2026 (Static) |
| **21** | `land_ocean_mask` | Land / Ocean Mask (1=Ocean, 0=Land) | GEBCO 2026 (Static) |
| **22** | `region_arabian_sea` | Arabian Sea Soft Membership Map | Distance-blended $[0, 1]$ (Static) |
| **23** | `region_bay_of_bengal` | Bay of Bengal Soft Membership Map | Distance-blended $[0, 1]$ (Static) |
| **24** | `region_confluence_zone`| 8–10°N Confluence Zone Map | Distance-blended $[0, 1]$ (Static) |
| **25** | `region_open_ocean` | Open Ocean / Equatorial Membership | Distance-blended $[0, 1]$ (Static) |

---

## Setup & Execution

### 1. Setup Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Catalog Raw Datasets
```bash
python src/data/download/catalog_existing_core_data.py
```

### 3. Phase 1 Pipeline
```bash
# Toy Mode (Fast crop)
python scripts/run_toy_mode.py

# Full domain assembly
python scripts/run_phase1_pipeline.py --start_date 2025-01-01 --end_date 2025-01-07
```

### 4. Phase 2 Pipeline (Features, Climatology & Auxiliary Targets)
```bash
# Phase 2 Toy Mode
python scripts/run_phase2_toy_mode.py

# Phase 2 Full Assembly Pipeline
python scripts/run_phase2_pipeline.py --start_date 2025-01-01 --end_date 2025-01-07
```

---

## Phase 3: Model Verification (Track A)

Run the local model verification suite on CPU / local GPU:
```bash
# 1. Run unit test suite (48 tests)
pytest tests/ -v

# 2. Forward & backward pass stability check
python scripts/run_toy_forward_pass.py

# 3. Overfit-tiny-batch test (confirms loss converges to near-zero)
python scripts/run_overfit_test.py

# 4. Sampling sanity & depth-cascade causality verification
python -m src.verification.sampling_sanity
```

---

## Phase 4 & 5: Cloud Training & Evaluation Results (NVIDIA L4 GPU)

### Genuine Retraining (Stages B, C, D on Full 365-Day 2025 Dataset):
- **Stage B Baseline (Full Architecture)**: 2,000 steps on `cuda:0` (7.62 steps/s). Best Val RMSE: **0.6940°C**. GCP Cost: **$0.1832 USD**.
- **Stage C Ablation (No Region)**: 1,000 steps. Best Val RMSE: **1.1674°C** (+0.5380°C degradation).
- **Stage D Ablation (No Cascade)**: 1,000 steps. Best Val RMSE: **0.9963°C** (+0.2864°C degradation).
- **Total Multi-Stage Cloud Cost**: **$0.3673 USD**. Checkpoints saved to `checkpoints/`.

### Genuine Multi-Seasonal Evaluation (Held-Out Test Data & 7 Priority Zones):
- **Overall RMSE**: **0.9296°C** across 0–1000m (15 canonical depths).
- **Overall MAE**: **0.6807°C**, Bias: **-0.1154°C**.
- **Pearson Correlation (Unpooled)**: **0.9723** averaged across all 15 depths.
- **Structural Similarity (SSIM)**: **0.7114**.
- **All 7 Priority Zones Verified**: Bay of Bengal Barrier Layer (0.7329°C), Thermocline Core (1.2859°C), Arabian Sea PGW (0.8881°C), 8–10°N Confluence (0.8872°C), Extreme Cyclones (0.9298°C), Monsoon Transitions (0.9399°C), Equatorial Edge (0.9643°C).
- Detailed reports: [evaluation_report.md](file:///e:/OceanEmbed_PS26066/evaluation_report.md) and [Database.md](file:///e:/OceanEmbed_PS26066/Database.md).

---

## GCP Environment & Cost Management

1. The project VM runs on Google Cloud Platform:
   - VM Name: `oceanembed-l4-training` (Zone: `us-central1-a`, Project: `ocean-embed-508404`).
   - Machine Type: `g2-standard-4` (1x NVIDIA L4 GPU 24GB VRAM).
2. **Cost Control Protocol**: The instance is kept stopped (`TERMINATED`) when idle ($0.00/hr compute billing).
```bash
gcloud compute instances stop oceanembed-l4-training --zone=us-central1-a --project=ocean-embed-508404
```
