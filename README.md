# OceanEmbed — 3D Subsurface Ocean Temperature Reconstruction via Physics-Conditioned Diffusion

**OceanEmbed** is a satellite embedding-based deep learning framework designed to reconstruct high-resolution 3D subsurface ocean temperature fields across 15 canonical depth levels ($0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\,\text{m}$) from multi-source 2D surface satellite observations over the North Indian Ocean domain ($2^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$).

---

## 🌟 Key Highlights

- **25-Channel Multi-Modal Surface Conditioning**: Fuses satellite altimetry (SLA/ADT), sea surface temperature (OSTIA SST), sea surface salinity (SMOS/SMAP), satellite wind stress (CCMP V3.1), OSCAR surface currents, derived geostrophic/ageostrophic dynamics, Ekman pumping, moisture flux ($E - P$), chlorophyll-a, river discharge plumes, and static geophysical masks.
- **Deep Architecture**: Context-conditioned diffusion architecture with depth-cascaded inductive bias and spatio-temporal ConvLSTM feature encoding.
- **Strict In-Situ Validation**: Validated against both reanalysis-assimilated profiles and **66 unassimilated GO-SHIP research cruise CTD stations (990 physical depth soundings)** across canonical Indian Ocean transects (Lines I01, I08N, I09N) achieving **$r = 0.9965$ vertical correlation** and **$0.32^\circ\text{C}$ MAE**.
- **Downstream Oceanographic Products**: Automated computation of Ocean Heat Content (OHC), Tropical Cyclone Heat Potential (TCHP), Mixed Layer Depth (MLD), Barrier Layer Thickness (BLT), and 3D Marine Heatwave (MHW) detection.

---

## 📊 25-Channel Harmonized Data Cube Architecture

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
| **20** | `bathymetry_log` | Log-scaled Seafloor Depth | GEBCO (Static) |
| **21** | `land_ocean_mask` | Land / Ocean Mask (1=Ocean, 0=Land) | GEBCO (Static) |
| **22** | `region_arabian_sea` | Arabian Sea Soft Membership Map | Distance-blended $[0, 1]$ (Static) |
| **23** | `region_bay_of_bengal` | Bay of Bengal Soft Membership Map | Distance-blended $[0, 1]$ (Static) |
| **24** | `region_confluence_zone`| 8–10°N Confluence Zone Map | Distance-blended $[0, 1]$ (Static) |
| **25** | `region_open_ocean` | Open Ocean / Equatorial Membership | Distance-blended $[0, 1]$ (Static) |

---

## 🛠️ Installation & Setup

### 1. Environment Setup
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

### 3. Data Processing & Feature Assembly Pipeline
```bash
# Toy Mode (Fast local test crop)
python scripts/run_toy_mode.py

# Full domain assembly pipeline
python scripts/run_phase1_pipeline.py --start_date 2025-01-01 --end_date 2025-01-07
python scripts/run_phase2_pipeline.py --start_date 2025-01-01 --end_date 2025-01-07
```

---

## 🧪 Model Verification & Testing

Run the full local verification and unit test suite:
```bash
# 1. Run unit test suite
pytest tests/ -v

# 2. Forward & backward pass stability check
python scripts/run_toy_forward_pass.py

# 3. Overfit-tiny-batch test
python scripts/run_overfit_test.py

# 4. Sampling sanity & depth-cascade causality verification
python -m src.verification.sampling_sanity
```

---

## 📈 Evaluation & Benchmarks

Run comprehensive evaluation across multi-seasonal partitions, priority oceanographic regimes, and independent in-situ datasets:

```bash
# Comprehensive model evaluation
python scripts/run_full_evaluation.py

# Independent CCHDO / GO-SHIP CTD in-situ validation
python scripts/evaluate_cchdo_goship_validation.py
```

---

## 🖥️ Live Scientific Dashboard & API

Launch the FastAPI backend and interactive scientific interface:

```bash
# Start FastAPI backend server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Launch interactive UI
streamlit run src/dashboard/app.py
```
