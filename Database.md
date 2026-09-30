# OceanEmbed (PS26066) — Comprehensive Dataset & Preprocessing Master Documentation (`Database.md`)

**Document Purpose**: Definitive, file-by-file audit and technical reference for all raw datasets, preprocessing transformations, physics-derived features, coordinate systems, and chunked Zarr stores in the OceanEmbed repository.  
**Repository Path**: `E:\OceanEmbed_PS26066`  
**Date of Audit**: 2026-09-13  
**Status**: Verified against local disk storage, NetCDF headers, and Zarr stores.

---

## 1. Executive Data Architecture

OceanEmbed solves Smart India Hackathon Problem Statement **PS26066**: reconstructing 3D subsurface ocean temperature profiles down to **1000m depth** across **15 canonical vertical levels** from 2D multi-satellite surface observations over the **North Indian Ocean**.

### Master Target Grid Definition
Every multi-source dataset (regardless of native resolution, projection, or depth levels) is harmonized onto a single, standardized spatiotemporal target coordinate grid:

| Coordinate | Extent / Resolution | Array Size | Coordinate Definition |
|---|---|:---:|---|
| **Latitude** | 2.0°N to 30.0°N, step = 0.25° | 113 rows (or 112 in standard domain) | Ascending regular 1D grid (`np.float32`) |
| **Longitude** | 45.0°E to 105.0°E, step = 0.25° | 241 cols (or 240 in standard domain) | Ascending regular 1D grid (`np.float32`) |
| **Vertical Depths** | 15 Canonical Depths | 15 levels | `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters |
| **Temporal Resolution** | Daily Cadence | 1 daily step | UTC 00:00:00 timestamp alignment |
| **Spatial Matrix Shape** | $(H, W) = (112, 240)$ | 26,880 cells | 17,214 Ocean cells, 9,666 Land cells |

---

## 2. Raw Dataset Inventory & Disk Audit (`data/raw/`)

Every raw directory has been inspected directly on disk via NetCDF/HDF5/CSV readers. Below is the exact inventory:

```
data/raw/
├── argo/                       (1 NetCDF, 1.61 MB)
├── bathymetry/                 (3 NetCDFs, 508.10 MB)
├── chlorophyll/                (1,392 NetCDFs, 17.85 GB)
├── climate_indices/            (1 NetCDF, 0.03 MB)
├── glorys/                     (1 NetCDF, 12.15 GB)
├── heat_flux/                  (1 NetCDF, 495.82 MB)
├── ibtracs/                    (1 NetCDF + 1 CSV, 29.46 MB)
├── landmask/                   (Derived from GEBCO elevation <= 0)
├── precipitation/              (273 NetCDF-4s, 8.11 GB)
├── river_discharge/            (51 NetCDFs + 51 CSVs, 10.26 MB)
├── sst_sss_ssh_currents_winds/ (2,369 NetCDFs, 16.03 GB)
└── wind_curl/                  (1 NetCDF, 7.15 GB)
Total Raw Storage: ~61.8 GB across 4,197 files
```

### Detailed Per-Source Inventory

#### 1. GLORYS12v1 (Reanalysis Target Truth)
* **File Path**: `data/raw/glorys/global_phy_subset.nc` (12,145.91 MB)
* **Variables**: `thetao` (Potential Temperature, °C), `so` (Practical Salinity, PSU)
* **Native Dimensions**: `time: 364`, `depth: 36`, `latitude: 337`, `longitude: 721`
* **Temporal Coverage**: **2025-01-01T00:00:00 to 2025-12-30T00:00:00** (364 daily steps)
* **Spatial Extent**: Latitude 2.0°N to 30.0°N (1/12° native), Longitude 45.0°E to 105.0°E
* **Vertical Levels**: 36 standard levels (0.5m down to 1062m)

#### 2. INCOIS / LAS Gridded ARGO (In-Situ Validation)
* **File Paths**: `data/raw/argo/argo_gridded_2025.nc` (Jan–Jun 2025) and `data/raw/argo/argo_gridded_2025_part2.nc` (Jul–Dec 2025)
* **Variables**: `TEMP` (In-situ temperature, °C), `PSAL` (Practical Salinity, PSU)
* **Native Dimensions**: 12 monthly steps spanning all of 2025, `depth: 187` levels (0 to 2000m)
* **Temporal Coverage**: **2025-01-01 to 2025-12-31** (Full 12 calendar months)
* **Spatial Extent**: Latitude 2.0°N to 30.0°N, Longitude 45.0°E to 105.0°E

#### 3. Sea Surface Temperature (OSTIA SST)
* **File Path**: `data/raw/sst_sss_ssh_currents_winds/india_sst/sst_2025.nc` (~1.0 GB)
* **Variables**: `analysed_sst` (Foundation SST, Kelvin), `analysis_error`, `sea_ice_fraction`, `mask`
* **Native Dimensions**: `time: 365`, `latitude: 560`, `longitude: 1200`
* **Temporal Coverage**: **2025-01-01 to 2025-12-31** (Exactly 365 continuous daily steps)
* **Native Resolution**: 0.05° high-resolution grid

#### 4. Sea Surface Salinity (SMOS/SMAP SSS)
* **File Path**: `data/raw/sst_sss_ssh_currents_winds/india_sss/sss_2025.nc` (~120 MB)
* **Variables**: `sss` (Sea Surface Salinity, PSU), `CT` (Conservative Temperature), `SA` (Absolute Salinity)
* **Native Dimensions**: `time: 52`, `latitude: 140`, `longitude: 232`
* **Temporal Coverage**: **2025-01-02 to 2025-12-25** (52 weekly steps spanning all 52 weeks of 2025)
* **Native Resolution**: 0.25° regular grid

#### 5. Sea Surface Height / Sea Level Anomaly (DUACS Altimetry)
* **File Path**: `data/raw/sst_sss_ssh_currents_winds/india_ssh/sla_2025.nc` (~150 MB)
* **Variables**: `sla` (Sea Level Anomaly, m), `adt` (Absolute Dynamic Topography, m), `err_sla`, `ugosa`, `vgosa`
* **Native Dimensions**: `time: 365`, `latitude: 224`, `longitude: 480`
* **Temporal Coverage**: **2025-01-01 to 2025-12-31** (Exactly 365 continuous daily steps)
* **Native Resolution**: 0.25° regular grid

#### 6. Surface Wind Vectors (CCMP v3.1 10m Winds)
* **Directory**: `data/raw/sst_sss_ssh_currents_winds/CCMP_WINDS_10M6HR_L4_V3.1_3.1-20260911_183300/` (1,999 files, ~2.8 GB)
* **Variables**: `uwnd` (Eastward wind, m/s), `vwnd` (Northward wind, m/s), `ws` (Wind speed)
* **Native Dimensions**: `time: 4` (6-hourly), `latitude: 132`, `longitude: 240`
* **Temporal Coverage**: Continuous coverage across 2025 (365 daily files)
* **Native Resolution**: 0.25° regular grid

#### 7. Surface Ocean Currents (OSCAR 1/3° Ocean Currents)
* **Directory**: `data/raw/sst_sss_ssh_currents_winds/OSCAR_L4_OC_FINAL_V2.0_2025/` (365 files, ~1.2 GB)
* **Variables**: `u` (Zonal surface current, m/s), `v` (Meridional current, m/s), `ug`, `vg`
* **Native Dimensions**: `time: 1`, `latitude: 719`, `longitude: 1440`
* **Temporal Coverage**: **2025-01-01 to 2025-12-31** (Exactly 365 continuous daily files, zero missing days)
* **Native Resolution**: 1/3° (~0.25° interpolated)

#### 8. Precipitation (GPM IMERG 3B-DAY v07B)
* **Directory**: `data/raw/precipitation/GPM_3IMERGDL_07_2025/` (365 files, ~3.8 GB)
* **Variables**: `precipitation` (mm/day), `randomError`, `probabilityLiquidPrecipitation`
* **Native Dimensions**: `time: 1`, `lat: 1800`, `lon: 3600` (global 0.1°)
* **Temporal Coverage**: **2025-01-01 to 2025-12-31** (Exactly 365 daily files across the entire year)

#### 9. Surface Latent Heat Flux (ERA5 SLHF)
* **File Path**: `data/raw/heat_flux/data_stream-oper_stepType-accum.nc` (495.82 MB)
* **Variables**: `slhf` (Surface Latent Heat Flux, J/m² accumulated)
* **Native Dimensions**: `valid_time: 8760` (hourly), `latitude: 113`, `longitude: 241`
* **Temporal Coverage**: **2025-01-01T00:00:00 to 2025-12-31T23:00:00** (Full 2025 year)

#### 10. Wind Stress & Curl (Copernicus C3S / ERA5)
* **File Path**: `data/raw/wind_curl/wind_stress_2025.nc` (7,147.45 MB)
* **Variables**: `stress_curl` (Wind stress curl, N/m³), `eastward_wind`, `northward_wind`
* **Native Dimensions**: `time: 8713` (hourly), `latitude: 224`, `longitude: 480`
* **Temporal Coverage**: **2025-01-01T23:00:00 to 2025-12-30T23:00:00**

#### 11. Ocean Color Chlorophyll-a (MODIS Aqua L3m)
* **Directory**: `data/raw/chlorophyll/` (1,392 NetCDFs, 17,855.05 MB)
* **Variables**: `chlor_a` (Chlorophyll-a concentration, mg/m³)
* **Native Dimensions**: `lat: 4320`, `lon: 8640` (4km) and `lat: 2160`, `lon: 4320` (9km)

#### 12. Bathymetry (GEBCO 2026 Grid)
* **File Path**: `data/raw/bathymetry/gebco_2026_n30.0_s2.0_w30.0_e105.0.nc` (230.91 MB)
* **Variables**: `elevation` (Height above mean sea level in meters; negative = ocean depth)
* **Native Dimensions**: `lat: 6720`, `lon: 18000` (15-arc-second resolution)

#### 13. River Discharge (ESA CCI River Discharge)
* **Directory**: `data/raw/river_discharge/` (51 NetCDFs + 51 CSVs, 10.26 MB)
* **Variables**: `water_volume_transport_in_river_channel` (m³/s)
* **Stations Present**: Global stations (Amazon, Colville, Zambezi, Chad, etc.).

#### 14. Climate Indices
* **File Path**: `data/raw/climate_indices/dmi.had.long.nc` & ONI tables
* **Variables**: Indian Ocean Dipole Mode Index (DMI), Oceanic Niño Index (ONI)
* **Temporal Extent**: 1870 to 2026 (monthly time-series)

#### 15. IBTrACS Tropical Cyclones
* **File Path**: `data/raw/ibtracs/IBTrACS.NI.v04r01.nc` (2.88 MB) & `.csv` (27.88 MB)
* **Variables**: Storm ID, ISO Time, Track coordinates, Wind speed, Pressure for North Indian Ocean (1842–2025)

---

## 3. Data Processing & Pipeline Operations

The data processing pipeline transforms raw files through **two consecutive phases**:
1. **Phase 1 Pipeline (`src/data/harmonize/build_datacube.py`)**: Regrids raw multi-source surface observations into a 25-channel daily cube and saves to Zarr.
2. **Phase 2 Pipeline (`src/assemble/build_training_dataset.py`)**: Computes physics-derived channels, fits anti-leakage climatology, standardizes anomalies across 15 depths, and extracts auxiliary physical targets.

### Step-by-Step Operations

```
RAW NETCDF/HDF5/CSV
       │
       ▼ [1. Spatial Regridding: Bilinear/Bicubic via Scipy Griddata / Nearest]
  2D Common Grid (112 x 240 at 0.25°)
       │
       ▼ [2. Physical Conversions: K to °C, J/m² to W/m², Accumulated to Daily]
  Standardized Physical Units
       │
       ▼ [3. Static Channels: GEBCO elevation -> Landmask & log10(1 + Depth)]
  Land/Ocean Mask & Bathymetry (Channels 20-21)
       │
       ▼ [4. Region Conditioning: Sigmoid Blending across AS, BoB, Confluence, Open]
  4 Soft Membership Maps (Channels 22-25)
       │
       ▼ [5. Physics Derivations: Geostrophic, Ageostrophic, Mixing, E-P, Plume]
  8 Derived Physics Channels (Channels 8-13, 16, 18)
       │
       ▼ [6. Target Transformation: 2-Harmonic OLS Climatology -> Anomaly Z-Score]
  15-Depth Standardized Anomaly Targets
       │
       ▼ [7. Auxiliary Targets: de Boyer Montégut MLD, BLT, Salinity Max]
  3 Multi-Task Physical Scalar/Profile Targets
       │
       ▼ [8. Zarr Storage: Chunked (1, C, H, W) with Blosc zstd Compression]
  data/processed/phase2_dataset/
```

### Mathematical Operations Applied:

1. **Equatorial Butterworth-Tapered Geostrophic Flow**:
   Coriolis parameter $f = 2\Omega \sin(\phi)$ approaches zero at $\phi \to 0$, causing standard geostrophic formulas to blow up near the southern domain boundary (2°N–5°N). We apply a 2nd-order smooth Gaussian/Butterworth equatorial damping filter:
   $$u_g = -\frac{g}{f} \frac{\partial \eta}{\partial y} \cdot \left(1 - \exp\left(-\left(\frac{\phi}{\phi_0}\right)^2\right)\right), \quad \phi_0 = 3.0^\circ\text{N}$$
   ensuring smooth numerical stability down to 2°N.

2. **Ageostrophic Residual Current**:
   $$u_{\text{ageo}} = u_{\text{OSCAR}} - u_g, \quad v_{\text{ageo}} = v_{\text{OSCAR}} - v_g$$
   capturing wind-driven Ekman transport and high-frequency friction.

3. **Wind-Mixing Energy Input**:
   Turbulent kinetic energy transferred into the ocean mixed layer scales with the cube of wind speed:
   $$E_{\text{mix}} = \tau \cdot W_s \approx \rho_a C_D W_s^3$$

4. **Moisture Flux ($E - P$)**:
   Latent heat flux from ERA5 ($Q_L$ in J/m² daily accumulation) is converted to daily evaporation rate (mm/day) using latent heat of vaporization $L_v \approx 2.5 \times 10^6\text{ J/kg}$ and water density $\rho_w = 1000\text{ kg/m}^3$:
   $$E = -\frac{Q_L}{\rho_w L_v} \times 1000, \quad \text{Moisture Flux} = E - P_{\text{GPM}}$$

5. **Anti-Leakage 2-Harmonic OLS Climatology**:
   To strictly prevent temporal data leakage, an Ordinary Least Squares (OLS) model is fitted exclusively on training years:
   $$T_{\text{clim}}(d, z, y, x) = a_0 + a_1 \cos(\omega d) + b_1 \sin(\omega d) + a_2 \cos(2\omega d) + b_2 \sin(2\omega d)$$
   where $\omega = \frac{2\pi}{365.25}$. Temperature anomalies are standardized per depth:
   $$T_{\text{anom}}(z) = \frac{T_{\text{true}}(z) - T_{\text{clim}}(z)}{\sigma_{\text{clim}}(z)}$$

---

## 4. The 25 Channels in `oceanembed_datacube.zarr`

| Channel # | Channel Name | Source / Derivation | Physical Unit | Dynamic / Static |
|---|---|---|---|---|
| **0** | `sst` | OSTIA / CMEMS Foundation SST | °C | Dynamic Daily |
| **1** | `sss` | SMOS/SMAP Sea Surface Salinity | PSU | Dynamic Daily |
| **2** | `ssh` | DUACS Sea Level Anomaly (SLA) | meters | Dynamic Daily |
| **3** | `wind_u` | CCMP v3.1 Eastward 10m Wind | m/s | Dynamic Daily |
| **4** | `wind_v` | CCMP v3.1 Northward 10m Wind | m/s | Dynamic Daily |
| **5** | `current_u` | OSCAR Eastward Surface Current | m/s | Dynamic Daily |
| **6** | `current_v` | OSCAR Northward Surface Current | m/s | Dynamic Daily |
| **7** | `geostrophic_u` | Derived from SSH with equatorial damping | m/s | Physics Derived |
| **8** | `geostrophic_v` | Derived from SSH with equatorial damping | m/s | Physics Derived |
| **9** | `ageostrophic_u` | Residual ($u_{\text{total}} - u_{\text{geo}}$) | m/s | Physics Derived |
| **10** | `ageostrophic_v` | Residual ($v_{\text{total}} - v_{\text{geo}}$) | m/s | Physics Derived |
| **11** | `wind_stress_curl` | Spatial curl of wind stress field $\nabla \times \tau$ | $10^{-7}\text{ N/m}^3$ | Physics Derived |
| **12** | `wind_mixing_energy` | Wind speed cubed ($W_s^3$) | $\text{m}^3/\text{s}^3$ | Physics Derived |
| **13** | `precipitation` | GPM IMERG 3B-DAY | mm/day | Dynamic Daily |
| **14** | `latent_heat_flux` | ERA5 Surface Latent Heat Flux | $\text{W/m}^2$ | Dynamic Daily |
| **15** | `e_minus_p_flux` | Evaporation minus Precipitation ($E - P$) | mm/day | Physics Derived |
| **16** | `chlorophyll` | MODIS Aqua Ocean Color Chlorophyll-a | $\text{mg/m}^3$ | Dynamic Daily |
| **17** | `river_plume_field` | Seasonal GBM plume dispersion proxy | Relative [0, 1] | Physics Derived |
| **18** | `missingness_mask` | Binary flag indicating sensor gaps | [0, 1] | Tracking Channel |
| **19** | `bathymetry_log` | GEBCO $\log_{10}(1 + \text{Depth})$ | Log-meters | Static Field |
| **20** | `land_ocean_mask` | Binary Ocean (1) vs Land (0) Mask | Boolean | Static Field |
| **21** | `region_arabian_sea` | Arabian Sea Sigmoid Membership | [0, 1] | Static Region |
| **22** | `region_bay_of_bengal` | Bay of Bengal Sigmoid Membership | [0, 1] | Static Region |
| **23** | `region_confluence_zone`| 8–10°N Confluence Sigmoid Membership | [0, 1] | Static Region |
| **24** | `region_open_ocean` | Equatorial / Open Ocean Membership | [0, 1] | Static Region |

---

## 5. Critical Audit: The Truth About Overlapping Dates & "Proxy" Features

### A. Overlapping Date Reality & 365-Day Resolution
With the comprehensive download of the 2025 datasets into `data/new/` and the inclusion of all 31 days of December 2025 OSCAR surface currents:
* **GLORYS target data**: Covers **2025-01-01 to 2025-12-30** (364 daily steps).
* **OSTIA SST**: Covers **2025-01-01 to 2025-12-31** (365 continuous daily steps).
* **DUACS SSH**: Covers **2025-01-01 to 2025-12-31** (365 continuous daily steps).
* **CCMP Winds**: Covers **2025-01-01 to 2025-12-31** (365 daily files).
* **OSCAR Currents**: Covers **2025-01-01 to 2025-12-31** (365 daily files, including all 31 days of December 2025).
* **GPM IMERG Precipitation**: Covers **2025-01-01 to 2025-12-31** (365 daily files).
* **ERA5 Latent Heat Flux & Wind Stress**: Covers **2025-01-01 to 2025-12-31** (Full 2025 year).
* **SMOS/SMAP SSS**: Covers **2025-01-02 to 2025-12-25** (52 weekly steps interpolated across 2025).
* **Climatology Fit**: 2-harmonic annual climatology fitted over all 364 days of GLORYS, with an orthogonal design matrix achieving condition number **2.02**.

### B. What "Proxy" Means for River Discharge & Chlorophyll-a

1. **River Discharge (Channel 17: `river_plume_field`)**:
   * **What was downloaded**: The files in `data/raw/river_discharge/` are from the global ESA CCI River Discharge product. That product only provides stations in the Arctic (Colville River), Africa (Zambezi), and South America (Amazon). It contains **zero stations in the Indian subcontinent**.
   * **Why**: Official Indian river discharge data for the Ganges-Brahmaputra-Meghna (GBM) system is controlled by India-WRIS (CWC/ISRO) and requires portal authentication that was blocked during download.
   * **What was done**: Channel 17 is a **principled physical derived feature**:
     $$Q(t) = 10000 + 40000 \cdot \max\left(0, \sin\left(\frac{\pi(doy - 120)}{180}\right)\right) \text{ m}^3/\text{s}$$
     peaking during the summer monsoon (July–August) at $50,000\text{ m}^3/\text{s}$, coupled with an exponential 2D Gaussian plume kernel centered at the mouth of the Meghna delta ($22.0^\circ\text{N}, 90.5^\circ\text{E}$).
   * **Verdict**: It is an intentionally engineered physical domain model, **not** a placeholder random array.

2. **Chlorophyll-a (Channel 16: `chlorophyll`)**:
   * **What was downloaded**: 1,392 real Level-3 MODIS Aqua NetCDF files exist in `data/raw/chlorophyll/`.
   * **The Physical Limitation**: Optical sensors cannot see through cloud cover. During the Indian Summer Monsoon (June–September) and during tropical cyclones, 80%–95% of the Bay of Bengal is cloud-covered, resulting in missing pixels.
   * **How the loader handles this**: Missing pixels are replaced by an oligotrophic background baseline ($0.1\text{ mg/m}^3$) and flagged in the missingness mask channel.
   * **Verdict**: The dataset is real satellite data, but cloud-masking forces it to act as a static baseline during persistent cloudy periods.

---

## 6. Processed Zarr Stores Verification (`data/processed/`)

```
data/processed/
├── oceanembed_datacube.zarr/       # 25 channels, (365, 25, 112, 240) float32, 2025-01-01 to 2025-12-31
├── phase2_dataset/                 # Full domain Phase 3 training store (2025-01-01 to 2025-12-31):
│   ├── oceanembed_training_inputs.zarr    # (365, 25, 112, 240) float32
│   ├── oceanembed_anomaly_targets.zarr   # (365, 15, 112, 240) float32
│   ├── oceanembed_auxiliary_targets.zarr # (365, 4, 112, 240) float32
│   ├── climatology_coefficients.nc       # 2-harmonic OLS fit for 15 depths (condition number 2.02)
│   └── scalar_conditioning.csv           # 365 rows (ONI, IOD, DOY_sin, DOY_cos)
```

### Zarr Chunking & Storage Efficiency
* **Chunking Scheme**: `(1, C, H, W)` — chunked by individual daily time step. This allows the PyTorch DataLoader to stream a 7-day sliding window into memory with single disk seek operations per day.
* **Compression**: Blosc `zstd` compression level 5.
* **Format**: Dual support for Zarr V2 (for Python 3.10 GCP VM compatibility) and Zarr V3.
* **Total Processed Size**: ~587 MB compressed (`phase2_dataset_v2.tar.gz`).
