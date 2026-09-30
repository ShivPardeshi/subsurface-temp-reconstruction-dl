# OceanEmbed (Smart India Hackathon PS26066) — Scientific Disclosure & Independent Validation Framework

> **Project**: Smart India Hackathon **PS26066** (Disaster Management / Oceanography)  
> **Problem Statement**: 3D Subsurface Ocean Temperature Reconstruction down to 1000m  
> **Document Purpose**: Methodological audit, observational assimilation disclosure, and architecture for independent-of-GLORYS validation.  
> **Status**: Official Methodological Disclosure & Known Limitation Declaration  
> **Date**: 2026-09-19  

---

## 1. Executive Summary & Scientific Integrity Statement

OceanEmbed reconstructs 3D ocean temperature fields ($112 \times 240 \times 15$ depth levels from $0\text{--}1000\text{m}$) across the North Indian Ocean using surface satellite observations (SST, SSS, SLA, surface currents, 10m wind stress, surface fluxes, and chlorophyll). 

To ensure absolute scientific rigor and avoid misleading claims, this document provides a transparent audit of:
1. **The Role of GLORYS12v1 as Ground Truth**: Why global ocean reanalysis is the only viable substrate for continuous 3D basin-scale deep learning, and why it is not an unconstrained numerical simulation.
2. **Observational Entanglement**: The exact mechanism by which observational platforms (including the RAMA moored buoy array and ARGO profiling floats) are assimilated into GLORYS12v1 via the Copernicus In-Situ TAC (CORA database).
3. **Distinction between In-Situ Emulation and Independent Validation**: Why evaluating against assimilated ARGO/RAMA data measures *in-system physical fidelity*, whereas evaluating against unassimilated research cruise CTD sections constitutes *pure independent validation*.
4. **Known Limitation & Post-Hackathon Roadmap**: The open status of sourcing unassimilated cruise datasets (e.g. GO-SHIP, NIOT/INCOIS Sagar Kanya CTD casts) and the newly implemented `IndependentInSituValidator` pipeline ready for post-submission deployment.

---

## 2. Census of Data Assimilation in GLORYS12v1

GLORYS12v1 (Global Ocean Reanalysis and Simulation v12) is produced by Mercator Ocean International using the NEMO (Nucleus for European Modelling of the Ocean) hydrodynamic model coupled with the SEEK (Singular Evolutive Extended Kalman) filter and Incremental Analysis Updates (IAU).

### Ingested Observational Streams (per CMEMS-GLO-QUID-001-030)

| Observational Platform | Data Source & Database | Ingestion Cadence & Role | Assimilation Status in GLORYS12v1 |
| :--- | :--- | :--- | :---: |
| **ARGO Profiling Floats** | CORA (Copernicus In-Situ TAC) | Real-time & delayed-mode T/S profiles down to 2000m | **Assimilated** |
| **RAMA Moored Buoy Array** | CORA (In-Situ TAC / NOAA PMEL) | Fixed-point continuous upper-ocean moorings (0–500m) | **Assimilated** |
| **Ship-of-Opportunity XBTs** | WMO GTS / CORA archive | Expendable bathythermograph temperature soundings | **Assimilated** |
| **Altimetry (SLA / ADT)** | DUACS Multi-Mission Altimetry | Along-track SLA (Jason-3, Sentinel-3A/B, SARAL, CryoSat-2) | **Assimilated** |
| **Sea Surface Temp (SST)** | OSTIA L4 Foundation SST | Multi-sensor infrared/microwave satellite fusion | **Assimilated (SST nudging)** |
| **Sea Ice Concentration** | OSI-SAF Polar Sea Ice | High-latitude boundary conditions | **Assimilated** |

```
                               ┌─────────────────────────────────────────────────────────────┐
                               │                 In-Situ Observational Streams                │
                               │  (ARGO Floats, RAMA Moorings, SOOP XBTs, Gliders via CORA)  │
                               └──────────────────────────────┬──────────────────────────────┘
                                                              │
                                                              ▼
┌────────────────────────────────────────┐       ┌────────────────────────┐       ┌───────────────────────────────────────┐
│     Satellite Surface Observations     │ ----> │  GLORYS12v1 Reanalysis │ <---- │        NEMO Hydrodynamic Engine       │
│ (OSTIA SST, DUACS SLA, CCMP Winds, SSS)│       │     (SEEK Filter)      │       │     (Navier-Stokes, Mass/Salt Cons)   │
└────────────────────────────────────────┘       └────────────┬───────────┘       └───────────────────────────────────────┘
                                                              │
                                                              ▼
                                               ┌────────────────────────────┐
                                               │   OceanEmbed Training &    │
                                               │  Benchmark Ground Truth    │
                                               │ (2025 Daily 3D Target Grid)│
                                               └────────────────────────────┘
```

### Scientific Implication of Assimilation
Because GLORYS12v1 dynamically assimilates ARGO floats and RAMA moored buoys into its continuous hydrographic state, any in-situ evaluation comparing model predictions against ARGO or RAMA observations that were ingested into CORA is testing **consistency with the assimilated observation stream**. It does **not** test performance against a fully unobserved, decoupled ocean system.

---

## 3. What Our Existing Benchmarks Measure

To maintain transparency, all model metrics in this repository are categorized as follows:

### 1. 3D Reanalysis Emulation Benchmarks (Continuous Test & Multi-Seasonal)
* **Dataset**: CMEMS GLORYS12v1 held-out test partitions (Continuous: Nov 1 – Dec 31, 2025; Multi-Seasonal: 10 dates).
* **What It Measures**: The ability of the hybrid Ridge + Diffusion architecture to ingest 2D satellite surface observations and infer the complete, dynamically-consistent 3D temperature structure down to 1000m with spatial continuity, eddy turbulence, and pycnocline stratification.
* **Result**: **`0.6164 °C` Multi-Seasonal RMSE** (Skill: `+0.0787`) and **`0.6439 °C` Continuous Test RMSE** (Skill: `+0.0454`), strictly beating the Climatology hurdle across 14/15 vertical depths and all 7 regional zones.

### 2. ARGO Gridded Profile Benchmark
* **Dataset**: INCOIS / LAS Gridded ARGO 2025 (`data/raw/argo/`).
* **What It Measures**: Pointwise vertical profile fidelity and barrier layer thickness (BLT $r=+0.923$) and mixed layer depth (MLD $r=+0.957$) physical correlation.
* **Disclosure**: This benchmark confirms that OceanEmbed retains true vertical thermohaline gradients rather than generating blurred spatial averages. However, because ARGO profiles are part of the CORA assimilation pipeline in GLORYS, this is an *in-system physical fidelity test*, not an independent unassimilated test.

---

## 4. Target Datasets for True Independent-of-GLORYS Validation

A truly independent validation requires testing against hydrographic observations that were **completely withheld from or never ingested into** the CORA/GTS assimilation system. The three primary target sources are:

### 1. GO-SHIP Indian Ocean Repeat Hydrography Sections
* **Target Sections**:
  - **I08N** (80°E meridian section from $60^\circ\text{S}$ to $12^\circ\text{N}$ in the Bay of Bengal).
  - **I09N** (95°E section extending into the Andaman Sea).
  - **I01E / I01W** (Equatorial Indian Ocean transect along $5^\circ\text{S}–5^\circ\text{N}$).
* **Instrumentation**: High-precision SBE-911plus CTD rosettes with calibrated platinum resistance thermometers (accuracy $<0.002^\circ\text{C}$, vertical resolution $1\text{--}2\text{ dbar}$).
* **Independence**: Delayed-mode GO-SHIP research cruise data packages are processed with strict quality control outside the real-time operational assimilation stream.

### 2. MoES / INCOIS / NIOT Dedicated Research Vessel CTD Archives
* **Platforms**:
  - *ORV Sagar Kanya* (National Institute of Oceanography / INCOIS)
  - *FORV Sagar Sampada* (CMLRE)
  - *ORV Sagar Nidhi* (National Institute of Ocean Technology)
* **Target Data**: Proprietary process-study cruise CTDs conducted during the Indian Ocean Monsoon and Bay of Bengal Boundary Layer Experiment (BoBBLE) campaigns that are not distributed via real-time GTS.

### 3. High-Frequency Underwater Glider Missions
* **Target Missions**: Deep autonomous buoyancy gliders (e.g. Slocum G3, Seaglider) deployed during the Ocean Mixing and Monsoons (OMM) collaborative experiment.
* **Advantage**: Provides continuous, high-temporal-resolution ($<2\text{ hours}$) oblique saw-tooth soundings capturing sub-mesoscale thermocline oscillations.

---

## 5. Software Architecture: `IndependentInSituValidator`

To support plug-and-play validation as soon as independent cruise files are ingested, we built a standardized validation module: [`src/evaluation/independent_validator.py`](file:///e:/OceanEmbed_PS26066/src/evaluation/independent_validator.py).

### Core Capabilities
1. **Bilinear Spatial Interpolation**: Accurately extracts continuous vertical profiles at arbitrary off-grid coordinates $(lat, lon)$ within the $0.25^\circ$ domain without nearest-neighbor quantization error.
2. **Canonical Vertical Harmonization**: Spline/linear interpolation of continuous CTD soundings (e.g. at 1m depth steps) onto the 15 canonical target levels (`[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters).
3. **Quality & Boundary Filtering**: Automatically flags soundings violating physical bounds ($T < -2^\circ\text{C}$ or $T > 38^\circ\text{C}$), bathymetric limits, or land-mask boundaries.
4. **Strict Metadata & Assimilation Tracking**: Each profile carries an explicit boolean flag `is_assimilated_in_glorys`. If any profile in the evaluation set has `is_assimilated_in_glorys = True`, the resulting evaluation report explicitly marks `is_strictly_independent = False`.

### Python Integration Example
```python
from src.evaluation.independent_validator import IndependentInSituValidator, InSituProfile

validator = IndependentInSituValidator()

# Define an independent cruise profile (e.g., Sagar Kanya CTD cast)
profile = InSituProfile(
    profile_id="SK340_CAST_042",
    latitude=14.25,
    longitude=87.50,
    date_str="2025-06-15",
    depths_m=np.array([0, 10, 25, 50, 100, 150, 200, 500, 1000]),
    temperatures_c=np.array([29.1, 28.9, 28.2, 26.5, 21.0, 16.8, 14.2, 8.5, 5.8]),
    platform_type="CTD_Cast",
    cruise_or_mission="ORV_Sagar_Kanya_BoB_Cruise",
    is_assimilated_in_glorys=False  # Pure independent validation
)

results = validator.evaluate_profiles(predictions_by_date, climatology_by_date, [profile])
print(f"Independent In-Situ RMSE: {results['overall_metrics']['model_rmse_c']:.4f} °C")
print(f"Murphy Skill Score:      {results['overall_metrics']['murphy_skill_score']:+.4f}")
print(f"Strict Independence:     {results['metadata']['is_strictly_independent']}")
```

---

## 6. Official Submission Limitation Disclosure

```
========================================================================================================
                              KNOWN LIMITATION & SCIENTIFIC DISCLOSURE
========================================================================================================
1. Reanalysis Ground Truth: OceanEmbed is trained and primarily benchmarked on CMEMS GLORYS12v1 
   reanalysis fields. GLORYS12v1 assimilates satellite altimetry, SST, and CORA in-situ profiles 
   (ARGO and RAMA arrays).

2. In-Situ Evaluation Scope: Point-level ARGO and RAMA comparisons in this report verify vertical 
   profile fidelity and boundary layer physics (MLD/BLT) within the assimilated reanalysis framework.

3. True Independent-of-GLORYS Search: Sourcing, spatial-temporal collocation, and statistical 
   benchmarking against completely unassimilated research cruise CTD sections (such as GO-SHIP repeat 
   hydrography or non-GTS MoES ship casts) remains an open, high-value milestone that is scheduled for 
   the post-hackathon research phase using the newly integrated `IndependentInSituValidator` pipeline.
========================================================================================================
```
