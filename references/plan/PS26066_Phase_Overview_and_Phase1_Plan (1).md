# PS26066 (OceanEmbed) — Development Plan: Overview + Phase 1 Detailed Spec
*Written for direct handoff to Antigravity. Phase 1 below is fully self-contained and implementation-ready.*

---

# PART A — Overview of All Phases

| Phase | Goal | Compute | Key deliverable |
|---|---|---|---|
| **1** | Environment setup + full data ingestion/harmonization pipeline | Laptop only | A clean, validated, 25-channel daily data cube on the common grid |
| **2** | Feature engineering: physics-derived channels, climatology, auxiliary targets, region masks | Laptop only (may be slow for full multi-year range — chunked/Dask processing recommended, GCP optional if too slow) | Full training-ready dataset: inputs + anomaly targets + auxiliary targets |
| **3** | Model implementation + toy-scale correctness verification | Laptop only for code/debugging. **Begin GCP VM setup here, in preparation, not yet in active use** | Verified forward/backward pass on dummy data; GCP environment ready to go |
| **4** | Real training on GCP | **GCP required and active** (L4 instance, per our compute plan) | Trained model checkpoint(s), training logs, first real validation numbers |
| **5** | Full evaluation suite: RMSE/corr/bias/skill-score/SSIM/spectral/heat-flux-consistency/calibration, sliced by all 7 priority zones | GCP (using trained checkpoint) or laptop (lighter, inference-only) | Complete, honest evaluation report |
| **6** | Downstream product layer (OHC/TCHP/MLD/marine-heatwave) + demo/dashboard | Laptop (inference-only workloads are light) | Working demo showing the disaster-management output, not just raw temperature grids |
| **7** | Integration testing, documentation, submission packaging | Laptop | Final, presentable, fully working system |

**This document covers Phase 1 in full detail.** Each subsequent phase will get its own equally detailed .md file once the prior phase is verified complete — building phases incrementally like this means every stage is checked working before the next one depends on it.

---

# PART B — Phase 1: Environment Setup & Data Ingestion Pipeline

## 1. Objective
Produce a single, clean, validated, chronologically-organized data cube containing all 25 input channels (7 SIH-mandated core + 18 derived/external) plus static channels, regridded to a common 0.25° daily grid over 2°N–30°N, 45°E–105°E, with explicit missingness tracking and land/ocean masking — ready to feed Phase 2's feature engineering. **No model code, no feature derivation (geostrophic currents, climatology, etc.) happens in this phase** — Phase 1 is purely: get every raw dataset, get it onto one common grid, validate it's correct.

## 2. Repository Structure
Antigravity should scaffold exactly this layout:

```
oceanembed/
├── .env.example                  # placeholder credentials file (never commit the real .env)
├── requirements.txt
├── README.md
├── configs/
│   ├── domain_config.yaml        # grid bounds, resolution, depth levels
│   └── data_sources_config.yaml  # per-source metadata: URLs, expected variable names, native resolution
├── data/
│   ├── raw/                      # untouched downloads, one subfolder per source
│   │   ├── glorys/
│   │   ├── argo/
│   │   ├── sst_sss_ssh_currents_winds/
│   │   ├── river_discharge/
│   │   ├── precipitation/
│   │   ├── heat_flux/
│   │   ├── wind_curl/
│   │   ├── chlorophyll/
│   │   ├── bathymetry/
│   │   ├── landmask/
│   │   ├── climate_indices/
│   │   └── ibtracs/
│   └── processed/                # final harmonized output (Zarr store, see §7)
├── src/
│   ├── data/
│   │   ├── download/             # one script per source, see §5
│   │   ├── harmonize/            # regridding, masking, missingness, cube assembly, see §6-7
│   │   └── validation/           # automated sanity checks, see §8
│   └── utils/
│       ├── grid.py               # single source of truth for target grid generation
│       ├── io_utils.py           # NetCDF/Zarr read-write helpers
│       └── logging_config.py
├── scripts/
│   ├── run_toy_mode.py           # fast small-scale end-to-end test, see §9
│   └── run_phase1_pipeline.py    # full orchestration entry point
└── tests/
    ├── test_grid.py
    ├── test_regrid.py
    ├── test_landmask.py
    └── test_datacube_shapes.py
```

## 3. Environment Setup
- Python 3.11, managed via `venv` or `conda`.
- Core dependencies for `requirements.txt`:
  - `xarray`, `netCDF4`, `numpy`, `scipy`, `pandas`, `dask` — core data handling and lazy/chunked processing for multi-year ranges.
  - `xesmf` (preferred) or `pyresample` — regridding between native and target grids. If `xesmf` proves hard to install (it has an ESMF binary dependency), fall back to `pyresample`.
  - `copernicusmarine` — official CMEMS Python client (for any remaining GLORYS/core-variable downloads).
  - `earthaccess` — official NASA Earthdata Python client (for IMERG, Ocean Color).
  - `argopy` — for direct ARGO access if needed beyond the INCOIS LAS product already downloaded.
  - `rioxarray`, `rasterio` — for GEBCO bathymetry and Natural Earth shapefile handling.
  - `cartopy`, `matplotlib` — for the sanity-check plots in §8.
  - `pytest` — for the test suite.
- `.env.example` should list placeholder entries for: `CMEMS_USERNAME`, `CMEMS_PASSWORD`, `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`, `MOSDAC_USERNAME`, `MOSDAC_PASSWORD`. **No script should ever hardcode credentials — always read from environment variables loaded via `.env`.**

## 4. Domain Configuration (`configs/domain_config.yaml`)
This is the single source of truth every other module reads from — define it once, reference everywhere, never hardcode grid values in individual scripts:

```yaml
domain:
  lat_min: 2.0        # extended south of nominal 5N for equatorial-edge padding fix
  lat_max: 30.0
  lon_min: 45.0
  lon_max: 105.0
  resolution: 0.25
  crop_lat_min_final: 5.0   # crop back to this for final delivery, post-Phase-1

depth_levels_m: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

temporal:
  frequency: daily
  # actual start/end dates set per-run via CLI args, not hardcoded here

region_boundaries:
  # Approximate polygon/threshold definitions for soft region-membership maps (used in Phase 2,
  # but the boundary logic should be defined here in Phase 1 alongside the rest of domain config)
  arabian_sea_lon_max: 77.0       # west of this, north of confluence zone
  bay_of_bengal_lon_min: 80.0     # east of this, north of confluence zone
  confluence_zone_lat_range: [8.0, 10.0]
  blend_distance_degrees: 2.0     # soft-edge blending width, not hard binary cutoff
```

`src/utils/grid.py` should expose a single function, e.g. `get_target_grid(config) -> (lat_array, lon_array)`, that every download/regrid script imports and uses — this guarantees every channel ends up on the *exact* same grid with no off-by-one or floating-point drift between modules.

## 5. Download Scripts (`src/data/download/`)

**First: a cataloging step for what's already downloaded.** Since the SIH-mandated core datasets (SST, SSS, SSH, currents, winds, GLORYS, ARGO) are already downloaded, write `catalog_existing_core_data.py` first — it scans the existing downloaded files, records their date coverage, native resolution, and variable names into a manifest file (`data/raw/core_data_manifest.json`), and flags any gaps in date coverage. This avoids blindly re-downloading and gives Phase 1 a clear picture of what's already in hand versus what Phase 1 still needs to fetch (the 11 external datasets).

**Then, one script per external source**, each following the same pattern:
- Reads credentials from `.env` where needed.
- Downloads to its designated `data/raw/<source>/` folder in native format/resolution — **do not regrid at download time**, keep raw downloads untouched for reproducibility/debugging.
- **Idempotent**: checks if the target file already exists before re-downloading.
- Logs exactly what was downloaded (date range, file size, source URL) to a per-source log file.
- On failure, logs the failure clearly and continues to the next file/date rather than crashing the whole run — a single missing day of one dataset shouldn't halt the entire pipeline.

Scripts to implement, each named for its source, with exact download instructions per the previously-provided `PS26066_External_Dataset_Download_Guide.md` (reference that file directly for URLs/steps — don't re-derive them):
- `download_river_discharge.py` (ESA CCI primary, GRDC/India-WRIS as configurable alternate sources)
- `download_precipitation.py` (IMERG via `earthaccess`, or MOSDAC MT-SAPHIR)
- `download_heat_flux.py` (OAFlux, direct FTP — no credentials needed)
- `download_wind_curl.py` (MOSDAC EOS-06 scatterometer L4AW product)
- `download_chlorophyll.py` (NASA Ocean Color via `earthaccess`, or MOSDAC OCM-3)
- `download_bathymetry.py` (GEBCO subset tool — this one requires an email-triggered download link, so this script should poll/wait for the email-delivered link or accept a manually-provided URL as a fallback)
- `download_landmask.py` (Natural Earth, direct download, no credentials)
- `download_climate_indices.py` (ONI table scrape/parse from NOAA CPC page, IOD/DMI from NOAA PSL — both are plain text/HTML, parse directly into a simple CSV of `date, ONI, DMI`)
- `download_ibtracs.py` (NOAA NCEI, North Indian basin CSV specifically)

## 6. Harmonization Module (`src/data/harmonize/`)

**`regrid.py`** — one general-purpose function `regrid_to_target(source_dataset, target_lat, target_lon, method="bilinear") -> regridded_dataset`, used by every source. Bilinear is the default (consistent with what most reviewed repos used successfully); leave `method` as a parameter so conservative regridding can be swapped in later without rewriting call sites, per the recognized limitation that bilinear can slightly misrepresent flux-conservation for some variables.

**`temporal_align.py`** — resamples every source to true daily resolution on the domain's date range. Sources natively coarser than daily (e.g. monthly climatology inputs, if any slip through) should be forward-filled or interpolated with the method explicitly logged, never silently.

**`missingness.py`** — for every dynamic channel, produces a same-shape binary mask (1 = valid, 0 = missing/cloud-obscured) **before** any gap-filling is applied elsewhere in the pipeline, and preserves this mask as its own channel per the architecture spec (channel #19) rather than letting missingness silently disappear into interpolated values.

**`landmask.py`** — derives the binary land/ocean mask from the regridded GEBCO bathymetry (elevation > 0 → land), and separately outputs log-scaled bathymetry as its own static channel. Applies masking consistently: land cells get a distinct, consistent fill value (not zero) across every channel, so the model never confuses "land" with "a real physical zero value."

**`region_masks.py`** — builds the four soft region-membership channels (Arabian Sea, Bay of Bengal, confluence zone, open-ocean/equatorial-edge) using the boundary parameters defined in `domain_config.yaml`, with smooth distance-based blending across the `blend_distance_degrees` transition width rather than hard binary edges — implement this as a straightforward distance-to-boundary calculation, normalized and smoothed (e.g. via a sigmoid or linear ramp over the blend distance), not a machine-learned classifier at this stage.

## 7. Data Cube Assembly (`src/data/harmonize/build_datacube.py`)
Orchestrates the full pipeline per day: load each source's regridded output, stack into the 25-channel array (per the exact channel ordering defined in the architecture spec §1.2–1.4), attach static channels, and write out.

**Output format: Zarr, not per-day NetCDF files.** A single chunked Zarr store (chunked by time, e.g. monthly chunks) is dramatically more efficient for the random-access, multi-year training reads Phase 2/3/4 will need, versus thousands of individual daily NetCDF files. Store at `data/processed/oceanembed_datacube.zarr`, with dimensions `(time, channel, lat, lon)` and channel names stored as coordinate metadata (not just integer indices) so every downstream script can reference channels by name, not position.

## 8. Automated Validation (`src/data/validation/sanity_checks.py`)
Run automatically at the end of every pipeline execution, not as an optional afterthought:
- **Shape check**: output matches `(T, 25, H, W)` per the domain config's expected grid dimensions.
- **No unexpected NaNs**: every non-land, non-explicitly-missing cell has a value; any unexpected NaN fails the check loudly.
- **Physical plausibility ranges per channel** — flag (don't silently clip) values outside reasonable bounds, e.g.:
  - SST: −2°C to 35°C
  - SSS: 25–40 psu
  - Bathymetry: sensible ocean-depth range for this domain
  - Precipitation, heat flux: non-negative where physically required
- **Land mask consistency**: land cells are masked identically across all channels (catches partial-masking bugs).
- **Spot-check visualization**: automatically generate and save a PNG map of SST and one derived channel for a randomly chosen day, for quick human visual sanity-checking — this is cheap to produce and catches obviously-wrong regridding (flipped latitude, shifted longitude, etc.) far faster than reading numbers.

## 9. Toy Mode (`scripts/run_toy_mode.py`)
A stripped-down run: a small 10°×10° sub-crop (e.g. just the Bay of Bengal core region), 7 days of data, all 25 channels, completing in well under a minute on the laptop. **This should be the first thing Antigravity runs and gets fully working end-to-end before attempting the full historical range** — it validates the entire pipeline's logic cheaply, so any bugs get caught fast rather than after a slow full-scale run fails partway through.

## 10. Tests (`tests/`)
- `test_grid.py`: confirms `get_target_grid()` produces the exact expected lat/lon arrays for the configured domain/resolution.
- `test_regrid.py`: regrids a known synthetic pattern (e.g. a simple sinusoidal field) onto the target grid and checks the result is reasonable (no wraparound, no flipped axes, correct magnitude).
- `test_landmask.py`: checks the derived land mask against a few manually-known points (e.g. a point well inland in India should be land; a point in the open Arabian Sea should be ocean).
- `test_datacube_shapes.py`: runs the toy pipeline and asserts the final Zarr output has the exact expected shape, channel names, and dtype.

## 11. Acceptance Criteria — Phase 1 is "done" when:
1. `python scripts/run_toy_mode.py` completes with no errors and produces a valid small Zarr cube passing every check in §8.
2. All download scripts run idempotently against real credentials (once registrations from the dataset guide come through) and populate `data/raw/` correctly, with gaps/failures clearly logged rather than silently skipped.
3. `python scripts/run_phase1_pipeline.py` run against at least one real full month of real data produces a correct, validated data cube for that month.
4. All tests in `tests/` pass.
5. `README.md` documents: how to set credentials in `.env`, how to run toy mode, how to run the full pipeline, and what each output file/directory contains.

## 12. What NOT to build yet (explicitly out of scope for Phase 1)
To keep this phase cleanly scoped and avoid Antigravity drifting into Phase 2/3 work prematurely:
- No geostrophic/ageostrophic current derivation, wind stress curl computation (beyond what's directly sourced from ISRO's product), E−P field computation, or river-plume influence field — these are Phase 2.
- No climatology fitting, no anomaly target computation — Phase 2.
- No auxiliary target derivation (MLD, BLT, salinity-max depth) — Phase 2.
- No model code of any kind — Phase 3.
- No GCP setup or usage — not needed until Phase 3 (prep) / Phase 4 (active use).

---

**Next step once Phase 1 is verified complete:** I'll write the equally detailed Phase 2 spec (feature engineering, climatology, auxiliary targets), which builds directly on the Zarr data cube this phase produces.
