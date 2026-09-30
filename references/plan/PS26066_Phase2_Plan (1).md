# PS26066 (OceanEmbed) — Phase 2 Detailed Spec: Feature Engineering, Climatology & Auxiliary Targets
*Written for direct handoff to Antigravity. Builds directly on Phase 1's validated Zarr data cube — do not start this phase until Phase 1's acceptance criteria are met.*

---

## 1. Objective
Phase 1 produced a harmonized cube with 17 of the architecture's 25 input channels (the 7 core + 3 external-regridded + missingness + 2 static + 4 region-membership), plus the raw river discharge, climate index, and IBTrACS data sitting alongside it, not yet integrated. Phase 2 has four jobs:

1. **Compute the 8 remaining physics-derived dynamic channels** to complete the full 25-channel input stack.
2. **Fit the harmonic seasonal climatology** per grid cell, per depth, from the *training-period-only* GLORYS data.
3. **Compute the anomaly training target** (what the model actually learns to predict) at all 15 depths.
4. **Derive the three auxiliary targets** (MLD, Bay-of-Bengal barrier layer thickness, Arabian Sea salinity-maximum depth/strength) that feed the architecture's Stage 5 auxiliary heads.

Output: a single, fully assembled, training-ready dataset that Phase 3's model code can load directly — no further feature engineering should be needed once this phase is done.

**Compute note:** this phase is still laptop-appropriate — everything here is array arithmetic and per-cell curve fitting, not model training. Full multi-year climatology fitting across the whole grid can be slow if done naively; use `dask` chunking (already a Phase 1 dependency) to parallelize across grid cells rather than looping in pure Python. If it's still too slow after chunking, a larger *CPU-only* cloud instance is a reasonable stopgap — this doesn't require GCP's GPU tier at all, so don't reach for the L4 setup we planned for Phase 4 here.

---

## 2. Repository Structure Additions
```
src/
├── features/
│   ├── geostrophic.py           # geostrophic U/V from SSH (thermal-wind/geostrophic balance)
│   ├── ageostrophic.py          # observed current − geostrophic current
│   ├── wind_derived.py          # wind stress curl (fallback derivation) + wind-mixing energy
│   ├── moisture_flux.py         # E−P field from precipitation + latent heat flux
│   └── river_plume.py           # distance-decay plume influence field from discharge
├── climatology/
│   ├── fit_climatology.py       # 2-harmonic seasonal fit per cell/depth, TRAINING YEARS ONLY
│   └── compute_anomaly.py       # anomaly target = GLORYS truth − climatology
├── auxiliary_targets/
│   ├── mixed_layer_depth.py     # de Boyer Montégut MLD
│   ├── barrier_layer_thickness.py  # ILD − MLD(density), Bay of Bengal
│   └── salinity_maximum.py      # depth + strength of subsurface salinity max, Arabian Sea
└── assemble/
    └── build_training_dataset.py  # final orchestration → Phase 3-ready output

scripts/
├── run_phase2_pipeline.py
└── run_phase2_toy_mode.py

tests/
├── test_geostrophic.py
├── test_climatology_fit.py
├── test_auxiliary_targets.py
└── test_training_dataset_shapes.py
```

---

## 3. Physics-Derived Dynamic Channels — Exact Formulas

### 3.1 Geostrophic currents (`geostrophic.py`) — channels 8–9
Standard geostrophic balance from sea surface height:
```
u_g = -(g / f) * ∂η/∂y
v_g =  (g / f) * ∂η/∂x
```
where `η` = SSH/SLA, `g` = 9.81 m/s², `f` = Coriolis parameter = `2·Ω·sin(latitude)`, `Ω` = 7.2921×10⁻⁵ rad/s. Compute `∂η/∂x` and `∂η/∂y` via centered finite differences on the grid.

**Critical implementation detail:** `f → 0` at the equator, making this formula numerically singular within a few degrees of it — exactly the equatorial-boundary issue flagged in our priority-zone planning. Do not let this produce `inf`/`NaN` values silently. Within roughly 2° of the equator, either (a) damp/blend the geostrophic estimate toward zero smoothly, or (b) mask it explicitly and rely on the open-ocean/equatorial-edge region-membership channel (already built in Phase 1) to flag these cells as low-confidence for this specific feature. Log how many cells were affected so this is visible, not hidden.

### 3.2 Ageostrophic currents (`ageostrophic.py`) — channels 10–11
```
ageo_u = observed_current_u − geostrophic_u
ageo_v = observed_current_v − geostrophic_v
```
Straightforward once 3.1 is done — this captures the wind-driven/Ekman component not explained by the pressure-driven geostrophic flow.

### 3.3 Wind stress curl fallback + wind-mixing energy (`wind_derived.py`) — channels 12–13
**Wind stress curl** — only compute this where Phase 1's ISRO scatterometer product download was incomplete or unavailable for a given date (check the Phase 1 manifest first; prefer the real ISRO-computed value wherever it exists, per our design principle of using ISRO's own value-added product rather than re-deriving it). Fallback formula when needed:
```
τ_x = ρ_air · C_d · |wind| · wind_u
τ_y = ρ_air · C_d · |wind| · wind_v
curl = ∂τ_y/∂x − ∂τ_x/∂y
```
`ρ_air ≈ 1.225 kg/m³`. Use a simple constant drag coefficient `C_d ≈ 1.3×10⁻³` for tractability rather than a wind-speed-dependent formulation — flag this as a known simplification in code comments, not something to over-engineer at this stage.

**Wind-mixing energy** — proportional to turbulent kinetic energy input:
```
wind_mixing_energy = wind_speed³, where wind_speed = sqrt(wind_u² + wind_v²)
```

### 3.4 E−P moisture flux (`moisture_flux.py`) — channel 16
Derive evaporation from latent heat flux:
```
E = latent_heat_flux / (ρ_water · L_v)
```
`ρ_water ≈ 1000 kg/m³`, `L_v ≈ 2.5×10⁶ J/kg` (constant approximation is fine here — don't over-engineer temperature dependence). Convert to the same units as precipitation (e.g. mm/day) before combining:
```
E_minus_P = E − precipitation
```
**Sign convention, worth getting right and documenting clearly in code:** positive E−P = net evaporative loss (Arabian Sea's regime, salinifying); negative E−P = net freshwater gain (Bay of Bengal's regime). This single field should show opposite-sign behavior between the two basins — a good sanity check to run visually (§7) once computed.

### 3.5 River plume influence field (`river_plume.py`) — channel 18
A distance-decay proxy from the Ganges-Brahmaputra-Meghna river mouth, scaled by real discharge magnitude:
```
plume_influence(x, y, t) = discharge(t) · exp(−distance(x, y, mouth_location) / decay_length_scale)
```
Define `mouth_location` as an approximate real coordinate for the main GBM delta outflow (roughly 21.5°N, 89.5°E — verify/refine against the actual river discharge dataset's station coordinates once downloaded, rather than treating this as exact). `decay_length_scale` is a tunable parameter (start around 200–300 km, adjustable).

**Be explicit in code comments that this is a simplified proxy, not a real transport/advection model** — a genuine plume follows coastal currents, doesn't decay radially. If time allows later, a real refinement would bias the decay ellipse along the coastal current direction using the current field we already have — note this as a possible Phase 2.5 enhancement, not a blocker now.

---

## 4. Climatology Fitting (`climatology/fit_climatology.py`)

Per grid cell, per depth level, fit a 2-harmonic seasonal cycle to **GLORYS training-period data only**:
```
Clim(x, y, d, doy) = a0 + a1·cos(2π·doy/365) + b1·sin(2π·doy/365) + a2·cos(4π·doy/365) + b2·sin(4π·doy/365)
```
Solve for `[a0, a1, b1, a2, b2]` via ordinary least squares per (grid cell, depth) pair, using `doy` (day-of-year) and the corresponding true temperature values from training years only.

**This is the single most important correctness requirement in this entire phase, worth its own warning: climatology must be fit exclusively on the designated training years (per our established chronological split, e.g. training years only, never validation/test years).** If validation or test period data leaks into the climatology fit, every downstream anomaly target and every reported accuracy number becomes optimistically biased — this would quietly undermine the entire "honestly validated, unlike prior attempts" story this whole project is built around. Add an explicit assertion in code that checks the date range passed into this function against the configured training-period boundary and raises an error if any out-of-range date is included — don't rely on the caller "remembering" to slice correctly.

Store the fitted coefficients (not just a lookup table of values) — this lets climatology be evaluated at any arbitrary day-of-year continuously, and keeps storage compact (5 coefficients × depth levels × grid cells, versus a full dense day-by-day table).

## 5. Anomaly Target Computation (`climatology/compute_anomaly.py`)
```
anomaly(x, y, d, t) = GLORYS_truth(x, y, d, t) − Clim(x, y, d, doy(t))
```
Computed at all 15 canonical depths, for the full date range (training, validation, and test periods alike — only the *climatology fit itself* is training-years-only; the anomaly target is computed for every period using that fixed, training-derived climatology). Store per-depth, matching the architecture's target structure.

---

## 6. Auxiliary Targets

### 6.1 Mixed Layer Depth (`mixed_layer_depth.py`)
Standard de Boyer Montégut definition: the depth at which temperature differs from the 10m reference temperature by more than `ΔT = 0.2°C`, found via linear interpolation between the two bracketing depth levels where the threshold is crossed (don't just snap to the nearest of the 15 discrete depth levels — interpolate for a smoother, more accurate target).

### 6.2 Bay of Bengal Barrier Layer Thickness (`barrier_layer_thickness.py`)
Requires two intermediate quantities, both from GLORYS temperature *and* salinity profiles:
- **Isothermal Layer Depth (ILD)**: depth where temperature drops by `0.2°C` from the 10m reference — a temperature-only criterion.
- **Mixed Layer Depth (density-based)**: depth where potential density increases by the density-equivalent of a `0.2°C` temperature change at the reference salinity — requires computing density from both temperature and salinity (a standard seawater equation of state; a simplified linear approximation is acceptable here, full TEOS-10 is not necessary for this auxiliary target specifically).
```
Barrier_Layer_Thickness = ILD − MLD_density
```
Only compute/supervise this target where Bay of Bengal region-membership > 0.5 (per Phase 1's region mask) — it's not a meaningful quantity elsewhere and shouldn't be force-fit globally.

### 6.3 Arabian Sea Salinity Maximum (`salinity_maximum.py`)
For each grid cell/day, scan the GLORYS salinity profile across depth within a plausible subsurface range (roughly 100–400m, deliberately excluding the surface layer to avoid spuriously picking up near-surface salinity peaks unrelated to the Persian Gulf Water signature). Record two values as the auxiliary target:
- **Depth** of the salinity maximum within that range.
- **Strength**: the salinity value at that maximum, expressed as an anomaly relative to the profile's surrounding baseline (not the raw absolute value), so the target reflects how pronounced the intrusion signature is, not just ambient salinity level.

Only compute/supervise where Arabian Sea region-membership > 0.5.

---

## 7. Final Assembly (`assemble/build_training_dataset.py`)
Orchestrates everything above into one Phase-3-ready output:
- The full 25-channel input stack (Phase 1's 17 + this phase's 8 derived channels), written to an updated/extended Zarr store.
- Climatology coefficients, stored separately (compact, per cell/depth).
- Anomaly targets at all 15 depths, aligned to the same time index as the input stack.
- The three auxiliary targets, each masked to its relevant region.
- Scalar conditioning table: ONI, IOD/DMI (from Phase 1's downloaded lookup tables), and cyclically-encoded day-of-year (`sin(2π·doy/365)`, `cos(2π·doy/365)`), indexed by date — a simple aligned table, not a spatial field.

**Sanity checks to run automatically at the end of this step (mirroring Phase 1's discipline, don't skip this):**
- Spot-check plot of the E−P field on a single day, confirming visually opposite signs between Arabian Sea and Bay of Bengal (§3.4's expected behavior) — this is a cheap, high-value check that would catch a sign-convention bug immediately.
- Spot-check plot of geostrophic current magnitude, confirming it's damped/masked near the equator rather than showing spurious extreme values.
- Confirm auxiliary targets are non-null only within their designated region masks, null elsewhere.
- Confirm climatology fit coefficients contain no NaNs over ocean cells (land cells expected to be masked/NaN, consistent with Phase 1).

---

## 8. Toy Mode (`scripts/run_phase2_toy_mode.py`)
Same philosophy as Phase 1: run the entire Phase 2 pipeline against Phase 1's toy-mode output (small crop, 7 days) first, confirming every derived feature, the climatology fit, and all three auxiliary targets compute correctly and quickly, before running against the full historical range.

---

## 9. Tests (`tests/`)
- `test_geostrophic.py`: verify geostrophic current computation against a simple synthetic SSH field with a known analytic gradient (e.g. a linear ramp), and explicitly test the equatorial-damping behavior near `f = 0`.
- `test_climatology_fit.py`: fit climatology against a synthetic sinusoidal time series with known coefficients, confirm the fit recovers them accurately — and explicitly test that passing an out-of-training-range date raises the expected error.
- `test_auxiliary_targets.py`: verify MLD/BLT/salinity-max computations against a few manually-constructed synthetic profiles with known correct answers (e.g. a profile with an obvious, hand-designed barrier layer).
- `test_training_dataset_shapes.py`: run toy mode end-to-end, assert final output has the correct channel count (25), correct auxiliary target shapes/masking, and correct scalar conditioning table alignment.

---

## 10. Acceptance Criteria — Phase 2 is "done" when:
1. `python scripts/run_phase2_toy_mode.py` completes with no errors and passes every check in §7.
2. All physics-derived channels compute correctly against real Phase 1 output for at least one real month, with the E−P sign-convention and equatorial-damping sanity checks visually confirmed.
3. Climatology is verifiably fit only on the designated training-period years (confirmed by the explicit assertion in §4, not just "trusted" by convention).
4. All three auxiliary targets are computed, correctly region-masked, and pass their synthetic-profile tests.
5. The final assembled dataset (§7) loads cleanly and has exactly the shape/structure Phase 3's model code expects, per the architecture specification.
6. All tests in `tests/` pass.

## 11. What NOT to Build Yet
- No model code, no ConvLSTM/U-Net/diffusion implementation — Phase 3.
- No GPU usage, no GCP setup or connection — still not needed. This phase is CPU-only work; hold off on touching GCP until Phase 3's preparation step.
- No training loop, no loss functions, no sampling logic — Phase 3/4.

---

**Next step once Phase 2 is verified complete:** Phase 3's spec — model implementation and toy-scale correctness verification, including where and how to begin GCP environment preparation in anticipation of Phase 4.
