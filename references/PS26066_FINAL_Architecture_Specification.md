# PS26066 (OceanEmbed) — FINAL ARCHITECTURE SPECIFICATION
*Implementation-ready. This document is scoped to architecture only — datasets and phased development planning follow as separate documents.*

---

## 0. Design Philosophy

This architecture is a deliberate synthesis, not an invention from scratch. Every major component traces to a specific, evidenced source:

| Component | Source |
|---|---|
| Depth-aware conditional diffusion (DDPM) as the core generative mechanism | Asefi et al. 2026 (real, current, peer-reviewed-track research) |
| Log-normalized continuous depth conditioning | Same paper — proven fix for a real boundary-learning bug |
| Climatology-anomaly prediction target, FiLM-style conditioning | sg721642 (real, trained) + TS-Cast (peer-reviewed, July 2026) |
| ConvLSTM temporal fusion over a multi-day window | sayan21m's real head-to-head evidence (~3× error reduction vs. single-snapshot) |
| Geostrophic/ageostrophic current decomposition, physics-derived features | Published literature + sg721642 |
| Depth-weighted loss targeting the thermocline core | AtulChaudharyy |
| Auxiliary physical prediction heads (MLD, barrier layer, salinity maximum) | sg721642's MLD/BLT heads, extended to our own Arabian-Sea-specific addition |
| Learnable adaptive loss weighting | Pinn-Ocean (Kendall-style homoscedastic weighting) |
| SSIM + Fourier spectral evaluation, heat-flux physical-consistency diagnostic | Asefi et al. 2026 |
| Downstream TCHP/MLD formulas | chilli-garlic-momo (verified textbook-correct) |
| **Depth-cascade autoregressive conditioning across the vertical dimension** | **Our own original contribution — not present in any reviewed source, including Asefi et al., whose depths are sampled independently of each other** |
| **Region-and-physics-tuned loss weighting (extra weight on 200–300m specifically in Arabian Sea pixels, for the Persian Gulf Water zone)** | **Our own original contribution** |

**One honest note before the detail below:** this is a genuinely ambitious system — diffusion models are more expensive to train and slower at inference than a plain regression network. Two things make it tractable rather than reckless: our spatial grid is small (roughly 100×240 cells, smaller than a single standard image), and we use DDIM fast sampling (detailed in §6) rather than full 1000-step sampling, which is what makes interactive inference realistic at all. Also worth noting: unlike Asefi et al.'s setting (99.9%-sparse raw satellite tracks), our PS-mandated inputs are already-gridded daily products — meaning our surface data is *dense*, not sparse. The ill-posedness we're solving for is specifically the surface-to-*depth* mapping, not incomplete surface coverage — if anything, this makes our version of the problem more tractable than the one the reference paper solved.

---

## 1. Full Input Specification

### 1.1 Spatial domain and grid
Ingest at **2°N–30°N, 45°E–105°E** (112 × 240 cells at 0.25°) — deliberately extended south of the PS's nominal 5°N boundary to give the model valid spatial context at the equatorial edge and avoid convolutional padding artifacts. **Crop final output back to the required 5°N–30°N (100 × 240) before evaluation/delivery.**

### 1.2 Dynamic (daily, gridded) input channels — 19 channels

| # | Channel | Type |
|---|---|---|
| 1 | SST | Required (PS) |
| 2 | SSS | Required (PS) |
| 3 | SSH / SLA | Required (PS) |
| 4 | Wind U | Required (PS) |
| 5 | Wind V | Required (PS) |
| 6 | Surface current U (observed) | Required (PS) |
| 7 | Surface current V (observed) | Required (PS) |
| 8 | Geostrophic current U | Derived (thermal-wind relation, from SSH) |
| 9 | Geostrophic current V | Derived |
| 10 | Ageostrophic current U | Derived (= observed − geostrophic) |
| 11 | Ageostrophic current V | Derived |
| 12 | Wind stress curl / Ekman pumping | Derived (or sourced directly from ISRO EOS-06 scatterometer L4AW product, preferred over self-derivation when available) |
| 13 | Wind-mixing energy | Derived (∝ wind speed³) |
| 14 | Precipitation | External |
| 15 | Latent heat flux | External (drives both Arabian Sea winter convection and E−P) |
| 16 | E − P (moisture flux) | Derived (from 14 + 15) |
| 17 | Chlorophyll-a | External |
| 18 | River-plume influence field | Derived (distance-weighted decay from GBM river mouth, scaled by real discharge magnitude — a spatial field, not a uniform broadcast scalar) |
| 19 | Missingness mask | Derived (1 = any core channel missing/cloud-obscured at this cell-day) |

### 1.3 Static (non-time-varying) channels — 2 channels
20. Bathymetry (log-scaled depth-to-seafloor, GEBCO)
21. Land/ocean mask (derived from bathymetry)

### 1.4 Region-membership channels — 4 soft spatial channels (static)
Rather than a single global "region embedding" scalar (which cannot represent a full grid where different cells belong to different regions simultaneously — a real limitation in at least one reviewed repo), region conditioning is encoded as **four soft, distance-blended membership maps**, each valued 0–1 with smooth transitions (no hard binary edges) across:
22. Arabian Sea membership
23. Bay of Bengal membership
24. 8–10°N confluence-zone membership
25. Open-ocean/equatorial-edge membership

### 1.5 Basin-wide scalar conditioning (not spatial — injected via FiLM/AdaGN)
- ONI index (El Niño/Southern Oscillation), current month
- IOD/DMI index (Indian Ocean Dipole), current month
- Day-of-year, cyclically encoded (sin/cos pair)

### 1.6 Per-query conditioning (used at the depth-sampling step, not part of the spatial input stack)
- **Log-normalized continuous depth identifier** *d* (per Asefi et al.'s exact fix for the boundary-learning bug — min-max normalization is explicitly rejected)
- **Harmonic climatology value** at (x, y, depth = d, day-of-year) — precomputed offline (2-harmonic seasonal fit per grid cell per depth, per sg721642's proven method)
- **Previously-sampled adjacent shallower depth's clean output** (our depth-cascade addition — omitted only for the first/shallowest depth in the sampling sequence)

**Total dynamic+static spatial channel count: 25.** Input tensor per training sample: `(T=7, C=25, H=112, W=240)` for the 7-day window (static channels broadcast identically across the T dimension), plus the scalar conditioning vector and per-query depth/climatology/previous-depth conditioning applied downstream.

---

## 2. Model Architecture

### 2.1 Stage 1 — Temporal-Spatial Context Encoder
Purpose: compress the 7-day, 25-channel input window into a single fused spatial context tensor, replacing Asefi et al.'s single-day (sparse) conditioning input with a temporally-aware dense one.

- A ConvLSTM stack processes the `(T=7, C=25, H=112, W=240)` sequence: 3 ConvLSTM layers, channel widths 32→64→64, 3×3 kernels, producing a final hidden state of shape `(64, 112, 240)` summarizing the week's evolution.
- This replaces sg721642's simpler "7-day window as extra channels" approach with a genuinely temporal architecture, consistent with the real evidence (sayan21m) that temporal processing, not just temporal *input*, is what drives the accuracy gain.
- Output: **context tensor** `u_cond` of shape `(64, 112, 240)` — this plays the same role as Asefi et al.'s `u_partial` conditioning input, but dense and temporally-informed rather than a single sparse snapshot.

### 2.2 Stage 2 — Climatology Module (offline, precomputed, not part of the trainable network)
- For every grid cell, every one of the 15 canonical depths, fit a 2-harmonic (annual + semi-annual) seasonal cycle from the training-period GLORYS data: `Clim(x, y, d, doy) = a0 + a1·cos(2π·doy/365) + b1·sin(2π·doy/365) + a2·cos(4π·doy/365) + b2·sin(4π·doy/365)`.
- Training target at every step is the **anomaly**: `y = T_true(x, y, d, t) − Clim(x, y, d, doy(t))`.
- Final delivered temperature: `T_pred = Clim(x, y, d, doy(t)) + anomaly_pred`.

### 2.3 Stage 3 — Depth-Aware Conditional Diffusion Denoiser
The core generative model, following Asefi et al.'s DDPM design, extended with FiLM/AdaGN conditioning and our depth-cascade addition.

**Backbone:** U-Net, 4 resolution stages (deeper than Asefi et al.'s 2-stage baseline, justified by our larger grid): encoder channel widths 32→64→128→256, each stage two 3×3 conv blocks + GroupNorm + SiLU activation, 2×2 downsampling; symmetric decoder with skip connections; bottleneck at 256 channels.

**Conditioning mechanism (AdaGN/FiLM, injected at every U-Net resolution stage, not just concatenated once):**
- `u_cond` (the 64-channel context tensor from Stage 1), spatially concatenated with the region-membership maps (4 channels) and static bathymetry/land-mask (2 channels) — **70 channels total spatially concatenated as the primary conditioning input.**
- Log-normalized depth identifier *d*, climatology value at *d*, ONI, IOD, day-of-year (sin/cos), and — when applicable — the previously-sampled shallower depth's clean anomaly value: all projected through a small MLP into a conditioning vector, injected via **Adaptive Group Normalization (AdaGN)** at every U-Net block (scale-and-shift the normalized feature maps), following standard practice in conditional diffusion literature rather than simple concatenation for these non-spatial or per-query variables.

**Output:** predicted noise `ε_θ(x_τ, τ, u_cond)` at each diffusion timestep τ, per the standard DDPM formulation.

### 2.4 Stage 4 — Depth-Cascade Autoregressive Sampling (our original contribution)
Asefi et al. sample each depth **independently**, conditioned only on the depth identifier. We extend this: depths are sampled **sequentially, shallow to deep**, and each depth's sampling is additionally conditioned on the **already-generated (clean) anomaly field of the immediately shallower depth**. This directly targets the universally-confirmed thermocline-core weakness (§ established across six independent sources in our prior research) by giving the model access to its own emerging vertical profile as evidence, not just the surface state, exactly where the surface-to-subsurface coupling is weakest.

Sampling order: 0 → 5 → 10 → 20 → 30 → 50 → 75 → 100 → 125 → 150 → 200 → 300 → 500 → 700 → 1000m.

### 2.5 Stage 5 — Auxiliary Prediction Heads
Attached to the Stage 1 context tensor (predicted once per sample, not per depth-query, since these are properties of the whole profile/day, not of a single depth):
- Mixed layer depth (MLD) — de Boyer Montégut definition, supervised against training-derived proxy
- Bay-of-Bengal barrier layer thickness (BLT) — supervised only where Bay-of-Bengal region-membership > 0.5
- **Arabian Sea salinity-maximum depth and strength (our addition, targeting the Persian Gulf Water intrusion)** — supervised only where Arabian Sea region-membership > 0.5, derived from training-period GLORYS salinity profiles the same way the BLT proxy is derived

Each head: 2-layer MLP on top of a global-average-pooled version of the context tensor, per-region-masked loss.

---

## 3. Loss Function

Total training loss, combined via **learnable homoscedastic weighting** (Pinn-Ocean's technique — each term's weight is itself a trained parameter, avoiding manual tuning):

```
L_total = exp(−w1)·L_diffusion + w1
        + exp(−w2)·L_aux + w2
        + exp(−w3)·L_physics + w3
```

**L_diffusion:** standard DDPM noise-prediction MSE, `E[||ε − ε_θ(x_τ, τ, u_cond)||²]`, but **multiplied by a depth-and-region-dependent weight** `α(d, region)`:
- Baseline weight 1.0 at all depths.
- 1.5× at 50–200m (thermocline core, all regions — per AtulChaudharyy's proven approach).
- Additional 1.3× multiplier specifically at 200–300m **within Arabian Sea region-membership** (targeting the Persian Gulf Water intrusion zone — our addition, informed by the real published finding that this is a distinctive, dynamically mobile feature).

**L_aux:** sum of MSE losses for the MLD, BLT, and salinity-maximum-depth/strength auxiliary heads, each masked to its relevant region where applicable.

**L_physics:** thermocline-depth-consistency loss (repo1's technique), applied to the denoiser's **predicted x̂₀ (clean-data estimate) at each training step**, not to a fully multi-step-sampled output — this is the standard, tractable way to apply a physics-consistency loss during diffusion training, since full iterative sampling isn't necessary (or straightforwardly differentiable) at every training step. Compares the depth of the temperature gradient's inflection point between predicted x̂₀ and ground truth.

**Note on the heat-flux (meridional transport) diagnostic:** following Asefi et al.'s own usage, this is treated as an **evaluation-time diagnostic, not a training loss** — computing it properly requires a full multi-depth sampling pass to get paired T and V fields, which is expensive to do every training step. It's used to check physical consistency of final outputs, described fully in §5.

---

## 4. Normalization

- All physical input channels: min-max normalized per-channel using training-set statistics (not global assumptions).
- **Depth identifier: log-normalized, explicitly not min-max** — per Asefi et al.'s documented finding that min-max normalization of depth caused the shallowest layer to fail to learn properly.
- Climatology-anomaly target: standardized (zero mean, unit variance) using training-set anomaly statistics, separately per depth level (anomaly variance differs substantially by depth).
- Region-membership maps: naturally bounded [0,1], no further normalization needed.

---

## 5. Inference / Sampling Procedure

1. Run Stage 1 (ConvLSTM context encoder) once per prediction request on the most recent 7-day input window → `u_cond`.
2. For each of the 15 canonical depths, in order (shallow to deep):
   a. Assemble conditioning: `u_cond` + region/static channels (spatial) + depth id, climatology, ONI/IOD/day-of-year, previous-depth output (non-spatial, via AdaGN).
   b. Sample using **DDIM (Denoising Diffusion Implicit Models)** with ~50 steps rather than full 1000-step ancestral sampling — necessary for realistic inference latency in an interactive demo, with negligible quality loss relative to full sampling per standard DDIM results.
   c. Store the clean anomaly output; add climatology to get the absolute temperature; this becomes the "previous depth" conditioning input for the next depth in the sequence.
3. **For uncertainty quantification:** repeat the entire 15-depth sampling sequence **N=10–20 times** per prediction (independent random noise seeds). The resulting ensemble of full vertical profiles gives a natural, non-parametric uncertainty distribution at every depth — mean and standard deviation computed directly from the ensemble, not from a separately-predicted variance term. This is the diffusion-native, more principled version of the heteroscedastic-uncertainty approach several of the 17 repos used.

---

## 6. Evaluation Protocol

Beyond RMSE/correlation/bias (the PS's own stated minimum), and beyond the R²-vs-baseline discipline established in our competitive research:

- **SSIM** (structural similarity) per depth — catches structural mismatch that pointwise RMSE misses.
- **Fourier spectral comparison** per depth — catches oversmoothing (a real, documented failure mode of deterministic baselines in the reference paper).
- **Meridional heat-flux consistency** (`q_v = ρ·c_p·V·T`, computed from a full sampled profile including velocity components if we extend scope to currents) — a coupled physical-plausibility check, not just per-variable accuracy.
- **Uncertainty calibration** — reliability diagrams checking whether the ensemble-derived confidence intervals actually contain the true value at the stated rate, against real held-out ARGO data.
- All of the above, sliced across the **seven priority zones** established in prior planning: Bay of Bengal barrier layer (0–30m), thermocline core (75–150m, both basins), Arabian Sea Persian-Gulf-Water zone (200–300m), the 8–10°N confluence zone, cyclone/extreme-event windows (IBTrACS-identified), monsoon transition windows, and the equatorial domain edge.
- Chronological train/val/test split (never random), and — pending the RAMA-buoy assimilation-status investigation — an explicit statement of how independent our final validation actually is.

---

## 7. Downstream Disaster Product Layer

Computed from the full ensemble of sampled profiles (so every downstream number carries an uncertainty band, not just a point value):
- **Ocean Heat Content**, integrated from the sampled profile.
- **Tropical Cyclone Heat Potential**, using chilli-garlic-momo's verified textbook-correct formula (26°C isotherm depth via interpolation, density/specific-heat-weighted integration, kJ/cm²).
- **Mixed Layer Depth**, via the standard de Boyer Montégut method, cross-checked against the dedicated MLD auxiliary head from Stage 5.
- **Marine heatwave flag**, comparing the reconstructed (not just surface) heat content anomaly against a threshold — the actual point of this whole system being genuinely useful for the Disaster Management theme, not just an ML benchmark.

Every one of these is reported as **mean ± uncertainty**, derived directly from the N-sample ensemble spread — e.g. "TCHP = 62 ± 8 kJ/cm², confidence reflecting local ARGO data density," not a bare number.

---

## What's next
With architecture locked, next we finalize the complete dataset list — cross-referencing exactly which of these 25 channels map to which real, named data products (CMEMS, MOSDAC, OAFlux, GRDC/India-WRIS, GEBCO, IBTrACS, etc.), their access method, and update cadence — before moving to the phase-wise development plan for Antigravity.
