# PS26066 — Architecture, Datasets & Methods: The Actual Design

This is the decisions document — what goes in, what happens to it, what comes out, and why. Organized so every choice traces back either to a proven technique (cited from our research) or a specific gap we're deliberately closing.

---

## 1. Final input variable set

### 1a. PS-mandated core (non-negotiable, the PS requires these)
SST, SSS, SSH/SLA, surface currents (U,V), surface winds (U,V) — 7 channels, 0.25°, daily, North Indian Ocean domain.

### 1b. Additional variables we're adding, and why each one is justified, not decorative

| Variable | Why it physically matters | Real source | Priority |
|---|---|---|---|
| **River discharge (Ganges-Brahmaputra-Meghna)** | Peer-reviewed literature (Durand et al. 2011, Akhil et al. 2014) directly ties GBM discharge variability to Bay of Bengal SSS and mixed-layer temperature anomalies, especially north of ~10°N — this is a primary driver of the barrier layer, not a minor factor. 2011 study found the 1998 discharge peak (2× the 1992 low) even penetrated into the southeastern Arabian Sea months later. | GRDC (Global Runoff Data Centre, WMO) — real-time/historical gauge data for GBM stations; MOSDAC also lists a dedicated "River Discharge" product as India's own alternative if GRDC's India/Bangladesh station coverage/latency is limiting (worth checking both) | **Must-have for Bay of Bengal accuracy** |
| **Precipitation** | Monsoon rainfall directly forces the SSS seasonal cycle (same Akhil et al. 2014 study: mixed-layer salt budget shows discharge *and* precipitation both drive the freshening) — independent of river discharge's effect. | GPM/IMERG (NASA), or India's own MT-SAPHIR-based rainfall product on MOSDAC | **Must-have for Bay of Bengal** |
| **Net surface heat flux** (shortwave + longwave + latent + sensible) | This is a real, direct physical gap in the PS's own input list — heat flux is what actually changes near-surface heat content day to day; SST alone shows the *result*, heat flux shows the *forcing*. Without it, the model has to infer forcing indirectly. | OAFlux (WHOI) or ERA5 | Strong-differentiator |
| **Wind stress curl / Ekman pumping** | Drives upwelling — critical for the Arabian Sea's seasonal upwelling signal specifically. | Can derive from wind U/V ourselves, **or** use ISRO's own EOS-06/Oceansat-3 scatterometer product directly — their L4AW "Analysed Winds" product already computes wind divergence and the vertical component of wind stress curl as a value-added output. Using ISRO's own computed product instead of re-deriving it ourselves is both less error-prone and a stronger "built on India's own space assets" narrative. | Must-have (cheap, ISRO does the work for us) |
| **Chlorophyll-a / ocean color** | Correlates with upwelling and stratification, especially in the Arabian Sea; also usable as a cross-check on our physical narrative (upwelling events should show both a temperature and a chlorophyll signature). | MODIS/VIIRS ocean color, or Oceansat-3's own OCM-3 (13-band ocean colour monitor, ISRO's own instrument) | Nice-to-have |
| **Bathymetry (static)** | Continental shelf regions physically cannot support a deep thermocline the way open ocean can; near-coast dynamics differ structurally. Also needed for land-masking regardless. | GEBCO (global bathymetry, free) | Must-have (also solves land-masking, which Pinn-Ocean explicitly dodged by picking a landless region) |
| **Large-scale climate indices (ONI, IOD/DMI)** | Basin-wide conditioning signal — El Niño and Indian Ocean Dipole phases measurably shift Arabian Sea/Bay of Bengal thermal structure on interannual timescales; a single scalar per day is cheap to add as a conditioning input. | Published monthly index values (NOAA/BOM), freely available, trivial to ingest | Nice-to-have, cheap |
| **IBTrACS cyclone track/intensity database** | Not a model input — a **labeling tool**. Lets us identify real historical cyclone windows in the Bay of Bengal/Arabian Sea to build the extreme-event-specific evaluation slice (the gap nobody else has closed) and to validate our downstream TCHP output against real storm behavior. | IBTrACS (NOAA), the standard global best-track archive | Must-have for the extreme-event evaluation piece specifically |

**Net effect:** PS asks for 7 channels; we're proposing roughly 11–13 depending on how aggressively we scope it, with every addition tied to a specific, citable physical mechanism — not "more data because more is better."

---

## 2. Derived features — the "relationships between inputs" layer

This is the physics-engineering step that happens **before** anything reaches the network, converting raw channels into physically meaningful derived quantities:

1. **Geostrophic currents from SSH** (thermal-wind relation) — proven in the literature (physically-guided fusion network, South China Sea) to outperform plain CNNs when added as input.
2. **Ageostrophic current component** (residual after removing geostrophic part from observed surface currents) — captures wind-driven Ekman transport separately from pressure-driven geostrophic flow; sg721642 already does this and it's a legitimate extension worth keeping.
3. **Ekman pumping velocity** from wind stress curl (either self-derived or pulled directly from ISRO's product, see above).
4. **Density/stratification proxy** from SST+SSS via a simplified equation of state (full TEOS-10 if we want to be rigorous, as Pinn-Ocean did) — gives the network an explicit stratification signal rather than making it infer density behavior implicitly from temperature and salinity separately.
5. **Seasonal harmonic climatology per depth/grid-cell** (1st + 2nd harmonic fit, the sg721642/TS-Cast approach) — this becomes the baseline the model predicts a *departure from*, not a raw input channel.
6. **Barrier layer thickness proxy** as an auxiliary target (not just an input) — computed from the climatology/training data itself (difference between isothermal and mixed layer depth), giving the network a Bay-of-Bengal-specific physical quantity to jointly learn, the way sg721642's auxiliary MLD/BLT heads do.

---

## 3. Model architecture — the actual synthesis

Rather than inventing a wholly new architecture, this combines the best individually-proven pieces from our research with the two genuinely open gaps:

**Backbone:** Full-basin (not patch-based) spatial encoder — a U-Net/CNN operating on the complete 100×240 grid (à la AtulChaudharyy and Nishad3107), not a 5×5 local-patch model (KaviBharathi643's approach) — full context matters for basin-scale features like eddies and monsoon current reversals that a small patch can't see.

**Temporal input:** 7-day rolling window of the surface-channel stack (sg721642's approach) rather than a single-day snapshot — gives the model access to recent trend, which matters given the ocean's memory.

**Regional conditioning:** A learned region/regime embedding (Arabian Sea / Bay of Bengal / open ocean / transition zone) injected at the network bottleneck — one model, region-aware, not separate models per basin. **Unlike every repo that added this, we will actually run and report the ablation** (with vs. without) rather than assume it helps.

**Prediction target:** Climatological anomaly, not absolute temperature — reconstructed at inference as `climatology + predicted_anomaly`. Proven approach (sg721642, TS-Cast), directly prevents the "cheat to a near-constant deep value" failure mode we found in KaviBharathi643's real results.

**Depth handling — our first genuinely original piece:** Rather than a fixed 15-channel output head (most repos) or an independent continuous-depth query per level (SatyaSailesh's Fourier decoder — elegant, but still predicts each depth independently), we use **depth-cascade conditioning**: predict shallow depths first, then feed each depth's predicted mean *and* uncertainty forward as additional context for the next depth down. This directly targets the thermocline-core failure mode confirmed five times over in our research — the model gets to use its own emerging vertical profile as evidence, not just the surface state, once you're past the point where surface signal alone is informative.

**Uncertainty:** Heteroscedastic — predict mean and log-variance in one pass (proven technique, several repos do this) — but **calibrated**, meaning we don't stop at predicting a variance, we check it against real held-out data with reliability diagrams. This is the second genuinely open gap nobody has closed.

**Loss function, combined from proven pieces:**
- Data-fidelity loss (Gaussian NLL, using the heteroscedastic output) on the anomaly prediction.
- Depth-weighted upweighting of the 50–200m band (AtulChaudharyy's proven approach, directly targeting the confirmed weak zone).
- Thermocline-consistency physics loss (repo1's approach — penalize mismatch between predicted and reference thermocline depth).
- Auxiliary losses for MLD/barrier-layer-thickness prediction (sg721642's approach), which forces the shared representation to encode physically meaningful structure, not just fit temperature directly.
- Learnable homoscedastic weighting between these loss terms (Pinn-Ocean's adaptive_loss technique) instead of manually tuned weights.

**Ocean masking:** Applied throughout via GEBCO bathymetry, so land cells never contribute to loss or get nonsensical predictions — closes the gap Pinn-Ocean dodged entirely by choosing a landless region.

---

## 4. Downstream output layer

Compute Ocean Heat Content and **Tropical Cyclone Heat Potential using chilli-garlic-momo's exact textbook-correct formula** (26°C isotherm depth via interpolation, density- and specific-heat-weighted integration of temperature excess above 26°C, converted to kJ/cm²) and mixed layer depth via the standard de Boyer Montégut threshold method — but, unlike every repo that attempted a downstream layer, driven by our own validated subsurface reconstruction (not a weather API, AtulChaudharyy's approach) and carrying the uncertainty band through to the final index, so a forecaster sees not just "TCHP = X" but "TCHP = X ± Y, confidence based on how much real ARGO data existed nearby."

---

## 5. Validation plan (the rigor layer nobody else has)

1. Chronological split (train/val/test by year, never randomly shuffled).
2. Test against held-out GLORYS (in-sample-style check).
3. Test against real gridded ARGO/INCOIS data (Nishad3107-style bootstrap confidence intervals per depth).
4. **Explicitly investigate whether an unassimilated Indian Ocean observation source exists** (RAMA moored buoys, pending confirmation of assimilation status) for a TS-Cast-style genuinely independent check — and state plainly in our reporting whether we could or couldn't establish full independence, rather than silently assuming ARGO validation settles the question.
5. Report accuracy sliced by depth × region × season, not a single number.
6. **Extreme-event slice**: use IBTrACS to identify real historical cyclone windows in our test period, and report accuracy specifically during those windows, separately from average-day accuracy.
7. Uncertainty calibration check: do our stated confidence intervals actually contain the true value at the stated rate, on held-out real data?

---

## Open decisions for us to make next

A few things genuinely need your input before this becomes final:
1. **Scope of additional inputs** — do we go for the full 11–13 channel set, or phase it (start with the 7 required + bathymetry + climatology, add river discharge/heat flux/chlorophyll only if time allows)? This affects data pipeline complexity a lot.
2. **Depth-cascade implementation** — sequential per-depth (slower, simplest to reason about) vs. a single pass with an attention mechanism over already-predicted shallower depths (faster, more complex) — worth deciding based on your team's comfort with the more complex version.
3. **How far to push the "independent validation" investigation** — this could be a rabbit hole; worth deciding how much time it's worth before we either find RAMA assimilation status or decide to state the limitation honestly and move on.
