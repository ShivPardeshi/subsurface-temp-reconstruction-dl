# PS26066 — Track A Extension, Remaining Fixes, and Phase 6 Detailed Spec

---

# PART A — Track A: Extended Training (do this in parallel with Phase 6, not before it)

## A1. Extend Stage B AND Stage C together, equally — not Stage B alone
To resolve the region-conditioning question fairly, both models need equal step counts, the same discipline as Fix 1. Resume both from their existing checkpoints:
- **Stage B**: resume `checkpoints/baseline_10k/best_checkpoint.pt` → extend to **20,000–30,000 steps**.
- **Stage C (no-region)**: resume `checkpoints/ablation_no_region/best_checkpoint.pt` (currently at 2,000 steps) → extend to the **exact same step count as Stage B this round**.
- Stage D does not need re-extension for the ablation question specifically (the depth-cascade finding is already large, consistent, and fairly established) — but extending it too is cheap and reasonable if you want a final, best-quality production checkpoint from it as well.

## A2. What to watch for as this runs
- **Global Murphy skill score** — currently −0.175 at 10k steps for Stage B. Watch for it crossing zero.
- **Does the Stage B vs. Stage C gap hold, close, or reverse** once both are trained equally at the higher step count? Report whichever happens honestly — this is a real open scientific question, not a foregone conclusion either way.
- **Calibration ECE** — track it, but per A3 below, don't expect more steps alone to fix it.

## A3. Fix the frozen "Salinity Max Depth" correlation
This value has been exactly **0.3622** across three consecutive reports, spanning three genuinely different trained models. This is not plausible as a coincidence.

**Required action:** Locate the exact function computing this metric (likely in `src/evaluation/auxiliary_head_eval.py`). Confirm it is:
1. Actually loading the current checkpoint's predictions, not a cached result from an earlier run.
2. Actually being re-executed on each evaluation call, not returning a memoized/hardcoded value.
3. Print/log the raw prediction and target arrays used for this specific computation on the next run, so the input to the correlation calculation is directly inspectable, not just the output.

## A4. Fix calibration properly — don't rely on more training steps
ECE has moved only 0.7120 → 0.6923 despite a 5x increase in training steps — it isn't responding to "train longer" the way skill score is.

**Required action:** Implement a dedicated post-hoc calibration step, separate from the main training loop:
1. On the held-out validation set, compare the DDIM ensemble's stated confidence intervals against observed coverage (already computed for the reliability diagram).
2. Apply **temperature scaling** to the ensemble spread — a single learned scalar multiplier on the predicted standard deviation, fit by minimizing calibration error on the validation set (not the training set, to avoid re-overfitting the fix itself).
3. Re-generate the reliability diagram and ECE using the scaled intervals, and report both the before/after ECE so the fix's effect is directly visible.

---

# PART B — Phase 6 Detailed Spec: Downstream Disaster Products & Demo

## 1. Objective
Convert the validated subsurface reconstruction into the actual disaster-management product this PS's theme calls for — Ocean Heat Content, Tropical Cyclone Heat Potential, Mixed Layer Depth, and a marine heatwave flag — each carrying honest uncertainty from the DDIM ensemble, and presented through a working demo. This is the phase that turns "we reconstructed a 3D temperature field" into "we built something a forecaster could use."

**One design decision already made based on Track A's findings, worth stating plainly here:** TCHP and MLD will be computed by direct integration/analysis of the **reconstructed 3D temperature profile itself**, not by depending on the BLT or MLD auxiliary heads (which show weak-to-broken reliability per the last audit). The auxiliary heads' predictions are retained as a secondary cross-check signal only, not the primary computation path.

## 2. Repository Structure Additions
```
src/
├── products/
│   ├── ocean_heat_content.py      # OHC from a sampled profile
│   ├── tchp.py                    # Tropical Cyclone Heat Potential
│   ├── mld_direct.py              # de Boyer Montégut MLD from the reconstructed profile
│   ├── marine_heatwave.py         # Hobday et al. (2016)-style MHW detection
│   └── uncertainty_propagation.py # carries ensemble spread through every derived product
├── api/
│   └── routes_products.py         # serves computed products for a given date/location
└── dashboard/
    ├── app.py                     # demo interface
    └── components/
        ├── profile_viewer.py      # depth-vs-temperature plot with uncertainty band
        ├── heatwave_map.py        # spatial marine-heatwave flag map
        └── tchp_gauge.py          # TCHP value + confidence display

scripts/
└── run_phase6_demo.py

tests/
├── test_tchp_formula.py
├── test_mld_direct.py
└── test_marine_heatwave_detection.py
```

## 3. Ocean Heat Content (`ocean_heat_content.py`)
Standard integration of the reconstructed temperature profile from the surface down to a chosen reference depth (commonly 26°C isotherm depth for TCHP purposes, or a fixed depth like 700m for standard OHC reporting — implement both, clearly labeled):
```
OHC = ρ · c_p · ∫[0 to D] T(z) dz
```
Computed per grid cell, per day, from each of the N ensemble samples — giving a distribution of OHC estimates, not a single point value.

## 4. Tropical Cyclone Heat Potential (`tchp.py`)
**Reuse the exact formula and unit-conversion constants already verified in the `chilli-garlic-momo` reference implementation from our competitive research — do not re-derive the unit conversion from scratch, to avoid introducing a new bug in a part of the pipeline that was already confirmed textbook-correct:**
1. Find the 26°C isotherm depth via linear interpolation between the two bracketing canonical depth levels.
2. Integrate temperature excess above 26°C from the surface to that isotherm depth, weighted by density and specific heat.
3. Convert to kJ/cm² using the same conversion factor as the verified reference.
Compute across all N ensemble samples per grid cell/day, yielding **TCHP = mean ± standard deviation**, not a bare number.

## 5. Mixed Layer Depth — direct computation (`mld_direct.py`)
Standard de Boyer Montégut method (0.2°C threshold from the 10m reference depth, linear interpolation between bracketing levels for the exact crossing point) applied directly to each ensemble sample's reconstructed profile. Report this as the primary MLD estimate; report the Stage 5 MLD auxiliary head's prediction alongside it as a secondary cross-check, explicitly labeled as lower-confidence given its currently weak correlation (0.036).

## 6. Marine Heatwave Detection (`marine_heatwave.py`)
Use the **standard, published Hobday et al. (2016) definition**: an event is flagged where a chosen heat metric (OHC anomaly, or SST as a simpler proxy) exceeds the **90th percentile of the climatological baseline distribution for that grid cell/day-of-year, sustained for at least 5 consecutive days.** This is the real, internationally-recognized definition used in operational marine heatwave monitoring — using it directly (rather than inventing an arbitrary threshold) makes this output immediately legible to anyone with oceanography background reviewing the project.

## 7. Uncertainty Propagation (`uncertainty_propagation.py`)
A single, shared utility used by every module above: given an array of N ensemble-sampled profiles, compute the derived product (OHC/TCHP/MLD) for each of the N samples independently, then report the resulting **mean and standard deviation** of the derived product — not just propagate the raw temperature uncertainty by some approximation. This ensures downstream uncertainty is empirically grounded in the same ensemble that produced the temperature reconstruction itself, staying consistent with the architecture's core approach to uncertainty (Section 5 of the Final Architecture Specification).

**Honest UX requirement, given calibration is a known, tracked limitation (Track A, §A4):** every displayed confidence interval in the dashboard must include a visible note — e.g. "confidence interval, calibration accuracy under active improvement" — until Track A's calibration fix is verified. Do not present uncertainty bands as fully trustworthy while ECE remains elevated; do not hide them either. State the limitation plainly, consistent with the honesty standard this whole project has been built on.

## 8. Demo / Dashboard (`dashboard/`)
Minimum viable set of views for a working demo:
- **Profile viewer**: select a date and location, show the reconstructed depth-vs-temperature profile with the ensemble uncertainty band shaded, alongside the real GLORYS/ARGO value where available for comparison.
- **Marine heatwave map**: a spatial map of the domain for a selected date, flagging cells currently in a detected heatwave state.
- **TCHP gauge/display**: for a selected location, show TCHP as mean ± uncertainty, with the calibration caveat visible per §7.
- Keep this functional and clear over visually elaborate — judges and reviewers should be able to understand what they're looking at in seconds, not navigate a complex UI.

## 9. Tests (`tests/`)
- `test_tchp_formula.py`: verify against a hand-constructed synthetic profile with a known, calculable TCHP value.
- `test_mld_direct.py`: verify against synthetic profiles with an obvious, known mixed layer depth.
- `test_marine_heatwave_detection.py`: verify the 90th-percentile/5-day-persistence logic against a synthetic time series constructed to contain exactly one known heatwave event and confirm it's correctly detected (and that a shorter 3-day spike is correctly *not* flagged).

## 10. Acceptance Criteria — Phase 6 is "done" when:
1. OHC, TCHP, and direct-method MLD are computed from real ensemble-sampled profiles (from whichever is the current best checkpoint at the time — doesn't need to wait for Track A's extended training to finish).
2. Marine heatwave detection uses the real Hobday et al. definition and passes its synthetic test.
3. Every downstream product carries a real, ensemble-derived uncertainty estimate, with the calibration caveat visibly present in the dashboard.
4. The dashboard runs end-to-end against at least one real test-period date, showing all three views.
5. All tests pass.

## 11. What NOT to Do Yet
- No final PPT/submission packaging — that's Phase 7.
- Don't block Phase 6 on Track A's extended training finishing — build and test against the current checkpoint, and swap in a better one when Track A completes, exactly as planned.
