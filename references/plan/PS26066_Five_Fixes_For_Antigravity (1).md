# PS26066 — Five Fixes Required Before Phase 6

*Context: Phase 5 evaluation on the real, full-year 2025 dataset is methodologically sound and trustworthy (confirmed via independent audit), but surfaced five specific, fixable issues. Resolve these and regenerate the evaluation report before proceeding to Phase 6.*

---

## Fix 1 — Equalize the ablation training budget
**Problem:** Stage B (baseline) trained for 2,000 steps. Stage C (no-region) and Stage D (no-cascade) each trained for only 1,000 steps. Any performance gap could be partly or entirely due to the unequal training budget, not the architectural difference being tested.

**Required action:** Re-run Stage C and Stage D for the **same 2,000 steps** as Stage B, using identical optimizer settings, learning rate schedule, and data ordering — the only difference between the three runs should be the architectural component being ablated. Save fresh checkpoints (`checkpoints/ablation_no_region/best_checkpoint.pt`, `checkpoints/ablation_no_cascade/best_checkpoint.pt`) and regenerate the ablation comparison table with the equalized runs.

---

## Fix 2 — Debug the two negative-correlation auxiliary heads
**Problem:** Two of the four Stage 5 auxiliary prediction heads currently show *negative* correlation with ground truth:
- Barrier Layer Thickness (BLT): correlation −0.71
- Salinity Max Strength: correlation −0.56

Negative correlation means these heads are predicting in the wrong direction more often than not — this points to a specific bug, not an undertraining issue.

**Required action:**
1. Check for a **sign-convention error** in how the BLT and salinity-max-strength targets are computed (`src/auxiliary_targets/barrier_layer_thickness.py`, `src/auxiliary_targets/salinity_maximum.py`) — confirm the formula matches the spec exactly (`BLT = ILD − MLD_density`; salinity-max strength = anomaly relative to surrounding baseline, not raw value).
2. Check for a **label/target mismatch** — confirm the auxiliary loss is being computed against the correctly-corresponding target tensor for each head, not accidentally swapped or misaligned.
3. Check the **region-masking logic** for these two heads (BLT masked to Bay of Bengal, salinity-max masked to Arabian Sea per the spec) — confirm the mask is being applied correctly and isn't inverted.
4. Re-evaluate both heads' correlation after the fix, before considering this resolved.

---

## Fix 3 — Train Stage B substantially longer, and watch two specific numbers
**Problem:** At 2,000 steps, the model currently underperforms a naive climatology baseline (global skill score −0.71) and shows severely overconfident uncertainty (calibration ECE 0.71).

**Required action:** Extend Stage B training well beyond 2,000 steps — 10,000–20,000 steps is a reasonable next target given real measured throughput (~7.6 steps/second implies this remains inexpensive; scale GCP budget tracking accordingly, expect roughly $1–2 total, not more). During/after this extended run, explicitly report:
- **Global Murphy skill score** — currently −0.71, the target is to see this cross above 0. This is the single most important number to track.
- **Calibration ECE** — currently 0.71 (overconfident), the target is a substantial drop toward a more typical well-calibrated range (well under 0.1 is ideal, but any large, consistent drop is meaningful progress).

If skill score is still meaningfully negative after this longer run, report that honestly rather than continuing to scale steps blindly — it would indicate a different issue (learning rate, model capacity, or a data problem) worth investigating separately rather than just "more training."

---

## Fix 4 — Restore per-depth skill scores to the evaluation report
**Problem:** The current evaluation report's depth-wise breakdown (Section 4) shows RMSE/MAE/bias/correlation/SSIM per depth, but omits the Murphy skill score per depth that earlier report versions included. This is the exact metric that exposed real problems twice before in this project — it needs to stay visible.

**Required action:** Add a skill-score-vs-climatology column back to the per-depth breakdown table, computed consistently with the global figure's methodology, for every regeneration of this report going forward.

---

## Fix 5 — Clarify the heat-flux consistency check's units and provide a reference baseline
**Problem:** The reported "Meridional Heat Flux RMSE: 746204.6 W/m²" has no reference value or unit context to sanity-check it against — unlike every other metric in the report, which is compared against a baseline (climatology, or a documented benchmark).

**Required action:** In the next report, state explicitly: (a) what physical quantity and spatial/temporal aggregation this number represents (e.g. pointwise flux vs. zonally-integrated transport), (b) the equivalent value computed using true GLORYS temperature instead of predicted temperature, as a reference point, and (c) confirm whether the reported 0.9996 pattern correlation for this diagnostic is computed per-depth or pooled across all 15 depths — given the earlier discovery that pooled correlation across depths trivially inflates due to the surface-to-deep temperature gradient, this needs the same unpooled treatment already applied to the main correlation metric.

---

## Once all five are complete
Regenerate the full Phase 5 evaluation report (all sections, not just the changed ones) and share it back for review before any Phase 6 work begins.
