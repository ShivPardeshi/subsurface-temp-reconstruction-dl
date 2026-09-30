# PS26066 — Ordered Action List: Stability Fix, Full Reference-Model Evaluation, Housekeeping

*Do these in order — each step is a reasonable prerequisite for the ones after it.*

---

## Step 1 — Fix the physics-loss numerical instability (code-level, do first)
In the physics-loss `x̂₀` reconstruction (`src/training/losses.py`), clamp `ᾱ_t` to a small positive epsilon before the division, or explicitly skip/down-weight the physics-loss term at very high noise timesteps (near `t=1000`) where the clean-data estimate is inherently unreliable regardless of model quality. This directly prevents the class of degenerate loss values seen in the diagnostic report (e.g., Total Loss = 4.79×10⁷ at Step 60).

**Verification required:** add a unit test (`tests/test_physics_loss_stability.py` or similar) that artificially sets `ᾱ_t` near zero and confirms the loss no longer blows up to a degenerate value — don't just assert the fix conceptually, prove it the same way every other fix in this project has needed to be proven.

**No dedicated re-training run is needed just to validate this fix.** It's expected to affect only a small number of pathological steps out of 20,000, not the overall training dynamics — let it apply naturally to whatever training happens next (Stage D's eventual second seed, an ensembling run, etc.) rather than spending a run solely to confirm it.

---

## Step 2 — Regenerate the full evaluation report against seed 42 post-Fix-A2 (the current reference checkpoint)
Rather than patching five individual numbers into the existing `evaluation_report.md`, regenerate the entire report fresh, end to end, using `checkpoints/baseline_20k_seed43...` — correction: using **`checkpoints/baseline_20k/best_checkpoint.pt` (seed 42, post-Fix-A2)** as the single reference checkpoint throughout every section. This includes, at minimum:

1. **Held-out test-set RMSE** (not just validation RMSE) — we have this for the pre-Fix-A2 20k run (0.7045°C) and the 2k baseline (0.9023°C), but not yet for the current best configuration.
2. **Auxiliary head correlations** (MLD, BLT, salinity-max depth, salinity-max strength) — we have these for seed 43 post-Fix-A2, not seed 42.
3. **Calibration ECE with Fix A1 and Fix A2 both active together** — the 0.50 ECE figure was measured with only Fix A1 active, before Fix A2 was properly integrated.
4. **A clean, final priority-zone breakdown** across all 7 zones, using the redefined 20–200m thermocline boundary.
5. **Heat-flux consistency check** re-confirmed against this specific checkpoint.
6. **Global RMSE, MAE, bias, correlation, SSIM, and skill score** — the full standard metric set, all against this one checkpoint, all in one place.
7. **Every other metric already established in the evaluation methodology** (per the original Phase 5 spec) that hasn't been explicitly re-listed above — the goal is one complete, internally consistent report, not a patchwork of numbers from different points in the fix history.

**Attach the exact checkpoint file path/hash and generation timestamp to this report** — this is the practice that would have caught several of the stale-number issues we've hit earlier in this project sooner, and this full regeneration is the ideal moment to make it standard going forward.

---

## Step 3 — Formally retire the stale pre-fix ablation figures
Update the internal project documentation (`PROJECT_MASTER_STATUS_AND_REPORT.md` and any other place these are cited) to replace the old 31.8% (depth-cascade) and 8.40% (region-conditioning) figures with the current, honestly-measured numbers from the fully-fixed pipeline. State plainly that the old figures were measured under a pipeline with since-corrected issues (unequal training budgets, unbalanced loss weighting, deterministic sampling) and should not be cited as current.

---

## Step 4 — Real GCP billing console check
Cross-check the internal per-run cost tracker's cumulative total against the actual GCP billing console directly. Given the number of 20k-scale runs that have now accumulated, this is a good, non-urgent habit checkpoint — total spend is almost certainly still trivial against the ₹40,000 credit, but worth confirming directly rather than trusting the internal estimate alone.

---

## Step 5 — Timeboxed independent-validation investigation
Spend one focused, bounded session (not open-ended) on the following:
1. Check GLORYS12v1's official CMEMS quality documentation for its exact, explicit list of assimilated in-situ observation platforms/networks.
2. Confirm whether RAMA moorings are explicitly on that list (expected, given the CORA in-situ database's known composition).
3. If RAMA is confirmed assimilated, spend limited additional effort searching for one alternative Indian Ocean observation source that is plausibly *not* part of standard global assimilation streams — for example, a specific INCOIS or National Institute of Oceanography research cruise dataset, or a delayed-mode field campaign not yet folded into CORA/GLORYS.
4. **If nothing concrete and practically accessible turns up within this bounded effort, stop searching and instead write down the limitation explicitly**: state that validation is against real observations that are themselves partial inputs to the training target, not a fully independent test, and that this is a known, disclosed limitation rather than an unresolved gap. This honest statement is itself an acceptable outcome — don't let this become an unbounded search.

---

## Deferred, not forgotten (no action right now)

- **Ensembling decision (single seed vs. multi-seed average) and Stage D's second seed** — both deliberately deferred until Step 2's full evaluation is complete, per your own reasoning. Revisit once those results are in hand.
- **Personal review of the Phase 6 dashboard** — deferred until the model itself is where you want it, per your call.
- **Physics-loss spike-count vs. final-quality correlation** — no dedicated action; just worth a passing glance once Step 1's fix is in and a future run's spike statistics are naturally available for comparison.
- **Second-seed-as-standard-practice** — agreed as a going-forward principle for any future architectural claims, not an action item on its own right now.

---

Once Step 2's regenerated report is generated let me know.