# PS26066 — Full Re-Verification Checklist, Outstanding Fixes & Second-Seed Validation

*Purpose: a complete, systematic check of the current implementation against every decision made across our planning documents — architecture spec, dataset guide, and Phases 1–6 — so nothing is silently hardcoded, stale, unused, or drifted from spec without us knowing. Also includes the second-seed 20k validation run, and a re-examination of the thermocline zone based on what the real data has actually shown us.*

---

## PART A — Fixes That Must Actually Land (not just be diagnosed)

### A1. Integrate the real calibration fix — not the isolated test
Enable `eta` in the 0.2–0.5 range in `DDIMSampler` for genuine sampling stochasticity (not just the post-hoc temperature-scaling multiplier), re-run evaluation against the **current production checkpoint**, and report the real resulting ECE. The 0.0255 figure in the audit document was a standalone proof-of-concept — it does not count as done until it's the number in an official evaluation report generated from the actual training/sampling pipeline. Update the "Fix 3: RESOLVED" label only once this is genuinely true end-to-end.

### A2. Apply the loss-scale normalization fix
Normalize auxiliary targets (MLD ÷ 50m, BLT ÷ 20m, salinity depth ÷ 100m, or similar) before computing MSE, so `loss_aux` starts on the same O(1) scale as `loss_diffusion`. Cheap, low-risk, and worth doing before the second-seed run so both seeds train under the corrected loss balance.

### A3. Formally document the BLT/auxiliary-head resolution
Write down explicitly (in the master status doc) that Barrier Layer Thickness and the general auxiliary-head reliability issue are being resolved via **direct 3D profile integration in Phase 6**, not via fixing the scalar heads themselves — this is a deliberate, accepted design decision now, not an open bug. State the reasoning: BLT's problem is a genuine physical decoupling (winter temperature inversions) that a single scalar head structurally can't resolve without 3D salinity input, so routing around it is the more robust answer.

---

## PART B — Re-Examine the Thermocline Zone Definition Against Real Data

The originally-hypothesized "thermocline core = 75–150m" zone was based on literature and early findings. **The real per-depth RMSE data we now have shows the actual struggle zone is broader than that** — elevated error is visible starting around 20–30m and doesn't fully subside until past 200m, not just 75–150m. This is a sharper, better-grounded observation than our original guess, now that we have real numbers to check it against.

**Required action:**
1. Re-examine the actual per-depth RMSE curve from the latest real evaluation run and determine the empirically-justified boundary of the hard zone (likely closer to 25–200m than 75–150m).
2. Update the **priority zone 2 definition** used in evaluation slicing to match the empirical boundary, not the original literature-based guess.
3. Update the **depth-weighted loss function's upweighted range** (`α(d, region)` in the loss spec) to match the same revised boundary — if the real struggle zone is wider than what's currently being upweighted, part of the zone may currently be getting no extra loss weight at all, which would itself help explain why error is elevated there.
4. Report the before/after RMSE in this zone once the loss weighting is corrected to match reality.

---

## PART C — Full Re-Verification Checklist (go through every item, report status for each — don't skip silently)

### Data & Pipeline
- [ ] Dump real summary statistics (mean, std, min, max, fraction-non-null) for **every one of the 25 input channels**, computed directly from the actual Zarr store used in the 20k training runs — confirm none are degenerate, accidentally zero-filled, or carrying suspiciously little variance (the same failure mode that caused the salinity-max-depth bug, but checked proactively this time across every channel, not just the one we happened to catch).
- [ ] Confirm the land/ocean mask channel (originally spec'd as channel 21) is genuinely present and wired into the conditioning input as intended — the audit's coastal-artifact finding suggests it may not be doing its job even if technically present; find out which.
- [ ] Re-confirm the climatology fit is still provably training-years-only (re-run the assertion check that raises an error on out-of-range dates — confirm it still exists and still passes).
- [ ] Re-confirm the chronological split is unchanged: train Jan–Aug, val Sep–Oct, test Nov–Dec 2025.

### Architecture
- [ ] Confirm the ConvLSTM context encoder is still exactly 3 layers, 32→64→64 channels, as specified.
- [ ] Confirm the U-Net denoiser is still 4 resolution stages, 32→64→128→256, with AdaGN conditioning injected at every block — not simplified to simple concatenation anywhere along the way.
- [ ] Confirm depth is still conditioned via **log-normalized** identifier, not min-max — re-check this specific detail hasn't regressed during any of the fixes made so far.
- [ ] Re-run the depth-cascade integrity check (corrupt the shallow-depth output, confirm deeper-depth predictions change) **against the current real trained model**, not just Phase 3's toy-scale version — confirm the cascade is genuinely doing something on real data, not just architecturally present.
- [ ] Confirm region-membership channels are still soft/distance-blended, not hard binary.

### Training
- [ ] **Report the actual current learned values of `w1`, `w2`, `w3`** (the adaptive loss weights) from the latest checkpoint. If any of them has converged to a value that makes its corresponding loss term's effective contribution negligible, that term may be silently doing nothing despite being "in the code" — this is exactly the kind of predefined-but-unused risk to rule out explicitly, not assume away.
- [ ] Confirm the physics-consistency (thermocline-depth-matching) loss is being computed on the model's `x̂₀` estimate as specified, and that its gradient is actually reaching the relevant parameters (not silently zeroed by a masking bug).

### Sampling / Inference
- [ ] Confirm the ensemble size actually used for uncertainty is genuinely N=10–20 independent samples, not fewer.
- [ ] Once A1 lands, confirm DDIM `eta` is genuinely > 0 during evaluation/inference, not left at the old default anywhere in the codebase (config files, hardcoded defaults, or dashboard-specific inference calls could each independently still have the old value).

### Evaluation
- [ ] For every metric in the next regenerated report, **attach the exact checkpoint file path/hash and generation timestamp used to compute it.** This is the single change most likely to prevent the stale-number class of bug from recurring silently again.
- [ ] Confirm the heat-flux-consistency correlation is computed per-depth (unpooled), not pooled across all 15 depths — this was never fully closed out with an explicit one-line confirmation.
- [ ] **Report the real held-out TEST-set RMSE for the 20,000-step runs**, not just "Best Val RMSE" — validation RMSE was used for checkpoint selection, so it can run slightly optimistic relative to genuinely unseen test data. We have this for the 2k runs (0.9023°C) but not yet for the 20k runs.

### Downstream Products (Phase 6)
- [ ] Confirm Phase 6's OHC/TCHP/MLD/MHW computations are pointed at the **current best checkpoint** (post-calibration-fix, post-loss-reweighting), not an earlier pre-fix checkpoint left wired in from before these fixes landed.

---

## PART D — Second-Seed Validation (approved, run before Phase 7)

Run Stage B and Stage C **again, both from scratch (step 0), 20,000 steps each, with a different random seed** than the first 20k run — same protocol as before (equal hardware, equal optimizer/schedule, only the seed differs.

**Report:**
1. Both seeds' results side by side for Stage B and Stage C.
2. Whether the region-conditioning advantage (Stage B beating Stage C) holds, shrinks, grows, or reverses with the new seed.
3. A combined verdict: if both seeds agree in direction, that's genuine validation. If they disagree, say so honestly — that would mean the effect is within seed-to-seed noise, which is itself an important, honest finding to report rather than something to paper over.

---

## Actions for you (not Antigravity) to do

1. **Approve this checklist and the second-seed run** — already done, per your message; just confirm Antigravity has the go-ahead to spend the additional ~$3.70 (two more 20k runs) against your GCP credit.
2. **Personally walk through the Phase 6 dashboard/demo yourself** once A1–A3 land and the checkpoint is refreshed — passing automated tests doesn't confirm it tells a clear, convincing story to someone seeing it cold. This needs your eyes, not just a test suite.
3. **Review Part B's proposed thermocline zone redefinition and sign off on it** — it affects both how we evaluate and how the loss function is weighted, so it's worth your explicit agreement before Antigravity changes it, not just Antigravity deciding unilaterally.
4. **Once the second-seed results come back, bring them to me before finalizing any "region conditioning validated" language for the PPT** — I'll check the same way I've checked everything else so far.
5. **Periodically cross-check the internal GPU-hour/cost tracker against your actual GCP billing console** — total spend so far is still trivial (roughly $5–6 all in), but this is good habit to keep, not a sign anything's wrong.
