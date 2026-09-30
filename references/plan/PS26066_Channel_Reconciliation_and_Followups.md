# PS26066 — Channel Reconciliation & Remaining Follow-Ups

*Most of the full re-verification checklist has now genuinely landed — this file covers one significant new finding and two smaller follow-ups, not a repeat of the whole prior checklist.*

---

## Priority 1: Reconcile the actual input channels against the original architecture spec

The real channel inventory (re-verification report, Item 1) does not match the 25-channel design in the Final Architecture Specification. Specifically:

**Present in the real implementation but not in the original spec:** SST spatial gradients (x/y), SST 7-day tendency, wind stress magnitude, SSH anomaly, an MLD diagnostic channel, temperature-at-MLD, heat/salt flux proxies, a temperature gradient at MLD, a Richardson number.

**In the original spec but missing from the real implementation:** geostrophic and ageostrophic currents derived from SSH, precipitation, chlorophyll-a, the river-plume influence field, and the explicit missingness-mask channel.

**Required action — a direct, explicit accounting, not a re-justification of why the new features are reasonable:**
1. State plainly when and why this substitution happened — was it a deliberate response to a real blocker (e.g. difficulty integrating precipitation/chlorophyll/river-discharge data cleanly), or did it happen incrementally without a clear single decision point?
2. Confirm whether the externally-downloaded precipitation, chlorophyll-a, and river-discharge datasets (from the dataset download guide) are sitting unused on disk, or were never actually pulled into the final training pipeline at all.
3. Give a reasoned recommendation: should we (a) add the missing spec'd features back alongside the current ones (increasing channel count), (b) formally adopt the current feature set as an intentional, superior redesign and update our own documentation to match it, or (c) some blend? This is a real decision for us to make together, not something to resolve unilaterally.
4. Specifically address geostrophic currents — this was flagged as a "must-have" in our original design because it's a literature-proven technique (physically-guided fusion networks outperforming plain surface-feature models). If it's genuinely absent, that's worth a deliberate decision, not a silent omission.

---

## Priority 2: Validate Fix A2 (loss normalization) on a real full training run, not just a unit test

Item 11's own numbers reveal that `checkpoints/baseline_20k/best_checkpoint.pt` — the checkpoint this entire report evaluates — was trained **before** Fix A2 was applied (`w2 = +1.693`, `exp(-w2) = 0.18×`, explicitly attributed to "unscaled MSE before Fix A2"). Fix A2 is currently verified only via `tests/test_losses.py` in isolation, the same gap we caught with the calibration fix last round.

**Required action:** once the seed-43 run (or a subsequent run) trains with Fix A2 active from step 0, report the resulting `w2` value and the updated auxiliary-head correlations (MLD, BLT, salinity-max depth/strength) from that real run — confirm the fix actually changes real training dynamics, not just that the loss function computes correctly in a unit test.

---

## Priority 3: Confirm SSIM methodology is unchanged
SSIM values jumped from roughly 0.71–0.88 (10k-step evaluation, several rounds ago) to 0.96–0.99 (this round's 20k evaluation). This may well be genuine — more training, plus several real fixes landing, could produce a real improvement this large. But given the size of the jump, get a one-line confirmation that the SSIM computation itself (window size, dynamic range normalization) hasn't changed between report versions, so this is a like-for-like comparison and not an artifact of a methodology change.

---

## What to bring back once seed 43 finishes
The comparison we actually need: does Stage B's advantage over Stage C hold at a second, independent random seed? Report both seeds side by side, and say plainly if they disagree — that would itself be an important, honest finding about how robust the region-conditioning effect really is.
