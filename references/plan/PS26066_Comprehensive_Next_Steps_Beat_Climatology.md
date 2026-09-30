# PS26066 — Comprehensive Next-Steps Plan: Closing the Climatology Gap

*Organized in phases: diagnose first, try the cheap/fast fixes next, only then consider expensive changes like more training. Includes everything from the eleventh audit plus additional items not covered there.*

---

## PHASE 1 — Diagnose before changing anything (fast, cheap, do these first)

### 1.1 Confirm the anomaly-prediction mechanism is actually intact
This is the single most important check in this entire plan, and it should happen before anything else. The whole architecture was designed around predicting the **anomaly relative to climatology**, not the absolute temperature — this was meant to make "beating climatology" the natural starting point, not a hard target to reach. Given we've already found one silent architectural regression this project (depth conditioning reverting to linear scaling) and one silent channel-set drift, **explicitly re-verify**: is the model's training target still genuinely `GLORYS_truth − climatology`, reconstructed at inference as `climatology + predicted_anomaly`? If this mechanism has degraded or been bypassed anywhere along the way, that alone could fully explain the persistent negative skill score — a model accidentally predicting absolute values (or predicting anomaly incorrectly) would naturally struggle to beat a baseline it's not actually being measured against correctly.

### 1.2 Check whether training had actually converged by step 20,000
Pull the raw validation-RMSE-vs-step curve for the current best checkpoint (seed 42, post-Fix-A2) and check: was RMSE still meaningfully decreasing at step 20,000, or had it plateaued? If still decreasing, more training is likely the highest-leverage single lever available. If it had plateaued, more steps alone won't help and the problem is more likely architectural/data-related — this determines which of the later phases matters most.

### 1.3 Confirm per-channel normalization is correctly implemented
Given the raw Zarr dump showed SST stored in Kelvin (~300) alongside wind stress curl (~1e-9) and latent heat flux (~1e6) — a huge scale range across the 25 input channels — explicitly confirm each channel is normalized independently (not globally) before being fed to the network. Poor multi-channel normalization is one of the most common, easy-to-overlook causes of underperformance in exactly this kind of multi-modal model, and it hasn't been explicitly checked since the Kelvin-units discovery a few rounds back.

### 1.4 Where exactly does the model lose to climatology?
Beyond the aggregate skill score, break down *where* skill score is negative versus positive — which depths, which regions, which seasons. We have per-depth skill scores already; extend this to check whether the losses are concentrated in low-variance conditions (consistent with the "climatology already captures most of the signal, model adds noise on top" pattern we've suspected since early in this project) or are spread evenly. This tells us whether to focus improvement effort narrowly or broadly.

### 1.5 Re-investigate the MLD auxiliary head correlation swing
(Carried over from the eleventh audit.) Correlation went from weakly positive (seed 43, +0.10) to strongly negative (seed 42, this report, −0.59) for a similar setup — check whether this is seed variance or something specific to this run.

---

## PHASE 2 — Cheap, fast improvement attempts (no new training required)

### 2.1 Test whether ensembling seed 42 and seed 43's existing checkpoints closes the gap
We already have both checkpoints. Before spending on more training, average their predictions and recompute the skill score for the ensemble. Ensembling often reduces noise enough to meaningfully improve aggregate metrics even when neither individual model clears the bar alone — this is a nearly-free experiment given both checkpoints already exist.

### 2.2 Try simple post-hoc bias correction
The model shows a persistent, real bias (currently a modest +0.02°C aggregate, larger at some depths in earlier reports). Fit a simple correction (e.g., per-depth, per-region bias offset) on the validation set and apply it to test predictions, the same general approach used for the calibration temperature-scaling fix. This is fast, low-risk, and directly attacks part of what drives skill score below zero — systematic bias hurts MSE-based skill score more than random error does.

### 2.3 Re-verify the climatology fit quality itself
Since skill score depends on both the model's error *and* the climatology baseline's own quality, confirm the harmonic climatology fits are genuinely well-converged (reasonable condition numbers, as checked much earlier in this project) for the current full-year dataset, not just spot-checked once early on.

---

## PHASE 3 — If Phase 1–2 don't close the gap: more substantial changes

### 3.1 Extend training with a properly adjusted schedule — not just "run more steps"
If Phase 1.2 shows the model was still improving at step 20,000, extend training. **Important: don't simply continue past step 20,000 if the learning rate schedule was designed to decay to its minimum exactly at that point** — a schedule built for a 20k budget won't behave the same way if naively extended. Either restart with a schedule designed for the new total step count (e.g., 40,000–50,000), or use a proper warm-restart approach.

### 3.2 Consider skill-focused loss reweighting
Rather than only upweighting specific depth bands (as the current depth-weighted loss does), consider a loss term that directly penalizes regions/depths where the model is currently losing to climatology more heavily — a more targeted way to close the specific gap we're trying to close, rather than a generic accuracy push.

### 3.3 Physically-consistent data augmentation
Given the effective training set is bounded at 359 sequences from a single year, consider whether careful, physically-reasonable augmentation (small spatial shifts/crops within the ocean domain, respecting land-sea boundaries and regional structure) could effectively expand the training signal without needing new real data. This is a real, standard technique, but needs care to avoid introducing physically nonsensical training examples.

---

## PHASE 4 — Reporting and presentation fixes (carried over from the eleventh audit)

4.1 Drop or clearly re-label the "Statistical R²" column so it can't be misread as contradicting the honest skill-score number.
4.2 Get the full continuous-period evaluation (not a 9-date/3-date spot check) so headline numbers are representative of the whole test window, not a selected subset.
4.3 Fix the small prose-vs-table numeric inconsistencies in Section 2's narrative text.
4.4 Do not present Section 9's benchmark comparison table anywhere until the model beats climatology on this same evaluation — right now it invites a question the honest answer to doesn't support yet.

---

## PHASE 5 — Can run in parallel with the above

### 5.1 The independent-of-GLORYS validation investigation
Still open from several rounds back — timeboxed, bounded search for a genuinely unassimilated Indian Ocean observation source, or an explicit, honest documentation of the limitation if nothing turns up. This matters more now, not less, precisely because the model is getting close enough that "is this real skill or GLORYS-mimicry" becomes the live question.

### 5.2 Re-run calibration verification after any model change
Every time the underlying model changes (bias correction, more training, ensembling), calibration (ECE) should be re-checked, not assumed to still hold from a prior checkpoint — calibration is a property of a specific trained model, not something that transfers automatically when the model changes.

### 5.3 Keep the GCP billing console check as a standing habit
Given Phase 3 in particular could add several more training runs, periodically confirm actual spend against the ₹40,000 credit directly, not just the internal tracker.

---

## Suggested order of actual execution

Do **Phase 1 completely first** — it's fast, cheap, and the answers directly determine which parts of Phase 3 are even worth attempting. **Phase 2 next**, since both items are nearly free given existing checkpoints. Only move to **Phase 3** if Phase 1–2 don't close the climatology gap on their own. **Phase 4 and 5** can happen anytime in parallel, since they don't depend on the outcome of the others.

Bring back Phase 1's findings first — what they show should shape exactly which Phase 3 items (if any) are worth pursuing, rather than attempting all of them speculatively.
