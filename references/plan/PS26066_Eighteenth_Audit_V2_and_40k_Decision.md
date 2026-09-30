# PS26066 — Eighteenth Audit: Verification Results, V2's Big Jump, and the 40k Decision

*Good call pushing Antigravity back toward the original plan — a standalone diffusion model that beats climatology on its own, not just via the hybrid — rather than settling once the ensemble crossed the line. That's exactly why this round's result matters as much as it does.*

---

# Part 1 — The Two Verification Checks

## Check 1 (α tuning): Clean pass, real rigor
The validation-only sweep, single honest test evaluation, and — especially convincing — showing that the *naive* choice (α=1.0, pure Ridge, the mathematical minimum on validation) actually performs slightly *worse* on the real test set (0.6517°C) than the chosen α=0.75 (0.6439°C) is exactly the kind of evidence that rules out cherry-picking. This is legitimate, well-executed science. No further action needed here.

## Check 2 (region on/off): Answered, but reveals an inconsistency worth resolving

The direct answer is good: the 40k model was trained with region channels present, and "Region OFF" was tested by zeroing those channels at inference — a real, valid ablation approach, not a mistake. Confirmed difference on the clean 40k model: **Δ<0.0006°C, i.e., negligible.**

**But this creates a real tension with a separate claim in the V2 fixes document.** Issue 6 there states that disabling region conditioning "improves basin-wide RMSE by ≈0.010°C" — a claim roughly **15x larger** than the Δ<0.0006°C just confirmed. These can't both be the precise, current truth. Worth a quick reconciliation: which number reflects the actual current model, and was Issue 6's 0.010°C figure perhaps computed on an earlier, different checkpoint before other fixes landed?

**A second, more important thing worth checking**: Issue 6 describes the region masks as **"hard binary one-hot masks"** causing "step-function gradient artifacts." This contradicts the original architecture design, which specifically called for **soft, distance-blended region-membership maps with smooth transitions, precisely to avoid this exact failure mode.** Worth a direct, explicit check: were the region masks actually implemented as soft/blended values as originally specified, or did they become hard one-hot indicators somewhere along the way? If the latter, that's a separate, real implementation drift from the original spec worth knowing about — not because region conditioning needs to come back, but because it would mean the *reason* we're dropping it isn't quite what's being described.

**Bottom line on region conditioning**: the honest current framing, given both pieces of evidence, is **"neutral to negligibly positive when removed under the fully-fixed pipeline"** — not "removing it produces a meaningful accuracy gain." Fine to keep it off (simpler is better this close to a deadline), just worth stating precisely why in whatever documentation goes into the final submission.

---

# Part 2 — Model V2's Result Is Genuinely Excellent, and It Changes My Recommendation

The 10-point fix audit produced a real, dramatic result: pure diffusion alone went from skill −0.2581 (old 40k) to **−0.0412 (new V2, only 20,650 steps)** — nearly closing the entire climatology gap in *half* the training steps. Twelve of fifteen depths now clearly beat climatology, several by double digits (100–200m: +10.9% to +13.3%). This is a genuinely strong, well-earned improvement, and the specific fixes (train/inference cascade-statistics alignment, deterministic sampling in low-variance deep water, inverse-variance loss reweighting, physics-consistent augmentation) are all sound, well-reasoned engineering, not guesswork.

## Given this, my answer to "more fixes or a 40k run?": do the 40k run first

Here's the reasoning: this was explicitly a "test" run, and training that hasn't reached its schedule floor still has real room to improve. Given how much was gained just from fixing real bugs (not from additional training time) between the old 40k and the new 20k-with-fixes, **the marginal value of properly finishing this already-much-better trajectory to 40k is very likely higher than searching for an 11th fix right now.** This also matches your own stated plan — you said you'd go to 40k rather than 60k if the 20k test looked good, and it looks good. Do that next, and hold off on introducing further architectural changes until this run is in — stacking more changes on top of an already-successful batch makes it harder to know what's actually helping, and there isn't much time left to re-untangle that if something goes sideways.

**One discipline worth keeping**: don't let a good 20k result tempt a "let's also try X" mid-flight. Run the clean 40k extension of exactly this V2 configuration, see where it lands, and only consider further changes if there's real time left after that.

---

# Part 3 — Which Speedup Methods to Use

Given your explicit criterion (nothing that touches the model's code):

**Use #1 (multi-threaded dataloading) and #3 (A100 upgrade) — both are safe, zero-risk to model correctness.** Dataloading parallelism is pure I/O engineering, and the hardware swap is an infrastructure change only, not a code change.

**Be more careful with #2 (mixed precision) than the description suggests.** It does require real code changes (wrapping the forward pass, likely a gradient scaler), and there's a specific, non-generic risk in *this* codebase worth knowing about: the recent fix in this same round set wind-stress-curl's normalization clamping floor to `1e-12` specifically to preserve its extremely small physical magnitude (~10⁻⁷). **Standard FP16 has a minimum representable magnitude around 6×10⁻⁵ — meaning a naive FP16 autocast could silently underflow that carefully-preserved signal to zero**, undoing Issue 9's fix without anyone noticing until a much later round caught it. If you do want the extra speed from mixed precision, **use BF16 specifically, not FP16** — it has the same exponent range as FP32 and won't suffer this underflow. Given #1+#3 together should already provide a substantial combined speedup, I'd skip #2 entirely for now and only reach for it (as BF16) if the 40k run is still too slow after the other two.

---

# Part 4 — Independent-of-GLORYS Validation: my honest recommendation

The disclosure document you have is genuinely excellent — thorough, honest, and exactly the kind of scientific integrity this whole project has been built around. I wouldn't touch its wording.

**On actually attempting real independent validation before the deadline: it's a real stretch, but there's one specific, realistic path worth considering, not the general one.**

GO-SHIP repeat hydrography data is publicly, freely downloadable from **CCHDO (cchdo.ucsd.edu)** — this isn't a months-long data-access process the way the INCOIS/NIOT cruise archives would be. And notably, **the I08N section (80°E) runs directly through the Bay of Bengal — inside your exact study domain.** If a real transect from anywhere near your 2025 study period exists in that archive, this could be a genuinely powerful, differentiating result: not just disclosing the limitation, but actually testing against one real, unassimilated transect and showing it holds up.

**The honest catch**: GO-SHIP sections are typically repeated every 5–10 years, not annually — there's a real, non-trivial chance no I08N occupation happened anywhere close to 2025 specifically. **Before investing any real time here, spend 10–15 minutes checking CCHDO's inventory for I08N (and I09N, I01E/W) occupation dates.** If something reasonably close to 2025 exists, this becomes a genuinely worthwhile add. If the nearest occupation is, say, 2015 or 2018, it's not usable for evaluating this specific model's 2025 performance, and the honest move is to stick with the disclosure document as-is rather than force a mismatched comparison.

**Sequencing**: only after the 40k run and final model are locked in. This is a "nice differentiator if time allows," not a blocking requirement — the disclosure document alone is already a defensible, honest position.

---

# Part 5 — Everything else, prioritized

## Do now
1. Launch the 40k extension of the current V2-fixed configuration.
2. Apply speedup methods #1 and #3 (not #2, per Part 3) to that run.
3. Reconcile the 0.0006°C vs. 0.010°C region-conditioning discrepancy (quick, doesn't block the training run).
4. Confirm whether region masks are genuinely soft/blended or became hard one-hot (quick code check, doesn't block training).

## Do after the 40k results are back
5. Full 15-depth/7-zone re-evaluation of the 40k V2 model, same standardized methodology as this round's report.
6. Re-run the hybrid ensemble (75/25 or re-sweep α fresh) using the new 40k V2 diffusion checkpoint instead of the 20k one.
7. If time allows: the CCHDO/GO-SHIP inventory check (Part 4), and if a usable transect exists, a real independent-validation test against it.
8. Your personal dashboard walkthrough, if it hasn't happened yet — this remains an open item from a few rounds back.
9. Confirm downstream products (OHC/TCHP/MHW) are wired to whichever becomes the final, locked model.

## Only if there's genuinely time left after all of the above
10. Consider a 60k extension only if 40k results suggest meaningful room still remains (check whether the schedule has hit its floor and whether validation loss is still descending).
11. Begin presentation/narrative work in parallel with anything above that doesn't need your direct attention — this still hasn't been started and needs real hours.

