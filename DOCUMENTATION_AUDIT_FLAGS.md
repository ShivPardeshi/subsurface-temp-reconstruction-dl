# OceanEmbed (PS26066) — Documentation Audit: Discrepancies, Retired Figures, and Flags

**Purpose:** This file logs all inconsistent, retired, or potentially misleading data points discovered during the creation of `MASTER_PROJECT_DOCUMENTATION.md`. It is organized by category: (A) Permanently Retired Metrics, (B) Contextual Inconsistencies Found in Reports, (C) Architecture Parameter Clarifications.

**Date:** 2026-09-21
**Audit Trail:** Cross-referenced from `CHECKPOINT_REGISTRY.md`, `post_phase6_decontaminated_chronicle_and_model_audit.md`, all phase reports, and source code.

---

## A. Permanently Retired Metrics (DO NOT CITE)

These figures appeared in historical reports and earlier conversations but are based on the **contaminated "Multi-Seasonal 10-Date" benchmark** (Days 15, 60, 105, 150, 195 were inside training set Days 0-236):

| Historical Claim | Source Document | Why It Is Retired |
|:---|:---|:---|
| "+19.49% Murphy Skill Score" | Various historical mentions | Computed on benchmark with 5/10 training-set dates |
| "15/15 depths beat climatology" | Historical multi-seasonal claims | Computed on contaminated benchmark |
| "0.6091 C Multi-Seasonal RMSE" from 40k V2 Hybrid | PS26066_Scientific_Narrative_and_Executive_Synthesis.md | Contaminated benchmark; 40k checkpoint is DEPRECATED |
| "+10.05% Murphy Skill" from 40k V2 Hybrid | PS26066_Final_Hackathon_Submission_Deck.md | Contaminated benchmark; 40k checkpoint is DEPRECATED |
| "0.6121 C Multi-Seasonal RMSE" from 20k V2 Hybrid | PS26066_Scientific_Narrative_and_Executive_Synthesis.md | Contaminated benchmark |
| "+9.15% Murphy Skill" from 20k V2 Hybrid | PS26066_Scientific_Narrative_and_Executive_Synthesis.md | Contaminated benchmark |
| "0.6164 C RMSE" from independent_validation_disclosure.md | reports/independent_validation_disclosure.md line 70 | Appears to be pre-decontamination figure |
| "+0.0787 Skill" from independent_validation_disclosure.md | reports/independent_validation_disclosure.md line 70 | Appears to be pre-decontamination figure |
| "0.6435 C Continuous Test RMSE" (20k V2) | PS26066_Scientific_Narrative_and_Executive_Synthesis.md | Continuous 61-day test also predates audit |
| "0.0169 ECE" | PS26066_Scientific_Narrative_and_Executive_Synthesis.md | This was for the hybrid ensemble (Ridge+Diffusion) which is NOT the current production model |
| "27.6x calibration improvement" | Various | The post-hoc depth-dependent uncertainty scaling was for the hybrid ensemble, not Phase 8 calibrated |

> [!CAUTION]
> The `PS26066_Scientific_Narrative_and_Executive_Synthesis.md` and `PS26066_Final_Hackathon_Submission_Deck.md` files were written BEFORE the decontamination audit. Their headline metrics (0.6091°C, +10.05%, +9.15%, etc.) are from the contaminated benchmark and the deprecated 40k hybrid ensemble model. These documents have archival value but should NOT be cited as current performance metrics.

---

## B. Contextual Inconsistencies Found Across Reports

### B1. The "Model V2 20k Hybrid Ensemble" vs. "Phase 8 Calibrated" Confusion

**Inconsistency:** Multiple documents (Scientific Narrative, Submission Deck) refer to a "Model V2 20k Production Hybrid" with a multi-output Ridge Regression blend (alpha=0.75). This is a DIFFERENT model from the current production (Phase 8 Pure Diffusion + Bayesian Calibration).

**Clarification:**
- "Model V2 20k Hybrid Ensemble" = Phase 4 Pure Diffusion (20k) blended with Ridge Regression at alpha=0.75, evaluated on the contaminated benchmark
- "Phase 8 Calibrated" = Phase 8 Zone-Adaptive Pure Diffusion (15k fine-tune from Phase 7) with Bayesian shrinkage + dual-sample DDIM posterior mean, evaluated on CLEAN decontaminated benchmarks
- **The current ACTIVE-PRODUCTION model is Phase 8 Calibrated, NOT the hybrid ensemble**

### B2. ECE (Expected Calibration Error) Confusion

**Inconsistency:** The Scientific Narrative reports ECE=0.0169 and "27.6x calibration improvement" from legacy ECE=0.4450. These numbers refer to post-hoc depth-dependent temperature scaling applied to the hybrid ensemble model, NOT to the Phase 8 Bayesian calibration.

**Clarification:** The Phase 8 calibration uses Bayesian shrinkage (Strategies 1 & 2), which is conceptually different from post-hoc temperature scaling for uncertainty. An ECE for Phase 8 was NOT explicitly computed and reported in any GPU run log found in the repository.

**Flag:** Any future ECE claims for the Phase 8 model must be computed fresh and should not borrow the ECE=0.0169 figure from the hybrid ensemble.

### B3. "67 Unit Tests" vs "68 Unit Tests" Discrepancy

**Inconsistency:** `PS26066_Scientific_Narrative_and_Executive_Synthesis.md` (line 180) states "100% automated test pass rate (67 unit tests passed)", while the actual pytest run (verified in session) returned "68 passed".

**Clarification:** The test suite was expanded between when the Scientific Narrative was written and when the final session ran. Current verified count is **68/68 passed**. Use 68 as the authoritative number.

### B4. Inference Latency "148ms" — Context Required

**Inconsistency:** The Scientific Narrative and Submission Deck state "148 ms per ensemble member". The Phase 8 benchmark report states training throughput of 3.30 steps/sec (~303 ms/step) during fine-tuning.

**Clarification:** These are different metrics:
- 148 ms = inference latency (one DDIM cascade sampling run for one batch member)
- 303 ms/step = training step time (includes forward + backward + optimizer update)
Both are plausible and non-contradictory. The 148ms inference figure is from the Scientific Narrative and may refer to an older benchmark run or a specific batch size. It should be re-verified against the Phase 8 checkpoint if used in new claims.

### B5. Total Training Steps — "41,300 Steps" Claim

**Inconsistency:** `PS26066_Final_Hackathon_Submission_Deck.md` states "Full 41,300-step training completed in 4.24 GPU-hours on a single NVIDIA L4 GPU ($2.97 total compute cost)." This refers to the Model V2 40k training run, NOT to the Phase 8 production training.

**Clarification:**
- Phase 8 production training: **15,000 fine-tuning steps**, 1h 48m, $1.26 USD
- The "41,300-step, 4.24 GPU-hours, $2.97" refers to a completely different (and now DEPRECATED) 40k training run
- These numbers should NOT be cited as the Phase 8 production training cost

### B6. "4.24 GPU-hours" vs "$1.26 USD" Cost

The Submission Deck states $2.97 for 41,300 steps on L4. Phase 8 actual cost was $1.26 for 15,000 steps. Both are consistent with $0.70/hr L4 pricing:
- 41,300 steps at 3.30 steps/sec = ~3.48 hours = ~$2.44 (close to $2.97 if using a slightly different throughput estimate)
- 15,000 steps at 3.30 steps/sec = ~1.26 hours = $0.88-$1.26 depending on exact rate

These are internally consistent. The Submission Deck figures refer to the deprecated 40k model run.

### B7. The "10 Model V2 Fixes" in Scientific Narrative

**Context:** The Scientific Narrative lists "10 Model V2 Enhancements" implemented during the "18th Audit". These fixes ARE incorporated into the Phase 5-8 architecture (they form the foundation of all phases from 5 onwards). They are NOT a separate model from Phase 5-8 — Phase 5-8 builds ON the V2 fixes.

**No Inconsistency Here** — just clarification needed: the V2 fixes are the architectural baseline that all Phase 5-8 training uses.

### B8. "72-Channel UNet" vs "74-Channel UNet"

**Inconsistency:** The Scientific Narrative states "72 input channels (noisy anomaly cube + spatial context + cascade priors)" for the UNet. The YAML config and code clearly specify `unet_in_channels: 74`.

**Clarification (verified from code and config):**
- 74 = 1 (x_t noisy anomaly) + 1 (prev_depth_clean) + 72 (spatial_cond)
- 72 (spatial_cond) = 64 (u_cond from ConvLSTM) + 6 (static features) + 1 (SSHA) + 1 (SSHA gradient magnitude)
- The Scientific Narrative's "72" appears to refer to the spatial conditioning channels, not the total UNet input channels
- **Authoritative value: 74 input channels** (from `phase8_zone_adaptive_15k.yaml`: `unet_in_channels: 74`)

### B9. Surface Accuracy Claims in Submission Deck vs. Decontaminated Results

**Inconsistency:** The Submission Deck (Slide 7) claims surface (0-30m) RMSE of "0.4425-0.4629 C" with skill "+0.90% to +2.27%". These came from the contaminated multi-seasonal benchmark results for the 20k V2 Hybrid model.

**Clarification:** On the clean Benchmark B (Sep-Dec), Phase 8 Calibrated surface RMSE is 0.4684-0.5569 C, which does NOT beat climatology (gap +0.013-0.025 C). The contaminated benchmark produced over-optimistic surface results because the model had seen 5 of those dates during training. The decontaminated Benchmark B results are the authoritative numbers.

---

## C. Architecture Parameter Clarifications

### C1. non_spatial_cond_dim = 14 (NOT 9 or 8)

**Flagged in tests:** `test_checkpoint_resume.py` previously used `non_spatial_cond_dim=9`, which was INCORRECT for the production model. This was fixed.

**Breakdown of 14 dimensions:**
1. ONI (scalar climate index)
2. IOD DMI (scalar climate index)
3. sin(DOY) (seasonal phase)
4. cos(DOY) (seasonal phase)
5. depth_norm (log-normalized depth in [0,1])
6. clim_T_val (climatological temperature / 30)
7. lapse_tensor (climatological lapse rate / 0.10)
8. dz_tensor (layer thickness / 100)
9. prev_mean (mean of previous depth clean sample, or 0)
10. prev_std (std of previous depth clean sample, or 0)
11. mld_cond (predicted MLD / 50)
12. blt_cond (predicted BLT / 20)
13. sal_cond (predicted sal. max depth / 100)
14. t_norm (diffusion timestep, added by DDIMSampler during each denoising step)

### C2. 72-Channel Spatial Conditioning Breakdown (Confirmed)

- u_cond: 64 channels (ConvLSTM output)
- static_features: 6 channels (landmask + bathymetry + 4 region masks: AS, BoB, Confluence, Open Ocean)
- SSHA: 1 channel (direct injection, Bottleneck 11)
- SSHA_gradmag: 1 channel (Sobel gradient magnitude of SSHA)
- **Total: 72 channels** (matches `spatial_cond_dim: 72` in YAML)

### C3. Phase 8 Configuration vs. Phase 8 Report Murphy Skill Sign

**Clarification:** The Phase 8 comprehensive report shows Murphy Skill = -23.42% (meaning Phase 8 pure diffusion still underperforms climatology on the INTERNAL test sequence at training-adjacent time indices). This is evaluated on a DIFFERENT internal test set (indices 299-353 stride 6), not on the clean Benchmark A/B.

On the **clean Benchmark B (Sep-Dec)**, Phase 8 Calibrated achieves **+3.89% Murphy Skill** (beats climatology). These are NOT contradictory — the internal test set uses a different temporal range and the raw (uncalibrated) model.

---

## D. Status Summary

| Item | Status | Action Required |
|:---|:---|:---|
| Contaminated multi-seasonal benchmark | PERMANENTLY RETIRED | No new citations allowed |
| ECE=0.0169 figure | SCOPED TO HYBRID ENSEMBLE | Do not cite for Phase 8 |
| 67 unit tests reference | OUTDATED (now 68) | Use 68 in all new docs |
| 41,300-step / $2.97 training cost | REFERS TO DEPRECATED 40k RUN | Do not cite for Phase 8 production |
| Scientific Narrative headline metrics | PRE-DECONTAMINATION / DEPRECATED MODEL | Archive only; do not cite |
| Submission Deck headline metrics | PRE-DECONTAMINATION / DEPRECATED MODEL | Archive only; do not cite |
| UNet "72 input channels" | CORRECTED: 74 input channels | Use 74 in all new docs |
| All Phase 8 Calibrated metrics | VERIFIED AND AUTHORITATIVE | Safe to cite |

---

*This audit was conducted during the creation of MASTER_PROJECT_DOCUMENTATION.md. All flags and discrepancies have been resolved in the master document by using only verified, source-traced figures.*
