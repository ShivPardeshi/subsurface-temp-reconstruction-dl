# PS26066 (OceanEmbed) — Phase 5 Detailed Spec: Full Evaluation Suite
*Written for direct handoff to Antigravity. Do not run this phase's evaluation scripts against anything other than a real Stage B (or C/D) checkpoint — the Stage A pilot checkpoint is a de-risking artifact, not a model worth reporting on.*

---

## 1. Objective
Produce the complete, honest accuracy picture this entire project has been built around — not a single RMSE number, but every metric and every slice we committed to across the research phase: standard metrics, skill-vs-baseline, structural/spectral fidelity, physical consistency, uncertainty calibration, all sliced across the seven priority zones, plus the honest ablation comparison (Stage B vs. C vs. D) and contextualization against real published/operational benchmarks. This phase's output is the actual evidence base for every claim the PPT will make.

---

## 2. Repository Structure Additions
```
src/
├── evaluation/
│   ├── metrics/
│   │   ├── basic_metrics.py        # RMSE, correlation, bias, R²
│   │   ├── skill_score.py          # Murphy skill score vs. climatology baseline
│   │   ├── ssim_metric.py          # structural similarity, per depth
│   │   ├── spectral_analysis.py    # Fourier spectral comparison
│   │   ├── heat_flux_consistency.py # meridional transport physical-consistency check
│   │   └── calibration.py          # reliability diagrams from ensemble spread
│   ├── slicing/
│   │   ├── priority_zones.py       # defines all 7 zone masks/time-windows
│   │   └── zone_evaluator.py       # applies any metric, sliced by zone
│   ├── auxiliary_head_eval.py      # accuracy of MLD / BLT / salinity-max predictions
│   ├── validation_independence/
│   │   └── rama_assimilation_check.py  # investigates & documents RAMA-vs-GLORYS status
│   ├── ablation_comparison.py      # Stage B vs. C vs. D, identical eval set
│   ├── benchmark_comparison.py     # contextualizes against ARMOR3D/isQG/CGKDN/TS-Cast
│   └── generate_report.py          # orchestrates everything into the final report
scripts/
├── run_full_evaluation.py
└── run_ablation_comparison.py
tests/
├── test_metrics_correctness.py     # each metric checked against known synthetic cases
└── test_priority_zone_masks.py
```

---

## 3. Standard Metrics (`metrics/basic_metrics.py`, `skill_score.py`)
- **RMSE, correlation, bias** — per depth, computed against both (a) held-out GLORYS test-period data and (b) real independent ARGO, kept as two clearly separate reported numbers, never merged into one — this distinction has mattered every time it's come up in our research (sg721642's 0.49°C vs. 1.90°C gap is the whole reason to keep these separate).
- **R²** — per depth, against the same climatology baseline used for the anomaly target in Phase 2. This is non-negotiable per our own research finding: KaviBharathi643's real results showed small raw RMSE masking negative R² below 300m, and we committed explicitly to not making that same mistake.
- **Murphy skill score** — `1 − (MSE_model / MSE_climatology)` — the rigorous "are we actually better than just guessing the climatological average" check that several real operational products (and most of the 17 repos) never computed. Positive means genuine skill beyond climatology; negative means the model is doing worse than the naive baseline at that depth/region/season.

## 4. Structural & Spectral Fidelity (`ssim_metric.py`, `spectral_analysis.py`)
Per Asefi et al.'s methodology — these exist specifically to catch a failure mode RMSE can miss:
- **SSIM** per depth, comparing predicted vs. true spatial fields — catches structural mismatch even when pointwise error looks acceptable.
- **Fourier spectral comparison** per depth — compute the 2D power spectrum of predicted vs. true fields, compare energy distribution across wavenumbers. A model that's quietly oversmoothing (reproducing large-scale patterns while losing fine-scale detail) will show a spectrum that drops off too fast at high wavenumbers relative to the truth — report this as a specific, visualizable diagnostic, not just a pass/fail number.

## 5. Physical Consistency Check (`heat_flux_consistency.py`)
Our model predicts temperature only (per the PS's actual requirement), not velocity — so this check uses the **real, input-derived current field** (geostrophic + ageostrophic, already computed in Phase 2) combined with the **predicted** temperature field, rather than requiring the model to jointly generate currents:
```
q_v = ρ · c_p · V_input · T_predicted
```
Compare this against the same quantity computed with `T_true` instead of `T_predicted`. This is still a legitimate, meaningful physical-plausibility check — it verifies the predicted temperature field is *consistent* with the real observed circulation, which a per-pixel-only RMSE check has no way of catching.

## 6. Uncertainty Calibration (`calibration.py`)
Using the N=10–20 sample ensemble from the DDIM+depth-cascade sampling (per the architecture's Stage 4/5), build **reliability diagrams**: bucket predictions by their stated confidence level (e.g. the ensemble's 80% interval), and check what fraction of real ARGO observations actually fall inside that interval. A well-calibrated model's 80% interval should contain the true value roughly 80% of the time — report the actual observed rate at several confidence levels, per depth, and flag whether the model is over-confident (intervals too narrow) or under-confident (too wide). This is the check that several of the 17 repos predicted uncertainty for but never verified — do not skip it.

## 7. Auxiliary Head Evaluation (`auxiliary_head_eval.py`)
Separately report accuracy for each of the three Stage 5 auxiliary predictions against their GLORYS-derived proxy truth:
- Mixed layer depth: MAE, correlation.
- Bay of Bengal barrier layer thickness: MAE, correlation, computed only within Bay of Bengal region-membership.
- Arabian Sea salinity-maximum depth and strength: MAE, correlation, computed only within Arabian Sea region-membership.

## 8. Priority Zone Slicing (`slicing/priority_zones.py`, `zone_evaluator.py`)
Every metric above must be computable sliced by each of the **seven priority zones** established across our planning, not just reported globally:
1. Bay of Bengal mixed-layer/barrier-layer zone (0–30m)
2. Thermocline core, both basins (75–150m)
3. Arabian Sea Persian-Gulf-Water zone (200–300m)
4. The 8–10°N confluence zone
5. Extreme-event windows (cyclone dates, from the downloaded IBTrACS North Indian basin data)
6. Monsoon transition windows (onset/withdrawal dates, from published IMD dates)
7. The equatorial domain edge (~5°N boundary region)

`zone_evaluator.py` should be a single generic function — `evaluate_metric_by_zone(metric_fn, predictions, truth, zone_masks) -> dict` — applied identically across every metric in §3–6, rather than writing bespoke slicing logic per metric. This guarantees consistency and makes it trivial to add an eighth zone later if needed.

## 9. Validation Independence Investigation (`validation_independence/rama_assimilation_check.py`)
Resolve the open question flagged since our TS-Cast research: **is RAMA moored buoy data assimilated into GLORYS?** Check Copernicus Marine's official GLORYS12v1 product documentation for its listed assimilated observation sources directly — this is a documentation-reading task, not an experiment. Document the finding explicitly in the evaluation report:
- If RAMA is *not* assimilated: use it as the genuinely independent validation set, exactly as TS-Cast used PIES sensors, and report those numbers as our strongest independence claim.
- If RAMA *is* assimilated (or documentation is ambiguous): state this plainly in the report rather than silently treating ARGO validation as fully independent — this honesty is itself the point, consistent with every prior document in this project.

## 10. Ablation Comparison (`ablation_comparison.py`)
Run the full §3–8 evaluation suite identically against **all three** checkpoints — Stage B (baseline), Stage C (no region-conditioning), Stage D (no depth-cascade) — and present them side by side in one comparison table per metric/zone. Report the real deltas honestly:
- If Stage C ≈ Stage B: region-conditioning isn't earning its complexity — say so.
- If Stage D ≈ Stage B: depth-cascade (our own original contribution) isn't earning its complexity — say so, and this would be a genuinely important, honest finding to report even though it's our own idea.
- Whatever the actual results are, this table is likely the single most scientifically credible piece of the entire submission, because so few teams in our research (zero, in fact) ever ran a real ablation and reported the honest result.

## 11. Benchmark Contextualization (`benchmark_comparison.py`)
Place our own numbers alongside the real reference points established during research, so accuracy claims aren't made in a vacuum:
- ARMOR3D's documented thermocline-zone RMSE behavior.
- ISRO's isQG limitations (6-monthly updates, Bay-of-Bengal-only, 10m vertical resolution).
- CGKDN's comparative table (CGKDN 0.590°C, IAP 0.598°C, ORAS5 0.690°C, DORS 0.723°C).
- TS-Cast's <1°C/<0.1psu upper-500m result and its independent-validation performance.

## 12. Final Report Generation (`generate_report.py`)
Orchestrates everything above into a single structured output (reuse the docx skill/pattern from our earlier competitive-analysis report if a formal document is wanted at this stage, or a clean markdown report otherwise) containing:
- Global metrics (all of §3–7).
- The full 7-zone breakdown.
- The ablation comparison table (§10).
- The benchmark contextualization (§11).
- The validation-independence finding (§9), stated plainly.
- All supporting plots: SSIM/spectral diagnostic plots, reliability diagrams, per-zone RMSE bar charts.

---

## 13. Tests (`tests/`)
- `test_metrics_correctness.py`: verify RMSE/correlation/bias/R²/skill-score against hand-constructed synthetic cases with known correct answers (e.g. a synthetic case where predictions exactly equal climatology should yield a skill score of exactly zero).
- `test_priority_zone_masks.py`: confirm each of the 7 zone masks/time-windows actually selects the expected grid cells/dates (e.g. the equatorial-edge mask shouldn't accidentally include cells at 20°N).

---

## 14. Acceptance Criteria — Phase 5 is "done" when:
1. All metrics in §3–7 are computed and reported for at least the Stage B baseline checkpoint, against both GLORYS and real ARGO.
2. All 7 priority-zone slices are computed for every metric, not just globally.
3. The RAMA assimilation-status investigation (§9) has a documented finding, whatever it turns out to be.
4. The Stage B/C/D ablation comparison (§10) is complete and reported honestly.
5. The benchmark contextualization (§11) is included.
6. `generate_report.py` produces one complete, coherent final report incorporating all of the above.
7. All tests in §13 pass.

## 15. What NOT to Do Yet
- No downstream disaster-product computation (OHC/TCHP/marine-heatwave flag) — that's Phase 6, and it consumes this phase's validated model, not the other way around.
- No PPT/presentation work — this phase produces the evidence base the PPT will later draw from, not the PPT itself.

---

**Next step once Phase 5 is complete:** Phase 6's spec — the downstream disaster-product layer (Ocean Heat Content, Tropical Cyclone Heat Potential via the verified chilli-garlic-momo formula, mixed layer depth cross-check, marine heatwave flagging) and the demo/dashboard that presents it.
