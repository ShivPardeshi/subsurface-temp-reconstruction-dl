# PS26066 (OceanEmbed) — Phase 4 Detailed Spec: Real Training on GCP
*Written for direct handoff to Antigravity. Do not begin this phase until Phase 3's acceptance criteria (especially the overfit-tiny-batch test) are fully met — this phase spends real GCP budget, and every prior phase existed specifically to de-risk this one.*

---

## 1. Objective
Train the full architecture on real, full-scale data, honestly — including running the region-conditioning and depth-cascade ablations we committed to, not just a single "does it work" run. Produce: trained checkpoints, complete training logs, and enough lightweight validation to select the best checkpoint. **Full rigorous evaluation (SSIM, spectral analysis, calibration, priority-zone slicing) is Phase 5, not this phase** — Phase 4's validation is just enough to make good decisions during training (is it converging, which checkpoint to keep), not the final honest reporting.

**This is the highest-risk, highest-cost phase in the whole project — treat it with the corresponding discipline.** Everything in §5 (staged execution) exists to avoid discovering a fundamental problem only after most of the budget is spent.

---

## 2. Repository Structure Additions
```
src/
├── training/
│   ├── train.py                    # extended: real dataset, resume-from-checkpoint, full logging
│   ├── config_registry/
│   │   ├── baseline_config.yaml    # full architecture, region-conditioning ON, depth-cascade ON
│   │   ├── no_region_ablation.yaml # region-conditioning OFF, everything else identical
│   │   └── no_cascade_ablation.yaml # depth-cascade OFF (independent per-depth, Asefi-style), everything else identical
│   ├── monitoring.py                # logs loss components, learnable weights, sample previews
│   └── budget_tracker.py            # logs cumulative GPU-hours and estimated cost per run
scripts/
├── gcp/
│   ├── resume_vm.sh                 # restart the Phase-3-stopped VM, verify environment intact
│   └── sync_full_dataset.sh         # transfer Phase 2's complete dataset to the GCP persistent disk
├── run_pilot_training.py            # Stage A: small-scale real-data de-risking run
└── run_full_training.py             # Stages B–D: full/ablation runs, config-selectable
tests/
└── test_checkpoint_resume.py        # verifies training can be killed and resumed without corruption
```

---

## 3. Pre-Flight: Resuming the GCP Environment
1. `gcloud compute instances start <instance-name>` — restart the VM Phase 3 left stopped.
2. Re-verify GPU visibility (`nvidia-smi`, the `torch.cuda.is_available()` check from Phase 3 §6) — confirm nothing changed since it was stopped.
3. Pull any code changes made locally since Phase 3 (`git pull`, or re-sync if the repo isn't hosted remotely).
4. Re-run the Phase 3 overfit-tiny-batch test **on the GCP VM specifically**, one more time, before touching real data — this confirms the environment itself hasn't drifted (driver updates, dependency changes) since it was last verified.

## 4. Data Transfer
Run `scripts/gcp/sync_full_dataset.sh` to move Phase 2's complete, full-history Zarr dataset (not just toy-mode output) onto the GCP persistent disk attached in Phase 3. Verify the transferred dataset's shape/checksum matches the laptop-side original before proceeding — a silently truncated or corrupted transfer here would be a very expensive thing to discover partway through a training run.

---

## 5. Staged Execution Plan — Do Not Skip Stage A

### Stage A — Small-scale real-data pilot run (`run_pilot_training.py`)
Before committing to the full multi-year, full-domain training run, run a short, cheap pilot: **real data** (not dummy/toy anymore), but a reduced scope — e.g. one full year, or a single sub-region (Bay of Bengal only), for a limited number of steps. Goals:
- Confirm the loss curve behaves the way the overfit test predicted it should — steadily decreasing, no divergence, no NaNs — now at real scale, with real data variability (which behaves differently than a 4-sample overfit).
- Confirm real GPU memory usage is within the L4's 24GB budget at your chosen batch size — if you hit out-of-memory here, better to discover it in a 30-minute pilot than 10 hours into the full run.
- Get a **real, small-scale GPU-hours-per-step measurement** — this is what lets you replace our earlier rough 80–250 GPU-hour estimate with an actual number specific to your setup, and re-forecast your GCP credit usage accordingly before committing further spend.
- **Do not proceed to Stage B until Stage A's loss curve looks healthy and the memory footprint is confirmed safe.**

### Stage B — Full baseline training run
The complete architecture as specified: region-conditioning on, depth-cascade on, full loss (diffusion + auxiliary + physics, adaptively weighted), full historical training-period data, full domain. Use `baseline_config.yaml`.

### Stage C — Region-conditioning ablation
Identical to Stage B in every other respect, but with region-conditioning channels removed (`no_region_ablation.yaml`). This is the honest ablation we committed to back in the architecture-planning discussion — **run it and report the real delta, whatever it turns out to be**, rather than assuming region-conditioning helps just because it's architecturally present.

### Stage D — Depth-cascade ablation
Identical to Stage B, but with independent per-depth sampling (no cascade conditioning — each depth sampled the way Asefi et al.'s original method does it) via `no_cascade_ablation.yaml`. This is the direct, honest test of whether our own original architectural contribution actually earns its complexity — run it, and be prepared to report honestly if the improvement is smaller than hoped, consistent with the standard of honesty this whole project has been built on.

**Sequencing note:** Stages C and D can run in either order, or in parallel if budget/quota allows two VMs briefly — they don't depend on each other, only on Stage B's baseline being complete for comparison.

---

## 6. Training Configuration
- **Batch size**: determined empirically in Stage A given the L4's 24GB — start conservative (e.g. batch size 4–8) and increase if memory allows, using gradient accumulation to simulate a larger effective batch size if needed.
- **Mixed precision (bf16 preferred over fp16 if the L4/driver stack supports it)** — meaningfully reduces memory pressure and speeds up training with minimal accuracy risk.
- **Gradient checkpointing** on the U-Net's deeper stages if memory remains tight after Stage A's findings — trades compute for memory, appropriate given we're on a single mid-tier GPU, not a cluster.
- **Diffusion timesteps**: train with the standard `T=1000` step noise schedule (cosine schedule, per Phase 3's `diffusion.py`); this doesn't conflict with using DDIM's ~50-step fast sampling at inference — training and sampling step counts are independent.
- **Optimizer**: AdamW, with a learning rate warmup period followed by cosine decay — standard, safe defaults for this model class; avoid exotic optimizer choices this late in the project.
- **Chronological split enforcement**: the dataset loader must draw training batches *only* from the designated training-period years — re-verify this explicitly at the start of this phase (not just trust Phase 2's climatology-fit assertion, which covered a different part of the pipeline).

## 7. Checkpointing & Spot-Instance Resume Strategy
Given the cost-efficiency case for Spot/preemptible instances (from our compute-budget planning), **training must be resumable from an interruption at any point**, not just at clean stopping points:
- Save a full checkpoint (model weights, optimizer state, current step count, the three learnable loss weights `w1/w2/w3`) **every N steps** (e.g. every 500–1000 steps, tuned so a preemption never loses more than a small fraction of an hour's work).
- `train.py` must detect an existing checkpoint on startup and resume from it automatically by default — this should be the normal way training is (re-)launched, not a special-case flag remembered only when needed.
- `tests/test_checkpoint_resume.py`: explicitly test that killing a training process mid-run and restarting it produces a continuous, consistent loss trajectory (no unexplained jump or reset) — verify this on a short run before trusting it for the real, multi-hour Stage B/C/D runs.

## 8. Monitoring & Logging (`training/monitoring.py`)
Track and log, at regular intervals throughout training:
- All individual loss components (`L_diffusion`, `L_aux`, `L_physics`) separately, not just the combined total — a single combined number can hide one component quietly failing while another compensates.
- The three learnable weights `w1, w2, w3` over time — watching how the model itself chooses to balance the loss terms is informative and a useful debugging signal if one term collapses to near-zero influence unexpectedly.
- **Periodic sample-generation previews**: every so often (e.g. every few thousand steps), run the actual DDIM + depth-cascade sampling loop on a fixed held-out validation sample and save a quick-look plot of the resulting depth profile against the true profile. This catches a model that's minimizing the training loss number while producing visibly wrong or degenerate profiles — a real risk with generative models that a loss curve alone won't always reveal.
- GPU utilization and elapsed wall-clock time, feeding directly into `budget_tracker.py`.

## 9. Budget Tracking (`training/budget_tracker.py`)
Log cumulative GPU-hours and estimated cost (against the L4's known hourly rate) after every training run, updated in a simple running log file. Cross-check this against your GCP billing console periodically (not just this internal estimate) and against the budget alert thresholds set up in Phase 3 — the internal tracker is a convenience for quick decisions, the GCP billing console is the source of truth for actual spend.

## 10. Lightweight Validation During Training
Not the full Phase 5 evaluation suite — just enough to make good decisions now:
- Basic RMSE and correlation against the held-out validation-period data, computed periodically (e.g. every epoch or every few thousand steps).
- **Checkpoint selection**: keep the checkpoint with the best validation RMSE, not simply the final/last checkpoint — training loss and validation performance can diverge, and this is a standard, important practice worth enforcing explicitly rather than assuming "latest is best."

---

## 11. Acceptance Criteria — Phase 4 is "done" when:
1. Stage A's pilot run completed with a healthy loss curve and a confirmed-safe memory footprint, and produced a real GPU-hours-per-step measurement.
2. Stage B (full baseline) completed, with checkpoints saved, monitored, and the best-validation-RMSE checkpoint identified.
3. Stage C (region-conditioning ablation) and Stage D (depth-cascade ablation) both completed, with their validation RMSE honestly compared against Stage B's baseline — whatever the result.
4. `test_checkpoint_resume.py` passes, confirming the training pipeline is genuinely interruption-safe.
5. Cumulative GPU-hours and cost are logged and cross-checked against actual GCP billing.
6. `README.md` documents: how to launch each stage, how training resumes automatically after interruption, and where checkpoints/logs are stored.

## 12. What NOT to Do Yet
- No full evaluation suite (SSIM, Fourier spectral analysis, heat-flux consistency, uncertainty calibration, priority-zone slicing) — that's Phase 5, deliberately kept separate so this phase's scope stays "get a trained, validated-enough-to-trust model," not "produce the final honest report."
- No downstream product layer (OHC/TCHP/marine-heatwave) — Phase 6.
- Don't leave the GCP VM running between stages if there's a natural pause (e.g. waiting on your own review of Stage A's results before launching Stage B) — stop it, resume it deliberately, exactly as disciplined as Phase 3's habit.

---

**Next step once Phase 4 is verified complete:** Phase 5's spec — the full, rigorous evaluation suite (SSIM, spectral analysis, heat-flux physical-consistency diagnostic, uncertainty calibration, and slicing every result across all seven priority zones), producing the honest, complete accuracy picture this whole project has been built around delivering.
