# PS26066 (OceanEmbed) — Phase 3 Detailed Spec: Model Implementation, Toy-Scale Verification & GCP Preparation
*Written for direct handoff to Antigravity. Builds on Phase 2's assembled training-ready dataset. This phase produces code correctness confidence, not a trained model — real training is Phase 4.*

---

## 1. Objective
Implement every model component defined in the Final Architecture Specification, verify it's *correct* (not yet *good*) using toy-scale data and standard ML-engineering sanity checks, and get the GCP environment fully prepared and verified — without spending real training compute yet. By the end of this phase: the code is trustworthy, and the GCP VM is a known-working, ready-to-go environment sitting stopped (and therefore costing nothing) until Phase 4 begins.

**Two separate tracks in this phase, sequenced deliberately:**
- **Track A (§2–5): model code**, developed and verified entirely on the laptop with dummy/toy data.
- **Track B (§6): GCP environment preparation**, done in parallel or after Track A — the VM should be provisioned, verified, and then *stopped*, not left running while Track A work continues.

---

## 2. Repository Structure Additions
```
src/
├── models/
│   ├── context_encoder.py     # Stage 1: ConvLSTM temporal-spatial encoder
│   ├── conditioning.py        # AdaGN/FiLM conditioning MLP + injection logic
│   ├── unet_denoiser.py       # Stage 3: U-Net noise-prediction backbone
│   ├── auxiliary_heads.py     # Stage 5: MLD / BLT / salinity-max heads
│   └── diffusion.py           # noise schedule, forward process, x0-prediction utilities
├── sampling/
│   ├── ddim_sampler.py        # Stage 4 mechanics: fast DDIM sampling
│   └── depth_cascade.py       # orchestrates sequential shallow→deep sampling
├── training/
│   ├── dataset.py             # PyTorch Dataset/DataLoader wrapping Phase 2 output
│   ├── losses.py              # L_diffusion (depth+region weighted), L_aux, L_physics, adaptive weighting
│   ├── train.py               # training loop entry point
│   └── checkpoint_utils.py    # save/resume, critical for Spot-instance interruption in Phase 4
└── verification/
    ├── overfit_tiny_batch.py  # the single most important test in this phase, see §5.2
    ├── shape_checks.py
    └── sampling_sanity.py

scripts/
├── run_toy_forward_pass.py
├── run_overfit_test.py
└── gcp/
    ├── setup_vm.sh            # provisioning script, see §6
    └── verify_gcp_env.py      # confirms GPU + environment parity with laptop

tests/
├── test_context_encoder_shapes.py
├── test_unet_denoiser_shapes.py
├── test_conditioning.py
├── test_losses.py
└── test_depth_cascade_order.py
```

---

## 3. Model Components — Exact Specification (pulled directly from the architecture spec, restated precisely so Antigravity implements the exact agreed design, not an approximation)

### 3.1 Context Encoder (`context_encoder.py`)
- Input: `(B, T=7, C=25, H=112, W=240)`.
- 3-layer ConvLSTM stack, channel widths `32 → 64 → 64`, `3×3` kernels, same-padding.
- Output: final hidden state `u_cond`, shape `(B, 64, H=112, W=240)`.

### 3.2 Conditioning Module (`conditioning.py`)
Two conditioning pathways, handled differently — **do not concatenate everything into one pile of channels; spatial and non-spatial conditioning are structurally different and should stay that way in code:**
- **Spatial conditioning** (concatenated as extra channels alongside `u_cond`): 4 region-membership maps + bathymetry + land mask = 70 total channels feeding the U-Net's input stage.
- **Non-spatial conditioning** (via AdaGN, injected at *every* U-Net resolution stage, not just once): log-normalized depth id, climatology value at that depth, ONI, IOD, day-of-year (sin/cos), and — except for the first/shallowest depth in a cascade sequence — the previous depth's clean output. All projected through a small shared MLP (e.g. 2 hidden layers, width 128) into per-stage scale-and-shift parameters.

### 3.3 U-Net Denoiser (`unet_denoiser.py`)
- 4 resolution stages. Encoder channel widths `32 → 64 → 128 → 256`.
- Each stage: two `3×3` conv blocks, each followed by GroupNorm + SiLU activation, with AdaGN conditioning injection (§3.2) applied at every block, not just stage boundaries.
- `2×2` downsampling between encoder stages; symmetric decoder with skip connections from corresponding encoder stages; bottleneck at 256 channels.
- Output: predicted noise `ε_θ`, same spatial shape as the (cropped, 100×240) target region.

### 3.4 Diffusion Mechanics (`diffusion.py`)
- Standard DDPM forward process: `x_τ = √(ᾱ_τ)·x_0 + √(1−ᾱ_τ)·ε`, with a cosine noise schedule (generally more stable than linear for this class of model — implement as the default, but keep it swappable).
- Provide both a noise-prediction interface and an **x̂₀-estimate reconstruction utility** (`predict_x0_from_noise`) — this is required by `L_physics` in §4, which operates on the model's clean-data estimate at each training step, not on a fully-sampled multi-step output.

### 3.5 Auxiliary Heads (`auxiliary_heads.py`)
- 3 heads (MLD, Bay-of-Bengal BLT, Arabian-Sea salinity-max depth+strength), each a 2-layer MLP on a global-average-pooled version of `u_cond`.
- Apply the region masks (already computed in Phase 1/2) when computing each head's loss — not when computing its prediction; the head always predicts a value everywhere, masking happens at the loss stage (§4), so the network isn't forced to predict garbage for masked-out regions during training in a way that destabilizes shared representations.

### 3.6 Sampling — DDIM + Depth Cascade (`sampling/`)
- `ddim_sampler.py`: standard DDIM sampling, ~50 steps by default (configurable), reused identically at every depth in the cascade.
- `depth_cascade.py`: orchestrates the fixed sampling order `0 → 5 → 10 → 20 → 30 → 50 → 75 → 100 → 125 → 150 → 200 → 300 → 500 → 700 → 1000m`, feeding each depth's clean sampled output forward as conditioning for the next. **This ordering must be enforced in code, not left to caller discretion** — provide a single `sample_full_profile()` function that's the only supported entry point for generating a complete depth profile, so it's impossible to accidentally call depths out of order.

---

## 4. Loss Function (`training/losses.py`)
Implement exactly as specified:
```
L_total = exp(−w1)·L_diffusion + w1
        + exp(−w2)·L_aux + w2
        + exp(−w3)·L_physics + w3
```
where `w1, w2, w3` are learnable `nn.Parameter` scalars (Pinn-Ocean-style adaptive weighting), initialized to 0.

- **`L_diffusion`**: standard noise-prediction MSE, multiplied by depth-and-region weight `α(d, region)`: baseline `1.0`; `1.5×` at 50–200m (all regions); an **additional** `1.3×` multiplier at 200–300m specifically where Arabian Sea region-membership > 0.5. Implement `α(d, region)` as its own small, clearly-named, independently-testable function — this exact weighting logic is one of our two original architectural contributions and deserves to be easy to point to and verify in code review, not buried inline.
- **`L_aux`**: sum of MSE losses across the 3 auxiliary heads, each masked to its relevant region mask before averaging.
- **`L_physics`**: thermocline-depth-consistency loss, computed on the model's predicted `x̂₀` (via `predict_x0_from_noise` from §3.4) at each training step — compares the depth of the temperature gradient's steepest point between predicted `x̂₀` and the true training target, **not** via a full multi-step sample (too expensive per training step).

---

## 5. Toy-Scale Verification — The Actual Point of This Phase

### 5.1 Forward/backward shape and stability checks (`verification/shape_checks.py`)
Feed dummy random tensors matching every expected shape from Phase 2's output through the full pipeline end to end (context encoder → conditioning → U-Net → single-depth noise prediction). Confirm:
- Every intermediate and final tensor shape matches expectations exactly.
- `loss.backward()` runs without error.
- No `NaN`/`Inf` appears in any gradient (check `torch.isfinite()` across all parameter gradients after backward).

### 5.2 The overfit-tiny-batch test (`verification/overfit_tiny_batch.py`) — the single most important check in this entire phase
Take a genuinely tiny set of real samples (2–4 samples, from Phase 2's toy-mode output) and train on **only these**, repeatedly, for many iterations (e.g. a few hundred steps). **The loss must drop to near-zero.** If it doesn't, something in the loss formula, gradient flow, or optimizer setup is wrong — and this is dramatically cheaper and faster to discover now, on 4 samples in a few minutes, than after hours or days of real training on real data reveal a model that never converges. This single test is standard, essential ML engineering practice and is not optional — do not proceed to Phase 4 without it passing cleanly.

### 5.3 Sampling loop verification (`verification/sampling_sanity.py`)
Using the model from the overfit test (§5.2), run the full DDIM + depth-cascade sampling loop end to end. Confirm:
- Output contains no `NaN`/`Inf`, and values fall in a physically plausible range once climatology is added back.
- The depth-cascade ordering is genuinely sequential and genuinely uses the previous depth's output (verify this directly — e.g. by checking that corrupting the shallow-depth output changes the deeper-depth predictions; if it doesn't, the conditioning isn't actually wired in correctly despite looking fine on paper).
- Running the full profile sample **N=10 times** with different random seeds produces outputs that vary sensibly — neither identical every time (a sign randomness isn't flowing through correctly) nor wildly, implausibly divergent (a sign of an unstable sampler).

---

## 6. GCP Environment Preparation (Track B)

This is where we set up GCP together, as planned — prepared and verified now, but **left stopped and unused until Phase 4 actually begins training.**

### Step-by-step
1. **Enable the Compute Engine API** in your GCP project console, if not already enabled.
2. **Check L4 GPU quota** in your target region: IAM & Admin → Quotas, filter by "NVIDIA L4 GPUs." If quota is 0, submit a quota increase request now — this can take time to approve, so do it at the start of this phase, not right before Phase 4.
3. **Provision the VM** (`scripts/gcp/setup_vm.sh`, a `gcloud` CLI script Antigravity should write and you run):
   - Machine type: `g2-standard-4` or `g2-standard-8` (1× L4 GPU attached).
   - Boot image: use a **Google Deep Learning VM image** with PyTorch and CUDA pre-installed — this saves significant manual driver/CUDA-version-matching setup versus a bare Ubuntu image.
   - Persistent disk: at minimum 200GB SSD, sized to comfortably hold the full Phase 2 dataset plus checkpoints with room to spare.
   - Attach the disk as persistent (not auto-delete-on-termination), so data/checkpoints survive VM stop/start cycles.
4. **Set a billing budget alert** in GCP Billing — thresholds at, e.g., ₹5,000 / ₹15,000 / ₹30,000 against your ₹40,000 credit, so you get warned well before running low.
5. **Configure SSH access** — either the `gcloud compute ssh` CLI command directly, or (recommended for a smoother coding experience) VS Code's Remote-SSH extension pointed at the VM's external IP, so the GCP machine feels like a normal remote dev environment.
6. **Sync the codebase** to the VM (`git clone` your repo, or `gcloud compute scp` if the repo isn't pushed anywhere yet) and install `requirements.txt` inside the VM's environment.
7. **Verify GPU visibility**: run `nvidia-smi` (confirms the driver sees the L4) and `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` (confirms PyTorch sees it too — these can disagree if CUDA versions mismatch, which is exactly why the Deep Learning VM image in step 3 is worth using rather than fighting driver installation by hand).
8. **Run `scripts/gcp/verify_gcp_env.py`** — this should be the same toy-scale forward/backward/overfit tests from §5, just run once on the GCP VM instead of the laptop, confirming the environment produces consistent results (not necessarily identical due to GPU non-determinism, but sane and stable) before we trust it for real training.
9. **Stop the VM** (`gcloud compute instances stop`, not just disconnecting) the moment step 8 passes. **This is the critical cost-control discipline we discussed — an idle running GPU VM is the most common way cloud budgets get wasted, and there's no reason for it to run at all between now and the start of Phase 4.**

---

## 7. Tests (`tests/`)
- `test_context_encoder_shapes.py`, `test_unet_denoiser_shapes.py`: confirm every module's input/output shapes match §3 exactly across a range of dummy batch sizes.
- `test_conditioning.py`: confirm AdaGN injection actually changes the output (i.e. passing different conditioning vectors produces different U-Net outputs for the same input — catches a conditioning pathway being silently disconnected).
- `test_losses.py`: confirm `α(d, region)` returns the exact expected weighting values at known (depth, region) combinations (e.g. assert it returns `1.5 × 1.3` at 250m within Arabian Sea membership).
- `test_depth_cascade_order.py`: confirm `sample_full_profile()` genuinely calls depths in the fixed correct order and genuinely passes each depth's output forward (not just that it runs without erroring).

---

## 8. Acceptance Criteria — Phase 3 is "done" when:
1. All shape/stability checks in §5.1 pass.
2. The overfit-tiny-batch test (§5.2) drives loss to near-zero on a handful of real toy samples — **non-negotiable, do not proceed to Phase 4 without this passing.**
3. The sampling sanity checks in §5.3 pass, including the "corrupt the shallow depth, confirm downstream depths change" check.
4. The GCP VM is provisioned, verified working (§6 step 8 passes), and **stopped**.
5. All tests in `tests/` pass.
6. `README.md` is updated with: how to run the toy/overfit verification suite locally, and how to start/stop/connect to the GCP VM.

## 9. What NOT to Do Yet
- No training on real, full-scale data — that's Phase 4, and it's the entire reason this phase exists first (catch bugs cheap, not expensive).
- Don't leave the GCP VM running "just in case" — stop it the moment §6 is verified.
- No hyperparameter tuning, no ablation studies (e.g. the region-conditioning ablation we planned) — those belong in Phase 4 once we know the code itself is correct.

---

**Next step once Phase 3 is verified complete:** Phase 4's spec — real training on GCP, including the checkpointing/resume strategy for Spot instances, monitoring setup, and the first honest validation numbers.
