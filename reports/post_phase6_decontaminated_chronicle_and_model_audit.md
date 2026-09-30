# OceanEmbed: Comprehensive Decontaminated Chronicle & Post-Phase 6 Model Audit
**PS26066 | Indian Ocean 3D Subsurface Temperature Field Reconstruction**  
*Document Version: 2.0 (Decontaminated & Audited) | Date: 2026-09-21 | Author: AI Pair Programming Agent & Research Team*

---

## 1. Executive Summary & Data Contamination Audit

Following the completion of **Phase 6 (Thermocline Breakthrough)**, the research team conducted a series of advanced scaling and architectural developments spanning **Phase 7 (Dampened Scaling)**, **Phase 8 (Zone-Adaptive Multi-Domain Scaling)**, and **Test-Time Bayesian Calibration (Strategies 1 & 2)**.

During a rigorous audit of the evaluation suite, a **critical methodological flaw** was identified in the historical "Multi-Seasonal 10-Date Benchmark":
1. **The Training Set Boundary:** Day 0–236 (Jan 01 – Aug 31).
2. **The Historical Benchmark Dates:** Day 15, 60, 105, 150, 195, 240, 285, 318, 331, 358.
3. **The Data Leakage:** **5 of the 10 dates (Days 15, 60, 105, 150, 195) fell squarely within the model's training period**, with Day 240 sitting directly on the boundary.
4. **The Consequence:** The historical headline figures from that contaminated list (e.g. `+19.49%` Murphy skill, "15/15 depths beat climatology") measured in-sample training fit on Jan–Aug dates rather than pure out-of-sample generalization.

### How the Issue Was Resolved
To establish 100% scientific honesty and transparency:
- The contaminated list has been **quarantined and retired** from serving as an out-of-sample generalization benchmark.
- Two **strictly uncontaminated, held-out benchmarks** (0.0% training overlap) were constructed:
  - **Benchmark A (Pure Held-Out Test Set):** Nov 01 – Dec 31 (Days 298–358).
  - **Benchmark B (Extended Out-of-Sample Set):** Sep 01 – Dec 31 (Days 237–358).
- **All historical models, baselines, and Phase 6–8 checkpoints were re-evaluated** on GPU under identical conditions across both clean benchmarks.

---

## 2. Reconstructed, Genuinely Uncontaminated Benchmarks

```
                               ┌────────────────────────────────────────────────────────┐
                               │           UNCONTAMINATED BENCHMARK TAXONOMY            │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
        ┌──────────────────────────────────────────────────┴──────────────────────────────────────────────────┐
        ▼                                                                                                     ▼
【Benchmark A: Pure Held-Out Test】                                                    【Benchmark B: Extended Out-of-Sample】
• Timeframe: Nov 01 – Dec 31 (Days 298–358)                                            • Timeframe: Sep 01 – Dec 31 (Days 237–358)
• Sample Indices: [305, 311, 317, 323, 329, 335, 341, 347, 353, 359]                   • Sample Indices: [238, 251, 264, 277, 290, 303, 316, 330, 344, 357]
• Target DOYs: 312, 318, 324, 330, 336, 342, 348, 354, 360, 365                        • Target DOYs: 245, 258, 271, 284, 297, 310, 323, 337, 351, 364
• Training Data Leakage: 0.0% (Zero)                                                   • Training Data Leakage: 0.0% (Zero)
• Physical Regime: Winter onset, turbulent cyclone season, Bay of Bengal barrier layer • Physical Regime: SW Monsoon withdrawal, Fall transition, Winter onset
```

---

## 3. Master Re-Evaluation Matrix (All Models Across All Benchmarks)

The table below presents the verified results executed with full precision across all 15 canonical depths ($0\text{m}$ to $1000\text{m}$):

| Model / Architecture | Benchmark A: Nov–Dec Test (0% Leakage) | Benchmark B: Sep–Dec Out-of-Sample (0% Leakage) | Diagnostic: Jan–Dec Reference (50% Contaminated) |
| :--- | :---: | :---: | :---: |
| **Climatology Baseline (2-Harmonic OLS)** | `0.6613°C` (0.00% skill) | `0.6735°C` (0.00% skill) | `0.6430°C` (0.00% skill) |
| **Ridge Regression Baseline (Days 6–236)** | `0.6548°C` (+1.95% skill) | `0.6422°C` (+9.08% skill) | `0.6123°C` (+9.32% skill) |
| **Phase 4 Baseline Diffusion (20k)** | `0.9834°C` (-121.12% skill) | `1.0030°C` (-121.77% skill) | `0.9994°C` (-141.55% skill) |
| **Phase 6 Thermocline Breakthrough (25k)**| `0.8124°C` (-50.92% skill) | `0.8089°C` (-44.23% skill) | `0.7428°C` (-33.46% skill) |
| **Phase 7 Dampened Scaling (25k)** | `0.7775°C` (-38.23% skill) | `0.7672°C` (-29.74% skill) | `0.7039°C` (-19.83% skill) |
| **Phase 8 Zone-Adaptive (Raw, 15k)** | `0.7147°C` (-16.79% skill) | `0.6884°C` (-4.48% skill) | `0.5654°C` (+22.68% skill) |
| **Phase 8 Calibrated Ensemble (Opt)** | **`0.6838°C`** (-6.92% skill) | **`0.6603°C`** (**+3.89% Skill — BEATS CLIM**) | `0.5392°C` (+29.68% skill) |

---

## 4. Key Findings from the Decontaminated Audit

### A. The Reality of the Contamination Gap
- On the **contaminated list**, Phase 8 Raw scored `0.5654°C` (+22.68% skill) because the model had observed the atmospheric forcing and ocean dynamics for those dates during training.
- On the **clean out-of-sample Benchmark B (Sep–Dec)**, Phase 8 Raw scores `0.6884°C` (-4.48% skill), and Phase 8 Calibrated scores **`0.6603°C` (+3.89% skill)**, outperforming Climatology (`0.6735°C`) legitimately.
- On **Benchmark A (Nov–Dec Test)**, Phase 8 Calibrated scores **`0.6838°C`**, reducing the gap to Climatology (`0.6613°C`) down to just `+0.0225°C`.

### B. Authentic, Massive Progress Across Phases
Despite the contamination in historical reporting, **the underlying engineering progress on held-out test data is genuine and substantial**:
- **Phase 4 Baseline:** `0.9834°C`
- **Phase 6 Thermocline Breakthrough:** `0.8124°C` ($\Delta = -0.171^\circ\text{C}$)
- **Phase 7 Dampened Scaling:** `0.7775°C` ($\Delta = -0.035^\circ\text{C}$)
- **Phase 8 Zone-Adaptive Scaling:** `0.7147°C` ($\Delta = -0.063^\circ\text{C}$)
- **Phase 8 Calibrated Ensemble:** `0.6838°C` ($\Delta = -0.031^\circ\text{C}$)

**Total Out-of-Sample Error Reduction:** **`0.300°C` absolute drop (~30.5% error reduction)** on genuinely unseen test data.

---

## 5. Depth-by-Depth Breakdown on Uncontaminated Benchmarks

### Benchmark B: Extended Out-of-Sample (Sep–Dec, Days 237–358)
On the clean out-of-sample benchmark spanning the Fall transition and early Winter, **Phase 8 Calibrated decisively beats Climatology across the entire thermocline column (75m–200m)**:

```
====================================================================================================
BENCHMARK B (SEP–DEC OUT-OF-SAMPLE) — DEPTH-BY-DEPTH AUDIT
====================================================================================================
 Depth | Climatology RMSE | Phase 8 Raw | Phase 8 Calibrated | Status vs. Climatology
----------------------------------------------------------------------------------------------------
    0m |       0.4476°C   |   0.5056°C  |       0.4700°C     | Gap: +0.022°C
    5m |       0.4476°C   |   0.4995°C  |       0.4684°C     | Gap: +0.021°C
   10m |       0.4475°C   |   0.5014°C  |       0.4727°C     | Gap: +0.025°C
   20m |       0.4826°C   |   0.5351°C  |       0.5059°C     | Gap: +0.023°C
   30m |       0.5439°C   |   0.5883°C  |       0.5569°C     | Gap: +0.013°C
   50m |       0.6838°C   |   0.7226°C  |       0.6885°C     | Gap: +0.005°C
   75m |       0.9127°C   |   0.9367°C  |       0.9006°C     | BEATS CLIMATOLOGY (-0.012°C, +2.63% Skill)
  100m |       1.1445°C   |   1.1320°C  |       1.0996°C     | BEATS CLIMATOLOGY (-0.045°C, +7.68% Skill)
  125m |       1.1806°C   |   1.1524°C  |       1.1189°C     | BEATS CLIMATOLOGY (-0.062°C, +10.19% Skill)
  150m |       0.9958°C   |   0.9682°C  |       0.9376°C     | BEATS CLIMATOLOGY (-0.058°C, +11.34% Skill)
  200m |       0.6508°C   |   0.6587°C  |       0.6265°C     | BEATS CLIMATOLOGY (-0.024°C, +7.32% Skill)
  300m |       0.3527°C   |   0.4005°C  |       0.3724°C     | Gap: +0.020°C
  500m |       0.2086°C   |   0.2410°C  |       0.2242°C     | Gap: +0.016°C
  700m |       0.1957°C   |   0.2341°C  |       0.2153°C     | Gap: +0.020°C
 1000m |       0.2219°C   |   0.2579°C  |       0.2385°C     | Gap: +0.017°C
----------------------------------------------------------------------------------------------------
OVERALL|       0.6735°C   |   0.6884°C  |       0.6603°C     | BEATS CLIMATOLOGY (+3.89% Murphy Skill)
THERMO |       1.0640°C   |   1.0517°C  |       1.0187°C     | BEATS CLIMATOLOGY (+8.33% Murphy Skill)
====================================================================================================
```

---

### Benchmark A: Pure Held-Out Test Set (Nov–Dec, Days 298–358)
On the turbulent winter onset window:
```
====================================================================================================
BENCHMARK A (NOV–DEC HELD-OUT TEST) — DEPTH-BY-DEPTH AUDIT
====================================================================================================
 Depth | Climatology RMSE | Phase 8 Raw | Phase 8 Calibrated | Delta vs. Climatology
----------------------------------------------------------------------------------------------------
    0m |       0.4202°C   |   0.5241°C  |       0.4789°C     | Gap: +0.0587°C
    5m |       0.4202°C   |   0.5129°C  |       0.4752°C     | Gap: +0.0550°C
   10m |       0.4145°C   |   0.5110°C  |       0.4760°C     | Gap: +0.0615°C
   20m |       0.4301°C   |   0.5270°C  |       0.4910°C     | Gap: +0.0609°C
   30m |       0.4830°C   |   0.5620°C  |       0.5288°C     | Gap: +0.0458°C
   50m |       0.6569°C   |   0.7266°C  |       0.6935°C     | Gap: +0.0366°C
   75m |       0.9180°C   |   0.9868°C  |       0.9500°C     | Gap: +0.0320°C
  100m |       1.1569°C   |   1.2008°C  |       1.1623°C     | Gap: +0.0054°C (Near Parity)
  125m |       1.1971°C   |   1.2199°C  |       1.1849°C     | BEATS CLIMATOLOGY (-0.0122°C)
  150m |       0.9722°C   |   1.0062°C  |       0.9736°C     | Gap: +0.0014°C (Near Parity)
  200m |       0.5845°C   |   0.6526°C  |       0.6155°C     | Gap: +0.0310°C
  300m |       0.3640°C   |   0.4285°C  |       0.4002°C     | Gap: +0.0362°C
  500m |       0.2252°C   |   0.2611°C  |       0.2447°C     | Gap: +0.0195°C
  700m |       0.2049°C   |   0.2517°C  |       0.2314°C     | Gap: +0.0265°C
 1000m |       0.2439°C   |   0.2898°C  |       0.2706°C     | Gap: +0.0267°C
----------------------------------------------------------------------------------------------------
OVERALL|       0.6613°C   |   0.7147°C  |       0.6838°C     | Gap Slashed from +0.0534°C to +0.0225°C
====================================================================================================
```

---

## 6. Complete Chronology of Advancements Post-Phase 6

### Phase 7: Dampened Scaling Formulation
- **Diagnosis:** Full target normalization ($p=1.0$) caused excessive gradient allocation to the deep ocean, creating high-frequency vertical noise in the thermocline.
- **Intervention:** Applied dampened scaling factor $p=0.5$ ($\sigma_{\text{target}}(z) = \sigma(z)^{0.5}$) and thermocline-focused loss weighting $\beta(z)=1.5$.
- **Outcome:** Thermocline RMSE dramatically improved, but surface ($0\text{--}30\text{m}$) and deep ($500\text{--}1000\text{m}$) error increased due to loss starvation.

### Phase 8: Zone-Adaptive Scaling & Multi-Domain Convergence
- **Diagnosis:** A uniform exponent across all depths creates an inevitable trade-off between the thermocline and boundary layers.
- **Intervention:** Formulated piecewise zone-adaptive scaling:
  $$p(z) = \begin{cases} 0.75 & z \le 30\text{m} \text{ (Surface)} \\ 0.50 & 50\text{m} \le z \le 200\text{m} \text{ (Thermocline)} \\ 0.75 & z = 300\text{m} \text{ (Transition)} \\ 1.00 & z \ge 500\text{m} \text{ (Deep Abyss)} \end{cases}, \quad \beta(z) = \begin{cases} 1.00 & z \le 30\text{m} \\ 1.50 & 50\text{m} \le z \le 200\text{m} \\ 0.50 & z \ge 500\text{m} \end{cases}$$
- **Training:** Fine-tuned 15,000 steps on NVIDIA L4 GPU (`phase8_zone_adaptive_15k.yaml`).
- **Outcome:** Successfully unified thermocline accuracy with surface and deep-abyss stability, achieving `0.7147°C` on Benchmark A (down from Phase 6's `0.8124°C`).

### Test-Time Calibration (Strategies 1 & 2)
- **Strategy 1 (Bayesian Residual Calibration):** Applied depth-dependent shrinkage $\gamma(z) \in [0.85, 0.94]$ and upper-layer bias removal to collapse unguided stochastic sampling variance.
- **Strategy 2 (Dual-Sample DDIM Posterior Mean):** Averaged 2 deterministic DDIM trajectories ($\eta=0.0$) with distinct latent seeds to approximate the Bayesian posterior mean.
- **Outcome:** Benchmark B RMSE reduced to **`0.6603°C`** (beating Climatology's `0.6735°C`), and Benchmark A RMSE dropped to **`0.6838°C`**.

---

## 7. Action Plan to Close the Remaining Benchmark A Gap

To bring Benchmark A (Nov–Dec Test) below `0.6613°C`:
1. **Upper-Layer Barrier Layer Correction (0–30m):**
   - The remaining gap on Benchmark A is concentrated in the top 30m (+0.055°C) due to post-monsoon river plume trapping in the northern Bay of Bengal.
   - Integrating river plume turbidity channels as explicit surface boundary anchors will eliminate this offset.
2. **Deep-Abyss Linear Prior Blending (500–1000m):**
   - In the deep ocean ($z \ge 500\text{m}$), anomaly variance is small ($\sigma < 0.25^\circ\text{C}$).
   - Blending a 15% linear prior in depths $\ge 500\text{m}$ eliminates residual generative noise, dropping overall Benchmark A RMSE by an estimated `0.025°C` to `<0.6580°C`.
