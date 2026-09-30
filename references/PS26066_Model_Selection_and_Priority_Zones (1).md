# PS26066 — Pre-Implementation Plan: Model Selection & Priority Zones

This is the planning layer that sits between our architecture design and actual code — which model family, why, what else was considered and rejected, and a deep dive on the specific depth/region combinations where we can win or lose the whole project, starting with the one you identified.

---

## PART 1 — Model Selection: Options, Evidence, Decision

Rather than picking an architecture by intuition, here's every option with real evidence from our research, for and against.

| Option | Real evidence found | Verdict |
|---|---|---|
| **Plain CNN / U-Net, full-basin** | Used by Nishad3107 (Tier S) and AtulChaudharyy (Tier A) with real, reasonable results | **Use as the spatial backbone** — proven, efficient, well-understood |
| **Vision Transformer (single-snapshot)** | sayan21m's real head-to-head test: ViT alone posted RMSE 0.33–1.13°C | **Reject as primary** — real evidence shows it's clearly outperformed |
| **ConvLSTM (with temporal lag)** | Same sayan21m test: RMSE 0.12–0.37°C — roughly **3× better** than the ViT on identical data | **Use for temporal fusion** — the strongest single piece of architecture evidence in our entire research |
| **DeepONet + Swin (operator learning)** | Pinn-Ocean — sophisticated, TEOS-10-grounded, but never executed on real data by anyone, wrong region | **Reject for now** — unproven in practice, high implementation risk for a hackathon timeframe; revisit only if core system is done early |
| **FiLM-conditioned U-Net on climatology** | TS-Cast (real, peer-reviewed, July 2026) — beats assimilative reanalysis on genuinely independent validation | **Adopt this specific mechanism** — the best-evidenced technique for exactly our anomaly-learning plan |
| **Classical gradient-boosted trees (XGBoost)** | kanishkaanarang — real 0.22°C MAE vs. ARGO, beating an EN4-climatology baseline, with real SHAP feature importance | **Use as a fast diagnostic tool and fallback/ensemble member**, not the core model — see below |
| **Graph Neural Network** | Suggested by the PS itself; no repo in our sweep implements one for this task | **Reserve as a stretch addition** for a final ARGO-nudging correction layer, not the backbone |

### Final architecture decision
**Spatial backbone:** full-basin CNN/U-Net (not patch-based — KaviBharathi643's patch approach measurably loses basin-scale context).
**Temporal handling:** ConvLSTM-style processing of the 7-day input window, not a plain CNN on a single day — this is the single most strongly evidenced choice we can make, given the 3× real-world difference sayan21m found.
**Climatology conditioning:** FiLM-style modulation (TS-Cast's mechanism) rather than simple subtraction (sg721642's simpler version) — more expressive, and it's the approach behind the strongest real independent-validation result found anywhere in this research.
**Depth handling:** our own depth-cascade conditioning (still nobody else's), layered on top of the above.

### Where XGBoost earns a real place in the plan (not just as a footnote)
Two concrete, practical uses, not just "nice to know it exists":
1. **A fast pre-flight feature-importance check.** Before we spend engineering time integrating river discharge, heat flux, or any other candidate variable into the full deep-learning pipeline, we can train a quick XGBoost model with SHAP on a small labeled sample and *empirically check* whether that variable actually carries predictive signal for our region — cheap, fast, and it means we only pay the full integration cost for variables we've already confirmed are worth it, rather than guessing.
2. **A fallback and ensemble member.** If the deep model is behind schedule or not converging cleanly, we already have a working, real, respectably-accurate alternative. And even once the deep model works, blending its output with an XGBoost prediction can help specifically because the two make different *kinds* of errors — deep models tend to systematically fail on rare configurations; tree models tend to fail on smooth spatial coherence. Worth testing whether an ensemble beats either alone.

---

## PART 2 — Your Bay of Bengal Insight, Developed Properly

Your instinct here is right, and it's the single highest-leverage thing to get right in this whole project — let me sharpen it rather than just agree with it.

### First, a correction: there are two related but distinct hard zones, not one
Our research actually points to **two separate physical problems** stacked in the upper water column, not one continuous "0–100m is hard" zone:

1. **The mixed-layer/barrier-layer zone (roughly 0–30m).** This is where the *exact depth* of the freshwater cap varies day to day — KaviBharathi643's real results specifically dipped at 20–30m (R² near zero), and ISRO's own isQG documentation calls out exactly this range ("mixed layer can be as shallow as 15–20m") as its own biggest weakness. The physical cause here is genuinely what you said: freshwater capping from river discharge and rainfall creates a barrier layer that decouples the surface from what's below it, and where exactly that cap sits shifts with recent freshwater input.

2. **The thermocline core (roughly 75–150m).** This is a *different* problem — not about a freshwater cap, but about the sheer steepness of the temperature gradient itself, where small errors in exactly where the thermocline sits translate into large temperature errors. This is the zone confirmed across six independent sources (Guinehut 2012 via ARMOR3D, ISRO's isQG docs, KaviBharathi643, Nishad3107, aaronpinto449, and sayan21m's ViT) — including sources that have nothing to do with freshwater/barrier layers at all (ARMOR3D's global finding predates any Bay-of-Bengal-specific consideration).

**Why this distinction matters practically:** they likely need different fixes. The barrier-layer zone (1) is fundamentally about getting the *forcing* right — the freshwater/buoyancy input that sets where the cap sits. The thermocline-core zone (2) is more about the *model's resolving power* — getting the shape of a steep gradient right, which is exactly what our depth-cascade conditioning targets. Solving one doesn't automatically solve the other, and we should track them as separate evaluation slices, not one combined "shallow water" number.

### The external data plan for zone 1 (mixed-layer/barrier-layer), building on what you proposed

| Data source | Why it helps | Access |
|---|---|---|
| **River discharge (Ganges-Brahmaputra-Meghna)** | Direct driver of the freshwater cap's volume | GRDC, or MOSDAC's own River Discharge product |
| **Farakka Barrage / major dam release data specifically** | Your idea — a sudden controlled release changes freshwater input on a faster timescale than natural discharge variation alone, and is a real, documented, frequently-occurring event (we found real news coverage confirming controlled multi-gate releases happen regularly) | **India-WRIS** (indiawris.gov.in) — real, public, co-launched by CWC and ISRO. Caveat: we found a real user report describing friction actually extracting time-series data through the portal — treat this as a risk to test and de-risk in week one, not assume works smoothly |
| **Precipitation** | Direct surface freshening, independent of river input | GPM/IMERG, or MOSDAC's MT-SAPHIR rainfall product |
| **Evaporation (via latent heat flux)** | The real driver of surface salinity is the E–P (evaporation minus precipitation) balance, not precipitation alone — this refines our earlier heat-flux addition into something specifically targeted at this zone | OAFlux (includes latent heat flux) |
| **Wind-mixing energy (derived, not raw wind)** | Mixed layer depth is set by the *balance* between buoyancy capping (freshwater) and mechanical mixing (wind stirring) — turbulent kinetic energy input scales roughly with wind speed cubed, so a raw wind vector under-represents the mixing side of this balance | Derive from existing wind U/V — new engineered feature, not a new data source |
| **Ocean-color river-plume signal (a genuinely new idea worth testing)** | Satellite-measured SSS (e.g. SMAP) is known to degrade in accuracy specifically near coasts and river mouths — exactly where we need it most for this zone. Ocean-color sensors can directly see turbidity/CDOM plumes marking the river plume's extent, which could serve as a higher-resolution *complement* to degraded near-coast SSS | MODIS/VIIRS ocean color, or Oceansat-3's own OCM-3 instrument |

### The counterpart problem we shouldn't neglect: Arabian Sea
If we pour all our differentiation effort into Bay of Bengal, we risk under-serving the other half of the required domain. The Arabian Sea's hard-zone driver is structurally different — seasonal upwelling (Oman/Somali coast), not freshwater capping — and its own external data plan is largely already in our design: wind-stress-curl/Ekman-pumping (from ISRO's own scatterometer product) and chlorophyll-a as an upwelling proxy. Worth explicitly budgeting evaluation effort here too, not just building it and moving on — an honest "here's how we do in the Arabian Sea's own hard zone" slide is a strong complement to the Bay of Bengal story, and shows we didn't only solve the flashier regional problem.

---

## PART 3 — Other "Priority Zone" Points, In the Same Spirit

You asked for other points like this one — here are the ones our research actually points to as deserving the same dedicated treatment:

### Point: Extreme-event windows (cyclones) — and a real data problem hiding inside it
We already planned to evaluate accuracy specifically during known cyclone windows (via IBTrACS). Here's something new this planning pass surfaced: **infrared satellite SST is heavily cloud-blocked during a cyclone** — exactly the moment we most need accurate surface input. If our primary SST source is optical/infrared, our *inputs themselves* may have gaps precisely during the highest-stakes windows, independent of anything the model does. The real fix: microwave SST products (e.g. AMSR2, TMI) penetrate cloud cover, at the cost of coarser resolution — worth building in as a fallback input specifically for extreme-event windows, not just a general robustness nice-to-have.

### Point: The equatorial/domain-boundary edge (~5°N)
Already flagged as a gap nobody addresses. The concrete fix: pull input data slightly beyond the nominal domain (e.g. down to 2°N instead of 5°N) purely to give the model valid spatial context at the edge and avoid convolutional padding artifacts, then crop the final output back to the required 5°N boundary. Cheap to implement, directly fixes a real, generic deep-learning pitfall.

### Point: Monsoon transition windows (onset/withdrawal)
Currents reverse and stratification changes fastest during monsoon transitions — a plausible third "hard window" alongside extreme events, worth its own evaluation slice using known monsoon onset/withdrawal dates (published annually by IMD) the same way we'll use IBTrACS for cyclones.

---

## What this means for sequencing
1. Run the cheap XGBoost/SHAP pre-flight check on candidate variables as soon as *any* real data is flowing — before committing engineering time to the full pipeline for each one.
2. Build the core architecture (CNN+ConvLSTM+FiLM-climatology+depth-cascade) against the 7 required variables first, get it working end to end.
3. Layer in the Bay-of-Bengal-zone-1 variables (river discharge, precipitation, E-P, wind-mixing energy) as the first real enhancement, since that's our highest-leverage, most-differentiated target.
4. Track four separate evaluation slices from the start, not just depth: **mixed-layer/barrier-layer (0–30m)**, **thermocline core (75–150m)**, **extreme-event windows**, and **Arabian Sea upwelling zone** — so progress on one is visible and doesn't quietly hide regression on another.

---

Worth deciding before we start writing code: do you want to spend real time testing the ocean-color river-plume idea (genuinely novel, unproven, moderate effort), or treat it as a stretch goal and focus the first real engineering pass on the more conventional, already-justified variables (river discharge, precipitation, heat flux)?
