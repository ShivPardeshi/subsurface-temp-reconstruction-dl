# PS26066 — Arabian Sea Deep-Dive & Real State-of-the-Art Architecture Research

Two things in this document go beyond everything before it: a proper physical breakdown of the Arabian Sea's own hard problems (not just "it has upwelling"), and — because you pushed back correctly on the model-selection research — an architecture recommendation grounded in real, current published research instead of extrapolation from 17 hackathon repos.

---

## PART 1 — The Arabian Sea's Real Problems, Parallel to Bay of Bengal

Your framing was right: we can't only solve Bay of Bengal. Here's what actually makes the Arabian Sea hard, and it turns out to be structurally *opposite* to Bay of Bengal in almost every way, not just "a different flavor of the same problem."

### Problem 1: The Persian Gulf Water intrusion — a subsurface salinity maximum with no Bay of Bengal equivalent
Real, published, well-quantified oceanography: intensely evaporated, dense water forms in the Persian Gulf and Red Sea (E–P budget in this region is enormous — the Persian Gulf alone sees surface temperatures swinging 20–30°C seasonally against a near-constant ~40 psu salinity), spills out through the Strait of Hormuz and Bab-el-Mandeb, and spreads into the Arabian Sea as a **subsurface salinity maximum sitting between 200–300m depth** — warm (>17°C) and highly saline (>36.2 psu), embedded within the thermocline. Critically:
- It's **seasonally mobile**: spreads south along the western boundary during winter monsoon, essentially absent from that path in summer.
- Real research (HYCOM modeling studies) shows **mesoscale eddies actively reshape its path** — this isn't a static feature, it moves around based on the same eddy field we're already trying to capture from SSH.
- This is a feature with **no Bay of Bengal analog at all** — Bay of Bengal's defining subsurface signature is a freshwater cap near the surface; the Arabian Sea's defining subsurface signature is a saline core sitting well below the surface. A model tuned to expect "surface-trapped anomalies" (learned from Bay of Bengal patterns) could genuinely miss this if regional conditioning isn't doing real work, not just nominal work.

### Problem 2: The Findlater Jet and Somali upwelling — a named, specific summer mechanism
The Arabian Sea's summer upwelling isn't generic wind-driven upwelling — it's driven by a specific, intense low-level atmospheric jet (the Findlater Jet / Somali Jet) that generates the "Great Whirl" gyre and pulls up cold water (17–18°C, a stark ~10°C+ contrast against 28–29°C surrounding surface water) along the Somali and Omani coasts. This vertically displaces the *entire* thermohaline structure along that coastline, not just the surface.

### Problem 3: Winter convective mixing — the literal opposite of Bay of Bengal's mechanism
This is the sharpest contrast, and it matters architecturally. In Bay of Bengal, the mixed layer stays *shallow* because a freshwater cap suppresses vertical mixing. In the Arabian Sea during winter, the mechanism runs the other way entirely: strong evaporative cooling plus already-high salinity makes surface water dense enough to sink and overturn convectively, **deepening** the mixed layer substantially through mid-basin convective mixing. Same general phenomenon we care about (mixed layer depth), opposite physical driver, opposite seasonal behavior, opposite depth response.

**Why this matters for architecture, not just data:** if we're relying on region-conditioning (an embedding fed to one shared network) to handle "Arabian Sea vs. Bay of Bengal," we're asking that embedding to learn two genuinely opposite physical regimes, not two variations on a theme. This raises the bar for what the region-conditioning actually needs to encode — worth explicitly testing (the honest ablation we already planned) whether a single shared embedding is expressive enough, or whether the mixed layer response specifically needs more capacity.

### Problem 4: The Arabian Sea's own quantified E–P imbalance — the unifying feature we should build once, not twice
The Arabian Sea is a region of strong net evaporation (E−P on the order of 150 cm/year), pushing surface salinity as high as 36 psu in the north — the mirror image of Bay of Bengal's strong net freshwater gain. **This is genuinely useful news for our design**, not just more complexity: rather than building a bespoke "freshwater feature" for Bay of Bengal and a separate bespoke "evaporation feature" for the Arabian Sea, we should build **one shared E−P (evaporation-minus-precipitation, or moisture flux) field across the whole domain**, computed once from precipitation and latent heat flux data. The same physical quantity, with opposite sign, explains both basins' defining behavior. This is more elegant than two ad hoc features and more defensible as "we understood the unifying physics," not "we patched two regions separately."

### External data for the Arabian Sea's hard zones

| Data needed | Why | Source |
|---|---|---|
| Wind stress curl / Ekman pumping (already planned) | Drives Findlater-Jet-linked Somali/Omani upwelling | ISRO's own Oceansat-3 scatterometer product |
| Latent heat flux (already planned, now doing double duty) | Drives both winter convective mixing (Arabian Sea) and the E−P balance (both basins) | OAFlux |
| Chlorophyll-a (already planned) | Upwelling proxy along the Somali/Omani coast | MODIS/VIIRS, or Oceansat-3's OCM-3 |
| A specific salinity-maximum-depth diagnostic (new) | Worth deriving a dedicated feature tracking the depth/strength of the subsurface salinity maximum from training-period GLORYS data, the same way we're building a barrier-layer-thickness auxiliary target for Bay of Bengal — the Persian Gulf Water intrusion is exactly as distinctive a signature and deserves the same auxiliary-head treatment | Derived from training data itself, no new external source needed |

### A bonus third zone this research surfaced: the ~8–10°N confluence
Real published research (Prasad, Ikeda & Kumar 2001) found that Persian Gulf Water spreading south along India's west coast during winter **laterally mixes with low-salinity Bay-of-Bengal-origin water specifically south of 10°N**. This means there's a genuine physical transition/mixing zone roughly where the two basins meet (the southern tip of India, the Lakshadweep Sea) — neither purely "Arabian Sea physics" nor purely "Bay of Bengal physics," but an actual confluence of both water masses. Worth tracking as its own evaluation slice alongside the others, and worth being aware that our region-conditioning embedding needs a genuine "transition zone" category, not just a hard Arabian-Sea-or-Bay-of-Bengal binary.

---

## PART 2 — Real State-of-the-Art Architecture Research (Beyond the 17 Repos)

You were right to push here. Everything in our architecture decision so far came from what hackathon teams happened to try. Here's what the actual current research literature is doing.

### The key paper: depth-aware conditional diffusion models (2026)
A very recent paper — "High-resolution probabilistic estimation of three-dimensional regional ocean dynamics from sparse surface observations" (Asefi, Wu, He & Chattopadhyay, UC Santa Cruz / NC State) — tackles almost exactly our problem: reconstructing full 3D ocean state (temperature, salinity, and both velocity components) from **extremely sparse** satellite SSH and SST, using a **depth-aware conditional Denoising Diffusion Probabilistic Model (DDPM)**, trained and tested on real satellite data and real GLORYS reanalysis (Gulf of Mexico, not our region, but the method transfers directly). Real code is publicly available (github.com/TACS-UCSC/Ocean3D_Estimation).

**Why this beats a plain deterministic model, and it's not a small effect:** the paper makes an important theoretical point worth internalizing — the mapping from ocean surface to ocean interior is **inherently ill-posed and non-unique**. A deterministic model (a plain CNN or U-Net) has to commit to one single answer even when the true underlying state is genuinely ambiguous given the surface data alone. Their own head-to-head comparison found deterministic baselines (UNet, FNO) either **oversmooth the fields or introduce spurious small-scale artifacts**, while the diffusion model preserved genuine multiscale variability closely matching real GLORYS structure — confirmed via Fourier spectral analysis, not just pointwise RMSE.

**Three specific, directly actionable technical findings, regardless of whether we adopt a full diffusion model:**

1. **Depth normalization matters more than it sounds like it should.** They found naive min-max normalization of the depth coordinate caused a real "boundary bias" — the shallowest layer (25m) specifically failed to be learned properly. **Log-normalizing the depth identifier fixed it.** This is a concrete, one-line, scientifically validated fix directly relevant to our own continuous/cascade depth conditioning — we should log-normalize depth, not min-max it, regardless of which architecture we end up using.

2. **Evaluate with SSIM and Fourier spectral analysis, not just RMSE/correlation/bias.** SSIM (Structural Similarity Index, borrowed from image quality assessment) catches whether the reconstructed *structure* matches, not just pointwise values. Fourier spectral comparison catches whether a model is quietly oversmoothing — reproducing large-scale patterns correctly while losing all the fine-scale (high-wavenumber) detail, a failure mode plain RMSE can completely miss. This is a more rigorous evaluation standard than anything in our 17-repo sweep or even most of the papers we'd previously reviewed, and it's cheap to add to our own evaluation pipeline.

3. **A genuine physical-consistency diagnostic: meridional heat flux.** They compute `q_v = ρ·cp·V·T` (density × specific heat × meridional velocity × temperature) as a coupled check — does the reconstructed temperature field, combined with the reconstructed velocity field, produce a physically sensible heat transport pattern? This is a stronger test than checking T and V accuracy separately, because it only passes if both fields are consistent *with each other*, not just individually plausible. This is directly adaptable to our own OHC/TCHP downstream layer as a training-time or evaluation-time consistency check, not just a final product.

### Other real prior art this search surfaced, worth knowing by name
- **SQG-MEOF-R** (Yan et al. 2020) — a dynamical-statistical hybrid combining surface quasi-geostrophic dynamics with multivariate empirical orthogonal function reconstruction, a real, established precedent for blending physics-based and statistical approaches, conceptually similar to but more rigorous than what any of the 17 hackathon repos attempted.
- **OceanNet** (Chattopadhyay et al., Scientific Reports 2024) — a "principled neural operator-based digital twin for regional oceans," from the same research group, relevant if we want to look further into neural-operator approaches specifically.
- **Bolton & Zanna (2019)** — the earliest CNN-based subsurface-from-surface inference in a simplified quasi-geostrophic setting, the true intellectual starting point of this entire line of research.

### Updated recommendation: a two-tier plan, not a single choice
Given real implementation-risk-versus-reward tradeoffs for a time-constrained team, here's how I'd actually sequence this:

**Tier 1 (build this first, it's the safe, working system):** the CNN + ConvLSTM temporal fusion + FiLM-climatology-conditioning + depth-cascade design from our previous planning document. Proven pieces, moderate complexity, real chance of being fully working and validated in time.

**Tier 2 (the genuinely ambitious upgrade, attempt only once Tier 1 works end-to-end):** replace the deterministic decoder with a depth-aware conditional diffusion model, following the Asefi et al. design directly — continuous log-normalized depth conditioning, sampling multiple times per prediction to get a natural probabilistic ensemble (uncertainty "for free," and more principled than a predicted variance). This is the single most scientifically current, best-evidenced option in all our research — but diffusion models are slower at inference (multiple denoising steps) and more finicky to train well in a short timeframe, so it should be a genuine stretch goal, not the foundation we bet the whole submission on.

**Adopt regardless of which tier we reach:** log-normalized depth conditioning (not min-max), SSIM + Fourier spectral evaluation alongside RMSE/correlation, and the heat-flux physical-consistency diagnostic. These three are cheap, don't require committing to diffusion models, and are more rigorous than anything in the 17 repos or most of the papers we'd reviewed before this pass.

---

## PART 3 — Explicit External-Dataset-to-Core-Dataset Relationship Map

You asked specifically how external data relates to the PS-mandated core inputs. Here's the explicit mapping — what combines with what, and where it enters the system:

| External input | Combines with (core PS variable) | Produces | Enters the model as |
|---|---|---|---|
| River discharge (GBM) + Precipitation | SSS | Freshwater-forcing context for the barrier layer | Extra input channel, concentrated in Bay of Bengal via region-conditioning |
| Latent heat flux + Precipitation | — (both, combined) | E−P moisture flux field (shared driver for both basins, opposite sign) | Single derived channel across the whole domain |
| Wind U/V (already core) | SSH (already core) | Geostrophic currents (thermal-wind relation) | Derived feature, replaces/supplements raw current channels |
| Wind U/V (already core) | — (self) | Wind stress curl / Ekman pumping | Derived feature, most relevant to Arabian Sea upwelling |
| Wind U/V (already core) | — (self) | Wind-mixing energy (~wind speed³) | Derived feature, feeds the mixed-layer-depth auxiliary head |
| Ocean color / chlorophyll | SST, wind stress curl | Upwelling cross-check signal | Extra input channel, most informative in Arabian Sea and along river plumes |
| Bathymetry (GEBCO) | — (static) | Land/ocean mask + shelf-depth context | Static channel + masking applied throughout |
| Climate indices (ONI, IOD) | — (scalar, basin-wide) | Interannual conditioning signal | Broadcast scalar, concatenated at the network bottleneck alongside the region embedding |
| IBTrACS cyclone tracks | — (labeling only) | Extreme-event evaluation windows | Not a model input — used to select evaluation slices |
| GEBCO + SST/SSS training-period statistics | — (derived from training data) | Barrier-layer-thickness proxy (BoB) and salinity-maximum-depth proxy (Arabian Sea) | Auxiliary prediction targets, not inputs — the network learns to predict these alongside temperature |

---

## Consolidated list of every "priority zone" identified so far
For tracking as separate evaluation slices, now including this pass's additions:
1. Bay of Bengal mixed-layer/barrier-layer zone (0–30m)
2. Thermocline core, both basins (75–150m)
3. Arabian Sea Persian-Gulf-Water intrusion zone (200–300m)
4. The ~8–10°N Arabian Sea / Bay of Bengal water-mass confluence zone
5. Extreme-event windows (cyclones, via IBTrACS) — plus the cloud-blocked-infrared-SST data problem specific to these windows
6. Monsoon transition windows (onset/withdrawal)
7. The equatorial domain boundary (~5°N edge)
