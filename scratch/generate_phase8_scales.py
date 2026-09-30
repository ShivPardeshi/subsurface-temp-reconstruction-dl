"""Generate zone-adaptive depth scale file for Phase 8.

Zone-adaptive exponents:
  Surface  (0-30m):   p=0.75  -> mild dampening, recover surface skill
  Thermocline(50-200m): p=0.50  -> IDENTICAL to Phase 7, preserve gains
  Transition (300m):  p=0.75  -> gradual bridge
  Deep     (500-1000m): p=1.00  -> full standardization = restore Phase 6 deep accuracy
"""

import json
import math
from pathlib import Path

# Original anomaly stds from Phase 7 scale file (anomaly_stds_original)
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
ANOMALY_STDS_ORIGINAL = [
    0.46143341064453125,   # 0m
    0.4485616087913513,    # 5m
    0.4490685760974884,    # 10m
    0.4773275554180145,    # 20m
    0.522144615650177,     # 30m
    0.6302638649940491,    # 50m
    0.854799211025238,     # 75m
    1.1007288694381714,    # 100m
    1.1202175617218018,    # 125m
    0.9349727034568787,    # 150m
    0.5934849977493286,    # 200m
    0.36516067385673523,   # 300m
    0.2279052436351776,    # 500m
    0.23208731412887573,   # 700m
    0.24380147457122803,   # 1000m
]

# Zone-adaptive exponents
SURFACE_DEPTHS    = {0, 5, 10, 20, 30}
THERMOCLINE_DEPTHS = {50, 75, 100, 125, 150, 200}
TRANSITION_DEPTHS  = {300}
DEEP_DEPTHS        = {500, 700, 1000}

ZONE_EXPONENTS = {
    "surface":     0.75,   # mild dampening
    "thermocline": 0.50,   # SAME AS PHASE 7 — protected
    "transition":  0.75,   # bridge
    "deep":        1.00,   # SAME AS PHASE 6 — restored
}

def get_zone(depth):
    if depth in SURFACE_DEPTHS:
        return "surface"
    elif depth in THERMOCLINE_DEPTHS:
        return "thermocline"
    elif depth in TRANSITION_DEPTHS:
        return "transition"
    elif depth in DEEP_DEPTHS:
        return "deep"
    raise ValueError(f"Unknown depth: {depth}")

# Compute zone-adaptive scales
zone_adaptive_stds = []
per_depth_exponents = []
print("Zone-Adaptive Scale File (Phase 8)")
print("=" * 70)
print(f"{'Depth':>6}  {'Zone':>12}  {'p':>4}  {'std_orig':>8}  {'std^p':>8}  {'vs P7 p0.5':>12}")
print("-" * 70)
for d, sigma in zip(DEPTHS, ANOMALY_STDS_ORIGINAL):
    zone = get_zone(d)
    p = ZONE_EXPONENTS[zone]
    scale = sigma ** p
    scale_p7 = sigma ** 0.5
    zone_adaptive_stds.append(scale)
    per_depth_exponents.append(p)
    mark = "SAME" if abs(scale - scale_p7) < 0.001 else f"{'UP' if scale < scale_p7 else 'DN'}{abs(scale-scale_p7):.4f}"
    print(f"{d:>6}m  {zone:>12}  {p:>4.2f}  {sigma:>8.4f}  {scale:>8.4f}  {mark:>12}")

print("=" * 70)

# Load other metadata from Phase 7 file
with open("data/processed/anomaly_depth_scales_dampened_p05.json") as f:
    p7_data = json.load(f)

# Build output
output = {
    "description": "Zone-Adaptive Depth Scales for Phase 8 Fine-Tuning",
    "strategy": {
        "surface_0_30m": "p=0.75 (mild dampening, recover Phase 6 surface skill)",
        "thermocline_50_200m": "p=0.50 (IDENTICAL to Phase 7 — thermocline gains protected)",
        "transition_300m": "p=0.75 (gradual bridge between thermocline and deep)",
        "deep_500_1000m": "p=1.00 (full standardization = RESTORE Phase 6 deep accuracy)"
    },
    "depths": DEPTHS,
    "anomaly_stds": zone_adaptive_stds,        # σ^p(d) — used during training & inference
    "anomaly_stds_original": ANOMALY_STDS_ORIGINAL,  # σ^1.0 — raw stds for reference
    "anomaly_means": p7_data["anomaly_means"],
    "clim_mean_temps": p7_data["clim_mean_temps"],
    "per_depth_exponents": per_depth_exponents,
    "zone_mapping": {
        "surface_depths": sorted(SURFACE_DEPTHS),
        "thermocline_depths": sorted(THERMOCLINE_DEPTHS),
        "transition_depths": sorted(TRANSITION_DEPTHS),
        "deep_depths": sorted(DEEP_DEPTHS),
    }
}

out_path = Path("data/processed/anomaly_depth_scales_zone_adaptive_p8.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to: {out_path}")

# Verification
print("\nVerification — Thermocline scales must match Phase 7 exactly:")
for i, (d, sigma) in enumerate(zip(DEPTHS, ANOMALY_STDS_ORIGINAL)):
    if d in THERMOCLINE_DEPTHS:
        p7_scale = sigma ** 0.5
        p8_scale = zone_adaptive_stds[i]
        match = "[OK] MATCH" if abs(p7_scale - p8_scale) < 1e-9 else "[FAIL] MISMATCH"
        print(f"  {d:>4}m: P7={p7_scale:.6f}  P8={p8_scale:.6f}  {match}")

print("\nVerification -- Deep ocean scales must match Phase 6 exactly (std^1.0):")
for i, (d, sigma) in enumerate(zip(DEPTHS, ANOMALY_STDS_ORIGINAL)):
    if d in DEEP_DEPTHS:
        p6_scale = sigma ** 1.0
        p8_scale = zone_adaptive_stds[i]
        match = "[OK] MATCH" if abs(p6_scale - p8_scale) < 1e-9 else "[FAIL] MISMATCH"
        print(f"  {d:>4}m: P6={p6_scale:.6f}  P8={p8_scale:.6f}  {match}")
