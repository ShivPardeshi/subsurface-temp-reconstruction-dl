"""OceanEmbed Unified CLI Inference Entry Point.

Usage:
    python scripts/infer.py --lat 15.0 --lon 65.0 --date 2025-11-28 --day-index 331
"""

import sys
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sampling.inference_service import OceanEmbedPredictor


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed 3D Temperature & Disaster Product Inference")
    parser.add_argument("--lat", type=float, default=15.0, help="Latitude (deg N, 2.0 to 30.0)")
    parser.add_argument("--lon", type=float, default=65.0, help="Longitude (deg E, 45.0 to 105.0)")
    parser.add_argument("--date", type=str, default="2025-11-28", help="Target date YYYY-MM-DD")
    parser.add_argument("--day-index", type=int, default=331, help="Day of year index (0-358)")
    parser.add_argument("--ensemble-size", type=int, default=10, help="Number of DDIM ensemble members")
    parser.add_argument("--alpha-ridge", type=float, default=0.75, help="Weight on Ridge baseline (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output file path")
    args = parser.parse_args()

    print("=" * 70)
    print("  OCEANEMBED (PS26066) -- 3D SUBSURFACE OCEAN TEMPERATURE RECONSTRUCTION")
    print("=" * 70)
    print(f"Target Coordinate: {args.lat:.2f}N, {args.lon:.2f}E")
    print(f"Evaluation Date:   {args.date} (Day Index {args.day_index})")
    print(f"Configuration:     Locked Hybrid Ridge + Diffusion Ensemble (Alpha={args.alpha_ridge})")
    print(f"Calibration:       Post-Hoc Depth-Dependent Calibrated (ECE=0.0727)")
    print("-" * 70)

    predictor = OceanEmbedPredictor()
    results = predictor.predict_location_products(
        lat=args.lat,
        lon=args.lon,
        date_str=args.date,
        day_index=args.day_index,
        ensemble_size=args.ensemble_size,
        alpha_ridge=args.alpha_ridge,
    )

    print("\n[1. RECONSTRUCTED 3D VERTICAL TEMPERATURE PROFILE (0-1000m)]")
    depths = results["profile"]["depths_m"]
    means = results["profile"]["mean_temperatures"]
    stds = results["profile"]["std_temperatures"]
    clims = results["climatology_profile"]
    print(f"{'Depth (m)':<10} | {'Reconstructed (deg C)':<22} | {'Uncertainty (+/-1 std)':<22} | {'Climatology (deg C)':<20}")
    print("-" * 80)
    for d, m, s, c in zip(depths, means, stds, clims):
        print(f"{d:<10} | {m:<22.2f} | +/-{s:<20.2f} | {c:<20.2f}")

    print("\n[2. DOWNSTREAM DISASTER MANAGEMENT PRODUCTS]")
    print(f"  * Tropical Cyclone Heat Potential (TCHP): {results['tchp']['mean_kj_cm2']:.2f} +/- {results['tchp']['std_kj_cm2']:.2f} kJ/cm^2 (95% CI: [{results['tchp']['ci_lower_95']:.2f}, {results['tchp']['ci_upper_95']:.2f}])")
    print(f"  * 26 deg C Isotherm Depth (D26):          {results['d26']['mean_meters']:.2f} +/- {results['d26']['std_meters']:.2f} m (95% CI: [{results['d26']['ci_lower_95']:.2f}, {results['d26']['ci_upper_95']:.2f}])")
    print(f"  * Direct Mixed Layer Depth (MLD):         {results['mld_direct']['mean_meters']:.2f} +/- {results['mld_direct']['std_meters']:.2f} m (95% CI: [{results['mld_direct']['ci_lower_95']:.2f}, {results['mld_direct']['ci_upper_95']:.2f}])")
    print(f"  * Ocean Heat Content to 700m (OHC):       {results['ohc_700']['mean_jm2']/1e9:.3f} +/- {results['ohc_700']['std_jm2']/1e9:.3f} x 10^9 J/m^2")

    print(f"\n[3. CALIBRATION & SCIENTIFIC DISCLOSURE]")
    print(f"  {results['calibration_note']}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull predictions exported to: {out_path}")


if __name__ == "__main__":
    main()
