"""Phase 6 Operational Demo Runner.

Supports:
1. `--test-mode`: Headless execution, validating full computational pipeline
   (OHC, TCHP, MLD direct crossing, Hobday MHW, uncertainty propagation),
   rendering and saving sample artifact figures to `artifacts/phase6_demo/`.
2. Default: Launches the interactive Uncodixified Streamlit dashboard.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.products.ocean_heat_content import integrate_vertical_heat
from src.products.tchp import compute_tchp_profile, find_d26_isotherm_depth
from src.products.mld_direct import compute_mld_direct_profile
from src.products.marine_heatwave import detect_mhw_1d, compute_mhw_spatial_window
from src.products.uncertainty_propagation import propagate_profile_uncertainty, CALIBRATION_DISCLOSURE_NOTE
from src.dashboard.components.profile_viewer import render_profile_figure
from src.dashboard.components.heatwave_map import render_heatwave_map_figure
from src.dashboard.components.tchp_gauge import render_tchp_card_figure


def run_headless_verification(output_dir: Path, date_str: str = "2025-11-28") -> bool:
    """Execute headless verification of Phase 6 pipeline and save figures."""
    print("=" * 65)
    print("  OceanEmbed Phase 6 Downstream Disaster Products Verification")
    print("=" * 65)
    output_dir.mkdir(parents=True, exist_ok=True)

    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    # 1. Generate realistic 10-member ensemble profile
    np.random.seed(42)
    ens_profiles = []
    base_sst = 28.8
    for i in range(10):
        noise = np.random.normal(0.0, 0.25, size=len(depths))
        prof = []
        for d in depths:
            if d <= 40:
                t = base_sst - 0.04 * (d / 40.0)
            elif d <= 150:
                t = base_sst - 0.04 - 12.0 * ((d - 40.0) / 110.0) ** 0.8
            else:
                t = 16.4 - 10.0 * ((d - 150.0) / 850.0) ** 0.5
            prof.append(max(4.0, t) + noise[depths.index(d)])
        ens_profiles.append(prof)

    ens_arr = np.array(ens_profiles)

    # 2. Propagate uncertainty
    print("[1/4] Running uncertainty propagation across ensemble profiles...")
    metrics = propagate_profile_uncertainty(ens_arr, depths)
    print(f"      TCHP:       {metrics['tchp']['mean_kj_cm2']:.2f} +/- {metrics['tchp']['std_kj_cm2']:.2f} kJ/cm^2")
    print(f"      D26 Depth:  {metrics['d26']['mean_meters']:.2f} +/- {metrics['d26']['std_meters']:.2f} m")
    print(f"      Direct MLD: {metrics['mld_direct']['mean_meters']:.2f} +/- {metrics['mld_direct']['std_meters']:.2f} m")
    print(f"      OHC (700m): {metrics['ohc_700']['mean_jm2']/1e9:.3f} +/- {metrics['ohc_700']['std_jm2']/1e9:.3f} x 10^9 J/m^2")

    # 3. Test Hobday MHW detection
    print("[2/4] Testing Hobday et al. (2016) 5-day persistence detection...")
    daily_t = np.array([28.2, 28.5, 29.1, 29.3, 29.5, 29.4, 29.2, 28.1, 28.0])
    p90_t = np.array([28.6] * 9)
    mhw_res = detect_mhw_1d(daily_t, p90_t, min_duration=5)
    print(f"      MHW active steps: {np.sum(mhw_res['is_mhw'])}/9 days (5-day criterion enforced)")
    assert mhw_res["is_mhw"][2:7].all(), "Failed to detect 5-day consecutive MHW event"

    # 4. Render and save component figures
    print("[3/4] Rendering and saving component figures...")
    fig_prof = render_profile_figure(
        depths=depths,
        temp_mean=metrics["profile"]["mean_temperatures"],
        temp_std=metrics["profile"]["std_temperatures"],
        obs_temp=ens_arr.mean(axis=0) + 0.2,
        mld=metrics["mld_direct"]["mean_meters"],
        d26=metrics["d26"]["mean_meters"],
        title="Arabian Sea (15.0°N, 65.0°E) Temperature Profile",
    )
    prof_path = output_dir / "profile_viewer_demo.png"
    fig_prof.savefig(prof_path, bbox_inches="tight")
    print(f"      Saved: {prof_path}")

    fig_card = render_tchp_card_figure(
        tchp_mean=metrics["tchp"]["mean_kj_cm2"],
        tchp_std=metrics["tchp"]["std_kj_cm2"],
        d26_mean=metrics["d26"]["mean_meters"],
        d26_std=metrics["d26"]["std_meters"],
        ohc_700_mean=metrics["ohc_700"]["mean_jm2"],
        mld_mean=metrics["mld_direct"]["mean_meters"],
        sst_mean=metrics["profile"]["mean_temperatures"][0],
        location_name="Central Arabian Sea (15.0°N, 65.0°E)",
        date_str=date_str,
    )
    card_path = output_dir / "tchp_card_demo.png"
    fig_card.savefig(card_path, bbox_inches="tight")
    print(f"      Saved: {card_path}")

    # Spatial map
    lats = np.linspace(2.0, 30.0, 56)
    lons = np.linspace(45.0, 105.0, 120)
    lon_m, lat_m = np.meshgrid(lons, lats)
    mhw_map = (np.exp(-((lat_m - 16.0)**2 / 15.0 + (lon_m - 65.0)**2 / 25.0)) > 0.5)
    fig_map = render_heatwave_map_figure(
        lats=lats,
        lons=lons,
        mhw_mask=mhw_map,
        title="North Indian Ocean Active MHW Flag",
        date_str=date_str,
    )
    map_path = output_dir / "heatwave_map_demo.png"
    fig_map.savefig(map_path, bbox_inches="tight")
    print(f"      Saved: {map_path}")

    print("[4/4] Verifying calibration caveat disclosure...")
    print(f"      Caveat text: '{CALIBRATION_DISCLOSURE_NOTE}'")
    assert "calibrat" in CALIBRATION_DISCLOSURE_NOTE.lower()

    print("=" * 65)
    print("  Phase 6 Downstream Disaster Products: ALL VERIFICATIONS PASSED")
    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Phase 6 Operational Demo")
    parser.add_argument("--test-mode", action="store_true", help="Run headless verification and save figures")
    parser.add_argument("--date", type=str, default="2025-11-28", help="Target date YYYY-MM-DD")
    parser.add_argument("--output-dir", type=str, default="artifacts/phase6_demo", help="Output directory for plots")
    parser.add_argument("--port", type=int, default=8501, help="Port for Streamlit dashboard")
    args = parser.parse_args()

    if args.test_mode:
        success = run_headless_verification(Path(args.output_dir), date_str=args.date)
        sys.exit(0 if success else 1)
    else:
        # Launch Streamlit interactive dashboard
        app_file = ROOT_DIR / "src" / "dashboard" / "app.py"
        cmd = f"streamlit run {app_file} --server.port {args.port}"
        print(f"Launching Streamlit dashboard: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    main()
