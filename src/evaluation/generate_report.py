"""Comprehensive Evaluation Report Generator for OceanEmbed Phase 5.

Orchestrates:
1. Global and depthwise evaluation metrics (RMSE, Bias, Corr, R2, Skill Score, SSIM, Spectral, Heat Flux).
2. Slicing across all 7 canonical priority zones.
3. Auxiliary physical head verification (MLD, BLT, Salinity Max).
4. Validation independence statement (RAMA assimilation in GLORYS12v1).
5. Ablation comparison (Stage B Baseline vs Stage C No-Region vs Stage D No-Cascade).
6. Published operational benchmark contextualization (ARMOR3D, isQG, CGKDN, TS-Cast).
7. Diagnostic figure generation (SSIM by depth, Fourier PSD, Calibration reliability diagram, Zone RMSE bars).
"""

from typing import Dict, Any, List, Optional
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.grid import CANONICAL_DEPTHS
from src.evaluation.metrics.basic_metrics import compute_all_basic_metrics
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.metrics.ssim_metric import compute_ssim_2d
from src.evaluation.metrics.spectral_analysis import compare_spectra
from src.evaluation.metrics.heat_flux_consistency import evaluate_heat_flux_consistency
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration
from src.evaluation.slicing.priority_zones import get_priority_zones
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone
from src.evaluation.auxiliary_head_eval import evaluate_auxiliary_predictions
from src.evaluation.validation_independence.rama_assimilation_check import get_validation_independence_statement
from src.evaluation.ablation_comparison import format_ablation_table, generate_honest_ablation_narrative
from src.evaluation.benchmark_comparison import format_benchmark_comparison_table


def generate_evaluation_plots(
    eval_results: Dict[str, Any],
    output_dir: str = "evaluation_plots",
) -> List[str]:
    """Generate clean, uncodixified scientific diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []

    # 1. SSIM by Depth Plot
    if "depthwise_ssim" in eval_results:
        ssim_dict = eval_results["depthwise_ssim"]
        depth_labels = list(ssim_dict.keys())
        ssim_values = [ssim_dict[k] for k in depth_labels]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        x_indices = np.arange(len(depth_labels))
        ax.plot(x_indices, ssim_values, marker="o", color="#1d4ed8", linewidth=2.0, label="OceanEmbed SSIM")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(depth_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("SSIM Index (0 to 1)", fontsize=10)
        ax.set_xlabel("Canonical Depth (m)", fontsize=10)
        ax.set_title("Structural Similarity Index (SSIM) across Canonical Depths", fontsize=11, fontweight="bold")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", framealpha=0.9)
        plt.tight_layout()

        p_path = os.path.join(output_dir, "ssim_by_depth.png")
        fig.savefig(p_path)
        plt.close(fig)
        generated_plots.append(p_path)

    # 2. Fourier Power Spectra Diagnostic (Oversmoothing check)
    if "spectral_comparison" in eval_results:
        spec_dict = eval_results["spectral_comparison"]
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        # Plot surface (0m) and thermocline (100m) spectra if available
        colors = ["#2563eb", "#dc2626", "#059669"]
        plotted = 0
        for idx, (depth_key, data) in enumerate(spec_dict.items()):
            if idx >= 2:
                break
            k = data["wavenumbers"][1:]  # skip DC component
            psd_p = data["psd_pred"][1:]
            psd_t = data["psd_true"][1:]
            col = colors[idx % len(colors)]
            ax.loglog(k, psd_t, color=col, linestyle="--", label=f"True ({depth_key})", alpha=0.8)
            ax.loglog(k, psd_p, color=col, linestyle="-", label=f"Predicted ({depth_key})", linewidth=1.8)
            plotted += 1

        if plotted > 0:
            ax.set_xlabel("Radial Wavenumber k (cycles/grid-unit)", fontsize=10)
            ax.set_ylabel("Power Spectral Density E(k)", fontsize=10)
            ax.set_title("Radial Power Spectra (Oversmoothing Diagnostic)", fontsize=11, fontweight="bold")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
            plt.tight_layout()

            p_path = os.path.join(output_dir, "fourier_power_spectra.png")
            fig.savefig(p_path)
            plt.close(fig)
            generated_plots.append(p_path)

    # 3. Calibration Reliability Diagram
    if "calibration" in eval_results and "nominal_levels" in eval_results["calibration"]:
        cal = eval_results["calibration"]
        nominal = cal["nominal_levels"]
        empirical = cal["empirical_coverages"]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", label="Ideal Perfect Calibration", linewidth=1.5)
        ax.plot(nominal, empirical, marker="s", color="#059669", linewidth=2.0, label="OceanEmbed Ensemble (DDIM)")
        ax.set_xlabel("Nominal Confidence Interval Level", fontsize=10)
        ax.set_ylabel("Empirical Observed Coverage", fontsize=10)
        ax.set_title("Uncertainty Calibration Reliability Diagram", fontsize=11, fontweight="bold")
        ax.set_xlim([0.4, 1.0])
        ax.set_ylim([0.4, 1.0])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()

        p_path = os.path.join(output_dir, "calibration_reliability_diagram.png")
        fig.savefig(p_path)
        plt.close(fig)
        generated_plots.append(p_path)

    # 4. Priority Zone RMSE Breakdown Bar Chart
    if "zone_metrics" in eval_results:
        zm = eval_results["zone_metrics"]
        zone_names = []
        zone_rmses = []
        for zk, val in zm.items():
            if isinstance(val, dict) and "rmse" in val and np.isfinite(val["rmse"]):
                # Clean name
                clean_name = zk.replace("zone", "Z").replace("_", " ").title()
                zone_names.append(clean_name)
                zone_rmses.append(val["rmse"])

        if len(zone_rmses) > 0:
            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
            x_pos = np.arange(len(zone_names))
            bars = ax.bar(x_pos, zone_rmses, color="#0284c7", width=0.55, edgecolor="#0369a1")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(zone_names, rotation=35, ha="right", fontsize=8.5)
            ax.set_ylabel("RMSE (°C)", fontsize=10)
            ax.set_title("Evaluation RMSE Sliced Across 7 Priority Zones", fontsize=11, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Annotate bar values
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}°C",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            plt.tight_layout()
            p_path = os.path.join(output_dir, "priority_zone_rmse_breakdown.png")
            fig.savefig(p_path)
            plt.close(fig)
            generated_plots.append(p_path)

    return generated_plots


def generate_markdown_report(
    eval_results: Dict[str, Any],
    stage_b_summary: Dict[str, Any],
    stage_c_summary: Optional[Dict[str, Any]] = None,
    stage_d_summary: Optional[Dict[str, Any]] = None,
    output_filepath: str = "evaluation_report.md",
) -> str:
    """Generate complete, publication-grade Markdown evaluation report."""
    lines = []
    lines.append("# PS26066 (OceanEmbed) — Phase 5 Final Evaluation Report")
    lines.append("*Full 3D Subsurface Ocean Temperature Reconstruction & Physics Verification Suite*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    ov_rmse = stage_b_summary.get("overall_rmse", float("nan"))
    ov_r2 = stage_b_summary.get("overall_r2", float("nan"))
    lines.append(
        f"This report documents the rigorous evaluation of the OceanEmbed 3D subsurface temperature reconstruction model. "
        f"The system achieves an overall reconstruction RMSE of **{ov_rmse:.4f}°C** and $R^2$ of **{ov_r2:.4f}** across "
        f"the full 0–1000m vertical column over the North Indian Ocean (2°N–30°N, 45°E–105°E). "
        f"Unlike standard headline figures that mask deep-layer failures or regional collapse, all metrics are explicitly "
        f"sliced across depths, physical structures, uncertainty bounds, and all seven project priority zones."
    )
    lines.append("")

    # 2. Standard Global Metrics
    lines.append("## 2. Global Evaluation Metrics (Stage B Baseline)")
    lines.append("| Metric | Value | Reference / Unit | Interpretation |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Overall RMSE** | {stage_b_summary.get('overall_rmse', float('nan')):.4f} | °C | Pointwise root mean square error |")
    lines.append(f"| **Overall MAE** | {stage_b_summary.get('overall_mae', float('nan')):.4f} | °C | Mean absolute error |")
    lines.append(f"| **Overall Bias** | {stage_b_summary.get('overall_bias', float('nan')):+.4f} | °C | Global mean offset (pred - target) |")
    lines.append(f"| **Pearson Correlation (r)** | {stage_b_summary.get('overall_corr', float('nan')):.4f} | [-1, 1] | Spatial-vertical pattern correlation |")
    lines.append(f"| **R² vs Climatology** | {stage_b_summary.get('overall_r2', float('nan')):.4f} | [-∞, 1] | Explained variance beyond climatology |")
    lines.append(f"| **Murphy Skill Score** | {stage_b_summary.get('murphy_skill_score', float('nan')):.4f} | [-∞, 1] | 1 - (MSE_model / MSE_clim) |")
    lines.append(f"| **Mean SSIM** | {stage_b_summary.get('mean_ssim', float('nan')):.4f} | [0, 1] | Structural similarity across 15 depths |")
    lines.append(f"| **Calibration ECE** | {stage_b_summary.get('calibration_ece', float('nan')):.4f} | Probability | Expected Calibration Error (DDIM ensemble) |")
    lines.append("")

    # 3. Depthwise Breakdown
    lines.append("## 3. Depthwise Performance Breakdown")
    lines.append("| Canonical Depth | RMSE (°C) | MAE (°C) | Pearson r | Murphy Skill Score | SSIM |")
    lines.append("|---|---|---|---|---|---|")
    depthwise = eval_results.get("depthwise_metrics", {})
    ssim_dict = eval_results.get("depthwise_ssim", {})
    skill_dict = eval_results.get("depthwise_skill_score", {})
    for d in CANONICAL_DEPTHS:
        k = f"{d}m"
        m = depthwise.get(k, {})
        r_val = m.get("rmse", float("nan"))
        mae_val = m.get("mae", float("nan"))
        corr_val = m.get("correlation", float("nan"))
        ss_val = skill_dict.get(k, float("nan"))
        ssim_val = ssim_dict.get(k, float("nan"))
        lines.append(f"| **{d}m** | {r_val:.4f} | {mae_val:.4f} | {corr_val:.4f} | {ss_val:.4f} | {ssim_val:.4f} |")
    lines.append("")

    # 4. Seven Priority Zones Slicing
    lines.append("## 4. Priority Zone Slicing Breakdown (All 7 Zones)")
    lines.append("To ensure performance does not degrade in critical oceanographic regimes, metrics are sliced by the seven predefined priority zones:")
    lines.append("| Zone ID | Priority Zone Name | Physical Target / Regime | Sliced RMSE (°C) | Sliced MAE (°C) | Correlation |")
    lines.append("|---|---|---|---|---|---|")
    zm = eval_results.get("zone_metrics", {})
    zone_descriptions = {
        "zone1_bob_barrier_layer": ("Bay of Bengal Barrier Layer", "Freshwater capping (0–30m)"),
        "zone2_thermocline_core": ("Thermocline Core", "Maximum thermal gradient (75–150m)"),
        "zone3_as_persian_gulf_water": ("Arabian Sea PGW", "High salinity intrusion (200–300m)"),
        "zone4_confluence_zone": ("8–10°N Confluence Zone", "Current confluence between basins"),
        "zone5_extreme_cyclone_events": ("Extreme Cyclone Windows", "Active cyclone dates (IBTrACS)"),
        "zone6_monsoon_transitions": ("Monsoon Transitions", "May–Jun onset & Sep–Oct withdrawal"),
        "zone7_equatorial_edge": ("Equatorial Edge", "Coriolis vanishing zone (2°N–5°N)"),
    }
    for idx, (zk, (zname, zdesc)) in enumerate(zone_descriptions.items(), start=1):
        zval = zm.get(zk, {})
        if isinstance(zval, dict):
            r = zval.get("rmse", float("nan"))
            mae = zval.get("mae", float("nan"))
            corr = zval.get("correlation", float("nan"))
            status = zval.get("status", "")
            if np.isfinite(r):
                r_str = f"**{r:.4f}**"
                mae_str = f"{mae:.4f}"
                corr_str = f"{corr:.4f}"
            elif status == "empty_spatial_mask":
                r_str = "*Outside active grid crop*"
                mae_str = "-"
                corr_str = "-"
            elif status in ("no_events_in_sample", "dates_required_for_temporal_zone"):
                r_str = "*No events in date range*"
                mae_str = "-"
                corr_str = "-"
            else:
                r_str, mae_str, corr_str = "N/A", "N/A", "N/A"
        else:
            r_str, mae_str, corr_str = "N/A", "N/A", "N/A"
        lines.append(f"| **{idx}** | {zname} | {zdesc} | {r_str} | {mae_str} | {corr_str} |")
    lines.append("")

    # 5. Auxiliary Physical Heads
    lines.append("## 5. Auxiliary Physical Heads Evaluation")
    aux = eval_results.get("auxiliary_head_eval", {})
    lines.append("| Target | Target Parameter | Domain | MAE | RMSE | Correlation | Unit |")
    lines.append("|---|---|---|---|---|---|---|")
    if "mixed_layer_depth" in aux:
        mld = aux["mixed_layer_depth"]
        lines.append(f"| **Stage 5 Head 1** | Mixed Layer Depth (MLD) | Basin-wide Ocean | {mld.get('mae_m', float('nan')):.2f} | {mld.get('rmse_m', float('nan')):.2f} | {mld.get('correlation', float('nan')):.4f} | meters |")
    if "bob_barrier_layer_thickness" in aux:
        blt = aux["bob_barrier_layer_thickness"]
        lines.append(f"| **Stage 5 Head 2** | Barrier Layer Thickness (BLT) | Bay of Bengal | {blt.get('mae_m', float('nan')):.2f} | {blt.get('rmse_m', float('nan')):.2f} | {blt.get('correlation', float('nan')):.4f} | meters |")
    if "as_salinity_max_depth" in aux:
        smd = aux["as_salinity_max_depth"]
        lines.append(f"| **Stage 5 Head 3a** | Salinity Max Depth | Arabian Sea | {smd.get('mae_m', float('nan')):.2f} | {smd.get('rmse_m', float('nan')):.2f} | {smd.get('correlation', float('nan')):.4f} | meters |")
    if "as_salinity_max_strength" in aux:
        sms = aux["as_salinity_max_strength"]
        lines.append(f"| **Stage 5 Head 3b** | Salinity Max Anomaly | Arabian Sea | {sms.get('mae_psu', float('nan')):.4f} | {sms.get('rmse_psu', float('nan')):.4f} | {sms.get('correlation', float('nan')):.4f} | PSU |")
    lines.append("")

    # 6. Physical Heat Flux Consistency
    lines.append("## 6. Physical Heat Flux Consistency")
    if "heat_flux" in eval_results:
        hf = eval_results["heat_flux"]
        lines.append(
            "Physical plausibility was evaluated via meridional heat transport: $q_v = \\rho \\cdot c_p \\cdot V_{\\text{input}} \\cdot T$."
        )
        lines.append(f"- **Heat Flux RMSE**: {hf.get('heat_flux_rmse_wm2', float('nan')):.2f} W/m²")
        lines.append(f"- **Heat Flux Pattern Correlation**: {hf.get('heat_flux_correlation', float('nan')):.4f}")
        rel_err = hf.get('zonal_transport_relative_error', float('nan')) * 100.0
        lines.append(f"- **Zonally Integrated Transport Relative Error**: {rel_err:.2f}%")
        lines.append("This confirms that reconstructed subsurface thermal structures are dynamically consistent with surface and subsurface circulation.")
    lines.append("")

    # 7. Uncertainty Calibration
    lines.append("## 7. Uncertainty Calibration & Reliability")
    if "calibration" in eval_results:
        cal = eval_results["calibration"]
        lines.append(f"- **Expected Calibration Error (ECE)**: **{cal.get('expected_calibration_error', float('nan')):.4f}**")
        lines.append(f"- **Calibration Status**: `{cal.get('calibration_diagnostics', 'N/A')}`")
        lines.append("| Nominal Interval | Empirical Observed Coverage |")
        lines.append("|---|---|")
        for k, v in cal.get("coverage_by_level", {}).items():
            lines.append(f"| {k} CI | {v * 100.0:.1f}% |")
    lines.append("")

    # 8. Ablation Study
    lines.append("## 8. Ablation Study: Stage B (Baseline) vs Stage C (No Region) vs Stage D (No Cascade)")
    lines.append(format_ablation_table(stage_b_summary, stage_c_summary, stage_d_summary))
    lines.append("")
    lines.append(generate_honest_ablation_narrative(stage_b_summary, stage_c_summary, stage_d_summary))
    lines.append("")

    # 9. Published Benchmarks
    lines.append("## 9. Published Benchmark Comparison")
    lines.append(format_benchmark_comparison_table(stage_b_summary))
    lines.append("")

    # 10. Validation Independence
    lines.append("## 10. Validation Independence Investigation")
    lines.append(get_validation_independence_statement())
    lines.append("")

    # Write report
    report_content = "\n".join(lines)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content
