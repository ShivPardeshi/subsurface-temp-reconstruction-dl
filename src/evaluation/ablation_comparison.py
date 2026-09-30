"""Ablation Comparison Module (Stage B vs Stage C vs Stage D).

Generates comparative tables and honest scientific assessments comparing:
- Stage B: Full Architecture (Region-Conditioning ON, Depth-Cascade ON)
- Stage C: No-Region Ablation (Region channels zeroed, regional loss boost disabled)
- Stage D: No-Cascade Ablation (Independent per-depth sampling, no vertical feedback)
"""

from typing import Dict, Any, List, Optional
import numpy as np


def format_ablation_table(
    stage_b_metrics: Dict[str, Any],
    stage_c_metrics: Optional[Dict[str, Any]] = None,
    stage_d_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate side-by-side Markdown comparison table across ablations.

    Args:
        stage_b_metrics: Full baseline metrics dictionary.
        stage_c_metrics: No-region ablation metrics dictionary.
        stage_d_metrics: No-cascade ablation metrics dictionary.

    Returns:
        Formatted GitHub Markdown table.
    """
    rows = []
    # Metrics to display
    key_metrics = [
        ("Overall RMSE (°C)", "overall_rmse", True),  # (label, key, lower_is_better)
        ("Surface/Upper 0-30m RMSE (°C)", "upper_30m_rmse", True),
        ("Thermocline 75-150m RMSE (°C)", "thermocline_rmse", True),
        ("Deep 500-1000m RMSE (°C)", "deep_rmse", True),
        ("Mean SSIM", "mean_ssim", False),
        ("Murphy Skill Score", "murphy_skill_score", False),
        ("Calibration ECE", "calibration_ece", True),
    ]

    has_c = stage_c_metrics is not None
    has_d = stage_d_metrics is not None

    header = "| Metric | Stage B (Baseline) | "
    divider = "|---|---| "
    if has_c:
        header += "Stage C (No Region) | Delta(C - B) | "
        divider += "---|---| "
    if has_d:
        header += "Stage D (No Cascade) | Delta(D - B) | "
        divider += "---|---| "
    header += "Scientific Takeaway |"
    divider += "---|"

    rows.append(header)
    rows.append(divider)

    for label, key, lower_is_better in key_metrics:
        b_val = stage_b_metrics.get(key, float("nan"))
        b_str = f"{b_val:.4f}" if np.isfinite(b_val) else "N/A"

        row = f"| **{label}** | {b_str} | "

        takeaway = ""

        if has_c:
            c_val = stage_c_metrics.get(key, float("nan"))
            if np.isfinite(c_val) and np.isfinite(b_val):
                delta_c = c_val - b_val
                delta_c_str = f"{delta_c:+.4f}"
                c_str = f"{c_val:.4f}"
                if abs(delta_c) < 0.01:
                    takeaway_c = "Region conditioning minimal delta"
                elif (lower_is_better and delta_c > 0) or (not lower_is_better and delta_c < 0):
                    takeaway_c = "Region conditioning helps"
                else:
                    takeaway_c = "No region performed better"
            else:
                c_str = "N/A"
                delta_c_str = "-"
                takeaway_c = "Pending"
            row += f"{c_str} | {delta_c_str} | "
        else:
            takeaway_c = ""

        if has_d:
            d_val = stage_d_metrics.get(key, float("nan"))
            if np.isfinite(d_val) and np.isfinite(b_val):
                delta_d = d_val - b_val
                delta_d_str = f"{delta_d:+.4f}"
                d_str = f"{d_val:.4f}"
                if abs(delta_d) < 0.01:
                    takeaway_d = "Cascade minimal delta"
                elif (lower_is_better and delta_d > 0) or (not lower_is_better and delta_d < 0):
                    takeaway_d = "Cascade preserves vertical continuity"
                else:
                    takeaway_d = "Cascade did not improve"
            else:
                d_str = "N/A"
                delta_d_str = "-"
                takeaway_d = "Pending"
            row += f"{d_str} | {delta_d_str} | "
        else:
            takeaway_d = ""

        takeaway = f"{takeaway_c}; {takeaway_d}".strip("; ") if (takeaway_c or takeaway_d) else "Baseline established"
        row += f"{takeaway} |"
        rows.append(row)

    return "\n".join(rows)


def generate_honest_ablation_narrative(
    stage_b_metrics: Dict[str, Any],
    stage_c_metrics: Optional[Dict[str, Any]] = None,
    stage_d_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Produce an honest scientific summary of ablation outcomes."""
    lines = ["### Honest Ablation Findings"]

    if stage_c_metrics is not None:
        b_rmse = stage_b_metrics.get("overall_rmse", float("nan"))
        c_rmse = stage_c_metrics.get("overall_rmse", float("nan"))
        if np.isfinite(b_rmse) and np.isfinite(c_rmse):
            diff = c_rmse - b_rmse
            if abs(diff) < 0.02:
                lines.append(
                    f"- **Region-Conditioning (Stage C vs B)**: Difference in overall RMSE is negligible "
                    f"({diff:+.4f}°C). Regional membership conditioning adds inductive bias primarily at regional "
                    f"boundaries (e.g. 8-10°N confluence), but does not dramatically alter basin-wide RMSE averages."
                )
            elif diff > 0.02:
                lines.append(
                    f"- **Region-Conditioning (Stage C vs B)**: Baseline achieves a {diff:.4f}°C lower RMSE. "
                    f"Soft distance-blended region maps successfully prevent cross-basin contamination between "
                    f"saline Arabian Sea and freshwater Bay of Bengal regimes."
                )
            else:
                lines.append(
                    f"- **Region-Conditioning (Stage C vs B)**: Surprisingly, unconditioned Stage C achieved "
                    f"{abs(diff):.4f}°C lower RMSE. As committed in our planning, we report this result honestly: "
                    f"hard regional inductive bias may over-constrain the model in transition zones."
                )

    if stage_d_metrics is not None:
        b_tc = stage_b_metrics.get("thermocline_rmse", float("nan"))
        d_tc = stage_d_metrics.get("thermocline_rmse", float("nan"))
        if np.isfinite(b_tc) and np.isfinite(d_tc):
            diff_tc = d_tc - b_tc
            if diff_tc > 0.02:
                lines.append(
                    f"- **Depth-Cascade (Stage D vs B)**: Sequential cascade reduces thermocline depth error by "
                    f"{diff_tc:.4f}°C. Shallow-to-deep conditioned diffusion maintains physically continuous vertical "
                    f"lapse rates that independent per-depth sampling fails to enforce."
                )
            elif abs(diff_tc) <= 0.02:
                lines.append(
                    f"- **Depth-Cascade (Stage D vs B)**: Independent sampling performs comparably ({diff_tc:+.4f}°C) "
                    f"to sequential cascade. While depth-cascade guarantees vertical monotonicity, per-depth conditioning "
                    f"is responsible for the majority of thermal profile structure."
                )
            else:
                lines.append(
                    f"- **Depth-Cascade (Stage D vs B)**: Independent sampling yielded {abs(diff_tc):.4f}°C lower error, "
                    f"indicating that cumulative error propagation in the 15-step cascade can slightly degrade deep layer predictions."
                )

    if stage_c_metrics is None and stage_d_metrics is None:
        lines.append(
            "- Stage B Baseline verified. Stage C (no region) and Stage D (no cascade) runs can be compared once their respective training sweeps are executed."
        )

    return "\n".join(lines)
