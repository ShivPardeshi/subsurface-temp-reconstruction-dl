"""Phase 6 Component: Tropical Cyclone Heat Potential (TCHP) Disaster Display.

Displays:
- TCHP metric with ensemble uncertainty (mean ± standard deviation in kJ/cm²)
- D26 Isotherm Depth (mean ± standard deviation in meters)
- Total Ocean Heat Content (OHC-700 in 10^9 J/m²)
- Mixed Layer Depth (MLD direct crossing in meters)
- Operational Cyclogenesis Intensity Risk Level
- MANDATORY Calibration Caveat Disclosure per Track A honest UX standard.
"""

from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def get_cyclone_intensity_risk(tchp_value: float) -> Dict[str, str]:
    """Classify cyclone intensification potential based on TCHP threshold.

    Operational thresholds:
    - < 50 kJ/cm²: Low / Insufficient heat for rapid intensification
    - 50 - 80 kJ/cm²: Moderate potential
    - > 80 kJ/cm²: High potential for rapid cyclogenesis / intensification
    """
    if tchp_value < 50.0:
        return {"level": "Low", "color": "#16a34a", "description": "Insufficient upper ocean heat for rapid intensification"}
    elif tchp_value <= 80.0:
        return {"level": "Moderate", "color": "#d97706", "description": "Favorable thermal energy for tropical cyclone maintenance"}
    else:
        return {"level": "High (Rapid Intensification Alert)", "color": "#dc2626", "description": "High upper ocean heat content supporting rapid cyclone intensification"}


def render_tchp_card_figure(
    tchp_mean: float,
    tchp_std: float,
    d26_mean: float,
    d26_std: float,
    ohc_700_mean: float,
    mld_mean: float,
    sst_mean: float,
    location_name: str = "Central Arabian Sea (15.0°N, 65.0°E)",
    date_str: str = "2025-11-28",
) -> Figure:
    """Generate a clean visual card for TCHP and disaster metrics.

    Adheres strictly to Uncodixified design: solid borders, readable sans-serif,
    clear hierarchy, no floating glass shells.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    ax.axis("off")

    risk = get_cyclone_intensity_risk(tchp_mean)

    # Card background and border
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                         facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5, zorder=1)
    ax.add_patch(rect)

    # Title & Location
    ax.text(0.06, 0.90, "Disaster Risk Assessment: Tropical Cyclone Heat Potential",
            fontsize=12, fontweight="bold", color="#0f172a", transform=ax.transAxes)
    ax.text(0.06, 0.83, f"Location: {location_name} | Date: {date_str}",
            fontsize=9, color="#64748b", transform=ax.transAxes)

    # Divider line
    ax.plot([0.06, 0.94], [0.80, 0.80], color="#e2e8f0", linewidth=1.2, transform=ax.transAxes)

    # Main TCHP Metric
    ax.text(0.06, 0.68, "TCHP (Excess Heat above 26°C):", fontsize=10, fontweight="medium", color="#334155", transform=ax.transAxes)
    tchp_str = f"{tchp_mean:.1f} ± {tchp_std:.1f} kJ/cm²"
    ax.text(0.06, 0.56, tchp_str, fontsize=20, fontweight="bold", color="#0f172a", transform=ax.transAxes)

    # Cyclogenesis Risk Tag
    risk_box = plt.Rectangle((0.55, 0.56), 0.38, 0.16, transform=ax.transAxes,
                             facecolor="#ffffff", edgecolor=risk["color"], linewidth=1.2, zorder=2)
    ax.add_patch(risk_box)
    ax.text(0.57, 0.66, f"Risk: {risk['level']}", fontsize=10, fontweight="bold", color=risk["color"], transform=ax.transAxes)
    ax.text(0.57, 0.59, risk["description"][:38] + "...", fontsize=7.5, color="#64748b", transform=ax.transAxes)

    # Secondary Metrics Row
    ax.plot([0.06, 0.94], [0.50, 0.50], color="#e2e8f0", linewidth=1.0, transform=ax.transAxes)

    col1_text = f"D26 Isotherm: {d26_mean:.1f} ± {d26_std:.1f} m"
    col2_text = f"OHC-700: {ohc_700_mean / 1e9:.2f} × 10⁹ J/m²"
    col3_text = f"Direct MLD: {mld_mean:.1f} m  |  SST: {sst_mean:.1f}°C"

    ax.text(0.06, 0.42, col1_text, fontsize=9, fontweight="medium", color="#1e293b", transform=ax.transAxes)
    ax.text(0.06, 0.34, col2_text, fontsize=9, fontweight="medium", color="#1e293b", transform=ax.transAxes)
    ax.text(0.06, 0.26, col3_text, fontsize=9, fontweight="medium", color="#1e293b", transform=ax.transAxes)

    # Divider before caveat
    ax.plot([0.06, 0.94], [0.20, 0.20], color="#e2e8f0", linewidth=1.0, transform=ax.transAxes)

    # Mandatory Calibration Caveat Note
    ax.text(
        0.06,
        0.11,
        "Honest UX Disclosure:\n"
        "Uncertainty intervals represent ensemble spread (±1σ). "
        "Calibration accuracy is currently under active refinement via post-hoc temperature scaling (Track A Fix A4).",
        fontsize=7.5,
        color="#64748b",
        style="italic",
        transform=ax.transAxes,
    )

    fig.tight_layout()
    return fig
