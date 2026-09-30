"""Phase 6 Component: Profile Viewer.

Visualizes vertical ocean temperature reconstruction:
- Mean profile across ensemble members
- Uncertainty ribbon (± 1 standard deviation)
- Ground truth reference (GLORYS / ARGO) when available
- Isotherm D26 and Mixed Layer Depth (MLD) annotations
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def render_profile_figure(
    depths: Union[List[float], np.ndarray],
    temp_mean: Union[List[float], np.ndarray],
    temp_std: Optional[Union[List[float], np.ndarray]] = None,
    obs_temp: Optional[Union[List[float], np.ndarray]] = None,
    mld: Optional[float] = None,
    d26: Optional[float] = None,
    title: str = "Vertical Temperature Profile with Ensemble Uncertainty",
    show_caveat: bool = True,
) -> Figure:
    """Generate a clean, publication-ready vertical profile figure.

    Depth is plotted on the inverted y-axis (0m at the top).

    Args:
        depths: Depth levels in meters (e.g. 0 to 1000m).
        temp_mean: Mean reconstructed temperature at each depth (°C).
        temp_std: Standard deviation across ensemble members at each depth (°C).
        obs_temp: Ground truth observation (GLORYS / ARGO) if available (°C).
        mld: Mixed layer depth in meters (optional annotation line).
        d26: Depth of 26°C isotherm in meters (optional annotation line).
        title: Plot title.
        show_caveat: Whether to display the calibration caveat in the footer.

    Returns:
        Matplotlib Figure object.
    """
    z = np.asarray(depths, dtype=np.float64)
    mu = np.asarray(temp_mean, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 7), dpi=120)

    # Shaded ensemble uncertainty band (±1 sigma)
    if temp_std is not None:
        sigma = np.asarray(temp_std, dtype=np.float64)
        ax.fill_betweenx(
            z,
            mu - sigma,
            mu + sigma,
            color="#2563eb",
            alpha=0.25,
            label="Ensemble Spread (±1σ)",
        )

    # Reconstructed mean profile
    ax.plot(mu, z, color="#1d4ed8", linewidth=2.0, label="Reconstructed Mean", zorder=4)

    # Ground truth observation
    if obs_temp is not None:
        obs = np.asarray(obs_temp, dtype=np.float64)
        ax.plot(
            obs,
            z,
            color="#16a34a",
            linestyle="--",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Ground Truth (GLORYS/ARGO)",
            zorder=5,
        )

    # 26°C isotherm indicator
    if d26 is not None and d26 > 0.0:
        ax.axhline(
            y=d26,
            color="#dc2626",
            linestyle=":",
            linewidth=1.4,
            label=f"D26 Isotherm ({d26:.1f} m)",
        )
        ax.scatter([26.0], [d26], color="#dc2626", s=30, zorder=6)

    # Mixed Layer Depth indicator
    if mld is not None and mld > 0.0:
        ax.axhline(
            y=mld,
            color="#d97706",
            linestyle="-.",
            linewidth=1.4,
            label=f"Direct MLD ({mld:.1f} m)",
        )

    # Styling: Inverted depth axis
    ax.set_ylim(max(z), 0)
    ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Depth (m)", fontsize=11, fontweight="medium")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, color="#94a3b8")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    if show_caveat:
        fig.text(
            0.5,
            0.01,
            "* Uncertainty ribbon represents ensemble standard deviation (±1σ). "
            "Calibration accuracy under active improvement via post-hoc temperature scaling.",
            ha="center",
            fontsize=7.5,
            color="#64748b",
            style="italic",
        )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig
