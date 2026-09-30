"""Phase 6 Component: Marine Heatwave Spatial Map.

Renders 2D spatial maps of the North Indian Ocean domain (2-30°N, 45-105°E)
showing active Marine Heatwave events identified using the Hobday et al. (2016)
operational standard (>= 90th percentile for >= 5 consecutive days).
"""

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm


def render_heatwave_map_figure(
    lats: np.ndarray,
    lons: np.ndarray,
    mhw_mask: np.ndarray,
    categories: Optional[np.ndarray] = None,
    land_mask: Optional[np.ndarray] = None,
    highlight_lat: Optional[float] = None,
    highlight_lon: Optional[float] = None,
    title: str = "Active Marine Heatwave Detection (Hobday et al. 2016)",
    date_str: str = "2025-11-28",
) -> Figure:
    """Generate a spatial map of active marine heatwaves in the North Indian Ocean.

    Args:
        lats: 1D array of latitudes (e.g. 112 cells from 2 to 30 N).
        lons: 1D array of longitudes (e.g. 240 cells from 45 to 105 E).
        mhw_mask: 2D boolean array (H, W), True if active heatwave.
        categories: Optional 2D integer array (H, W): 0=None, 1=Moderate, 2=Strong, 3=Severe, 4=Extreme.
        land_mask: Optional 2D boolean array (H, W), True if land.
        highlight_lat: Optional latitude of selected point to highlight.
        highlight_lon: Optional longitude of selected point to highlight.
        title: Plot title.
        date_str: Target evaluation date.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    # Base display array
    H, W = mhw_mask.shape
    display_grid = np.zeros((H, W), dtype=np.float32)

    if categories is not None:
        display_grid = np.asarray(categories, dtype=np.float32)
    else:
        display_grid = np.where(mhw_mask, 1.0, 0.0).astype(np.float32)

    # Mask land as NaN or distinct value
    if land_mask is not None:
        display_grid = np.where(land_mask, -1.0, display_grid)

    # Colormap: -1: Land (slate gray), 0: Normal Ocean (navy/blue-gray),
    # 1: Moderate (yellow-orange), 2: Strong (orange-red), 3: Severe (crimson), 4: Extreme (dark red)
    colors = ["#334155", "#0f172a", "#f59e0b", "#ea580c", "#dc2626", "#7f1d1d"]
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Meshgrid extent
    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]

    im = ax.imshow(
        display_grid,
        extent=extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
        aspect="auto",
    )

    # Mark highlighted location if provided
    if highlight_lat is not None and highlight_lon is not None:
        ax.scatter(
            [highlight_lon],
            [highlight_lat],
            s=80,
            facecolors="none",
            edgecolors="#38bdf8",
            linewidth=2.0,
            zorder=10,
            label=f"Selected ({highlight_lat:.1f}°N, {highlight_lon:.1f}°E)",
        )
        ax.legend(loc="upper right", framealpha=0.85, fontsize=8)

    ax.set_xlabel("Longitude (°E)", fontsize=10, fontweight="medium")
    ax.set_ylabel("Latitude (°N)", fontsize=10, fontweight="medium")
    ax.set_title(f"{title} — {date_str}", fontsize=11, fontweight="bold", pad=10)

    # Clean grid lines
    ax.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")

    # Custom colorbar legend
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.14, shrink=0.75, ticks=[-1, 0, 1, 2, 3, 4])
    cbar.ax.set_xticklabels(["Land", "Normal", "Cat I (Mod)", "Cat II (Str)", "Cat III (Sev)", "Cat IV (Ext)"], fontsize=8)

    fig.tight_layout()
    return fig
