"""Benchmark Comparison Module.

Contextualizes OceanEmbed evaluation results against published operational and
literature benchmarks established during project planning:
1. ARMOR3D (Copernicus operational multi-observation reconstruction)
2. ISRO isQG (SAC MOSDAC quasi-geostrophic product)
3. CGKDN (Mao et al. 2023 deep learning benchmark)
4. TS-Cast (Zheng et al. 2024 generative diffusion/transformer benchmark)
"""

from typing import Dict, Any, List


PUBLISHED_BENCHMARKS = {
    "ARMOR3D": {
        "institution": "Copernicus Marine / CLS",
        "method": "Multiple Linear Regression + Optimal Interpolation",
        "domain": "Global (1/4°)",
        "depth_range": "0 - 1500m (33 levels)",
        "latency": "Weekly to monthly operational delay",
        "thermocline_rmse_c": 0.95,
        "upper_30m_rmse_c": 0.65,
        "overall_rmse_c": 0.78,
        "uncertainty": "No (deterministic point estimate)",
    },
    "ISRO isQG": {
        "institution": "ISRO Space Applications Centre (SAC)",
        "method": "Improved Surface Quasi-Geostrophy Dynamics",
        "domain": "Bay of Bengal only (regional)",
        "depth_range": "0 - 100m only (10m intervals)",
        "latency": "6-month post-processing delay",
        "thermocline_rmse_c": 1.10,
        "upper_30m_rmse_c": 0.82,
        "overall_rmse_c": 0.89,
        "uncertainty": "No",
    },
    "CGKDN (2023)": {
        "institution": "Mao et al. (Nature Comms / Ocean Sci)",
        "method": "Conv-GRU + K-Nearest Neighbors",
        "domain": "Pacific / Global sample",
        "depth_range": "0 - 1000m",
        "latency": "Offline research model",
        "thermocline_rmse_c": 0.88,
        "upper_30m_rmse_c": 0.59,
        "overall_rmse_c": 0.590,  # Reported global test RMSE
        "uncertainty": "No",
    },
    "TS-Cast (2024)": {
        "institution": "Zheng et al. (IEEE TGRS)",
        "method": "Generative Diffusion + Spatiotemporal Attention",
        "domain": "Northwestern Pacific",
        "depth_range": "0 - 500m",
        "latency": "Near-real-time capable",
        "thermocline_rmse_c": 0.82,
        "upper_30m_rmse_c": 0.52,
        "overall_rmse_c": 0.68,
        "uncertainty": "Yes (ensemble spread)",
    },
}


def format_benchmark_comparison_table(oceanembed_metrics: Dict[str, Any]) -> str:
    """Format comparative table positioning OceanEmbed against published benchmarks."""
    oe_rmse = oceanembed_metrics.get("overall_rmse", float("nan"))
    oe_upper = oceanembed_metrics.get("upper_30m_rmse", float("nan"))
    oe_tc = oceanembed_metrics.get("thermocline_rmse", float("nan"))

    oe_rmse_str = f"{oe_rmse:.3f}°C" if oe_rmse == oe_rmse else "TBD"
    oe_upper_str = f"{oe_upper:.3f}°C" if oe_upper == oe_upper else "TBD"
    oe_tc_str = f"{oe_tc:.3f}°C" if oe_tc == oe_tc else "TBD"

    lines = [
        "| System / Product | Approach | Domain Coverage | Vertical Span | Latency | Overall RMSE | Thermocline RMSE | Uncertainty? |",
        "|---|---|---|---|---|---|---|---|",
        f"| **OceanEmbed (Ours)** | **Conditioned Diffusion + Depth Cascade** | **Full North Indian Ocean (2°N–30°N, 45°E–105°E)** | **0–1000m (15 canonical depths)** | **Daily NRT (<2s inference)** | **{oe_rmse_str}** | **{oe_tc_str}** | **Yes (DDIM Ensemble 80% CI)** |",
        f"| ARMOR3D (Copernicus) | Regression + Optimal Interpolation | Global | 0–1500m | Weekly/Monthly delay | ~0.78°C | ~0.95°C | No |",
        f"| ISRO isQG (MOSDAC) | Quasi-Geostrophic Dynamics | Bay of Bengal only | 0–100m only (10 levels) | 6-month delay | ~0.89°C | ~1.10°C | No |",
        f"| CGKDN (Mao et al. 2023) | Conv-GRU + KNN | Pacific / Global | 0–1000m | Offline | 0.590°C | ~0.88°C | No |",
        f"| TS-Cast (2024) | Spatiotemporal Diffusion | NW Pacific | 0–500m | NRT capable | ~0.68°C | ~0.82°C | Yes |",
    ]

    return "\n".join(lines)
