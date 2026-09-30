"""Evaluation metrics module for OceanEmbed."""

from .basic_metrics import (
    compute_rmse,
    compute_bias,
    compute_correlation,
    compute_r2,
    compute_all_basic_metrics,
)
from .skill_score import compute_murphy_skill_score
from .ssim_metric import compute_ssim_2d, compute_profile_ssim
from .spectral_analysis import compute_radial_power_spectrum, compare_spectra
from .heat_flux_consistency import compute_meridional_heat_flux, evaluate_heat_flux_consistency
from .calibration import evaluate_ensemble_calibration

__all__ = [
    "compute_rmse",
    "compute_bias",
    "compute_correlation",
    "compute_r2",
    "compute_all_basic_metrics",
    "compute_murphy_skill_score",
    "compute_ssim_2d",
    "compute_profile_ssim",
    "compute_radial_power_spectrum",
    "compare_spectra",
    "compute_meridional_heat_flux",
    "evaluate_heat_flux_consistency",
    "evaluate_ensemble_calibration",
]
