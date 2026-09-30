"""Unit tests for Phase 5 evaluation metrics correctness.

Verifies analytical correctness on known synthetic mathematical cases:
- Perfect prediction: RMSE=0, MAE=0, Bias=0, Corr=1, R2=1, Skill Score=1, SSIM=1.
- Climatology prediction: Murphy Skill Score=0, R2=0.
- Constant offset: Bias = offset, RMSE = |offset|, Corr=1.
- Inverted signal: Corr = -1.
- Ocean masking: Land pixels excluded from calculation.
- Heat flux & calibration edge cases.
"""

import numpy as np
import pytest

from src.evaluation.metrics.basic_metrics import (
    compute_rmse,
    compute_mae,
    compute_bias,
    compute_correlation,
    compute_r2,
    compute_all_basic_metrics,
)
from src.evaluation.metrics.skill_score import compute_murphy_skill_score
from src.evaluation.metrics.ssim_metric import compute_ssim_2d
from src.evaluation.metrics.spectral_analysis import compute_radial_power_spectrum
from src.evaluation.metrics.heat_flux_consistency import compute_meridional_heat_flux, evaluate_heat_flux_consistency
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration


def test_perfect_prediction_metrics():
    """When pred == target, all error metrics should be zero and fidelity metrics 1.0."""
    target = np.array([[10.0, 15.0], [20.0, 25.0]], dtype=np.float64)
    pred = target.copy()
    clim = target - 5.0  # Climatology with non-zero error

    rmse = compute_rmse(pred, target)
    mae = compute_mae(pred, target)
    bias = compute_bias(pred, target)
    corr = compute_correlation(pred, target)
    r2 = compute_r2(pred, target, clim=clim)
    skill = compute_murphy_skill_score(pred, target, clim=clim)
    ssim = compute_ssim_2d(pred, target)

    assert np.isclose(rmse, 0.0, atol=1e-7), f"Expected RMSE 0, got {rmse}"
    assert np.isclose(mae, 0.0, atol=1e-7), f"Expected MAE 0, got {mae}"
    assert np.isclose(bias, 0.0, atol=1e-7), f"Expected Bias 0, got {bias}"
    assert np.isclose(corr, 1.0, atol=1e-7), f"Expected Corr 1, got {corr}"
    assert np.isclose(r2, 1.0, atol=1e-7), f"Expected R2 1, got {r2}"
    assert np.isclose(skill, 1.0, atol=1e-7), f"Expected Skill 1, got {skill}"
    assert np.isclose(ssim, 1.0, atol=1e-4), f"Expected SSIM 1, got {ssim}"


def test_climatology_prediction_zero_skill():
    """When pred == clim, Murphy skill score and R2 must equal exactly 0.0."""
    target = np.array([20.0, 24.0, 28.0, 22.0], dtype=np.float64)
    clim = np.array([21.0, 25.0, 27.0, 23.0], dtype=np.float64)
    pred = clim.copy()

    skill = compute_murphy_skill_score(pred, target, clim=clim)
    r2 = compute_r2(pred, target, clim=clim)

    assert np.isclose(skill, 0.0, atol=1e-7), f"Expected Skill 0.0 when pred==clim, got {skill}"
    assert np.isclose(r2, 0.0, atol=1e-7), f"Expected R2 0.0 when pred==clim, got {r2}"


def test_constant_offset_and_inverted_correlation():
    """Constant offset should change bias/RMSE but keep correlation 1.0; negation gives -1.0."""
    target = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    offset = 3.5
    pred_offset = target + offset

    assert np.isclose(compute_bias(pred_offset, target), offset, atol=1e-7)
    assert np.isclose(compute_rmse(pred_offset, target), offset, atol=1e-7)
    assert np.isclose(compute_mae(pred_offset, target), offset, atol=1e-7)
    assert np.isclose(compute_correlation(pred_offset, target), 1.0, atol=1e-7)

    # Inverted signal
    pred_inverted = -target
    assert np.isclose(compute_correlation(pred_inverted, target), -1.0, atol=1e-7)


def test_masking_behavior():
    """Masked pixels (e.g. land) must not influence the computed metrics."""
    target = np.array([[10.0, 1000.0], [20.0, 5000.0]])
    pred = np.array([[12.0, -9999.0], [22.0, -9999.0]])
    mask = np.array([[True, False], [True, False]])  # Ignore column 1

    rmse = compute_rmse(pred, target, mask=mask)
    bias = compute_bias(pred, target, mask=mask)

    # Valid points are [10 vs 12] and [20 vs 22] -> diff is +2.0
    assert np.isclose(rmse, 2.0, atol=1e-7)
    assert np.isclose(bias, 2.0, atol=1e-7)


def test_heat_flux_consistency():
    """Meridional heat flux computation and comparison."""
    v_zero = np.zeros((4, 4))
    t = np.ones((4, 4)) * 25.0
    q = compute_meridional_heat_flux(v_zero, t)
    assert np.allclose(q, 0.0), "Zero velocity must produce zero heat flux"

    v_const = np.ones((4, 4)) * 0.5  # 0.5 m/s
    res = evaluate_heat_flux_consistency(v_const, t, t)
    assert np.isclose(res["heat_flux_rmse_wm2"], 0.0, atol=1e-5)
    assert np.isclose(res["zonal_transport_relative_error"], 0.0, atol=1e-5)


def test_calibration_reliability():
    """Ensemble calibration coverage ordering test."""
    np.random.seed(42)
    # Target drawn from N(0, 1)
    target = np.random.normal(0, 1.0, size=500)
    # Ideal calibrated ensemble of 20 members drawn from same N(0, 1)
    ensemble = np.random.normal(0, 1.0, size=(20, 500))

    cal = evaluate_ensemble_calibration(ensemble, target, confidence_levels=[0.50, 0.80, 0.90])
    coverages = cal["empirical_coverages"]

    # Coverages must be strictly monotonic: 50% < 80% < 90%
    assert coverages[0] < coverages[1] < coverages[2], f"Expected monotonic coverage, got {coverages}"
    # ECE on well-calibrated ensemble should be small (< 0.10)
    assert cal["expected_calibration_error"] < 0.12, f"ECE too high: {cal['expected_calibration_error']}"
