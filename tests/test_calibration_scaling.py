"""Unit tests for Post-Hoc Uncertainty Calibration Scaling."""

import numpy as np
import pytest
from src.evaluation.calibration_scaling import PostHocCalibrator
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration


def test_post_hoc_calibrator_synthetic():
    np.random.seed(42)
    n_ens = 10
    n_dates = 5
    n_depths = 15
    h, w = 20, 30

    # True values
    t_val = np.random.randn(n_dates, n_depths, h, w).astype(np.float32)

    # Overconfident ensemble: error is ~0.70, but ensemble spread is only ~0.10
    mu_ens = t_val + 0.70 * np.random.randn(n_dates, n_depths, h, w).astype(np.float32)
    ens_val = mu_ens[None, ...] + 0.10 * np.random.randn(n_ens, n_dates, n_depths, h, w).astype(np.float32)

    ocean_mask = np.ones((h, w), dtype=bool)

    # Check uncalibrated ECE on date 0
    raw_cal = evaluate_ensemble_calibration(ens_val[:, 0, ...], t_val[0, ...], mask=ocean_mask)
    assert raw_cal["expected_calibration_error"] > 0.40
    assert "overconfident" in raw_cal["calibration_diagnostics"]

    # Fit calibrator
    calibrator = PostHocCalibrator()
    fit_res = calibrator.fit(ens_val, t_val, mask=ocean_mask)
    assert calibrator.is_fitted
    assert len(fit_res["s_factors"]) == n_depths
    assert len(fit_res["sigma_res"]) == n_depths

    # Calibrate ensemble for date 0
    ens_cal = calibrator.calibrate_ensemble(ens_val[:, 0, ...])
    assert ens_cal.shape == (n_ens, n_depths, h, w)

    # Check calibrated ECE
    cal_res = evaluate_ensemble_calibration(ens_cal, t_val[0, ...], mask=ocean_mask)
    assert cal_res["expected_calibration_error"] < 0.15
    print(f"Calibrated ECE: {cal_res['expected_calibration_error']:.4f} vs Raw: {raw_cal['expected_calibration_error']:.4f}")
