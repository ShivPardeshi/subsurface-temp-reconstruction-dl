"""Temperature Scaling Calibration Module for Ensemble Spread.

Implements post-hoc calibration of DDIM ensemble spread:
    y_cal_i = mean(y) + s * (y_i - mean(y))
where s > 0 is a single learned scalar multiplier fit on the held-out validation set
by minimizing Expected Calibration Error (ECE).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.optimize import minimize_scalar

from src.evaluation.metrics.basic_metrics import _to_numpy
from src.evaluation.metrics.calibration import evaluate_ensemble_calibration


def scale_ensemble_spread(
    ensemble_preds: Union[np.ndarray, list],
    scale_factor: float,
) -> np.ndarray:
    """Scale ensemble predictions around their empirical mean by a scalar factor.

    Args:
        ensemble_preds: (N, ...) array of N ensemble predictions.
        scale_factor: s > 0 multiplier on the ensemble spread.

    Returns:
        Calibrated ensemble predictions of the same shape.
    """
    ens = _to_numpy(ensemble_preds)
    mean_ens = np.mean(ens, axis=0, keepdims=True)
    spread = ens - mean_ens
    calibrated_ens = mean_ens + scale_factor * spread
    return calibrated_ens.astype(np.float32)


def fit_temperature_scale(
    val_ensemble_preds: Union[np.ndarray, list],
    val_target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    confidence_levels: Optional[List[float]] = None,
) -> float:
    """Fit optimal temperature scale factor on validation data.

    Minimizes ECE over the nominal confidence levels.

    Args:
        val_ensemble_preds: (N, ...) validation ensemble predictions.
        val_target: (...) matching ground truth observations.
        mask: Optional spatial/depth valid mask.
        confidence_levels: List of nominal confidence levels (default [0.50, 0.68, 0.80, 0.90, 0.95]).

    Returns:
        Optimal scale factor s* (float).
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95]

    val_ens = _to_numpy(val_ensemble_preds)
    val_t = _to_numpy(val_target)

    def objective(s: float) -> float:
        if s <= 0.0:
            return 1.0
        cal_ens = scale_ensemble_spread(val_ens, s)
        res = evaluate_ensemble_calibration(
            ensemble_preds=cal_ens,
            target=val_t,
            confidence_levels=confidence_levels,
            mask=mask,
        )
        return float(res.get("expected_calibration_error", 1.0))

    opt = minimize_scalar(objective, bounds=(0.5, 50.0), method="bounded")
    best_s = float(opt.x) if opt.success else 1.0
    return max(0.1, best_s)


def calibrate_and_evaluate(
    val_ensemble: Union[np.ndarray, list],
    val_target: np.ndarray,
    test_ensemble: Union[np.ndarray, list],
    test_target: np.ndarray,
    val_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Fit temperature scaling on validation set and evaluate before/after on test set.

    Returns:
        Dictionary containing:
        - 'scale_factor': learned scalar s*
        - 'raw_ece': ECE before calibration on test set
        - 'calibrated_ece': ECE after calibration on test set
        - 'raw_coverages': empirical coverage before calibration
        - 'calibrated_coverages': empirical coverage after calibration
        - 'nominal_levels': nominal confidence levels
        - 'raw_diagnostics': diagnostic string before
        - 'calibrated_diagnostics': diagnostic string after
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.68, 0.80, 0.90, 0.95]

    # 1. Fit on validation set
    s_star = fit_temperature_scale(
        val_ensemble_preds=val_ensemble,
        val_target=val_target,
        mask=val_mask,
        confidence_levels=confidence_levels,
    )

    # 2. Evaluate uncalibrated on test set
    raw_eval = evaluate_ensemble_calibration(
        ensemble_preds=test_ensemble,
        target=test_target,
        confidence_levels=confidence_levels,
        mask=test_mask,
    )

    # 3. Apply calibrated spread and evaluate on test set
    cal_test_ens = scale_ensemble_spread(test_ensemble, s_star)
    cal_eval = evaluate_ensemble_calibration(
        ensemble_preds=cal_test_ens,
        target=test_target,
        confidence_levels=confidence_levels,
        mask=test_mask,
    )

    return {
        "scale_factor": s_star,
        "raw_ece": raw_eval.get("expected_calibration_error", 0.0),
        "calibrated_ece": cal_eval.get("expected_calibration_error", 0.0),
        "raw_coverages": raw_eval.get("empirical_coverages", []),
        "calibrated_coverages": cal_eval.get("empirical_coverages", []),
        "nominal_levels": confidence_levels,
        "raw_diagnostics": raw_eval.get("calibration_diagnostics", ""),
        "calibrated_diagnostics": cal_eval.get("calibration_diagnostics", ""),
    }


# Convenient alias
run_post_hoc_calibration_pipeline = calibrate_and_evaluate


def test_temperature_scaling():
    """Verify that temperature scaling recovers proper coverage on synthetic overconfident predictions."""
    np.random.seed(42)
    N, C, H, W = 10, 5, 20, 20
    # True error standard deviation is 0.8
    true_sigma = 0.8
    mean_val = 25.0 + np.random.randn(C, H, W)
    target = mean_val + np.random.randn(C, H, W) * true_sigma

    # Overconfident ensemble spread has sigma ~ 0.05
    raw_sigma = 0.05
    ensemble = mean_val[None, ...] + np.random.randn(N, C, H, W) * raw_sigma

    res = run_post_hoc_calibration_pipeline(
        val_ensemble=ensemble,
        val_target=target,
        test_ensemble=ensemble,
        test_target=target,
    )

    print(f"Fitted Scale Factor: {res['scale_factor']:.2f}")
    print(f"Raw ECE:             {res['raw_ece']:.4f}")
    print(f"Calibrated ECE:      {res['calibrated_ece']:.4f}")
    assert res["calibrated_ece"] < res["raw_ece"], "Calibrated ECE must be lower than raw ECE"
    assert res["calibrated_ece"] < 0.10, "Calibrated ECE should achieve < 0.10"
    print("test_temperature_scaling: SUCCESS")

