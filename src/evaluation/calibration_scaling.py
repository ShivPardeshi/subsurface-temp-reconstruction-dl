"""Post-Hoc Uncertainty Calibration Scaling Module.

Implements depth-dependent temperature scaling and residual variance calibration:
    sigma^2_cal(z) = s(z)^2 * sigma^2_ens(z) + sigma^2_res(z)

Fits parameters s(z) and sigma^2_res(z) on validation data to achieve well-calibrated
empirical coverage across all confidence intervals (ECE < 0.05).
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import scipy.optimize as opt

from src.evaluation.metrics.calibration import evaluate_ensemble_calibration
from src.evaluation.metrics.basic_metrics import _to_numpy


class PostHocCalibrator:
    """Post-hoc uncertainty calibrator fitting depth-wise scaling and residual variance."""

    def __init__(self, canonical_depths: Optional[List[float]] = None):
        self.depths = canonical_depths or [
            0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
        ]
        self.num_depths = len(self.depths)
        self.s_factors: np.ndarray = np.ones(self.num_depths, dtype=np.float32)
        self.sigma_res: np.ndarray = np.zeros(self.num_depths, dtype=np.float32)
        self.is_fitted: bool = False

    def fit(
        self,
        ensemble_preds: np.ndarray,  # (N_ens, N_dates, D, H, W) or (N_ens, D, H, W)
        targets: np.ndarray,         # (N_dates, D, H, W) or (D, H, W)
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit s(z) and sigma_res(z) on validation data using NLL minimization.

        Args:
            ensemble_preds: Array of ensemble predictions.
            targets: Array of true ocean temperatures.
            mask: Ocean mask boolean array.
        """
        ens = _to_numpy(ensemble_preds)
        t = _to_numpy(targets)

        if ens.ndim == 4:
            # (N_ens, D, H, W) -> add date dimension: (N_ens, 1, D, H, W)
            ens = ens[:, None, ...]
            t = t[None, ...]

        n_ens, n_dates, num_d, h, w = ens.shape
        mu_ens = np.mean(ens, axis=0)  # (N_dates, D, H, W)
        var_ens = np.var(ens, axis=0, ddof=1)  # (N_dates, D, H, W)
        error_sq = (t - mu_ens) ** 2  # (N_dates, D, H, W)

        s_list = []
        res_list = []
        fit_diagnostics = {}

        for d in range(num_d):
            d_m = mask if mask is not None else np.ones((h, w), dtype=bool)
            d_var = var_ens[:, d, :, :][:, d_m].flatten()
            d_err2 = error_sq[:, d, :, :][:, d_m].flatten()

            valid = np.isfinite(d_var) & np.isfinite(d_err2)
            d_var = d_var[valid]
            d_err2 = d_err2[valid]

            mean_err2 = float(np.mean(d_err2))
            mean_var = float(np.mean(d_var))

            # Initial guess: s=1.0, sigma_res^2 = max(0, mean_err2 - mean_var)
            init_res2 = max(1e-6, mean_err2 - mean_var)
            init_s = 1.0

            # Optimize Gaussian NLL: 0.5 * sum(log(s^2 * var + res2) + err2 / (s^2 * var + res2))
            def nll_obj(params):
                log_s, log_res2 = params
                s_val = np.exp(log_s)
                res2_val = np.exp(log_res2)
                var_cal = (s_val ** 2) * d_var + res2_val
                return np.mean(0.5 * np.log(var_cal) + 0.5 * d_err2 / var_cal)

            try:
                res = opt.minimize(
                    nll_obj,
                    x0=[np.log(init_s), np.log(init_res2)],
                    method="Nelder-Mead",
                    options={"maxiter": 200},
                )
                opt_s = float(np.exp(res.x[0]))
                opt_res = float(np.sqrt(np.exp(res.x[1])))
            except Exception:
                opt_s = init_s
                opt_res = float(np.sqrt(init_res2))

            s_list.append(opt_s)
            res_list.append(opt_res)
            fit_diagnostics[f"{self.depths[d]}m"] = {
                "depth_m": self.depths[d],
                "raw_spread_mean": float(np.sqrt(mean_var)),
                "raw_rmse": float(np.sqrt(mean_err2)),
                "fitted_s": opt_s,
                "fitted_sigma_res": opt_res,
                "calibrated_spread": float(np.sqrt(opt_s**2 * mean_var + opt_res**2)),
            }

        self.s_factors = np.array(s_list, dtype=np.float32)
        self.sigma_res = np.array(res_list, dtype=np.float32)
        self.is_fitted = True

        return {
            "s_factors": self.s_factors.tolist(),
            "sigma_res": self.sigma_res.tolist(),
            "depthwise_diagnostics": fit_diagnostics,
        }

    def calibrate_ensemble(
        self,
        ensemble_preds: np.ndarray,  # (N_ens, D, H, W)
    ) -> np.ndarray:
        """Apply calibration scaling: expand ensemble spread around the mean.

        y_cal = mu_ens + (sigma_cal / sigma_ens) * (y - mu_ens)
        """
        if not self.is_fitted:
            raise RuntimeError("PostHocCalibrator must be fitted before calling calibrate_ensemble")

        ens = _to_numpy(ensemble_preds).copy()
        is_5d = (ens.ndim == 5)
        if is_5d:
            # (N_ens, N_dates, D, H, W)
            s_broadcast = self.s_factors[None, None, :, None, None]
            res_broadcast = self.sigma_res[None, None, :, None, None]
        else:
            # (N_ens, D, H, W)
            s_broadcast = self.s_factors[None, :, None, None]
            res_broadcast = self.sigma_res[None, :, None, None]

        mu_ens = np.mean(ens, axis=0, keepdims=True)
        std_ens = np.std(ens, axis=0, ddof=1, keepdims=True)
        std_ens = np.maximum(std_ens, 1e-6)

        # sigma_cal = sqrt(s^2 * sigma_ens^2 + sigma_res^2)
        std_cal = np.sqrt((s_broadcast ** 2) * (std_ens ** 2) + (res_broadcast ** 2))

        # Scaled ensemble members
        ens_cal = mu_ens + (std_cal / std_ens) * (ens - mu_ens)
        return ens_cal
