"""
Tests for 2-harmonic climatology fitting and anti-leakage assertions.
"""

import numpy as np
import pandas as pd
import pytest
from src.climatology.fit_climatology import fit_harmonic_climatology
from src.climatology.compute_anomaly import evaluate_climatology, compute_temperature_anomaly

def test_harmonic_fit_recovery():
    # Construct synthetic 365-day time series with known parameters: a0=25, a1=3, b1=2, a2=1, b2=-0.5
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    doy = dates.dayofyear.values
    omega = 2.0 * np.pi / 365.25

    true_params = [25.0, 3.0, 2.0, 1.0, -0.5]
    y = (true_params[0] +
         true_params[1] * np.cos(omega * doy) +
         true_params[2] * np.sin(omega * doy) +
         true_params[3] * np.cos(2.0 * omega * doy) +
         true_params[4] * np.sin(2.0 * omega * doy)) # (366,)

    y_tensor = y[:, np.newaxis, np.newaxis, np.newaxis] # (366, 1, 1, 1)

    fitted_coeffs = fit_harmonic_climatology(y_tensor, dates) # (5, 1, 1, 1)
    recovered = fitted_coeffs[:, 0, 0, 0]

    assert np.allclose(recovered, true_params, atol=1e-3)

def test_anti_leakage_assertion():
    # Dates spanning 2024 and 2025
    dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")
    y = np.ones((len(dates), 1, 1, 1), dtype=np.float32)

    # Climatology fitting with cutoff 2024-12-31 should fail if dates include 2025
    with pytest.raises(ValueError, match="Data leakage detected"):
        fit_harmonic_climatology(y, dates, training_end_date="2024-12-31")
