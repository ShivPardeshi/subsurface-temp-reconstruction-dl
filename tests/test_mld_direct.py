"""Unit tests for direct profile-based Mixed Layer Depth (MLD) calculation."""

import numpy as np
import pytest
from src.products.mld_direct import compute_profile_mld_direct, compute_mld_field


def test_analytical_linear_crossing_mld():
    """Verify de Boyer Montégut linear crossing on known synthetic profile.

    Setup:
    T(10m) = 28.0°C.
    Threshold = 28.0 - 0.2 = 27.8°C.
    T(30m) = 28.0°C.
    T(50m) = 27.6°C. (Drop of 0.4°C over 20m interval).
    Crossing where T = 27.8°C:
    z = 30 + (28.0 - 27.8)/(28.0 - 27.6) * (50 - 30) = 30 + 0.5 * 20 = 40.0m.
    """
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    temps = [28.1, 28.05, 28.0, 28.0, 28.0, 27.6, 26.5, 24.0, 20.0, 16.0]

    res = compute_profile_mld_direct(np.array(temps), depths)
    assert res["ref_temp_c"] == 28.0
    assert res["thresh_temp_c"] == 27.8
    assert pytest.approx(res["mld_direct_m"], abs=0.1) == 40.0


def test_deep_mixed_layer():
    """Verify MLD when the entire profile is well-mixed down to deep levels."""
    depths = [0, 5, 10, 20, 30, 50, 100, 200]
    temps = [25.0, 25.0, 25.0, 25.0, 25.0, 24.95, 24.90, 24.85]  # Max drop is only 0.15°C

    res = compute_profile_mld_direct(np.array(temps), depths)
    assert res["mld_direct_m"] == 200.0


def test_auxiliary_head_cross_check_logging():
    """Verify that auxiliary head predictions are included and labeled as secondary cross-check."""
    depths = [0, 5, 10, 20, 30, 50, 100]
    temps = [28.0, 28.0, 28.0, 27.9, 27.5, 26.0, 20.0]

    res = compute_profile_mld_direct(np.array(temps), depths, aux_head_pred=35.2)
    assert res["aux_head_mld_m"] == 35.2
    assert "secondary cross-check" in res["aux_head_confidence"]
