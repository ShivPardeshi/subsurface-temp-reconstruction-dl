"""Unit tests for Tropical Cyclone Heat Potential (TCHP) calculation."""

import numpy as np
import pytest
from src.products.tchp import (
    compute_profile_tchp,
    find_d26_isotherm_depth,
    compute_tchp_field,
    DEFAULT_RHO,
    DEFAULT_CP,
    J_M2_TO_KJ_CM2,
)


def test_analytical_linear_thermocline_tchp():
    """Verify TCHP against an analytically solvable linear thermocline profile.

    Profile:
    T(0m) = 30.0°C, drops linearly to 26.0°C at exactly D26 = 50.0m.
    Below 50m, T drops to 15°C at 200m, etc.

    Analytical Solution:
    excess(z) = 4.0 * (1 - z/50)
    integral_0^50 excess(z) dz = 4 * 25 = 100.0 K*m
    TCHP = rho * c_p * 100.0 * 1e-7 = 1025 * 3990 * 100 * 1e-7 = 40.8975 kJ/cm^2
    """
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    # Linear drop from 30°C to 26°C at 50m: T(z) = 30 - 4*(z/50)
    temps = [30.0 - 4.0 * (z / 50.0) if z <= 50 else 26.0 - 11.0 * ((z - 50) / 150.0) for z in depths]

    d26 = find_d26_isotherm_depth(np.array(temps), depths)
    assert pytest.approx(d26, abs=0.1) == 50.0

    tchp, d26_ret = compute_profile_tchp(np.array(temps), depths)
    expected_tchp = DEFAULT_RHO * DEFAULT_CP * 100.0 * J_M2_TO_KJ_CM2  # 40.8975 kJ/cm^2
    assert pytest.approx(tchp, rel=1e-3) == expected_tchp
    assert pytest.approx(d26_ret, abs=0.1) == 50.0


def test_cold_sst_zero_tchp():
    """Verify that if SST < 26.0°C, TCHP is strictly 0.0 kJ/cm^2 and D26 is 0.0m."""
    depths = [0, 10, 20, 50, 100]
    temps = [24.5, 24.0, 23.5, 20.0, 15.0]

    tchp, d26 = compute_profile_tchp(np.array(temps), depths)
    assert tchp == 0.0
    assert d26 == 0.0


def test_deep_isotherm_crossing_interpolation():
    """Verify linear interpolation when 26°C crossing falls strictly between depth levels."""
    depths = [0, 20, 50, 100]
    # At 20m, T = 28°C; at 50m, T = 24°C.
    # Crossing 26°C should be at exactly midway: 20 + 0.5 * 30 = 35.0m
    temps = [28.0, 28.0, 24.0, 18.0]

    d26 = find_d26_isotherm_depth(np.array(temps), depths)
    assert pytest.approx(d26, abs=0.01) == 35.0


def test_tchp_field_shape_and_masking():
    """Verify 2D spatial field calculation and landmasking."""
    field = np.full((15, 10, 10), 28.0, dtype=np.float32)
    # Put 25°C below 50m
    field[5:, ...] = 20.0  # depths >= 50m are cold

    mask = np.ones((10, 10), dtype=bool)
    mask[0, 0] = False  # Land cell

    tchp_map, d26_map = compute_tchp_field(field, mask=mask)
    assert tchp_map.shape == (10, 10)
    assert d26_map.shape == (10, 10)
    assert tchp_map[0, 0] == 0.0  # Masked land cell
    assert tchp_map[5, 5] > 0.0   # Ocean cell
