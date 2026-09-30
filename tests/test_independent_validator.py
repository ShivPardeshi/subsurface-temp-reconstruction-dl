"""
Unit tests for IndependentInSituValidator.
"""

import pytest
import numpy as np
from src.evaluation.independent_validator import (
    IndependentInSituValidator,
    InSituProfile,
)
from src.utils.grid import CANONICAL_DEPTHS, get_target_grid


def test_bilinear_interpolation():
    lat_grid, lon_grid = get_target_grid()
    validator = IndependentInSituValidator(lat_grid=lat_grid, lon_grid=lon_grid)

    # Synthetic 3D cube (15, H, W) where T(z, y, x) = z_val + y*0.1 + x*0.01
    H = len(lat_grid)
    W = len(lon_grid)
    D = len(CANONICAL_DEPTHS)

    pred_cube = np.zeros((D, H, W), dtype=np.float32)
    for d in range(D):
        pred_cube[d, :, :] = float(CANONICAL_DEPTHS[d])

    # Test center point
    test_lat = float(lat_grid[10] + 0.125)
    test_lon = float(lon_grid[20] + 0.125)

    profile = validator.bilinear_interpolate_profile(pred_cube, test_lat, test_lon)
    assert profile is not None
    assert profile.shape == (15,)
    for d in range(D):
        assert pytest.approx(profile[d], abs=1e-4) == float(CANONICAL_DEPTHS[d])


def test_out_of_bounds_interpolation():
    lat_grid, lon_grid = get_target_grid()
    validator = IndependentInSituValidator(lat_grid=lat_grid, lon_grid=lon_grid)
    pred_cube = np.zeros((15, len(lat_grid), len(lon_grid)), dtype=np.float32)

    # Far out of domain
    assert validator.bilinear_interpolate_profile(pred_cube, 45.0, 10.0) is None
    assert validator.bilinear_interpolate_profile(pred_cube, -10.0, 80.0) is None


def test_interpolate_obs_to_canonical_depths():
    validator = IndependentInSituValidator()

    # Synthetic CTD sounding at irregular depths
    obs_depths = np.array([0, 8, 25, 60, 120, 250, 600, 1050], dtype=np.float32)
    # Linear temperature profile from 28°C down to 5°C
    obs_temps = 28.0 - (23.0 / 1050.0) * obs_depths

    interp_t = validator.interpolate_obs_to_canonical_depths(obs_depths, obs_temps)
    assert interp_t.shape == (15,)
    assert pytest.approx(interp_t[0], abs=0.1) == 28.0
    assert np.all(np.isfinite(interp_t))


def test_evaluate_profiles_end_to_end():
    lat_grid, lon_grid = get_target_grid()
    validator = IndependentInSituValidator(lat_grid=lat_grid, lon_grid=lon_grid)
    H, W, D = len(lat_grid), len(lon_grid), len(CANONICAL_DEPTHS)

    # Build mock prediction and climatology
    pred_cube = np.full((D, H, W), 20.0, dtype=np.float32)
    clim_cube = np.full((D, H, W), 22.0, dtype=np.float32)

    date = "2025-05-15"
    preds_by_date = {date: pred_cube}
    clim_by_date = {date: clim_cube}

    # In-situ observation exactly matching model (20°C everywhere)
    obs_depths = np.array(CANONICAL_DEPTHS, dtype=np.float32)
    obs_temps = np.full_like(obs_depths, 20.0)

    profile_independent = InSituProfile(
        profile_id="CRUISE_001_CAST_12",
        latitude=15.0,
        longitude=85.0,
        date_str=date,
        depths_m=obs_depths,
        temperatures_c=obs_temps,
        platform_type="CTD_Cast",
        cruise_or_mission="Sagar_Kanya_SK340",
        is_assimilated_in_glorys=False,
    )

    results = validator.evaluate_profiles(preds_by_date, clim_by_date, [profile_independent])
    assert results["status"] == "SUCCESS"
    assert results["metadata"]["matched_profiles"] == 1
    assert results["metadata"]["is_strictly_independent"] is True
    assert pytest.approx(results["overall_metrics"]["model_rmse_c"], abs=1e-4) == 0.0
    assert pytest.approx(results["overall_metrics"]["climatology_rmse_c"], abs=1e-4) == 2.0
    assert pytest.approx(results["overall_metrics"]["murphy_skill_score"], abs=1e-4) == 1.0
