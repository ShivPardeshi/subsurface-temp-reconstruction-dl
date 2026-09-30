"""
Tests for auxiliary physical targets: MLD, Barrier Layer Thickness, Salinity Maximum.
"""

import numpy as np
import pytest
from src.auxiliary_targets.mixed_layer_depth import compute_mld_profile
from src.auxiliary_targets.barrier_layer_thickness import compute_barrier_layer_profile
from src.auxiliary_targets.salinity_maximum import compute_salinity_maximum_profile

def test_mld_linear_interpolation():
    depths = np.array([0, 10, 20, 30, 50, 75, 100], dtype=np.float32)
    # 10m temp = 28.0°C. Drops by 0.2°C (to 27.8°C) halfway between 20m (27.9°C) and 30m (27.7°C) -> 25m
    temp_profile = np.array([28.2, 28.0, 27.9, 27.7, 26.0, 24.0, 20.0], dtype=np.float32)

    mld = compute_mld_profile(temp_profile, depths, delta_t_threshold=0.2, ref_depth=10.0)
    assert np.isclose(mld, 25.0, atol=0.1)

def test_barrier_layer_profile():
    depths = np.array([0, 10, 20, 30, 50, 75, 100], dtype=np.float32)
    # Deep isothermal layer (ILD ~ 50m) with shallow halocline/freshwater cap (MLD_rho ~ 20m)
    temp_profile = np.array([28.0, 28.0, 28.0, 28.0, 27.8, 25.0, 20.0], dtype=np.float32) # ILD ~ 50m
    sal_profile = np.array([31.0, 31.0, 34.0, 35.0, 35.0, 35.0, 35.0], dtype=np.float32)   # Salinity jump at 20m

    blt = compute_barrier_layer_profile(temp_profile, sal_profile, depths, ref_depth=10.0)
    # BLT should be positive (> 15m)
    assert blt > 15.0

def test_salinity_maximum_pgw_profile():
    depths = np.array([0, 50, 100, 150, 200, 250, 300, 400, 500], dtype=np.float32)
    # Salinity peak at 250m
    sal_profile = np.array([35.5, 35.6, 35.8, 36.1, 36.4, 36.9, 36.3, 35.8, 35.4], dtype=np.float32)

    s_depth, s_strength = compute_salinity_maximum_profile(sal_profile, depths, min_depth=100.0, max_depth=400.0)
    assert np.isclose(s_depth, 250.0)
    assert s_strength > 0.5
