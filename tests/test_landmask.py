"""
Tests for landmask and bathymetry derivation.
"""

import numpy as np
import pytest
from src.data.harmonize.landmask import derive_land_ocean_mask_and_bathymetry

def test_land_ocean_classification():
    # Synthetic elevation where < 0 is ocean, >= 0 is land
    elev = np.array([
        [-4000.0, -2000.0, -100.0],
        [0.0, 500.0, 2000.0]
    ], dtype=np.float32)

    log_bath, ocean_mask = derive_land_ocean_mask_and_bathymetry(elev)

    # Ocean mask should be 1 for negative elevation, 0 for positive
    expected_mask = np.array([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)
    assert np.array_equal(ocean_mask, expected_mask)

    # Log bathymetry should be log10(1 + depth) for ocean and 0 for land
    assert np.isclose(log_bath[0, 0], np.log10(4001.0))
    assert np.isclose(log_bath[0, 1], np.log10(2001.0))
    assert log_bath[1, 0] == 0.0
    assert log_bath[1, 1] == 0.0
