"""
Tests for grid generation utility.
"""

import numpy as np
import pytest
from src.utils.grid import get_target_grid, get_depth_levels, load_domain_config

def test_full_target_grid_shape_and_bounds():
    config = load_domain_config()
    lat, lon = get_target_grid(config, toy_mode=False)

    # Check shapes: (112,) and (240,)
    assert lat.shape == (112,), f"Expected lat shape (112,), got {lat.shape}"
    assert lon.shape == (240,), f"Expected lon shape (240,), got {lon.shape}"

    # Check bounds
    assert np.isclose(lat[0], 2.0)
    assert np.isclose(lat[-1], 29.75)
    assert np.isclose(lon[0], 45.0)
    assert np.isclose(lon[-1], 104.75)

    # Check step resolution
    lat_diffs = np.diff(lat)
    lon_diffs = np.diff(lon)
    assert np.allclose(lat_diffs, 0.25)
    assert np.allclose(lon_diffs, 0.25)

def test_toy_grid():
    lat, lon = get_target_grid(toy_mode=True)
    assert len(lat) > 0 and len(lon) > 0
    assert lat[0] >= 12.0 and lat[-1] <= 22.0
    assert lon[0] >= 82.0 and lon[-1] <= 92.0

def test_canonical_depths():
    depths = get_depth_levels()
    expected = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    assert len(depths) == 15
    assert np.array_equal(depths, expected)
