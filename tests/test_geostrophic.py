"""
Tests for Geostrophic and Ageostrophic currents.
"""

import numpy as np
import pytest
from src.features.geostrophic import compute_geostrophic_currents
from src.features.ageostrophic import compute_ageostrophic_currents

def test_geostrophic_equatorial_tapering():
    # SSH slope that would otherwise cause infinity at latitude 0
    lat = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 10.0, 20.0], dtype=np.float32)
    lon = np.array([60.0, 61.0, 62.0], dtype=np.float32)
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    # Linear slope d_eta / dy = 0.01 m per degree
    ssh = 0.01 * lat_mesh

    u_g, v_g = compute_geostrophic_currents(ssh, lat, lon, equatorial_blend_deg=2.0)

    # Assert no NaNs or Infinities
    assert not np.isnan(u_g).any()
    assert not np.isnan(v_g).any()
    assert not np.isinf(u_g).any()
    assert not np.isinf(v_g).any()

    # Velocity at equator (lat index 2) should be 0 due to tapering
    assert np.isclose(u_g[2, 1], 0.0, atol=1e-3)
    assert np.isclose(v_g[2, 1], 0.0, atol=1e-3)

    # Away from equator, geostrophic velocity should be non-zero
    assert np.abs(u_g[-1, 1]) > 0.0

def test_ageostrophic_residual():
    u_obs = np.array([[2.0, 3.0], [1.0, 0.5]], dtype=np.float32)
    v_obs = np.array([[1.0, -1.0], [0.0, 0.2]], dtype=np.float32)
    u_geo = np.array([[1.5, 2.0], [0.8, 0.5]], dtype=np.float32)
    v_geo = np.array([[0.5, -0.5], [0.0, 0.1]], dtype=np.float32)

    u_a, v_a = compute_ageostrophic_currents(u_obs, v_obs, u_geo, v_geo)

    assert np.allclose(u_a, [[0.5, 1.0], [0.2, 0.0]])
    assert np.allclose(v_a, [[0.5, -0.5], [0.0, 0.1]])
