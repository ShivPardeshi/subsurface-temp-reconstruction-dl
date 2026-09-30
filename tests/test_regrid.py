"""
Tests for regridding module.
"""

import numpy as np
import pytest
from src.data.harmonize.regrid import regrid_2d_array

def test_regrid_identity():
    # When source and target grids match exactly
    src_lat = np.linspace(10.0, 20.0, 20)
    src_lon = np.linspace(70.0, 80.0, 20)
    lon_mesh, lat_mesh = np.meshgrid(src_lon, src_lat)
    synthetic_field = np.sin(lat_mesh) + np.cos(lon_mesh)

    regridded = regrid_2d_array(synthetic_field, src_lat, src_lon, src_lat, src_lon)
    assert regridded.shape == (20, 20)
    assert np.allclose(regridded, synthetic_field, atol=1e-5)

def test_regrid_rescaling():
    # Coarse to fine regrid
    src_lat = np.linspace(2.0, 30.0, 50)
    src_lon = np.linspace(45.0, 105.0, 60)
    lon_mesh, lat_mesh = np.meshgrid(src_lon, src_lat)
    synthetic_field = np.sin(lat_mesh / 10.0)

    target_lat = np.linspace(2.0, 30.0, 112)
    target_lon = np.linspace(45.0, 105.0, 240)

    regridded = regrid_2d_array(synthetic_field, src_lat, src_lon, target_lat, target_lon)
    assert regridded.shape == (112, 240)
    assert not np.isnan(regridded).any()
    assert np.isclose(regridded.min(), synthetic_field.min(), atol=0.05)
    assert np.isclose(regridded.max(), synthetic_field.max(), atol=0.05)

def test_regrid_descending_latitude():
    # Handing source grids where latitude is in descending order (e.g. 30 -> 2)
    src_lat = np.linspace(30.0, 2.0, 30)
    src_lon = np.linspace(45.0, 105.0, 40)
    lon_mesh, lat_mesh = np.meshgrid(src_lon, src_lat)
    synthetic_field = lat_mesh

    target_lat = np.linspace(2.0, 30.0, 112)
    target_lon = np.linspace(45.0, 105.0, 240)

    regridded = regrid_2d_array(synthetic_field, src_lat, src_lon, target_lat, target_lon)
    assert regridded.shape == (112, 240)
    # Latitude should increase along axis 0
    assert regridded[0, 0] < regridded[-1, 0]
