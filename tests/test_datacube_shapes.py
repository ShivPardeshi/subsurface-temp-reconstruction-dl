"""
Tests for Data Cube shape and channel consistency.
"""

import numpy as np
import pytest
from src.data.harmonize.build_datacube import DataCubeBuilder, CHANNEL_NAMES

def test_datacube_toy_shape():
    builder = DataCubeBuilder(toy_mode=True)
    dates = ["2024-10-01", "2024-10-02"]
    ds = builder.build_cube_for_dates(dates)

    cube = ds["cube"]
    assert cube.dims == ("time", "channel", "lat", "lon")
    assert cube.shape[0] == 2
    assert cube.shape[1] == 25
    assert cube.shape[2] == len(builder.target_lat)
    assert cube.shape[3] == len(builder.target_lon)

    # Check coordinate labels
    assert list(ds["channel"].values) == CHANNEL_NAMES
    assert len(ds["time"]) == 2
