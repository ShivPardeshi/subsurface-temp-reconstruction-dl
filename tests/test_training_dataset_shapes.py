"""
Tests for Phase 2 Training Dataset assembly and shapes.
"""

import os
import pytest
import xarray as xr
import pandas as pd
from src.assemble.build_training_dataset import Phase2DatasetBuilder

def test_phase2_builder_toy_components(tmp_path):
    builder = Phase2DatasetBuilder(toy_mode=True)
    dates = ["2024-10-01", "2024-10-02"]
    out_dir = str(tmp_path / "phase2_test")

    out_paths = builder.build_all_components(
        date_list=dates,
        output_dir=out_dir,
        training_end_date="2024-12-31"
    )

    # 1. Inputs Zarr
    ds_in = xr.open_zarr(out_paths["inputs_zarr"])
    assert ds_in["inputs"].shape == (2, 25, len(builder.target_lat), len(builder.target_lon))

    # 2. Anomaly Zarr
    ds_anom = xr.open_zarr(out_paths["anomaly_zarr"])
    assert ds_anom["anomaly"].shape == (2, 15, len(builder.target_lat), len(builder.target_lon))

    # 3. Auxiliary Zarr
    ds_aux = xr.open_zarr(out_paths["auxiliary_zarr"])
    assert ds_aux["auxiliary_targets"].shape == (2, 4, len(builder.target_lat), len(builder.target_lon))

    # 4. Scalar conditioning
    df_scalar = pd.read_csv(out_paths["scalar_csv"])
    assert len(df_scalar) == 2
    assert "sin_doy" in df_scalar.columns
    assert "cos_doy" in df_scalar.columns
