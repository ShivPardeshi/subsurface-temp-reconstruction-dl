"""
Temporal alignment utilities.
Standardizes multi-source temporal sequences to daily frequency.
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import List, Tuple, Union, Optional
from datetime import datetime

def generate_daily_time_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Generate daily DatetimeIndex between start_date and end_date (inclusive).
    """
    return pd.date_range(start=start_date, end=end_date, freq="D")

def extract_daily_slice(
    ds: xr.Dataset,
    target_date: Union[str, pd.Timestamp],
    time_coord_name: str = "time"
) -> xr.Dataset:
    """
    Extracts or averages data for a specific target day.
    """
    target_dt = pd.to_datetime(target_date)
    start_dt = target_dt.floor("D")
    end_dt = target_dt.ceil("D")

    if time_coord_name not in ds.coords and time_coord_name not in ds.dims:
        return ds

    time_vals = pd.to_datetime(ds[time_coord_name].values)
    mask = (time_vals >= start_dt) & (time_vals < end_dt + pd.Timedelta(days=1))

    if not np.any(mask):
        # Fallback: nearest timestamp if within 1 day
        time_diffs = np.abs(time_vals - target_dt)
        min_idx = np.argmin(time_diffs)
        if time_diffs[min_idx] <= pd.Timedelta(days=1):
            return ds.isel({time_coord_name: min_idx})
        raise KeyError(f"No valid observations found for date {target_date}")

    selected = ds.isel({time_coord_name: np.where(mask)[0]})
    if len(selected[time_coord_name]) > 1:
        # Average sub-daily observations (e.g. 6-hourly winds to daily mean)
        return selected.mean(dim=time_coord_name, keep_attrs=True)
    else:
        return selected.squeeze(dim=time_coord_name)
