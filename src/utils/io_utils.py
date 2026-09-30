"""
I/O utilities for reading NetCDF and reading/writing chunked Zarr datasets.
"""

import os
import xarray as xr
import zarr
import numpy as np
from typing import Optional, Dict, Any, List

def open_netcdf_safe(filepath: str, decode_times: bool = True, **kwargs) -> xr.Dataset:
    """
    Safely opens a NetCDF file, trying h5netcdf and scipy engines.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    engines_to_try = ["h5netcdf", "scipy"]
    last_error = None

    for engine in engines_to_try:
        try:
            return xr.open_dataset(filepath, engine=engine, decode_times=decode_times, **kwargs)
        except Exception as e:
            last_error = e

    # Try with decode_times=False if time decoding failed
    for engine in engines_to_try:
        try:
            return xr.open_dataset(filepath, engine=engine, decode_times=False, **kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to open NetCDF file {filepath}. Last error: {last_error}")

def save_datacube_to_zarr(
    ds: xr.Dataset,
    zarr_path: str,
    time_chunk_size: int = 30,
    overwrite: bool = True
) -> None:
    """
    Saves an xarray Dataset containing the data cube to a chunked Zarr store.

    Args:
        ds: xarray Dataset with dimensions (time, channel, lat, lon)
        zarr_path: Output directory path for the Zarr store
        time_chunk_size: Chunk size along the time dimension
        overwrite: If True, replaces existing Zarr store
    """
    os.makedirs(os.path.dirname(os.path.abspath(zarr_path)), exist_ok=True)

    # Configure chunking
    chunks = {}
    if "time" in ds.dims:
        chunks["time"] = min(time_chunk_size, len(ds.time))
    if "channel" in ds.dims:
        chunks["channel"] = len(ds.channel)
    if "lat" in ds.dims:
        chunks["lat"] = len(ds.lat)
    if "lon" in ds.dims:
        chunks["lon"] = len(ds.lon)

    ds_chunked = ds.chunk(chunks)

    mode = "w" if overwrite else "w-"
    
    # Try using numcodecs for compression if installed
    try:
        from numcodecs import Blosc
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        encoding = {
            var_name: {"compressor": compressor} for var_name in ds_chunked.data_vars
        }
        ds_chunked.to_zarr(zarr_path, mode=mode, encoding=encoding, consolidated=True)
    except Exception:
        # Fallback to standard xarray Zarr serialization
        ds_chunked.to_zarr(zarr_path, mode=mode, consolidated=True)

def open_datacube_zarr(zarr_path: str) -> xr.Dataset:
    """
    Opens an assembled Zarr data cube with consolidated metadata.
    """
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    return xr.open_zarr(zarr_path, consolidated=True)
