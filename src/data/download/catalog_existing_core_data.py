"""
Scans and catalogs all downloaded raw datasets in data/raw/
Outputs a comprehensive manifest: data/raw/core_data_manifest.json
"""

import os
import sys
import glob
import json
from datetime import datetime
from typing import Dict, Any, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import xarray as xr

from src.utils.io_utils import open_netcdf_safe
from src.utils.logging_config import setup_logger

logger = setup_logger("catalog_raw_data")

RAW_DATA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
)

def inspect_netcdf(filepath: str) -> Dict[str, Any]:
    """Extract metadata, coordinates, variables and date ranges from a NetCDF file."""
    meta = {
        "file": os.path.basename(filepath),
        "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
        "variables": [],
        "dims": {},
        "time_min": None,
        "time_max": None,
        "time_steps": 0,
        "lat_range": None,
        "lon_range": None,
    }

    try:
        ds = open_netcdf_safe(filepath)
        meta["variables"] = list(ds.data_vars.keys())
        meta["dims"] = {str(k): int(v) for k, v in ds.sizes.items()}

        # Find time coord
        for t_name in ["time", "valid_time", "Time"]:
            if t_name in ds.coords or t_name in ds.dims:
                t_vals = ds[t_name].values
                if len(t_vals) > 0:
                    meta["time_steps"] = len(t_vals)
                    try:
                        meta["time_min"] = str(t_vals.min())[:19]
                        meta["time_max"] = str(t_vals.max())[:19]
                    except Exception:
                        meta["time_min"] = str(t_vals[0])
                        meta["time_max"] = str(t_vals[-1])
                break

        # Find lat/lon coords
        lat_arr = None
        for lat_name in ["latitude", "lat", "LATITUDE", "Lat"]:
            if lat_name in ds.coords:
                lat_arr = ds[lat_name].values
                break
        if lat_arr is not None and len(lat_arr) > 0:
            meta["lat_range"] = [float(np.nanmin(lat_arr)), float(np.nanmax(lat_arr))]

        lon_arr = None
        for lon_name in ["longitude", "lon", "LONGITUDE", "Lon"]:
            if lon_name in ds.coords:
                lon_arr = ds[lon_name].values
                break
        if lon_arr is not None and len(lon_arr) > 0:
            meta["lon_range"] = [float(np.nanmin(lon_arr)), float(np.nanmax(lon_arr))]

        ds.close()
    except Exception as e:
        meta["error"] = str(e)

    return meta

def catalog_dataset_folder(folder_path: str, name: str) -> Dict[str, Any]:
    """Catalog all files inside a dataset folder."""
    logger.info(f"Scanning category: {name} in {folder_path}...")
    if not os.path.exists(folder_path):
        return {"status": "MISSING", "file_count": 0, "total_size_mb": 0}

    all_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.startswith("._") and f.endswith((".nc", ".nc4", ".csv", ".txt")):
                all_files.append(os.path.join(root, f))

    if not all_files:
        return {"status": "EMPTY", "file_count": 0, "total_size_mb": 0}

    total_size = sum(os.path.getsize(f) for f in all_files) / (1024 * 1024)

    # Sample inspect NetCDF files (first and last)
    nc_files = [f for f in all_files if f.endswith((".nc", ".nc4"))]
    samples = []
    if nc_files:
        samples.append(inspect_netcdf(nc_files[0]))
        if len(nc_files) > 1:
            samples.append(inspect_netcdf(nc_files[-1]))

    return {
        "status": "PRESENT",
        "file_count": len(all_files),
        "total_size_mb": round(total_size, 2),
        "sample_metadata": samples,
    }

def generate_manifest(raw_root: str = RAW_DATA_ROOT) -> Dict[str, Any]:
    """Generates and writes core_data_manifest.json."""
    logger.info(f"Starting raw data cataloging from: {raw_root}")

    categories = [
        "glorys",
        "argo",
        "sst_sss_ssh_currents_winds",
        "river_discharge",
        "precipitation",
        "heat_flux",
        "wind_curl",
        "chlorophyll",
        "bathymetry",
        "landmask",
        "climate_indices",
        "ibtracs",
    ]

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "raw_root": raw_root,
        "categories": {},
    }

    for cat in categories:
        cat_path = os.path.join(raw_root, cat)
        manifest["categories"][cat] = catalog_dataset_folder(cat_path, cat)

    output_path = os.path.join(raw_root, "core_data_manifest.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Catalog saved to {output_path}")
    return manifest

if __name__ == "__main__":
    generate_manifest()
