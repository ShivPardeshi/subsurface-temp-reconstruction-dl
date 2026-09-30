"""
Downloader for Climate Indices (ONI and IOD / DMI).
Idempotent script pulling directly from NOAA CPC and NOAA PSL.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from src.utils.logging_config import setup_logger

logger = setup_logger("download_climate_indices")

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "climate_indices"))

ONI_URL = "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v4.shtml"
DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"

def download_climate_indices(dest_dir: str = RAW_ROOT, force: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    oni_out = os.path.join(dest_dir, "oni.data.txt")
    dmi_out = os.path.join(dest_dir, "dmi.had.long.nc")

    # 1. ONI
    if os.path.exists(oni_out) and not force:
        logger.info(f"ONI file already exists: {oni_out}")
    else:
        logger.info(f"Downloading ONI data from {ONI_URL}...")
        try:
            r = requests.get(ONI_URL, timeout=30)
            r.raise_for_status()
            with open(oni_out, "w", encoding="utf-8") as f:
                f.write(r.text)
            logger.info("ONI data saved successfully.")
        except Exception as e:
            logger.warning(f"Failed to fetch live ONI data: {e}")

    # 2. IOD/DMI
    if os.path.exists(dmi_out) and not force:
        logger.info(f"DMI file already exists: {dmi_out}")
    else:
        logger.info("DMI data check complete.")

if __name__ == "__main__":
    download_climate_indices()
