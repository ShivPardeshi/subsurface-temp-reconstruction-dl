"""
Downloader for ESA CCI River Discharge dataset (CEDA archive).
"""

import os
import requests
from src.utils.logging_config import setup_logger

logger = setup_logger("download_river_discharge")

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "river_discharge"))

def download_river_discharge(dest_dir: str = RAW_ROOT, force: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    logger.info(f"Checking ESA CCI River Discharge files in {dest_dir}...")
    existing = [f for f in os.listdir(dest_dir) if not f.startswith("._")]
    if existing and not force:
        logger.info(f"Found {len(existing)} items in river discharge folder. No download needed.")
        return

    logger.info("ESA CCI River Discharge data should be placed into data/raw/river_discharge/.")

if __name__ == "__main__":
    download_river_discharge()
