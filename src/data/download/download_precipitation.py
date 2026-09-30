"""
Ingest and download verification helpers for external datasets.
"""

import os
from src.utils.logging_config import setup_logger

logger = setup_logger("download_external")

def verify_folder(folder_name: str) -> None:
    raw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw"))
    p = os.path.join(raw_root, folder_name)
    os.makedirs(p, exist_ok=True)
    files = [f for f in os.listdir(p) if not f.startswith("._")]
    logger.info(f"Verified dataset {folder_name}: {len(files)} files present.")

if __name__ == "__main__":
    for cat in ["precipitation", "wind_curl", "chlorophyll", "bathymetry", "landmask"]:
        verify_folder(cat)
