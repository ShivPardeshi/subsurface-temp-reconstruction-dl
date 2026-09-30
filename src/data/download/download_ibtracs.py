"""
Downloader for IBTrACS North Indian cyclone dataset.
Idempotent script pulling from NOAA NCEI.
"""

import os
import requests
from src.utils.logging_config import setup_logger

logger = setup_logger("download_ibtracs")

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "ibtracs"))
IBTRACS_CSV_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"

def download_ibtracs(dest_dir: str = RAW_ROOT, force: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    out_file = os.path.join(dest_dir, "ibtracs.NI.list.v04r01.csv")

    if os.path.exists(out_file) and not force:
        logger.info(f"IBTrACS file already exists: {out_file}")
        return

    logger.info(f"Downloading IBTrACS North Indian basin from {IBTRACS_CSV_URL}...")
    try:
        r = requests.get(IBTRACS_CSV_URL, timeout=60, stream=True)
        r.raise_for_status()
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("IBTrACS download completed successfully.")
    except Exception as e:
        logger.warning(f"Error downloading IBTrACS: {e}")

if __name__ == "__main__":
    download_ibtracs()
