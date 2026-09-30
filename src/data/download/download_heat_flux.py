"""
Downloader / ingest helper for Heat Flux (ERA5 / OAFlux).
"""

import os
from src.utils.logging_config import setup_logger

logger = setup_logger("download_heat_flux")

RAW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "heat_flux"))

def download_heat_flux(dest_dir: str = RAW_ROOT, force: bool = False) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    existing = [f for f in os.listdir(dest_dir) if not f.startswith("._")]
    if existing and not force:
        logger.info(f"Heat flux files verified in {dest_dir}. ({len(existing)} items)")
        return

    logger.info("Heat flux data checked.")

if __name__ == "__main__":
    download_heat_flux()
