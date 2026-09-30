"""
Data Cube Assembly Module for OceanEmbed Phase 1.
Harmonizes multi-source daily surface satellite observations into a 25-channel data cube
and writes out to a standardized chunked Zarr store.
"""

import os
import time
import glob
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Any, List, Optional, Tuple

from src.utils.grid import get_target_grid, load_domain_config
from src.utils.io_utils import open_netcdf_safe, save_datacube_to_zarr
from src.utils.logging_config import setup_logger
from src.data.harmonize.regrid import regrid_2d_array
from src.data.harmonize.landmask import load_and_regrid_gebco, derive_land_ocean_mask_and_bathymetry
from src.data.harmonize.region_masks import generate_region_masks
from src.data.harmonize.missingness import compute_missingness_channel

logger = setup_logger("build_datacube")

CHANNEL_NAMES = [
    "sst",                          # 1
    "sss",                          # 2
    "ssh",                          # 3
    "wind_u",                       # 4
    "wind_v",                       # 5
    "current_u",                    # 6
    "current_v",                    # 7
    "geostrophic_u",                # 8 (Phase 2 derived placeholder)
    "geostrophic_v",                # 9 (Phase 2 derived placeholder)
    "ageostrophic_u",               # 10 (Phase 2 derived placeholder)
    "ageostrophic_v",               # 11 (Phase 2 derived placeholder)
    "wind_stress_curl",             # 12
    "wind_mixing_energy",           # 13
    "precipitation",                # 14
    "latent_heat_flux",             # 15
    "e_minus_p_flux",               # 16 (Phase 2 derived placeholder)
    "chlorophyll",                  # 17
    "river_plume_field",            # 18 (Phase 2 derived placeholder)
    "missingness_mask",             # 19
    "bathymetry_log",               # 20 (Static)
    "land_ocean_mask",              # 21 (Static)
    "region_arabian_sea",           # 22 (Static)
    "region_bay_of_bengal",         # 23 (Static)
    "region_confluence_zone",       # 24 (Static)
    "region_open_ocean"             # 25 (Static)
]

RAW_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
)

class DataCubeBuilder:
    def __init__(self, config_path: Optional[str] = None, toy_mode: bool = False, raw_root: str = RAW_ROOT):
        self.config = load_domain_config(config_path) if config_path else load_domain_config()
        self.toy_mode = toy_mode
        self.raw_root = raw_root
        self.target_lat, self.target_lon = get_target_grid(self.config, toy_mode=toy_mode)
        self.H = len(self.target_lat)
        self.W = len(self.target_lon)
        self.n_channels = len(CHANNEL_NAMES)

        logger.info(f"Initialized DataCubeBuilder (toy_mode={toy_mode}, grid=({self.H}, {self.W}), channels={self.n_channels})")
        self._init_static_channels()

    def _init_static_channels(self):
        """Precomputes static channels 20-25."""
        # 1. Bathymetry & Land Mask (Channels 20 & 21)
        gebco_candidates = glob.glob(os.path.join(self.raw_root, "bathymetry", "*", "*sub_ice*.nc")) + \
                           glob.glob(os.path.join(self.raw_root, "bathymetry", "*", "*.nc"))
        if gebco_candidates:
            gebco_path = gebco_candidates[0]
            try:
                elev = load_and_regrid_gebco(gebco_path, self.target_lat, self.target_lon)
                self.log_bathymetry, self.ocean_mask = derive_land_ocean_mask_and_bathymetry(elev)
            except Exception as e:
                logger.warning(f"Failed to load GEBCO from {gebco_path}: {e}. Creating synthetic bathymetry.")
                self._synthetic_bathymetry()
        else:
            logger.warning("No GEBCO bathymetry file found in raw. Creating synthetic bathymetry.")
            self._synthetic_bathymetry()

        # 2. Soft Region Membership Maps (Channels 22, 23, 24, 25)
        self.as_mask, self.bob_mask, self.conf_mask, self.open_mask = generate_region_masks(
            self.target_lat, self.target_lon, self.config
        )

        # 3. High-performance file caching and indexing
        self._init_file_caches()

    def _synthetic_bathymetry(self):
        # Fallback ocean mask (assume Indian landmass between lat 8-28N and lon 72-88E)
        lat_mesh, lon_mesh = np.meshgrid(self.target_lon, self.target_lat)
        land = (lat_mesh >= 8.5) & (lat_mesh <= 26.0) & (lon_mesh >= 72.5) & (lon_mesh <= 87.0)
        self.ocean_mask = (~land).astype(np.float32)
        depth = np.where(self.ocean_mask > 0.5, 3000.0, 0.0)
        self.log_bathymetry = np.where(self.ocean_mask > 0.5, np.log10(1.0 + depth), 0.0).astype(np.float32)

    def assemble_day(self, date_str: str) -> np.ndarray:
        """
        Assembles a 25-channel 2D spatial stack for a specific date: (25, H, W).
        """
        channels = np.zeros((self.n_channels, self.H, self.W), dtype=np.float32)
        dt = pd.to_datetime(date_str)
        date_tag = dt.strftime("%Y%m%d")

        core_channels = []

        # 1. SST (OSTIA)
        sst_arr = self._load_sst(dt)
        channels[0] = sst_arr
        core_channels.append(sst_arr)

        # 2. SSS (SMAP/SMOS)
        sss_arr = self._load_sss(dt)
        channels[1] = sss_arr
        core_channels.append(sss_arr)

        # 3. SSH (DUACS)
        ssh_arr = self._load_ssh(dt)
        channels[2] = ssh_arr
        core_channels.append(ssh_arr)

        # 4 & 5. Winds U & V (CCMP)
        w_u, w_v = self._load_winds(dt)
        channels[3] = w_u
        channels[4] = w_v
        core_channels.extend([w_u, w_v])

        # 6 & 7. Currents U & V (OSCAR)
        c_u, c_v = self._load_currents(dt)
        channels[5] = c_u
        channels[6] = c_v
        core_channels.extend([c_u, c_v])

        # 8-11: Geostrophic / Ageostrophic placeholders (Phase 2 derived)
        channels[7] = np.zeros((self.H, self.W), dtype=np.float32)
        channels[8] = np.zeros((self.H, self.W), dtype=np.float32)
        channels[9] = np.zeros((self.H, self.W), dtype=np.float32)
        channels[10] = np.zeros((self.H, self.W), dtype=np.float32)

        # 12. Wind Stress Curl / Ekman pumping
        channels[11] = self._load_wind_curl(dt)

        # 13. Wind-mixing energy (~ |v|^3)
        wind_speed_sq = np.square(np.nan_to_num(w_u)) + np.square(np.nan_to_num(w_v))
        channels[12] = (wind_speed_sq * np.sqrt(wind_speed_sq)).astype(np.float32)

        # 14. Precipitation (GPM IMERG)
        channels[13] = self._load_precipitation(dt)

        # 15. Latent Heat Flux (ERA5)
        channels[14] = self._load_heat_flux(dt)

        # 16. E - P moisture flux (Phase 2 placeholder)
        channels[15] = np.zeros((self.H, self.W), dtype=np.float32)

        # 17. Chlorophyll-a
        channels[16] = self._load_chlorophyll(dt)

        # 18. River plume influence field (Phase 2 placeholder)
        channels[17] = np.zeros((self.H, self.W), dtype=np.float32)

        # 19. Missingness Mask
        channels[18] = compute_missingness_channel(core_channels, self.ocean_mask)

        # 20. Static Bathymetry (log-scaled)
        channels[19] = self.log_bathymetry

        # 21. Static Land/Ocean Mask
        channels[20] = self.ocean_mask

        # 22-25. Soft Region Membership Maps
        channels[21] = self.as_mask
        channels[22] = self.bob_mask
        channels[23] = self.conf_mask
        channels[24] = self.open_mask

        # Mask land cells to 0 across dynamic channels
        for c in range(19):
            channels[c] = np.where(self.ocean_mask > 0.5, channels[c], 0.0)

        return channels

    def _init_file_caches(self):
        """Initializes multi-time dataset handles and indexes daily filenames."""
        self._cached_sst = None
        self._sst_times = None
        self._cached_sss = None
        self._sss_times = None
        self._cached_ssh = None
        self._ssh_times = None
        self._cached_hf = None
        self._hf_times = None
        self._hf_tcoord = "time"
        self._cached_curl = None
        self._curl_times = None
        self._curl_tcoord = "time"
        self._curl_var = None

        self._wind_map = {}
        self._current_map = {}
        self._precip_map = {}
        self._chlor_map = {}

        if self.toy_mode:
            return

        try:
            # 1. SST
            sst_files = glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "india_sst", "*.nc"))
            if sst_files:
                self._cached_sst = open_netcdf_safe(sst_files[0])
                self._sst_times = pd.to_datetime(self._cached_sst["time"].values)

            # 2. SSS
            sss_files = glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "india_sss", "*.nc"))
            if sss_files:
                self._cached_sss = open_netcdf_safe(sss_files[0])
                self._sss_times = pd.to_datetime(self._cached_sss["time"].values)

            # 3. SSH
            ssh_files = glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "india_ssh", "*.nc"))
            if ssh_files:
                self._cached_ssh = open_netcdf_safe(ssh_files[0])
                self._ssh_times = pd.to_datetime(self._cached_ssh["time"].values)

            # 4. Latent Heat Flux
            hf_files = glob.glob(os.path.join(self.raw_root, "heat_flux", "*", "*.nc"))
            if hf_files:
                self._cached_hf = open_netcdf_safe(hf_files[0])
                self._hf_tcoord = "valid_time" if "valid_time" in self._cached_hf.coords else "time"
                self._hf_times = pd.to_datetime(self._cached_hf[self._hf_tcoord].values)

            # 5. Wind Stress Curl
            curl_files = glob.glob(os.path.join(self.raw_root, "wind_curl", "*.nc"))
            if curl_files:
                self._cached_curl = open_netcdf_safe(curl_files[0])
                self._curl_tcoord = "time" if "time" in self._cached_curl.coords else list(self._cached_curl.coords.keys())[0]
                self._curl_times = pd.to_datetime(self._cached_curl[self._curl_tcoord].values)
                self._curl_var = list(self._cached_curl.data_vars.keys())[0]

            # Index daily files for instant lookup
            for f in glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "*CCMP*", "*.nc*")):
                base = os.path.basename(f)
                for token in base.split("."):
                    for sub in token.split("_"):
                        if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                            self._wind_map[sub] = f
                            break

            for f in glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "*OSCAR*", "*.nc")):
                base = os.path.basename(f)
                for token in base.split("_"):
                    sub = token.replace(".nc", "")
                    if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                        self._current_map[sub] = f
                        break

            for f in glob.glob(os.path.join(self.raw_root, "precipitation", "*GPM*", "*.nc*")):
                base = os.path.basename(f)
                for token in base.split("."):
                    sub = token.split("-")[0]
                    if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                        self._precip_map[sub] = f
                        break

            for f in glob.glob(os.path.join(self.raw_root, "chlorophyll", "*", "*.nc")):
                base = os.path.basename(f)
                for token in base.split("."):
                    for sub in token.split("_"):
                        if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                            self._chlor_map[sub] = f
                            break
            logger.info(f"Initialized caches & maps: Winds={len(self._wind_map)}, OSCAR={len(self._current_map)}, GPM={len(self._precip_map)}, Chlor={len(self._chlor_map)}")
        except Exception as e:
            logger.warning(f"Error during file cache initialization: {e}")

    def build_cube_for_dates(self, date_list: List[str]) -> xr.Dataset:
        """
        Builds full (T, 25, H, W) data cube across date_list with periodic progress logging.
        """
        logger.info(f"Building data cube across {len(date_list)} days ({date_list[0]} to {date_list[-1]})...")
        time_index = pd.to_datetime(date_list)
        all_days = []
        t_start = time.time()

        for i, d_str in enumerate(date_list):
            day_stack = self.assemble_day(d_str)
            all_days.append(day_stack)
            if (i + 1) % 30 == 0 or (i + 1) == len(date_list):
                elapsed = time.time() - t_start
                rate = elapsed / (i + 1)
                eta = rate * (len(date_list) - (i + 1))
                logger.info(f"  Processed {i+1}/{len(date_list)} days ({d_str}) | Elapsed: {elapsed:.1f}s | Rate: {rate:.2f}s/day | ETA: {eta:.1f}s")

        data_array = np.stack(all_days, axis=0) # (T, 25, H, W)

        ds = xr.Dataset(
            data_vars={
                "cube": (("time", "channel", "lat", "lon"), data_array)
            },
            coords={
                "time": time_index,
                "channel": CHANNEL_NAMES,
                "lat": self.target_lat,
                "lon": self.target_lon
            },
            attrs={
                "title": "OceanEmbed Phase 1 Harmonized Surface Data Cube",
                "domain": f"{self.target_lat.min():.2f}N to {self.target_lat.max():.2f}N, {self.target_lon.min():.2f}E to {self.target_lon.max():.2f}E",
                "resolution": "0.25 degree",
                "channel_count": self.n_channels,
                "toy_mode": str(self.toy_mode)
            }
        )
        return ds

    # --- Source loader helpers (cached & fast) ---
    def _load_sst(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_sst is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._sst_times - dt)))
                src_lat = self._cached_sst["latitude"].values
                src_lon = self._cached_sst["longitude"].values
                raw_slice = self._cached_sst["analysed_sst"].isel(time=match_idx).values
                return regrid_2d_array(raw_slice, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"SST load error: {e}")
        return np.full((self.H, self.W), 298.15, dtype=np.float32)

    def _load_sss(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_sss is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._sss_times - dt)))
                src_lat = self._cached_sss["latitude"].values
                src_lon = self._cached_sss["longitude"].values
                var = "sss" if "sss" in self._cached_sss else ("sos" if "sos" in self._cached_sss else list(self._cached_sss.data_vars.keys())[0])
                raw_slice = self._cached_sss[var].isel(time=match_idx).squeeze().values
                return regrid_2d_array(raw_slice, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"SSS load error: {e}")
        return np.full((self.H, self.W), 35.0, dtype=np.float32)

    def _load_ssh(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_ssh is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._ssh_times - dt)))
                src_lat = self._cached_ssh["latitude"].values
                src_lon = self._cached_ssh["longitude"].values
                var = "sla" if "sla" in self._cached_ssh else "adt"
                raw_slice = self._cached_ssh[var].isel(time=match_idx).values
                return regrid_2d_array(raw_slice, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"SSH load error: {e}")
        return np.zeros((self.H, self.W), dtype=np.float32)

    def _load_winds(self, dt: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        tag = dt.strftime("%Y%m%d")
        f = self._wind_map.get(tag)
        if f:
            try:
                ds = open_netcdf_safe(f)
                src_lat = ds["latitude"].values
                src_lon = ds["longitude"].values
                u_mean = ds["uwnd"].mean(dim="time").values if "time" in ds.dims else ds["uwnd"].values
                v_mean = ds["vwnd"].mean(dim="time").values if "time" in ds.dims else ds["vwnd"].values
                ds.close()
                u_out = regrid_2d_array(u_mean, src_lat, src_lon, self.target_lat, self.target_lon)
                v_out = regrid_2d_array(v_mean, src_lat, src_lon, self.target_lat, self.target_lon)
                return u_out, v_out
            except Exception as e:
                logger.debug(f"Winds load error: {e}")
        return np.zeros((self.H, self.W), dtype=np.float32), np.zeros((self.H, self.W), dtype=np.float32)

    def _load_currents(self, dt: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        tag = dt.strftime("%Y%m%d")
        f = self._current_map.get(tag)
        if f:
            try:
                ds = open_netcdf_safe(f, decode_times=False)
                lat_var = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else "lat")
                lon_var = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else "lon")
                src_lat = ds[lat_var].values
                src_lon = ds[lon_var].values
                u_val = ds["u"].squeeze().values
                v_val = ds["v"].squeeze().values
                ds.close()
                if u_val.shape == (len(src_lon), len(src_lat)):
                    u_val = u_val.T
                    v_val = v_val.T
                u_out = regrid_2d_array(u_val, src_lat, src_lon, self.target_lat, self.target_lon)
                v_out = regrid_2d_array(v_val, src_lat, src_lon, self.target_lat, self.target_lon)
                return u_out, v_out
            except Exception as e:
                logger.debug(f"Currents load error: {e}")
        return np.zeros((self.H, self.W), dtype=np.float32), np.zeros((self.H, self.W), dtype=np.float32)

    def _load_precipitation(self, dt: pd.Timestamp) -> np.ndarray:
        tag = dt.strftime("%Y%m%d")
        f = self._precip_map.get(tag)
        if f:
            try:
                ds = open_netcdf_safe(f)
                src_lat = ds["lat"].values
                src_lon = ds["lon"].values
                raw_precip = ds["precipitation"].squeeze().values
                if raw_precip.shape == (len(src_lon), len(src_lat)):
                    raw_precip = raw_precip.T
                ds.close()
                return regrid_2d_array(raw_precip, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"Precipitation load error: {e}")
        return np.zeros((self.H, self.W), dtype=np.float32)

    def _load_heat_flux(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_hf is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._hf_times - dt)))
                src_lat = self._cached_hf["latitude"].values
                src_lon = self._cached_hf["longitude"].values
                val = self._cached_hf["slhf"].isel({self._hf_tcoord: match_idx}).values
                return regrid_2d_array(val, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"Heat flux load error: {e}")
        return np.zeros((self.H, self.W), dtype=np.float32)

    def _load_chlorophyll(self, dt: pd.Timestamp) -> np.ndarray:
        tag = dt.strftime("%Y%m%d")
        f = self._chlor_map.get(tag)
        if f:
            try:
                ds = open_netcdf_safe(f)
                src_lat = ds["lat"].values
                src_lon = ds["lon"].values
                raw_val = ds["chlor_a"].values
                ds.close()
                return regrid_2d_array(raw_val, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"Chlorophyll load error: {e}")
        return np.full((self.H, self.W), 0.1, dtype=np.float32)

    def _load_wind_curl(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_curl is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._curl_times - dt)))
                lat_var = "latitude" if "latitude" in self._cached_curl else "lat"
                lon_var = "longitude" if "longitude" in self._cached_curl else "lon"
                src_lat = self._cached_curl[lat_var].values
                src_lon = self._cached_curl[lon_var].values
                val = self._cached_curl[self._curl_var].isel({self._curl_tcoord: match_idx}).values
                return regrid_2d_array(val, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                logger.debug(f"Wind curl load error: {e}")
        return np.zeros((self.H, self.W), dtype=np.float32)
