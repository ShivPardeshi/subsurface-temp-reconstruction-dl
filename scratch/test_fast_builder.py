import os, sys, time, glob
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import xarray as xr
from src.utils.io_utils import open_netcdf_safe
from src.data.harmonize.regrid import regrid_2d_array
from src.data.harmonize.build_datacube import DataCubeBuilder

class FastDataCubeBuilder(DataCubeBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        t0 = time.time()
        # 1. Cache multi-time NetCDFs
        # SST
        sst_files = glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "india_sst", "*.nc"))
        self._cached_sst = open_netcdf_safe(sst_files[0]) if sst_files else None
        self._sst_times = pd.to_datetime(self._cached_sst["time"].values) if self._cached_sst else None

        # SSS
        sss_files = glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "india_sss", "*.nc"))
        self._cached_sss = open_netcdf_safe(sss_files[0]) if sss_files else None
        self._sss_times = pd.to_datetime(self._cached_sss["time"].values) if self._cached_sss else None

        # SSH
        ssh_files = glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "india_ssh", "*.nc"))
        self._cached_ssh = open_netcdf_safe(ssh_files[0]) if ssh_files else None
        self._ssh_times = pd.to_datetime(self._cached_ssh["time"].values) if self._cached_ssh else None

        # Heat Flux
        hf_files = glob.glob(os.path.join(self.raw_root, "heat_flux", "*", "*.nc"))
        self._cached_hf = open_netcdf_safe(hf_files[0]) if hf_files else None
        hf_tcoord = "valid_time" if (self._cached_hf and "valid_time" in self._cached_hf.coords) else "time"
        self._hf_tcoord = hf_tcoord
        self._hf_times = pd.to_datetime(self._cached_hf[hf_tcoord].values) if self._cached_hf else None

        # Wind Stress Curl
        curl_files = glob.glob(os.path.join(self.raw_root, "wind_curl", "*.nc"))
        self._cached_curl = open_netcdf_safe(curl_files[0]) if curl_files else None
        curl_tcoord = "time" if (self._cached_curl and "time" in self._cached_curl.coords) else (list(self._cached_curl.coords.keys())[0] if self._cached_curl else None)
        self._curl_tcoord = curl_tcoord
        self._curl_times = pd.to_datetime(self._cached_curl[curl_tcoord].values) if self._cached_curl else None

        # Index daily files for instant O(1) dict lookup
        # CCMP Winds
        self._wind_map = {}
        for f in glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "*CCMP*", "*.nc*")):
            base = os.path.basename(f)
            # Find 8-digit date in filename
            for token in base.split("."):
                for sub in token.split("_"):
                    if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                        self._wind_map[sub] = f
                        break

        # OSCAR Currents
        self._current_map = {}
        for f in glob.glob(os.path.join(self.raw_root, "sst_sss_ssh_currents_winds", "*OSCAR*", "*.nc")):
            base = os.path.basename(f)
            for token in base.split("_"):
                sub = token.replace(".nc", "")
                if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                    self._current_map[sub] = f
                    break

        # GPM IMERG
        self._precip_map = {}
        for f in glob.glob(os.path.join(self.raw_root, "precipitation", "*GPM*", "*.nc*")):
            base = os.path.basename(f)
            for token in base.split("."):
                sub = token.split("-")[0]
                if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                    self._precip_map[sub] = f
                    break

        # MODIS Chlorophyll
        self._chlor_map = {}
        for f in glob.glob(os.path.join(self.raw_root, "chlorophyll", "*", "*.nc")):
            base = os.path.basename(f)
            for token in base.split("."):
                for sub in token.split("_"):
                    if len(sub) == 8 and sub.isdigit() and sub.startswith("2025"):
                        self._chlor_map[sub] = f
                        break

        print(f"FastDataCubeBuilder init & indexing took {time.time() - t0:.2f}s")
        print(f"Indexed: Winds={len(self._wind_map)}, OSCAR={len(self._current_map)}, GPM={len(self._precip_map)}, Chlor={len(self._chlor_map)}")

    def _load_sst(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_sst is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._sst_times - dt)))
                src_lat = self._cached_sst["latitude"].values
                src_lon = self._cached_sst["longitude"].values
                raw_slice = self._cached_sst["analysed_sst"].isel(time=match_idx).values
                return regrid_2d_array(raw_slice, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                pass
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
                pass
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
                pass
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
                pass
        return np.zeros((self.H, self.W), dtype=np.float32)

    def _load_wind_curl(self, dt: pd.Timestamp) -> np.ndarray:
        if self._cached_curl is not None:
            try:
                match_idx = int(np.argmin(np.abs(self._curl_times - dt)))
                lat_var = "latitude" if "latitude" in self._cached_curl else "lat"
                lon_var = "longitude" if "longitude" in self._cached_curl else "lon"
                src_lat = self._cached_curl[lat_var].values
                src_lon = self._cached_curl[lon_var].values
                var = list(self._cached_curl.data_vars.keys())[0]
                val = self._cached_curl[var].isel({self._curl_tcoord: match_idx}).values
                return regrid_2d_array(val, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception as e:
                pass
        return np.zeros((self.H, self.W), dtype=np.float32)

    def _load_winds(self, dt: pd.Timestamp):
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
                return regrid_2d_array(u_mean, src_lat, src_lon, self.target_lat, self.target_lon), regrid_2d_array(v_mean, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception:
                pass
        return np.zeros((self.H, self.W), dtype=np.float32), np.zeros((self.H, self.W), dtype=np.float32)

    def _load_currents(self, dt: pd.Timestamp):
        tag = dt.strftime("%Y%m%d")
        f = self._current_map.get(tag)
        if f:
            try:
                ds = open_netcdf_safe(f, decode_times=False)
                lat_var = "lat" if "lat" in ds.coords else "latitude"
                lon_var = "lon" if "lon" in ds.coords else "longitude"
                src_lat = ds[lat_var].values
                src_lon = ds[lon_var].values
                u_val = ds["u"].squeeze().values
                v_val = ds["v"].squeeze().values
                ds.close()
                if u_val.shape == (len(src_lon), len(src_lat)):
                    u_val = u_val.T
                    v_val = v_val.T
                return regrid_2d_array(u_val, src_lat, src_lon, self.target_lat, self.target_lon), regrid_2d_array(v_val, src_lat, src_lon, self.target_lat, self.target_lon)
            except Exception:
                pass
        return np.zeros((self.H, self.W), dtype=np.float32), np.zeros((self.H, self.W), dtype=np.float32)

    def _load_precipitation(self, dt: pd.Timestamp):
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
            except Exception:
                pass
        return np.zeros((self.H, self.W), dtype=np.float32)

    def _load_chlorophyll(self, dt: pd.Timestamp):
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
            except Exception:
                pass
        return np.full((self.H, self.W), 0.1, dtype=np.float32)

if __name__ == "__main__":
    builder = FastDataCubeBuilder(toy_mode=False)
    # Test 5 days
    dates = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]
    t0 = time.time()
    for d in dates:
        td0 = time.time()
        arr = builder.assemble_day(d)
        print(f"Day {d} assembled in {time.time() - td0:.3f}s")
    print(f"Total for 5 days: {time.time() - t0:.2f}s (avg {(time.time() - t0)/5:.3f}s/day)")
