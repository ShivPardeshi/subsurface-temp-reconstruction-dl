import sys
sys.path.insert(0, ".")
import numpy as np
import time
from src.data.harmonize.regrid import regrid_2d_array

src_lat = np.linspace(2.0, 30.0, 140)
src_lon = np.linspace(45.0, 105.0, 232)
target_lat = np.linspace(2.0, 30.0, 112)
target_lon = np.linspace(45.0, 105.0, 240)

# Create synthetic data with 30% NaNs
data = np.random.randn(len(src_lat), len(src_lon)).astype(np.float32)
data[data < -0.5] = np.nan

# Scipy regrid_2d_array
t0 = time.time()
res_scipy = regrid_2d_array(data, src_lat, src_lon, target_lat, target_lon, method="linear")
t_scipy = time.time() - t0

# Fast Bilinear with Weights
t1 = time.time()
lat_sort_idx = np.argsort(src_lat)
lon_sort_idx = np.argsort(src_lon)
s_lat = src_lat[lat_sort_idx]
s_lon = src_lon[lon_sort_idx]
s_data = data[lat_sort_idx, :][:, lon_sort_idx]

lat_idx = np.clip(np.searchsorted(s_lat, target_lat) - 1, 0, len(s_lat) - 2)
lat_frac = (target_lat - s_lat[lat_idx]) / (s_lat[lat_idx + 1] - s_lat[lat_idx])

lon_idx = np.clip(np.searchsorted(s_lon, target_lon) - 1, 0, len(s_lon) - 2)
lon_frac = (target_lon - s_lon[lon_idx]) / (s_lon[lon_idx + 1] - s_lon[lon_idx])

w00 = ((1.0 - lat_frac[:, None]) * (1.0 - lon_frac[None, :])).astype(np.float32)
w10 = (lat_frac[:, None] * (1.0 - lon_frac[None, :])).astype(np.float32)
w01 = ((1.0 - lat_frac[:, None]) * lon_frac[None, :]).astype(np.float32)
w11 = (lat_frac[:, None] * lon_frac[None, :]).astype(np.float32)

nan_mask = np.isnan(s_data)
clean_data = np.where(nan_mask, 0.0, s_data).astype(np.float32)
valid_mask = np.where(nan_mask, 0.0, 1.0).astype(np.float32)

val_interp = (
    w00 * clean_data[lat_idx[:, None], lon_idx[None, :]] +
    w10 * clean_data[(lat_idx + 1)[:, None], lon_idx[None, :]] +
    w01 * clean_data[lat_idx[:, None], (lon_idx + 1)[None, :]] +
    w11 * clean_data[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
)

weight_interp = (
    w00 * valid_mask[lat_idx[:, None], lon_idx[None, :]] +
    w10 * valid_mask[(lat_idx + 1)[:, None], lon_idx[None, :]] +
    w01 * valid_mask[lat_idx[:, None], (lon_idx + 1)[None, :]] +
    w11 * valid_mask[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
)

res_fast = np.where(weight_interp > 0.5, val_interp / np.maximum(weight_interp, 1e-6), np.nan)
t_fast = time.time() - t1

valid_both = (~np.isnan(res_scipy)) & (~np.isnan(res_fast))
diff = np.abs(res_scipy[valid_both] - res_fast[valid_both])
print(f"Scipy time: {t_scipy*1000:.1f}ms, Fast Bilinear time: {t_fast*1000:.2f}ms")
print(f"NaN match: {np.array_equal(np.isnan(res_scipy), np.isnan(res_fast))}")
print(f"Max difference on valid points: {np.max(diff):.2e}")
