import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from src.data.harmonize.regrid import regrid_2d_array

src_lat = np.linspace(2.0, 30.0, 56)
src_lon = np.linspace(45.0, 105.0, 120)
target_lat = np.linspace(2.0, 30.0, 112)
target_lon = np.linspace(45.0, 105.0, 240)

data = np.random.randn(len(src_lat), len(src_lon)).astype(np.float32)

# Scipy regrid_2d_array
res_scipy = regrid_2d_array(data, src_lat, src_lon, target_lat, target_lon, method="linear")

# Bilinear precompute
lat_idx = np.searchsorted(src_lat, target_lat) - 1
lat_idx = np.clip(lat_idx, 0, len(src_lat) - 2)
lat_frac = (target_lat - src_lat[lat_idx]) / (src_lat[lat_idx + 1] - src_lat[lat_idx])

lon_idx = np.searchsorted(src_lon, target_lon) - 1
lon_idx = np.clip(lon_idx, 0, len(src_lon) - 2)
lon_frac = (target_lon - src_lon[lon_idx]) / (src_lon[lon_idx + 1] - src_lon[lon_idx])

w00 = (1.0 - lat_frac[:, None]) * (1.0 - lon_frac[None, :])
w10 = lat_frac[:, None] * (1.0 - lon_frac[None, :])
w01 = (1.0 - lat_frac[:, None]) * lon_frac[None, :]
w11 = lat_frac[:, None] * lon_frac[None, :]

res_fast = (
    w00 * data[lat_idx[:, None], lon_idx[None, :]] +
    w10 * data[(lat_idx + 1)[:, None], lon_idx[None, :]] +
    w01 * data[lat_idx[:, None], (lon_idx + 1)[None, :]] +
    w11 * data[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
)

max_diff = np.max(np.abs(res_scipy - res_fast))
print(f"Max difference between Scipy and Fast Bilinear: {max_diff:.2e}")
assert max_diff < 1e-5, f"Difference too large: {max_diff}"
print("VERIFICATION SUCCESSFUL: Numerically identical!")
