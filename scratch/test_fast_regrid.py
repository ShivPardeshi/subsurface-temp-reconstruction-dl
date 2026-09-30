import numpy as np
import time

# Target grid
target_lat = np.linspace(2.0, 30.0, 112)
target_lon = np.linspace(45.0, 105.0, 240)

# Source grid (e.g. SST 560 x 1200)
src_lat = np.linspace(2.0, 30.0, 560)
src_lon = np.linspace(45.0, 105.0, 1200)

# Precompute bilinear indices and weights
# For each target_lat, find surrounding src_lat indices
lat_idx = np.searchsorted(src_lat, target_lat) - 1
lat_idx = np.clip(lat_idx, 0, len(src_lat) - 2)
lat_frac = (target_lat - src_lat[lat_idx]) / (src_lat[lat_idx + 1] - src_lat[lat_idx])

lon_idx = np.searchsorted(src_lon, target_lon) - 1
lon_idx = np.clip(lon_idx, 0, len(src_lon) - 2)
lon_frac = (target_lon - src_lon[lon_idx]) / (src_lon[lon_idx + 1] - src_lon[lon_idx])

# Meshgrid of weights
w00 = (1.0 - lat_frac[:, None]) * (1.0 - lon_frac[None, :])
w10 = lat_frac[:, None] * (1.0 - lon_frac[None, :])
w01 = (1.0 - lat_frac[:, None]) * lon_frac[None, :]
w11 = lat_frac[:, None] * lon_frac[None, :]

# Test interpolation of a 2D slice
data = np.random.randn(len(src_lat), len(src_lon)).astype(np.float32)

t0 = time.time()
for _ in range(100):
    res = (
        w00 * data[lat_idx[:, None], lon_idx[None, :]] +
        w10 * data[(lat_idx + 1)[:, None], lon_idx[None, :]] +
        w01 * data[lat_idx[:, None], (lon_idx + 1)[None, :]] +
        w11 * data[(lat_idx + 1)[:, None], (lon_idx + 1)[None, :]]
    )
t_elapsed = time.time() - t0
print(f"100 precomputed bilinear regrids took {t_elapsed:.4f}s ({t_elapsed/100*1000:.2f}ms per regrid)!")
