import os
import shutil
import time
import xarray as xr

src_dir = "data/processed/phase2_dataset"
dst_dir = "data/processed/phase2_dataset_v2"
os.makedirs(dst_dir, exist_ok=True)

for name in [
    "oceanembed_training_inputs.zarr",
    "oceanembed_anomaly_targets.zarr",
    "oceanembed_auxiliary_targets.zarr",
]:
    t0 = time.time()
    print(f"Converting {name} to Zarr V2...")
    s_path = os.path.join(src_dir, name)
    d_path = os.path.join(dst_dir, name)
    if os.path.exists(d_path):
        shutil.rmtree(d_path)
    ds = xr.open_zarr(s_path)
    for v in ds.variables.values():
        v.encoding.clear()
    ds.to_zarr(d_path, zarr_format=2, mode="w")
    print(f"Done {name} in {time.time() - t0:.1f}s")

for f in ["climatology_coefficients.nc", "scalar_conditioning.csv"]:
    shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

print("Conversion complete!")
