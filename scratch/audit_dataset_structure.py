import pandas as pd
import zarr
import xarray as xr

df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")
print(f"Total rows in scalar_conditioning.csv: {len(df)}")
print(df.head())
print(df.tail())

in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
print(f"inputs shape: {in_zarr['inputs'].shape}")

tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
print(f"anomaly shape: {tgt_zarr['anomaly'].shape}")
