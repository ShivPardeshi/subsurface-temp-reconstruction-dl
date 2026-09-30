import zarr
import numpy as np

z = zarr.open('data/processed/phase2_dataset/oceanembed_training_inputs.zarr', mode='r')

print("=== ZARR STORE ROOT KEYS ===")
print(list(z.keys()))

print("\n=== ZARR CHANNELS ARRAY z['channel'][:] ===")
channels = list(z['channel'][:])
for idx, ch in enumerate(channels):
    print(f"[{idx:02d}] {ch}")

print("\n=== INPUTS TENSOR SHAPE & DTYPE ===")
inputs = z['inputs']
print(f"Shape: {inputs.shape}, Dtype: {inputs.dtype}")

print("\n=== PER-CHANNEL RAW TRACE (Days 0-30, Ocean Grid Cells) ===")
ocean_mask = inputs[0, 20, :, :] > 0.5
print(f"{'Ch':<4} {'Channel Name':<22} {'Mean':<14} {'Std':<14} {'Min':<14} {'Max':<14}")
print("-" * 80)
for idx in range(len(channels)):
    name = channels[idx]
    arr = inputs[:30, idx, :, :]
    vals = arr[:, ocean_mask]
    mean_v = float(np.nanmean(vals))
    std_v = float(np.nanstd(vals))
    min_v = float(np.nanmin(vals))
    max_v = float(np.nanmax(vals))
    print(f"{idx:<4} {name:<22} {mean_v:<14.6e} {std_v:<14.6e} {min_v:<14.6e} {max_v:<14.6e}")
