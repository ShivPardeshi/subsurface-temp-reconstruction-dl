import zarr
import numpy as np

z = zarr.open('data/processed/phase2_dataset/oceanembed_training_inputs.zarr', mode='r')
inputs = z['inputs']
channels = [str(c) for c in z['channel'][:]]
print('Inputs shape:', inputs.shape, 'dtype:', inputs.dtype)
channel_names = channels

header = f"{'Ch':<3} | {'Name':<24} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Non-Null Frac':<12}"
print(header)
print('-' * len(header))

for ch in range(inputs.shape[1]):
    data = inputs[:, ch, :, :]
    total_pixels = data.size
    valid = ~np.isnan(data)
    non_null_frac = float(np.count_nonzero(valid) / total_pixels)
    valid_data = data[valid]
    if valid_data.size > 0:
        mean_val = float(np.mean(valid_data))
        std_val = float(np.std(valid_data))
        min_val = float(np.min(valid_data))
        max_val = float(np.max(valid_data))
    else:
        mean_val = std_val = min_val = max_val = 0.0
    name = channel_names[ch] if ch < len(channel_names) else f'ch_{ch}'
    print(f"{ch:<3} | {name:<24} | {mean_val:<10.4f} | {std_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f} | {non_null_frac:<12.4f}")
