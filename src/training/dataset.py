"""PyTorch Dataset and DataLoader for OceanEmbed.

Wraps Phase 2 Zarr stores (training inputs, anomaly targets, auxiliary targets)
and scalar conditioning tables into 7-day spatiotemporal sequence batches.
"""

from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import zarr

from src.utils.grid import CANONICAL_DEPTHS


class OceanEmbedDataset(Dataset):
    """OceanEmbed Spatiotemporal Dataset wrapping Phase 2 preprocessed stores."""

    def __init__(
        self,
        inputs_zarr_path: Union[str, Path],
        anomaly_targets_zarr_path: Union[str, Path],
        aux_targets_zarr_path: Union[str, Path],
        scalar_csv_path: Union[str, Path],
        sequence_length: int = 7,
        augment: bool = False,
        standardize_targets: bool = True,
        depth_scales_path: Optional[Union[str, Path]] = "data/processed/anomaly_depth_scales.json",
    ):
        self.inputs_zarr_path = Path(inputs_zarr_path)
        self.anomaly_targets_zarr_path = Path(anomaly_targets_zarr_path)
        self.aux_targets_zarr_path = Path(aux_targets_zarr_path)
        self.scalar_csv_path = Path(scalar_csv_path)
        self.seq_len = sequence_length
        self.augment = augment
        self.standardize_targets = standardize_targets

        # Load per-depth anomaly scale factors and climatological mean temperatures
        scales_file = Path(depth_scales_path) if depth_scales_path else None
        if scales_file and scales_file.exists():
            import json
            with open(scales_file, "r") as f:
                scales_data = json.load(f)
            self.anomaly_stds = np.array(scales_data["anomaly_stds"], dtype=np.float32)[:, None, None]
            self.clim_mean_temps = np.array(scales_data.get("clim_mean_temps", [20.0] * 15), dtype=np.float32)
        else:
            self.anomaly_stds = np.ones((15, 1, 1), dtype=np.float32)
            self.clim_mean_temps = np.full(15, 20.0, dtype=np.float32)

        # Open Zarr stores
        self.inputs_root = zarr.open_group(str(self.inputs_zarr_path), mode="r")
        if "inputs" in self.inputs_root:
            self.inputs_data = self.inputs_root["inputs"]
        elif "datacube" in self.inputs_root:
            self.inputs_data = self.inputs_root["datacube"]
        else:
            raise KeyError(f"No inputs or datacube array in {self.inputs_zarr_path}")

        self.anomaly_root = zarr.open_group(str(self.anomaly_targets_zarr_path), mode="r")
        if "anomaly" in self.anomaly_root:
            self.anomaly_data = self.anomaly_root["anomaly"]
        elif "anomaly_targets" in self.anomaly_root:
            self.anomaly_data = self.anomaly_root["anomaly_targets"]
        else:
            raise KeyError(f"No anomaly or anomaly_targets array in {self.anomaly_targets_zarr_path}")

        self.aux_root = zarr.open_group(str(self.aux_targets_zarr_path), mode="r")
        if "auxiliary_targets" in self.aux_root:
            self.aux_data = self.aux_root["auxiliary_targets"]
        elif "aux_target" in self.aux_root:
            self.aux_data = self.aux_root["aux_target"]
        else:
            raise KeyError(f"No auxiliary array in {self.aux_targets_zarr_path}")

        # Load scalar table
        self.scalar_df = pd.read_csv(self.scalar_csv_path)

        self.num_days = self.inputs_data.shape[0]
        # Number of valid sequences (window of seq_len)
        self.num_samples = max(0, self.num_days - self.seq_len + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Fetch a single 7-day sequence and corresponding target day.

        Args:
            idx: Sequence index.

        Returns:
            Dictionary with tensors:
            - 'x_seq': (T=7, C=25, H, W)
            - 'static_features': (6, H, W) channels 19-24 (landmask, bathy, 4 regions)
            - 'anomaly_target': (15, H, W)
            - 'aux_targets': (4, H, W)
            - 'scalar_cond': (4,) [sin_doy, cos_doy, oni, iod]
        """
        start_idx = idx
        end_idx = idx + self.seq_len
        target_day_idx = end_idx - 1

        # Slice 7-day inputs
        x_seq_np = np.asarray(self.inputs_data[start_idx:end_idx], dtype=np.float32)
        x_seq_np = np.nan_to_num(x_seq_np, nan=0.0)

        # Anomaly targets at target day (15, H, W)
        anomaly_np = np.asarray(self.anomaly_data[target_day_idx], dtype=np.float32)
        anomaly_np = np.nan_to_num(anomaly_np, nan=0.0)

        # Auxiliary targets at target day (4, H, W)
        aux_np = np.asarray(self.aux_data[target_day_idx], dtype=np.float32)
        aux_np = np.nan_to_num(aux_np, nan=0.0)

        # Physically-consistent data augmentation (Item 3.3, enabled only during training)
        if self.augment:
            # 1. Atmospheric forcing scale jitter (+/- 3% across channels 3-16)
            forcing_scale = np.random.uniform(0.97, 1.03, size=(1, 14, 1, 1)).astype(np.float32)
            x_seq_np[:, 3:17] = x_seq_np[:, 3:17] * forcing_scale

            # 2. Emulate satellite altimetry / IR cloud cover patch dropout on SST (0) & SSH (2)
            if np.random.rand() < 0.30:
                h, w = x_seq_np.shape[-2:]
                ph, pw = np.random.randint(8, 16), np.random.randint(12, 24)
                top = np.random.randint(0, h - ph)
                left = np.random.randint(0, w - pw)
                rand_t = np.random.randint(0, self.seq_len)
                x_seq_np[rand_t, 0, top : top + ph, left : left + pw] = 0.0
                x_seq_np[rand_t, 2, top : top + ph, left : left + pw] = 0.0

            # 3. Horizontal spatial translation jitter with strict land-mask zero clamping
            if np.random.rand() < 0.50:
                shift_lon = int(np.random.choice([-2, -1, 1, 2]))
                x_seq_np = np.roll(x_seq_np, shift=shift_lon, axis=-1)
                anomaly_np = np.roll(anomaly_np, shift=shift_lon, axis=-1)
                aux_np = np.roll(aux_np, shift=shift_lon, axis=-1)

                # Re-enforce zero-clamping on land cells (is_ocean == False / land_ocean_mask < 0.5)
                # Ocean data over valid ocean water is fully preserved; land cells are clamped to 0.0
                is_land = (x_seq_np[-1, 20] < 0.5)
                for ch in range(19):
                    x_seq_np[:, ch, is_land] = 0.0
                for d in range(anomaly_np.shape[0]):
                    anomaly_np[d, is_land] = 0.0
                for a in range(aux_np.shape[0]):
                    aux_np[a, is_land] = 0.0

        # Standardize anomaly targets by per-depth scale factor to equalize SNR across water column
        if self.standardize_targets:
            anomaly_np = anomaly_np / self.anomaly_stds

        # Static features from the target day: channels [20, 19, 21, 22, 23, 24] (6 channels)
        # (landmask, bathymetry, arabian_sea, bay_of_bengal, confluence_zone, open_ocean)
        static_indices = [20, 19, 21, 22, 23, 24]
        static_np = x_seq_np[-1, static_indices]

        # Scalar conditioning
        row = self.scalar_df.iloc[target_day_idx]
        sin_doy = float(row.get("sin_doy", 0.0))
        cos_doy = float(row.get("cos_doy", 1.0))
        oni = float(row.get("oni_index", 0.0))
        iod = float(row.get("iod_dmi_index", 0.0))
        scalar_vec = np.array([sin_doy, cos_doy, oni, iod], dtype=np.float32)

        return {
            "x_seq": torch.from_numpy(x_seq_np),
            "static_features": torch.from_numpy(static_np),
            "anomaly_target": torch.from_numpy(anomaly_np),
            "aux_targets": torch.from_numpy(aux_np),
            "scalar_cond": torch.from_numpy(scalar_vec),
            "clim_mean_temps": torch.from_numpy(self.clim_mean_temps),
            "anomaly_stds": torch.from_numpy(self.anomaly_stds.squeeze()),
        }


def create_dataloader(
    dataset: OceanEmbedDataset,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
