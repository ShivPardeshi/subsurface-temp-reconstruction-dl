import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import zarr
import math

in_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_training_inputs.zarr", mode="r")
tgt_zarr = zarr.open("data/processed/phase2_dataset/oceanembed_anomaly_targets.zarr", mode="r")
scalar_df = pd.read_csv("data/processed/phase2_dataset/scalar_conditioning.csv")

lat = in_zarr["lat"][:]
lon = in_zarr["lon"][:]
ocean_mask = (in_zarr["inputs"][0, 20] > 0.5)
feature_channels = [0, 1, 2, 3, 4, 11, 14, 15, 19, 21, 22, 23, 24]
train_days = list(range(6, 237))

n_ocean = int(np.sum(ocean_mask))
lat_mesh, lon_mesh = np.meshgrid(lat, lon, indexing="ij")
lat_pts = lat_mesh[ocean_mask][None, :]
lon_pts = lon_mesh[ocean_mask][None, :]

X_list, Y_list = [], []
for t_day in train_days:
    seq = np.nan_to_num(in_zarr["inputs"][t_day - 6 : t_day + 1], nan=0.0)
    anom = np.nan_to_num(tgt_zarr["anomaly"][t_day].copy(), nan=0.0)
    anom[0] = anom[1]

    sin_doy = float(scalar_df.loc[t_day, "sin_doy"])
    cos_doy = float(scalar_df.loc[t_day, "cos_doy"])
    oni = float(scalar_df.loc[t_day, "oni_index"])
    iod = float(scalar_df.loc[t_day, "iod_dmi_index"])

    curr_feat = seq[-1, feature_channels][:, ocean_mask]
    mean_feat = np.mean(seq[:, feature_channels], axis=0)[:, ocean_mask]
    diff_feat = (seq[-1, feature_channels] - seq[0, feature_channels])[:, ocean_mask]
    scalars = np.array([[sin_doy], [cos_doy], [oni], [iod]]) * np.ones((4, n_ocean))

    x_day = np.vstack([curr_feat, mean_feat, diff_feat, lat_pts, lon_pts, scalars]).T
    y_day = anom[:, ocean_mask].T
    X_list.append(x_day)
    Y_list.append(y_day)

X_train = np.nan_to_num(np.vstack(X_list), nan=0.0)
Y_train = np.nan_to_num(np.vstack(Y_list), nan=0.0)

mean_X = np.mean(X_train, axis=0, keepdims=True)
std_X = np.std(X_train, axis=0, keepdims=True)
std_X[std_X < 1e-6] = 1.0
X_train_norm = (X_train - mean_X) / std_X
X_train_b = np.hstack([X_train_norm, np.ones((X_train_norm.shape[0], 1))])

alpha = 100.0
XtX = X_train_b.T @ X_train_b + alpha * np.eye(X_train_b.shape[1])
XtY = X_train_b.T @ Y_train
W = np.linalg.solve(XtX, XtY)

def eval_ridge(indices, name):
    sqs = []
    for s_idx in indices:
        seq = np.nan_to_num(in_zarr["inputs"][s_idx - 6 : s_idx + 1], nan=0.0)
        anom = np.nan_to_num(tgt_zarr["anomaly"][s_idx].copy(), nan=0.0)
        anom[0] = anom[1]

        sin_doy = float(scalar_df.loc[s_idx, "sin_doy"])
        cos_doy = float(scalar_df.loc[s_idx, "cos_doy"])
        oni = float(scalar_df.loc[s_idx, "oni_index"])
        iod = float(scalar_df.loc[s_idx, "iod_dmi_index"])

        curr_feat = seq[-1, feature_channels][:, ocean_mask]
        mean_feat = np.mean(seq[:, feature_channels], axis=0)[:, ocean_mask]
        diff_feat = (seq[-1, feature_channels] - seq[0, feature_channels])[:, ocean_mask]
        scalars = np.array([[sin_doy], [cos_doy], [oni], [iod]]) * np.ones((4, n_ocean))

        x_day = np.vstack([curr_feat, mean_feat, diff_feat, lat_pts, lon_pts, scalars]).T
        x_norm = (x_day - mean_X) / std_X
        x_b = np.hstack([x_norm, np.ones((x_norm.shape[0], 1))])
        y_pred = (x_b @ W).T
        y_true = anom[:, ocean_mask]
        sqs.extend(((y_pred - y_true)**2).flatten().tolist())
    rmse = math.sqrt(np.mean(sqs))
    print(f"{name} Ridge RMSE: {rmse:.4f}°C")

eval_ridge([305, 311, 317, 323, 329, 335, 341, 347, 353, 359], "Benchmark A (Nov-Dec)")
eval_ridge([238, 251, 264, 277, 290, 303, 316, 330, 344, 357], "Benchmark B (Sep-Dec)")
eval_ridge([14, 59, 104, 149, 194, 239, 284, 317, 330, 357], "Contaminated Diagnostic (Jan-Dec)")
