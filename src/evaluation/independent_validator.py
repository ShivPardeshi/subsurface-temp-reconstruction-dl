"""
Independent In-Situ Validation Module for OceanEmbed.

This module provides tools for validating 3D ocean temperature reconstructions
against external, unassimilated observational datasets (e.g., GO-SHIP repeat hydrography,
research vessel CTD casts, autonomous gliders, or non-CORA in-situ profiles).

Methodological Integrity Note:
GLORYS12v1 reanalysis assimilates ARGO floats and RAMA moorings via the CORA database.
To assess generalization beyond reanalysis emulation, this validator provides
bilinear spatial interpolation, vertical profile harmonization, quality-flag filtering,
and skill-score evaluation against truly independent observational platforms.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

from src.utils.grid import CANONICAL_DEPTHS, get_target_grid
from src.evaluation.metrics.basic_metrics import (
    compute_rmse,
    compute_mae,
    compute_bias,
    compute_correlation_1d,
)
from src.evaluation.metrics.skill_score import compute_murphy_skill_score


@dataclass
class InSituProfile:
    """Represents a single in-situ vertical profile (e.g., from CTD or float)."""
    profile_id: str
    latitude: float
    longitude: float
    date_str: str  # ISO 'YYYY-MM-DD'
    depths_m: np.ndarray  # 1D array of observed depth levels
    temperatures_c: np.ndarray  # 1D array of observed in-situ temperatures (°C)
    platform_type: str = "CTD_Cast"  # e.g., 'CTD_Cast', 'Glider', 'Argo_Withheld', 'Mooring'
    cruise_or_mission: str = "Unknown"
    is_assimilated_in_glorys: bool = False  # Strict disclosure flag
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndependentInSituValidator:
    """
    Validator for collocating and benchmarking OceanEmbed reconstructions against
    independent (unassimilated) observational hydrographic profiles.
    """

    def __init__(
        self,
        lat_grid: Optional[np.ndarray] = None,
        lon_grid: Optional[np.ndarray] = None,
        canonical_depths: Optional[List[float]] = None,
        ocean_mask: Optional[np.ndarray] = None,
    ):
        """
        Initialize the independent validator.

        Args:
            lat_grid: 1D array of grid latitudes (default: 0.25° grid 2.0 to 30.0°N).
            lon_grid: 1D array of grid longitudes (default: 0.25° grid 45.0 to 105.0°E).
            canonical_depths: List of target depth levels in meters.
            ocean_mask: 2D boolean array (H, W) where True = ocean, False = land.
        """
        if lat_grid is None or lon_grid is None:
            self.lat_grid, self.lon_grid = get_target_grid()
        else:
            self.lat_grid = np.asarray(lat_grid, dtype=np.float32)
            self.lon_grid = np.asarray(lon_grid, dtype=np.float32)

        self.canonical_depths = np.array(
            canonical_depths if canonical_depths is not None else CANONICAL_DEPTHS,
            dtype=np.float32,
        )
        self.ocean_mask = ocean_mask

    def bilinear_interpolate_profile(
        self,
        pred_cube_3d: np.ndarray,
        lat: float,
        lon: float,
    ) -> Optional[np.ndarray]:
        """
        Extract a 1D vertical temperature profile at (lat, lon) from a 3D field (D, H, W)
        using bilinear spatial interpolation.

        Args:
            pred_cube_3d: 3D array of shape (15, H, W) representing the predicted field.
            lat: Target latitude in degrees North.
            lon: Target longitude in degrees East.

        Returns:
            1D array of shape (15,) containing interpolated temperatures, or None if out of bounds.
        """
        # Check domain boundaries
        if (
            lat < self.lat_grid[0]
            or lat > self.lat_grid[-1]
            or lon < self.lon_grid[0]
            or lon > self.lon_grid[-1]
        ):
            return None

        # Find grid indices for bounding box
        lat_res = float(self.lat_grid[1] - self.lat_grid[0])
        lon_res = float(self.lon_grid[1] - self.lon_grid[0])

        i_lat = (lat - self.lat_grid[0]) / lat_res
        i_lon = (lon - self.lon_grid[0]) / lon_res

        i0 = int(np.clip(np.floor(i_lat), 0, len(self.lat_grid) - 2))
        i1 = i0 + 1
        j0 = int(np.clip(np.floor(i_lon), 0, len(self.lon_grid) - 2))
        j1 = j0 + 1

        w_lat = i_lat - i0
        w_lon = i_lon - j0

        # Extract 4 corner profiles: (15, 2, 2)
        p00 = pred_cube_3d[:, i0, j0]
        p01 = pred_cube_3d[:, i0, j1]
        p10 = pred_cube_3d[:, i1, j0]
        p11 = pred_cube_3d[:, i1, j1]

        # Bilinear combination
        profile = (
            (1.0 - w_lat) * (1.0 - w_lon) * p00
            + (1.0 - w_lat) * w_lon * p01
            + w_lat * (1.0 - w_lon) * p10
            + w_lat * w_lon * p11
        )
        return profile

    def interpolate_obs_to_canonical_depths(
        self,
        obs_depths: np.ndarray,
        obs_temps: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate observed in-situ temperatures onto the 15 canonical depths.

        Args:
            obs_depths: 1D array of measured depth levels (meters).
            obs_temps: 1D array of measured temperatures (°C).

        Returns:
            1D array of shape (len(canonical_depths),) with NaN for unreached depths.
        """
        # Sort by depth ascending
        order = np.argsort(obs_depths)
        z = obs_depths[order]
        t = obs_temps[order]

        # Filter valid values
        valid = np.isfinite(z) & np.isfinite(t) & (t > -3.0) & (t < 40.0)
        z = z[valid]
        t = t[valid]

        if len(z) < 2:
            return np.full_like(self.canonical_depths, np.nan)

        # 1D linear interpolation bounded by max observed depth
        max_obs_z = float(np.max(z))
        min_obs_z = float(np.min(z))

        interp_t = np.interp(self.canonical_depths, z, t, left=t[0], right=np.nan)
        # Mask depths deeper than maximum observed sounding
        interp_t[self.canonical_depths > max_obs_z + 10.0] = np.nan
        interp_t[self.canonical_depths < min_obs_z - 5.0] = np.nan

        return interp_t

    def evaluate_profiles(
        self,
        predictions_by_date: Dict[str, np.ndarray],
        climatology_by_date: Dict[str, np.ndarray],
        in_situ_profiles: List[InSituProfile],
    ) -> Dict[str, Any]:
        """
        Evaluate predicted fields against a collection of independent in-situ profiles.

        Args:
            predictions_by_date: Dict mapping 'YYYY-MM-DD' to 3D array (15, H, W).
            climatology_by_date: Dict mapping 'YYYY-MM-DD' to 3D array (15, H, W).
            in_situ_profiles: List of InSituProfile objects.

        Returns:
            Dict containing detailed evaluation statistics and breakdown.
        """
        model_preds_all: List[float] = []
        obs_temps_all: List[float] = []
        clim_preds_all: List[float] = []

        per_depth_model: Dict[int, List[float]] = {int(z): [] for z in self.canonical_depths}
        per_depth_obs: Dict[int, List[float]] = {int(z): [] for z in self.canonical_depths}
        per_depth_clim: Dict[int, List[float]] = {int(z): [] for z in self.canonical_depths}

        matched_count = 0
        skipped_count = 0
        assimilated_count = 0

        for prof in in_situ_profiles:
            if prof.is_assimilated_in_glorys:
                assimilated_count += 1

            date = prof.date_str
            if date not in predictions_by_date or date not in climatology_by_date:
                skipped_count += 1
                continue

            pred_cube = predictions_by_date[date]
            clim_cube = climatology_by_date[date]

            # Spatially interpolate model & climatology to profile (lat, lon)
            p_prof = self.bilinear_interpolate_profile(pred_cube, prof.latitude, prof.longitude)
            c_prof = self.bilinear_interpolate_profile(clim_cube, prof.latitude, prof.longitude)

            if p_prof is None or c_prof is None:
                skipped_count += 1
                continue

            # Vertically interpolate in-situ observations onto canonical depths
            obs_canonical = self.interpolate_obs_to_canonical_depths(
                prof.depths_m, prof.temperatures_c
            )

            # Record valid depth pairs
            for idx, z in enumerate(self.canonical_depths):
                t_obs = float(obs_canonical[idx])
                t_pred = float(p_prof[idx])
                t_clim = float(c_prof[idx])

                if np.isfinite(t_obs) and np.isfinite(t_pred) and np.isfinite(t_clim):
                    z_int = int(z)
                    per_depth_model[z_int].append(t_pred)
                    per_depth_obs[z_int].append(t_obs)
                    per_depth_clim[z_int].append(t_clim)

                    model_preds_all.append(t_pred)
                    obs_temps_all.append(t_obs)
                    clim_preds_all.append(t_clim)

            matched_count += 1

        if len(model_preds_all) == 0:
            return {
                "status": "NO_VALID_MATCHES",
                "matched_profiles": 0,
                "skipped_profiles": skipped_count,
            }

        arr_p = np.array(model_preds_all)
        arr_t = np.array(obs_temps_all)
        arr_c = np.array(clim_preds_all)

        overall_rmse = float(compute_rmse(arr_p, arr_t))
        overall_mae = float(compute_mae(arr_p, arr_t))
        overall_bias = float(compute_bias(arr_p, arr_t))
        overall_corr = float(compute_correlation_1d(arr_p, arr_t))
        clim_rmse = float(compute_rmse(arr_c, arr_t))
        murphy_skill = float(compute_murphy_skill_score(arr_p, arr_t, arr_c))

        per_depth_summary = {}
        for z in self.canonical_depths:
            z_int = int(z)
            p_z = np.array(per_depth_model[z_int])
            o_z = np.array(per_depth_obs[z_int])
            c_z = np.array(per_depth_clim[z_int])
            if len(p_z) > 0:
                z_rmse = float(compute_rmse(p_z, o_z))
                z_clim_rmse = float(compute_rmse(c_z, o_z))
                per_depth_summary[str(z_int)] = {
                    "num_soundings": len(p_z),
                    "model_rmse": z_rmse,
                    "climatology_rmse": z_clim_rmse,
                    "skill_score": float(compute_murphy_skill_score(p_z, o_z, c_z)),
                    "bias": float(compute_bias(p_z, o_z)),
                }

        return {
            "status": "SUCCESS",
            "metadata": {
                "matched_profiles": matched_count,
                "skipped_profiles": skipped_count,
                "assimilated_profiles_count": assimilated_count,
                "total_soundings": len(model_preds_all),
                "is_strictly_independent": (assimilated_count == 0),
            },
            "overall_metrics": {
                "model_rmse_c": overall_rmse,
                "climatology_rmse_c": clim_rmse,
                "murphy_skill_score": murphy_skill,
                "mae_c": overall_mae,
                "bias_c": overall_bias,
                "correlation": overall_corr,
            },
            "depth_profile_metrics": per_depth_summary,
        }
