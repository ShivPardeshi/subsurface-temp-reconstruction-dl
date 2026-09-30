"""Marine Heatwave (MHW) Detection Module (Hobday et al., 2016 Definition).

Implements the internationally recognized standard definition of Marine Heatwaves:
A discrete prolonged anomalously warm water event characterized by:
1. Threshold: Temperature (SST or surface-layer heat) >= 90th percentile of the
   climatological distribution for that calendar day.
2. Persistence: Exceedance sustained for at least 5 consecutive days (min_duration = 5).
3. Intensity Categories:
   - Category I (Moderate): 1x to 2x (Threshold - Climatology)
   - Category II (Strong): 2x to 3x
   - Category III (Severe): 3x to 4x
   - Category IV (Extreme): >= 4x
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np


def detect_mhw_1d(
    values: np.ndarray,
    thresholds_90th: np.ndarray,
    climatology_means: Optional[np.ndarray] = None,
    min_duration: int = 5,
) -> Dict[str, Any]:
    """Detect marine heatwave events in a 1D temporal series for a single grid cell.

    Args:
        values: 1D array of daily metric values (e.g. daily SST or 0-30m mean temp) over time T.
        thresholds_90th: 1D array of matching 90th percentile baseline thresholds for each day.
        climatology_means: Optional 1D array of mean climatological values for category scaling.
        min_duration: Minimum consecutive days required to declare an MHW event (Hobday standard = 5).

    Returns:
        Dictionary containing:
        - 'is_mhw': Boolean array of shape (T,) indicating active MHW days.
        - 'event_count': Number of discrete MHW events detected.
        - 'events': List of event dicts with start_idx, end_idx, duration, max_intensity, category.
        - 'categories': List of category strings for each time step ('None', 'Category I (Moderate)', etc.).
    """
    v = np.asarray(values, dtype=np.float64)
    th = np.asarray(thresholds_90th, dtype=np.float64)
    T = len(v)

    if climatology_means is not None:
        clim = np.asarray(climatology_means, dtype=np.float64)
    else:
        # Default: assume threshold is ~1.5°C above mean if not provided
        clim = th - 1.5

    # 1. Identify raw exceedance days
    exceedance = (v >= th)

    is_mhw = np.zeros(T, dtype=bool)
    categories = ["None"] * T
    events = []

    # 2. Run-length grouping of consecutive True intervals
    i = 0
    while i < T:
        if exceedance[i]:
            start = i
            while i < T and exceedance[i]:
                i += 1
            end = i  # exclusive
            duration = end - start

            # 3. Apply the 5-day persistence filter (Hobday et al.)
            if duration >= min_duration:
                is_mhw[start:end] = True
                event_v = v[start:end]
                event_th = th[start:end]
                event_clim = clim[start:end]

                diff_th_clim = np.maximum(0.2, event_th - event_clim)
                intensity_multipliers = (event_v - event_clim) / diff_th_clim
                max_mult = float(np.max(intensity_multipliers))

                # Categorize event based on maximum intensity
                if max_mult < 2.0:
                    cat_name = "Category I (Moderate)"
                elif max_mult < 3.0:
                    cat_name = "Category II (Strong)"
                elif max_mult < 4.0:
                    cat_name = "Category III (Severe)"
                else:
                    cat_name = "Category IV (Extreme)"

                for t_step in range(start, end):
                    step_mult = float(intensity_multipliers[t_step - start])
                    if step_mult < 2.0:
                        categories[t_step] = "Category I (Moderate)"
                    elif step_mult < 3.0:
                        categories[t_step] = "Category II (Strong)"
                    elif step_mult < 4.0:
                        categories[t_step] = "Category III (Severe)"
                    else:
                        categories[t_step] = "Category IV (Extreme)"

                events.append({
                    "start_idx": start,
                    "end_idx": end - 1,
                    "duration_days": duration,
                    "max_intensity_excess_c": float(np.max(event_v - event_th)),
                    "category": cat_name,
                })
        else:
            i += 1

    return {
        "is_mhw": is_mhw,
        "event_count": len(events),
        "events": events,
        "categories": categories,
    }


def detect_mhw_spatial(
    sequence_3d: np.ndarray,
    thresholds_90th_3d: np.ndarray,
    min_duration: int = 5,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute active Marine Heatwave spatial flag map for the final day of a sliding temporal window.

    Args:
        sequence_3d: Array of shape (T, H, W) of daily temperatures.
        thresholds_90th_3d: Array of shape (T, H, W) of 90th percentile thresholds.
        min_duration: Minimum persistence length in days (Hobday et al. = 5).
        mask: Optional boolean ocean mask (H, W). True = ocean.

    Returns:
        Boolean 2D array (H, W) where True = cell is in an active MHW on day T-1.
    """
    T, H, W = sequence_3d.shape
    active_mhw_map = np.zeros((H, W), dtype=bool)

    if T < min_duration:
        return active_mhw_map

    # Check if last `min_duration` days all exceed 90th threshold
    window_vals = sequence_3d[-min_duration:]       # (min_duration, H, W)
    window_thresh = thresholds_90th_3d[-min_duration:] # (min_duration, H, W)

    exceedance_window = (window_vals >= window_thresh) # (min_duration, H, W)
    # Cell is in active MHW if all consecutive days in the tail window exceed threshold
    all_exceed = np.all(exceedance_window, axis=0)     # (H, W)

    if mask is not None:
        all_exceed = all_exceed & mask

    return all_exceed


# Convenient alias
compute_mhw_spatial_window = detect_mhw_spatial

