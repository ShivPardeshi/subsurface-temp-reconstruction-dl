"""Priority Zones Definition Module.

Defines spatial, depth, and temporal slicing definitions for the 7 canonical
priority zones established in OceanEmbed planning:
  1. Bay of Bengal mixed-layer/barrier-layer zone (0–30m)
  2. Thermocline core, both basins (20–200m)
  3. Arabian Sea Persian-Gulf-Water zone (200–300m)
  4. The 8–10°N confluence zone
  5. Extreme-event windows (cyclone dates from IBTrACS)
  6. Monsoon transition windows (onset & withdrawal dates)
  7. The equatorial domain edge (~2°N–5°N boundary region)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from src.utils.grid import get_target_grid, CANONICAL_DEPTHS, get_grid_mesh
from src.data.harmonize.region_masks import generate_region_masks


@dataclass
class PriorityZone:
    """Represents a priority evaluation zone with spatial, vertical, and temporal filters."""
    zone_id: int
    name: str
    description: str
    spatial_mask: np.ndarray          # 2D boolean array (H, W)
    depth_indices: List[int]          # List of indices into CANONICAL_DEPTHS
    depth_values: List[float]         # Actual depths in meters
    time_filter_fn: Optional[Callable[[Union[pd.DatetimeIndex, np.ndarray, List[str]]], np.ndarray]] = None


def is_cyclone_date(dates: Union[pd.DatetimeIndex, np.ndarray, List[str]], ibtracs_csv_path: Optional[str] = None) -> np.ndarray:
    """Determine whether given timestamps coincide with North Indian Ocean cyclone events."""
    d_index = pd.to_datetime(dates)

    # Known intense North Indian Ocean cyclone windows (2020-2023) as default ground truth
    known_cyclone_intervals = [
        ("2020-05-16", "2020-05-21"),  # Amphan
        ("2020-05-31", "2020-06-04"),  # Nisarga
        ("2020-11-21", "2020-11-26"),  # Nivar
        ("2020-11-30", "2020-12-04"),  # Burevi
        ("2021-05-14", "2021-05-19"),  # Tauktae
        ("2021-05-23", "2021-05-28"),  # Yaas
        ("2021-09-24", "2021-09-28"),  # Gulab
        ("2021-12-02", "2021-12-06"),  # Jawad
        ("2022-05-07", "2022-05-12"),  # Asani
        ("2022-10-22", "2022-10-25"),  # Sitrang
        ("2022-12-06", "2022-12-10"),  # Mandous
        ("2023-05-09", "2023-05-15"),  # Mocha
        ("2023-06-06", "2023-06-19"),  # Biparjoy
        ("2023-10-20", "2023-10-25"),  # Tej / Hamoon
        ("2023-12-01", "2023-12-06"),  # Michaung
    ]

    mask = np.zeros(len(d_index), dtype=bool)

    # If IBTrACS CSV is available, augment with official records
    if ibtracs_csv_path and os.path.exists(ibtracs_csv_path):
        try:
            df = pd.read_csv(ibtracs_csv_path, low_memory=False, skiprows=[1])
            if "ISO_TIME" in df.columns:
                c_times = pd.to_datetime(df["ISO_TIME"], errors="coerce").dropna().dt.date.unique()
                c_dates_set = set(c_times)
                for idx, dt in enumerate(d_index):
                    if dt.date() in c_dates_set:
                        mask[idx] = True
                return mask
        except Exception:
            pass

    # Standard interval matching
    for start_str, end_str in known_cyclone_intervals:
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        mask |= ((d_index >= start_dt) & (d_index <= end_dt))

    return mask


def is_monsoon_transition(dates: Union[pd.DatetimeIndex, np.ndarray, List[str]]) -> np.ndarray:
    """Determine whether timestamps fall in Monsoon Transition windows:
    - Pre-monsoon onset window: May 15 to June 15
    - Post-monsoon withdrawal window: September 15 to October 15
    """
    d_index = pd.to_datetime(dates)
    mask = np.zeros(len(d_index), dtype=bool)

    for idx, dt in enumerate(d_index):
        month = dt.month
        day = dt.day
        # May 15 - June 15
        onset = (month == 5 and day >= 15) or (month == 6 and day <= 15)
        # Sept 15 - Oct 15
        withdrawal = (month == 9 and day >= 15) or (month == 10 and day <= 15)
        if onset or withdrawal:
            mask[idx] = True

    return mask


def get_priority_zones(
    lat: Optional[np.ndarray] = None,
    lon: Optional[np.ndarray] = None,
    toy_mode: bool = False,
    ibtracs_csv_path: Optional[str] = None,
) -> Dict[str, PriorityZone]:
    """Instantiate and return all 7 priority zones with calibrated masks and bounds.

    Returns:
        Ordered dictionary of {zone_slug: PriorityZone}.
    """
    if lat is None or lon is None:
        lat, lon = get_target_grid(toy_mode=toy_mode)

    if ibtracs_csv_path is None:
        default_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw", "ibtracs", "ibtracs.NI.list.v04r01.csv"))
        if os.path.exists(default_csv):
            ibtracs_csv_path = default_csv

    lat_mesh, lon_mesh = get_grid_mesh(lat, lon)
    as_mask, bob_mask, conf_mask, open_mask = generate_region_masks(lat, lon)

    # 1. Bay of Bengal barrier layer zone (0-30m)
    # Depths: 0, 5, 10, 20, 30m
    bob_depth_indices = [i for i, d in enumerate(CANONICAL_DEPTHS) if d <= 30]
    bob_depth_vals = [CANONICAL_DEPTHS[i] for i in bob_depth_indices]
    bob_spatial = bob_mask >= 0.25

    # 2. Thermocline core (20-200m, both basins)
    # Empirically redefined from 75-150m to capture full elevated thermal gradient and error bracket
    # Depths: 20, 30, 50, 75, 100, 125, 150, 200m
    tc_depth_indices = [i for i, d in enumerate(CANONICAL_DEPTHS) if 20 <= d <= 200]
    tc_depth_vals = [CANONICAL_DEPTHS[i] for i in tc_depth_indices]
    tc_spatial = (as_mask + bob_mask + open_mask + conf_mask) > 0.1

    # 3. Arabian Sea Persian-Gulf-Water zone (200-300m)
    # Depths: 200, 300m
    pgw_depth_indices = [i for i, d in enumerate(CANONICAL_DEPTHS) if 200 <= d <= 300]
    pgw_depth_vals = [CANONICAL_DEPTHS[i] for i in pgw_depth_indices]
    pgw_spatial = as_mask >= 0.25

    # 4. 8-10°N Confluence zone (all depths)
    all_depth_indices = list(range(len(CANONICAL_DEPTHS)))
    all_depth_vals = list(CANONICAL_DEPTHS)
    conf_spatial = (lat_mesh >= 8.0) & (lat_mesh <= 10.0) & (conf_mask >= 0.15)
    if not np.any(conf_spatial):
        # Fallback if in toy mode cropped away from 8°N
        conf_spatial = conf_mask >= 0.15

    # 5. Extreme-event windows (all ocean cells, cyclone dates)
    ocean_spatial = (as_mask + bob_mask + open_mask + conf_mask) > 0.05
    cyclone_filter = lambda dates: is_cyclone_date(dates, ibtracs_csv_path=ibtracs_csv_path)

    # 6. Monsoon transition windows (all ocean cells, May-Jun & Sep-Oct dates)
    monsoon_filter = lambda dates: is_monsoon_transition(dates)

    # 7. Equatorial domain edge (~2°N-5°N boundary region)
    eq_spatial = (lat_mesh <= 5.0) & ocean_spatial
    if not np.any(eq_spatial):
        # Fallback if cropped in toy mode: bottom 10% latitude rows
        eq_spatial = np.zeros_like(lat_mesh, dtype=bool)
        eq_spatial[: max(1, len(lat) // 10), :] = True
        eq_spatial = eq_spatial & ocean_spatial

    zones = {
        "zone1_bob_barrier_layer": PriorityZone(
            zone_id=1,
            name="Bay of Bengal Barrier Layer (0-30m)",
            description="Freshwater plume barrier layer capping in Bay of Bengal (0 to 30m depth)",
            spatial_mask=bob_spatial,
            depth_indices=bob_depth_indices,
            depth_values=bob_depth_vals,
            time_filter_fn=None,
        ),
        "zone2_thermocline_core": PriorityZone(
            zone_id=2,
            name="Thermocline Core (75-150m)",
            description="High vertical thermal gradient zone across Arabian Sea and Bay of Bengal (75 to 150m depth)",
            spatial_mask=tc_spatial,
            depth_indices=tc_depth_indices,
            depth_values=tc_depth_vals,
            time_filter_fn=None,
        ),
        "zone3_as_persian_gulf_water": PriorityZone(
            zone_id=3,
            name="Arabian Sea PGW Zone (200-300m)",
            description="High-salinity Persian Gulf Water outflow intrusion in Arabian Sea (200 to 300m depth)",
            spatial_mask=pgw_spatial,
            depth_indices=pgw_depth_indices,
            depth_values=pgw_depth_vals,
            time_filter_fn=None,
        ),
        "zone4_confluence_zone": PriorityZone(
            zone_id=4,
            name="8-10°N Confluence Zone",
            description="Dynamic current confluence separating Arabian Sea and Bay of Bengal (8°N - 10°N)",
            spatial_mask=conf_spatial,
            depth_indices=all_depth_indices,
            depth_values=all_depth_vals,
            time_filter_fn=None,
        ),
        "zone5_extreme_cyclone_events": PriorityZone(
            zone_id=5,
            name="Extreme Event Windows (Cyclones)",
            description="Active North Indian Ocean tropical cyclone periods (IBTrACS identified windows)",
            spatial_mask=ocean_spatial,
            depth_indices=all_depth_indices,
            depth_values=all_depth_vals,
            time_filter_fn=cyclone_filter,
        ),
        "zone6_monsoon_transitions": PriorityZone(
            zone_id=6,
            name="Monsoon Transition Windows",
            description="Pre-monsoon onset (May 15-Jun 15) and Post-monsoon withdrawal (Sep 15-Oct 15) periods",
            spatial_mask=ocean_spatial,
            depth_indices=all_depth_indices,
            depth_values=all_depth_vals,
            time_filter_fn=monsoon_filter,
        ),
        "zone7_equatorial_edge": PriorityZone(
            zone_id=7,
            name="Equatorial Boundary Edge (2-5°N)",
            description="Boundary region near 2°N - 5°N where Coriolis force approaches zero and boundary artifacts occur",
            spatial_mask=eq_spatial,
            depth_indices=all_depth_indices,
            depth_values=all_depth_vals,
            time_filter_fn=None,
        ),
    }

    return zones
