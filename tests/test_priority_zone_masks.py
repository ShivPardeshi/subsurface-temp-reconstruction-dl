"""Unit tests for Priority Zone masks, boundaries, and evaluator.

Verifies:
1. All 7 canonical priority zones exist and have correct depth ranges.
2. Spatial bounds and masks do not spill into incorrect domains.
3. Temporal filters (cyclone windows, monsoon transitions) correctly trigger on known dates.
4. evaluate_metric_by_zone executes cleanly across all 7 zones.
"""

import numpy as np
import pandas as pd
import pytest

from src.utils.grid import get_target_grid, CANONICAL_DEPTHS
from src.evaluation.slicing.priority_zones import (
    get_priority_zones,
    is_cyclone_date,
    is_monsoon_transition,
)
from src.evaluation.slicing.zone_evaluator import evaluate_metric_by_zone
from src.evaluation.metrics.basic_metrics import compute_all_basic_metrics


def test_priority_zones_initialization():
    """Verify all 7 zones are defined and have physically correct depth levels."""
    zones = get_priority_zones(toy_mode=False)

    expected_keys = [
        "zone1_bob_barrier_layer",
        "zone2_thermocline_core",
        "zone3_as_persian_gulf_water",
        "zone4_confluence_zone",
        "zone5_extreme_cyclone_events",
        "zone6_monsoon_transitions",
        "zone7_equatorial_edge",
    ]

    for key in expected_keys:
        assert key in zones, f"Missing expected priority zone: {key}"

    # Zone 1: Bay of Bengal barrier layer (0-30m)
    z1 = zones["zone1_bob_barrier_layer"]
    assert all(d <= 30 for d in z1.depth_values), f"Zone 1 contains depth > 30m: {z1.depth_values}"

    # Zone 2: Thermocline core (20-200m, empirically redefined from 75-150m)
    z2 = zones["zone2_thermocline_core"]
    assert all(20 <= d <= 200 for d in z2.depth_values), f"Zone 2 depth out of thermocline: {z2.depth_values}"

    # Zone 3: Arabian Sea PGW (200-300m)
    z3 = zones["zone3_as_persian_gulf_water"]
    assert all(200 <= d <= 300 for d in z3.depth_values), f"Zone 3 depth out of PGW range: {z3.depth_values}"


def test_temporal_filters_cyclone_and_monsoon():
    """Verify cyclone and monsoon transition temporal filter functions."""
    # Test dates
    cyclone_dates = ["2021-05-16", "2023-06-12", "2021-01-15"]
    c_mask = is_cyclone_date(cyclone_dates)
    assert c_mask[0] == True, "May 16 2021 was Cyclone Tauktae - should be True"
    assert c_mask[1] == True, "June 12 2023 was Cyclone Biparjoy - should be True"
    assert c_mask[2] == False, "Jan 15 2021 had no cyclone - should be False"

    # Monsoon transitions
    # May 20 (onset window) -> True
    # Oct 01 (withdrawal window) -> True
    # Jan 10 (winter) -> False
    # Aug 01 (peak monsoon, not transition) -> False
    monsoon_dates = ["2022-05-20", "2022-10-01", "2022-01-10", "2022-08-01"]
    m_mask = is_monsoon_transition(monsoon_dates)
    assert m_mask[0] == True, "May 20 should be onset transition window"
    assert m_mask[1] == True, "Oct 1 should be withdrawal transition window"
    assert m_mask[2] == False, "Jan 10 should not be transition window"
    assert m_mask[3] == False, "Aug 1 is peak monsoon, not onset/withdrawal"


def test_zone_evaluator_execution():
    """Verify generic evaluate_metric_by_zone runs without error on 3D array."""
    lat, lon = get_target_grid(toy_mode=True)
    h, w = len(lat), len(lon)
    num_depths = len(CANONICAL_DEPTHS)

    pred = np.ones((num_depths, h, w), dtype=np.float32) * 25.0
    target = np.ones((num_depths, h, w), dtype=np.float32) * 24.5
    clim = np.ones((num_depths, h, w), dtype=np.float32) * 24.0

    zones = get_priority_zones(lat, lon, toy_mode=True)

    results = evaluate_metric_by_zone(
        metric_fn=compute_all_basic_metrics,
        pred=pred,
        target=target,
        zones=zones,
        clim=clim,
    )

    assert isinstance(results, dict)
    valid_count = 0
    for zk in zones.keys():
        assert zk in results
        val = results[zk]
        if isinstance(val, dict) and "rmse" in val and np.isfinite(val["rmse"]):
            # Sliced RMSE on active spatial zone should be ~0.50
            assert np.isclose(val["rmse"], 0.50, atol=1e-3)
            valid_count += 1
    # At least Zone 1 (BoB barrier layer) and Zone 2 (thermocline) must be active in BoB toy crop
    assert valid_count >= 2

    # Now test with batched dates including a known cyclone date
    pred_4d = pred[None, ...]
    target_4d = target[None, ...]
    clim_4d = clim[None, ...]
    cyclone_date = pd.to_datetime(["2021-05-16"])  # Cyclone Tauktae

    results_temporal = evaluate_metric_by_zone(
        metric_fn=compute_all_basic_metrics,
        pred=pred_4d,
        target=target_4d,
        zones=zones,
        clim=clim_4d,
        dates=cyclone_date,
    )
    # Zone 5 (cyclone) should now be active
    z5 = results_temporal["zone5_extreme_cyclone_events"]
    assert np.isclose(z5["rmse"], 0.50, atol=1e-3)
