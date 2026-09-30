"""Unit tests for Hobday et al. (2016) Marine Heatwave detection."""

import numpy as np
import pytest
from src.products.marine_heatwave import detect_mhw_1d, detect_mhw_spatial


def test_hobday_5_day_persistence_filter():
    """Verify that a 5-day event is flagged, but a 3-day spike is NOT flagged as an MHW."""
    # 15 days total:
    # Days 0-1: normal (28.0°C)
    # Days 2-4: 3-day warm spike (29.5°C >= 29.0°C) -> DURATION 3 < 5 -> NOT MHW
    # Days 5-6: normal (28.0°C)
    # Days 7-12: 6-day warm event (29.5°C >= 29.0°C) -> DURATION 6 >= 5 -> ACTIVE MHW
    # Days 13-14: normal (28.0°C)
    temps = np.array([
        28.0, 28.0,
        29.5, 29.5, 29.5,
        28.0, 28.0,
        29.5, 29.5, 29.5, 29.5, 29.5, 29.5,
        28.0, 28.0,
    ])
    th_90 = np.full(15, 29.0)
    clim = np.full(15, 27.5)

    res = detect_mhw_1d(temps, th_90, climatology_means=clim, min_duration=5)

    # Days 2-4 should be False (short spike rejected)
    assert not np.any(res["is_mhw"][2:5])

    # Days 7-12 should be True (6-day persistent event accepted)
    assert np.all(res["is_mhw"][7:13])

    # Overall discrete event count should be exactly 1
    assert res["event_count"] == 1
    assert res["events"][0]["duration_days"] == 6
    assert "Category" in res["events"][0]["category"]


def test_mhw_spatial_window_flagging():
    """Verify spatial MHW flagging on a sliding 3D time sequence."""
    # 7 days, 4x4 spatial domain
    seq = np.full((7, 4, 4), 28.0, dtype=np.float32)
    th_90 = np.full((7, 4, 4), 29.0, dtype=np.float32)

    # Make cell (1, 1) exceed threshold for all last 5 days
    seq[-5:, 1, 1] = 30.0

    # Make cell (2, 2) exceed threshold for only last 3 days
    seq[-3:, 2, 2] = 30.0

    mask = np.ones((4, 4), dtype=bool)
    active_map = detect_mhw_spatial(seq, th_90, min_duration=5, mask=mask)

    assert active_map[1, 1] == True  # 5-day event active
    assert active_map[2, 2] == False # Only 3 days, not yet MHW
    assert active_map[0, 0] == False # Normal cell
