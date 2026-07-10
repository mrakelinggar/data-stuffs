"""
Tests for src/data_ingestion/weather.py (ROADMAP Phase 5).

Network-free: exercises the pure gap-flagging/cleaning helpers directly and
the cache-skip behavior of `_download_lcd_csv` via a mocked `requests.get`.
No test depends on a live NOAA fetch.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from weather import _clean_numeric, _download_lcd_csv, _flag_long_gaps


def test_flag_long_gaps_short_gap_unflagged():
    # 2-hour gap at index 3-4, below the 3-hour threshold -- should not be flagged.
    is_na = pd.Series([False, False, False, True, True, False, False])
    flagged = _flag_long_gaps(is_na, max_ffill_hours=3)
    assert not flagged.any()


def test_flag_long_gaps_long_gap_flagged():
    # 4-hour gap at index 2-5, at/above the 3-hour threshold -- should be flagged.
    is_na = pd.Series([False, False, True, True, True, True, False])
    flagged = _flag_long_gaps(is_na, max_ffill_hours=3)
    expected = pd.Series([False, False, True, True, True, True, False])
    pd.testing.assert_series_equal(flagged, expected)


def test_flag_long_gaps_no_missing_values():
    is_na = pd.Series([False, False, False])
    flagged = _flag_long_gaps(is_na, max_ffill_hours=3)
    assert not flagged.any()


def test_clean_numeric_trace_precip_maps_to_zero():
    series = pd.Series(["T", "0.12", "0.00s"])
    cleaned = _clean_numeric(series)
    assert cleaned.tolist() == pytest.approx([0.0, 0.12, 0.0])


def test_clean_numeric_strips_trailing_flag_chars():
    series = pd.Series(["72s", "45"])
    cleaned = _clean_numeric(series)
    assert cleaned.tolist() == pytest.approx([72.0, 45.0])


def test_clean_numeric_unparseable_becomes_nan():
    series = pd.Series(["", "abc"])
    cleaned = _clean_numeric(series)
    assert cleaned.isna().all()


def test_download_lcd_csv_skips_if_cached(tmp_path: Path, mocker):
    dest = tmp_path / "cached.csv"
    dest.write_text("STATION,DATE\n72505394728,2024-01-01T00:00:00\n")

    mock_get = mocker.patch("weather.requests.get")
    result = _download_lcd_csv("72505394728", 2024, dest)

    assert result == dest
    mock_get.assert_not_called()


def test_download_lcd_csv_cleans_up_partial_file_on_failure(tmp_path: Path, mocker):
    dest = tmp_path / "new.csv"

    mock_get = mocker.patch("weather.requests.get")
    mock_get.side_effect = ConnectionError("network unreachable")

    with pytest.raises(ConnectionError):
        _download_lcd_csv("72505394728", 2024, dest)

    assert not dest.exists()
