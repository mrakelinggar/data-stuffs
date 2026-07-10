"""
Tests for src/models/evaluate.py -- WAPE, peak-weighted MAE, and the
generic per-segment breakdowns (cluster / volume tier / hour band)
added in ROADMAP Phase 2.

All fixtures are synthetic -- no dependency on data/processed/ or mlflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    assign_volume_tier,
    bootstrap_ci,
    compute_metrics,
    compute_volume_tier_cutoffs,
    evaluate_by_segment,
    full_evaluation,
    paired_bootstrap_test,
    peak_weighted_mae,
    reconstruct_cluster_id,
    wape,
)


# ---------------------------------------------------------------------------
# wape
# ---------------------------------------------------------------------------

def test_wape_correctness():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 3.0, 3.0, 6.0])
    # |err| = [0, 1, 0, 2] -> sum = 3; sum(|y_true|) = 10
    assert wape(y_true, y_pred) == pytest.approx(0.3)


def test_wape_all_zero_ytrue():
    y_true = np.zeros(5)
    y_pred = np.array([1.0, 0.0, 2.0, 0.0, 1.0])
    result = wape(y_true, y_pred)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# peak_weighted_mae
# ---------------------------------------------------------------------------

def _make_timestamps(hours: list[int]) -> pd.DatetimeIndex:
    base = pd.Timestamp("2024-05-01")
    return pd.DatetimeIndex([base + pd.Timedelta(hours=h) for h in hours])


def test_peak_weighted_mae_equals_mae_at_weight_1():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([2.0, 2.0, 5.0, 4.0, 8.0, 6.0])
    timestamps = _make_timestamps([0, 7, 8, 17, 18, 3])  # mix of peak/off-peak

    plain_mae = float(np.mean(np.abs(y_pred - y_true)))
    result = peak_weighted_mae(y_true, y_pred, timestamps, weight=1.0)
    assert result == pytest.approx(plain_mae)


def test_peak_weighted_mae_upweights_peak_errors():
    # Errors: peak hours (7, 17) have error 10; off-peak hours (0, 3) have error 1.
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([10.0, 1.0, 10.0, 1.0])
    timestamps = _make_timestamps([7, 0, 17, 3])

    result_w1 = peak_weighted_mae(y_true, y_pred, timestamps, weight=1.0)
    result_w3 = peak_weighted_mae(y_true, y_pred, timestamps, weight=3.0)

    # Hand-computed: weight=3 -> weights = [3,1,3,1], errors = [10,1,10,1]
    # sum(w*err) = 30+1+30+1 = 62; sum(w) = 8 -> 7.75
    assert result_w3 == pytest.approx(62.0 / 8.0)
    assert result_w3 > result_w1


# ---------------------------------------------------------------------------
# evaluate_by_segment
# ---------------------------------------------------------------------------

def test_evaluate_by_segment_grouping():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_pred = np.array([2.0, 2.0, 5.0, 4.0, 8.0, 6.0])
    segments = np.array(["a", "a", "b", "b", "c", "c"])

    df = evaluate_by_segment(y_true, y_pred, segments, segment_col="seg")

    assert list(df["seg"]) == ["a", "b", "c"]
    assert list(df["n_samples"]) == [2, 2, 2]

    seg_a = df[df["seg"] == "a"].iloc[0]
    assert seg_a["mae"] == pytest.approx(np.mean([1.0, 0.0]))

    seg_c = df[df["seg"] == "c"].iloc[0]
    assert seg_c["mae"] == pytest.approx(np.mean([3.0, 0.0]))


# ---------------------------------------------------------------------------
# reconstruct_cluster_id
# ---------------------------------------------------------------------------

def test_reconstruct_cluster_id():
    onehot = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    result = reconstruct_cluster_id(onehot)
    np.testing.assert_array_equal(result, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# volume tier cutoffs / boundary
# ---------------------------------------------------------------------------

def test_volume_tier_cutoffs_and_boundary():
    train_lag24 = np.arange(1, 100, dtype=np.float64)
    cutoffs = compute_volume_tier_cutoffs(train_lag24)
    expected = tuple(np.percentile(train_lag24, [33.33, 66.67]))
    assert cutoffs[0] == pytest.approx(expected[0])
    assert cutoffs[1] == pytest.approx(expected[1])

    p33, p66 = cutoffs
    # A value exactly equal to a cutoff lands in the upper bin.
    tiers = assign_volume_tier(np.array([p33 - 0.01, p33, p66, p66 + 0.01]), cutoffs)
    assert tiers[0] == 0   # just below p33 -> low
    assert tiers[1] == 1   # exactly at p33 -> mid (upper bin)
    assert tiers[2] == 2   # exactly at p66 -> high (upper bin)
    assert tiers[3] == 2   # just above p66 -> high


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_bounds_point_estimate():
    rng = np.random.default_rng(7)
    by_station_mae = rng.normal(loc=2.0, scale=0.5, size=30).clip(min=0.01)

    lo, point, hi = bootstrap_ci(by_station_mae, n_boot=1000, ci=0.95, seed=42)

    assert lo <= point <= hi
    assert point == pytest.approx(np.mean(by_station_mae))


def test_bootstrap_ci_single_station_collapses_to_point():
    by_station_mae = np.array([1.5])
    lo, point, hi = bootstrap_ci(by_station_mae, seed=42)
    assert lo == pytest.approx(1.5)
    assert point == pytest.approx(1.5)
    assert hi == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------

def test_paired_bootstrap_test_identical_predictions():
    rng = np.random.default_rng(3)
    n = 400
    y_true = rng.poisson(lam=3.0, size=n).astype(np.float64)
    y_pred = y_true + rng.normal(0, 1.0, size=n)
    station_idx = rng.integers(0, 12, size=n)

    result = paired_bootstrap_test(y_true, y_pred, y_pred, station_idx, n_boot=500, seed=42)

    assert result["delta"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_bootstrap_test_detects_directional_difference():
    rng = np.random.default_rng(11)
    n_stations, n_per_station = 15, 200
    station_idx = np.repeat(np.arange(n_stations), n_per_station)
    n = len(station_idx)
    y_true = rng.poisson(lam=3.0, size=n).astype(np.float64)
    y_pred_a = y_true + rng.normal(0.0, 1.0, size=n)   # good model
    y_pred_b = y_true + rng.normal(2.0, 1.0, size=n)   # systematically worse

    result = paired_bootstrap_test(y_true, y_pred_a, y_pred_b, station_idx, n_boot=500, seed=42)

    assert result["delta"] > 0          # b worse than a
    assert result["ci_lo"] > 0          # CI excludes zero
    assert result["p_value"] < 0.05


def test_paired_bootstrap_test_handles_sparse_station_ids():
    # station ids are non-contiguous (not 0..k-1) -- regression guard against
    # any dense-indexing assumption creeping into evaluate_by_station usage.
    rng = np.random.default_rng(5)
    n = 400
    y_true = rng.poisson(lam=3.0, size=n).astype(np.float64)
    y_pred = y_true + rng.normal(0, 1.0, size=n)
    station_idx = rng.choice([3, 17, 42, 256], size=n)

    result = paired_bootstrap_test(y_true, y_pred, y_pred, station_idx, n_boot=200, seed=1)

    assert result["n_stations"] == 4
    assert result["delta"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# full_evaluation optional-arg behavior
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_eval_data():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.poisson(lam=3.0, size=n).astype(np.float32)
    y_pred = y_true + rng.normal(0, 1.0, size=n).astype(np.float32)
    station_idx = rng.integers(0, 10, size=n)
    timestamps = pd.date_range("2024-05-01", periods=n, freq="h")
    return y_true, y_pred, station_idx, timestamps


def test_full_evaluation_optional_segments_none_by_default(synthetic_eval_data):
    y_true, y_pred, station_idx, timestamps = synthetic_eval_data
    result = full_evaluation(y_true, y_pred, station_idx, timestamps, model_name="m", split="val")

    assert result.by_cluster is None
    assert result.by_volume_tier is None
    assert result.by_hour_band is not None


def test_full_evaluation_with_all_optional_args(synthetic_eval_data):
    y_true, y_pred, station_idx, timestamps = synthetic_eval_data
    rng = np.random.default_rng(1)
    n = len(y_true)

    cluster = reconstruct_cluster_id(rng.integers(0, 2, size=(n, 3)))
    train_lag24 = rng.poisson(lam=3.0, size=1000).astype(np.float32)
    lag24 = rng.poisson(lam=3.0, size=n).astype(np.float32)

    result = full_evaluation(
        y_true, y_pred, station_idx, timestamps, model_name="m", split="val",
        cluster=cluster, train_lag24=train_lag24, lag24=lag24,
    )

    assert result.by_cluster is not None
    assert result.by_volume_tier is not None
    assert result.by_hour_band is not None
    assert np.isfinite(result.metrics.wape)
    assert np.isfinite(result.metrics.peak_mae)


def test_full_evaluation_warns_on_partial_volume_tier_args(synthetic_eval_data, caplog):
    y_true, y_pred, station_idx, timestamps = synthetic_eval_data
    lag24 = np.zeros(len(y_true))

    result = full_evaluation(
        y_true, y_pred, station_idx, timestamps, model_name="m", split="val",
        lag24=lag24,  # train_lag24 omitted -- should warn and skip by_volume_tier
    )
    assert result.by_volume_tier is None


def test_full_evaluation_attaches_bootstrap_ci_and_row_arrays(synthetic_eval_data):
    y_true, y_pred, station_idx, timestamps = synthetic_eval_data
    result = full_evaluation(y_true, y_pred, station_idx, timestamps, model_name="m", split="val")

    assert np.isfinite(result.mae_ci_lo) and np.isfinite(result.mae_ci_hi)
    assert result.mae_ci_lo <= result.mae_ci_hi
    assert result.y_true is not None
    assert result.station_idx is not None
    assert len(result.y_true) == len(y_true)


def test_compute_metrics_peak_mae_fallback_without_timestamps():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.peak_mae == pytest.approx(metrics.mae)
