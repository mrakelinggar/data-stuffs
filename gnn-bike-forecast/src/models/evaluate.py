"""
evaluate.py
-----------
Shared evaluation suite for all Phase 4 models.

Exposes:
  - compute_metrics()        : MAE, RMSE, MAPE (with zero-demand masking)
  - evaluate_by_station()    : per-station MAE / RMSE breakdown
  - evaluate_by_hour()       : per-hour MAE / RMSE breakdown
  - bootstrap_ci()           : station-level bootstrap CI on mean MAE
  - paired_bootstrap_test()  : paired bootstrap significance test between two models
  - full_evaluation()        : runs the above, returns a single EvalResult

All functions accept numpy arrays. Models are responsible for converting
tensors to numpy before calling here.

MAPE policy: samples where y_true < 1 are excluded from MAPE. Coverage
(fraction of samples included) is reported alongside the metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAPE_THRESHOLD = 1.0  # exclude samples where y_true < this from MAPE


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Scalar metrics for a single model / split."""
    mae:           float
    rmse:          float
    mape:          float
    mape_coverage: float   # fraction of samples included in MAPE
    wape:          float   # sum(|err|) / sum(|y_true|) -- headline metric
    peak_mae:      float   # MAE weighted 3x during peak hours (business KPI)


@dataclass
class EvalResult:
    """Full evaluation output for one model run."""
    metrics:        Metrics
    by_station:     pd.DataFrame   # columns: station_idx, mae, rmse, n_samples
    by_hour:        pd.DataFrame   # columns: hour, mae, rmse, n_samples
    split:          str
    model_name:     str
    by_hour_band:   pd.DataFrame           # columns: segment, mae, rmse, n_samples (always computed)
    by_cluster:     Optional[pd.DataFrame] = None  # only if `cluster` was passed to full_evaluation()
    by_volume_tier: Optional[pd.DataFrame] = None  # only if `train_lag24`+`lag24` were passed
    mae_ci_lo:      float = float("nan")   # station-level bootstrap CI lower bound (see bootstrap_ci)
    mae_ci_hi:      float = float("nan")   # station-level bootstrap CI upper bound
    y_true:         Optional[np.ndarray] = None  # row-level, for log_predictions_to_mlflow
    y_pred:         Optional[np.ndarray] = None
    station_idx:    Optional[np.ndarray] = None
    timestamps:     Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

DEFAULT_PEAK_HOURS = (7, 8, 9, 17, 18, 19)
DEFAULT_PEAK_WEIGHT = 3.0

DEFAULT_N_BOOT = 1000
DEFAULT_CI = 0.95
DEFAULT_BOOT_SEED = 42


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error: sum(|err|) / sum(|y_true|).

    Headline metric (see CLAUDE.md Evaluation contract) -- unlike MAPE, this
    is unmasked (no y_true >= 1 exclusion), so it doesn't hide bias on quiet
    stations.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    denom = np.sum(np.abs(y_true))
    if denom == 0:
        logger.warning("sum(|y_true|) == 0 -- WAPE set to NaN")
        return float("nan")

    return float(np.sum(np.abs(y_pred - y_true)) / denom)


def peak_weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    peak_hours: tuple[int, ...] = DEFAULT_PEAK_HOURS,
    weight: float = DEFAULT_PEAK_WEIGHT,
) -> float:
    """
    MAE with `weight`x upweighting for samples falling in `peak_hours`.

    Business KPI (see CLAUDE.md): rebalancing errors during evening rush hour
    are the most operationally expensive. At weight=1.0 this is identical to
    plain MAE.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    hours  = pd.to_datetime(timestamps).hour.values

    w = np.where(np.isin(hours, peak_hours), weight, 1.0)
    errors = np.abs(y_pred - y_true)

    return float(np.sum(w * errors) / np.sum(w))


def reconstruct_cluster_id(onehot: np.ndarray) -> np.ndarray:
    """
    Reconstruct the implicit dropped-first cluster category from a one-hot
    array of cluster_0/1/2 columns.

    Parameters
    ----------
    onehot : (N, 3) array -- cluster_0, cluster_1, cluster_2 one-hot columns

    Returns
    -------
    (N,) int array in {0, 1, 2, 3}. A row that is all-zero across the three
    columns is the implicit 4th category (id 3).
    """
    onehot = np.asarray(onehot)
    return np.where(onehot.sum(axis=1) == 0, 3, np.argmax(onehot, axis=1))


def compute_volume_tier_cutoffs(train_lag24: np.ndarray) -> tuple[float, float]:
    """
    Compute tercile (33rd/66th percentile) cutoffs from TRAIN lag_24 values.

    Must be computed once from train and reused unchanged for val/test --
    never recomputed per split.
    """
    p33, p66 = np.percentile(np.asarray(train_lag24), [33.33, 66.67])
    return float(p33), float(p66)


def assign_volume_tier(lag24: np.ndarray, cutoffs: tuple[float, float]) -> np.ndarray:
    """
    Assign each row to a volume tier (0=low, 1=mid, 2=high) using fixed
    train-derived cutoffs. A value exactly equal to a cutoff lands in the
    upper bin (np.digitize default right=False).
    """
    return np.digitize(np.asarray(lag24), cutoffs)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    peak_hours: tuple[int, ...] = DEFAULT_PEAK_HOURS,
    peak_weight: float = DEFAULT_PEAK_WEIGHT,
) -> Metrics:
    """
    Compute MAE, RMSE, MAPE, WAPE, and peak-weighted MAE over flat arrays.

    Parameters
    ----------
    y_true     : (N,) float array
    y_pred     : (N,) float array
    timestamps : optional (N,) datetime-like array -- required for a true
                 peak-weighted MAE; if omitted, peak_mae falls back to plain
                 MAE (used by per-epoch training-loop monitoring, which
                 doesn't need peak weighting).

    Returns
    -------
    Metrics dataclass
    """
    y_true = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float32).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}"
        )

    errors = y_pred - y_true

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: exclude near-zero demand samples
    mask          = y_true >= MAPE_THRESHOLD
    mape_coverage = float(mask.sum() / len(y_true))

    if mask.sum() == 0:
        logger.warning("No samples with y_true >= %.1f -- MAPE set to NaN", MAPE_THRESHOLD)
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs(errors[mask] / y_true[mask])) * 100
        )

    w = wape(y_true, y_pred)
    if timestamps is not None:
        peak = peak_weighted_mae(y_true, y_pred, timestamps, peak_hours=peak_hours, weight=peak_weight)
    else:
        peak = mae

    return Metrics(
        mae=mae, rmse=rmse, mape=mape, mape_coverage=mape_coverage,
        wape=w, peak_mae=peak,
    )


# ---------------------------------------------------------------------------
# Per-station breakdown
# ---------------------------------------------------------------------------

def evaluate_by_station(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    station_idx: np.ndarray,
) -> pd.DataFrame:
    """
    Compute MAE and RMSE for each station.

    Parameters
    ----------
    y_true      : (N,) float array
    y_pred      : (N,) float array
    station_idx : (N,) int array — 0-based station index

    Returns
    -------
    DataFrame with columns [station_idx, mae, rmse, n_samples],
    sorted by mae descending (worst stations first).
    """
    y_true      = np.asarray(y_true,      dtype=np.float32).ravel()
    y_pred      = np.asarray(y_pred,      dtype=np.float32).ravel()
    station_idx = np.asarray(station_idx, dtype=np.int64).ravel()

    unique_stations = np.unique(station_idx)
    records = []

    for s in unique_stations:
        mask    = station_idx == s
        yt      = y_true[mask]
        yp      = y_pred[mask]
        errors  = yp - yt
        mae     = float(np.mean(np.abs(errors)))
        rmse    = float(np.sqrt(np.mean(errors ** 2)))
        records.append({"station_idx": int(s), "mae": mae, "rmse": rmse, "n_samples": int(mask.sum())})

    df = pd.DataFrame(records).sort_values("mae", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def bootstrap_ci(
    by_station_mae: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    ci: float = DEFAULT_CI,
    seed: int = DEFAULT_BOOT_SEED,
) -> tuple[float, float, float]:
    """
    Station-level bootstrap CI on the (macro-averaged) mean of per-station MAE.

    Resamples STATIONS with replacement -- not rows -- from `by_station_mae`
    (e.g. EvalResult.by_station["mae"].values), for spatial-correlation-aware
    CIs: rows from the same station are correlated (shared demand pattern),
    so row-level resampling would understate uncertainty.

    Note: `point` is mean(by_station_mae) -- a macro-average across stations,
    unweighted by each station's n_samples. This generally will NOT equal
    Metrics.mae (the row-level/micro-averaged MAE) when stations have unequal
    sample counts -- that's expected, not a bug.

    Returns
    -------
    (lo, point, hi)
    """
    arr = np.asarray(by_station_mae, dtype=np.float64).ravel()
    n = arr.shape[0]

    if n == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(np.mean(arr))

    if n == 1:
        logger.warning("bootstrap_ci got n=1 station -- CI collapses to the point estimate")
        return point, point, point

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)

    alpha = 1.0 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), point, float(hi)


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    station_idx: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> dict:
    """
    Paired bootstrap test of the per-station MAE delta (b - a) between two
    models evaluated on the SAME rows (y_true/y_pred_a/y_pred_b/station_idx
    must already be row-aligned -- same length, same row meaning).

    For a cross-run comparison (two separately trained models with
    independently-shaped prediction arrays), the caller must first merge on
    (station_idx, timestamp) to produce these aligned arrays -- this function
    does not do that alignment itself.

    Resamples stations with replacement, bootstraps mean(MAE_b) - mean(MAE_a),
    and derives a two-sided empirical p-value from the resulting delta
    distribution: p = 2 * min(P(delta_boot <= 0), P(delta_boot >= 0)), clipped
    to [0, 1].

    Returns
    -------
    dict with keys: delta, p_value, ci_lo, ci_hi, n_stations
    """
    y_true      = np.asarray(y_true,      dtype=np.float64).ravel()
    y_pred_a    = np.asarray(y_pred_a,    dtype=np.float64).ravel()
    y_pred_b    = np.asarray(y_pred_b,    dtype=np.float64).ravel()
    station_idx = np.asarray(station_idx, dtype=np.int64).ravel()

    by_station_a = evaluate_by_station(y_true, y_pred_a, station_idx).set_index("station_idx")["mae"]
    by_station_b = evaluate_by_station(y_true, y_pred_b, station_idx).set_index("station_idx")["mae"]

    # Both are derived from the identical station_idx array, so they share
    # the same station set by construction; .align defensively anyway.
    mae_a, mae_b = by_station_a.align(by_station_b, join="inner")
    mae_a = mae_a.values
    mae_b = mae_b.values
    n = len(mae_a)

    if n == 0:
        return {"delta": float("nan"), "p_value": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"), "n_stations": 0}

    delta = float(np.mean(mae_b) - np.mean(mae_a))

    if n == 1:
        logger.warning("paired_bootstrap_test got n=1 station -- CI/p-value undefined")
        return {"delta": delta, "p_value": float("nan"),
                "ci_lo": delta, "ci_hi": delta, "n_stations": 1}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_delta = mae_b[idx].mean(axis=1) - mae_a[idx].mean(axis=1)

    ci_lo, ci_hi = np.percentile(boot_delta, [2.5, 97.5])
    p_ge = float(np.mean(boot_delta >= 0))
    p_le = float(np.mean(boot_delta <= 0))
    p_value = float(min(1.0, 2 * min(p_ge, p_le)))

    return {
        "delta":      delta,
        "p_value":    p_value,
        "ci_lo":      float(ci_lo),
        "ci_hi":      float(ci_hi),
        "n_stations": n,
    }


# ---------------------------------------------------------------------------
# Per-hour breakdown
# ---------------------------------------------------------------------------

def evaluate_by_hour(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    timestamps: np.ndarray,
) -> pd.DataFrame:
    """
    Compute MAE and RMSE broken down by hour of day (0-23).

    Parameters
    ----------
    y_true     : (N,) float array
    y_pred     : (N,) float array
    timestamps : (N,) array of datetime-like values

    Returns
    -------
    DataFrame with columns [hour, mae, rmse, n_samples],
    sorted by hour ascending.
    """
    y_true     = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred     = np.asarray(y_pred, dtype=np.float32).ravel()
    timestamps = pd.to_datetime(timestamps)
    hours      = timestamps.hour.values

    records = []
    for h in range(24):
        mask   = hours == h
        if mask.sum() == 0:
            continue
        yt     = y_true[mask]
        yp     = y_pred[mask]
        errors = yp - yt
        mae    = float(np.mean(np.abs(errors)))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        records.append({"hour": int(h), "mae": mae, "rmse": rmse, "n_samples": int(mask.sum())})

    df = pd.DataFrame(records).sort_values("hour").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Generic per-segment breakdown (cluster / volume tier / hour band)
# ---------------------------------------------------------------------------

def evaluate_by_segment(
    y_true:         np.ndarray,
    y_pred:         np.ndarray,
    segment_values: np.ndarray,
    segment_col:    str = "segment",
) -> pd.DataFrame:
    """
    Compute MAE and RMSE for each unique value in `segment_values`.

    Generalizes the evaluate_by_station / evaluate_by_hour pattern to an
    arbitrary segmentation (cluster id, volume tier, hour band, ...).

    Parameters
    ----------
    y_true         : (N,) float array
    y_pred         : (N,) float array
    segment_values : (N,) array -- any hashable/comparable segment label

    Returns
    -------
    DataFrame with columns [segment_col, mae, rmse, n_samples],
    sorted by segment_col ascending.
    """
    y_true         = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred         = np.asarray(y_pred, dtype=np.float32).ravel()
    segment_values = np.asarray(segment_values).ravel()

    unique_segments = np.unique(segment_values)
    records = []

    for seg in unique_segments:
        mask   = segment_values == seg
        yt     = y_true[mask]
        yp     = y_pred[mask]
        errors = yp - yt
        mae    = float(np.mean(np.abs(errors)))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        records.append({segment_col: seg, "mae": mae, "rmse": rmse, "n_samples": int(mask.sum())})

    df = pd.DataFrame(records).sort_values(segment_col).reset_index(drop=True)
    return df


_HOUR_BAND_LABELS = {
    **{h: "off_peak" for h in (0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23)},
    **{h: "am_peak" for h in (7, 8, 9)},
    **{h: "mid" for h in (10, 11, 12, 13, 14, 15, 16)},
    **{h: "pm_peak" for h in (17, 18, 19)},
}


def _assign_hour_band(timestamps: np.ndarray) -> np.ndarray:
    hours = pd.to_datetime(timestamps).hour.values
    return np.array([_HOUR_BAND_LABELS[h] for h in hours])


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def full_evaluation(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    station_idx: np.ndarray,
    timestamps:  np.ndarray,
    model_name:  str,
    split:       str = "val",
    cluster:     Optional[np.ndarray] = None,
    train_lag24: Optional[np.ndarray] = None,
    lag24:       Optional[np.ndarray] = None,
    n_boot:      int = DEFAULT_N_BOOT,
    ci:          float = DEFAULT_CI,
    boot_seed:   int = DEFAULT_BOOT_SEED,
) -> EvalResult:
    """
    Run all evaluations and return a single EvalResult.

    Parameters
    ----------
    y_true      : (N,) float array
    y_pred      : (N,) float array
    station_idx : (N,) int array
    timestamps  : (N,) datetime-like array
    model_name  : name to tag results with (e.g. "gcn_adj_knn")
    split       : which split this is ("train", "val", "test")
    cluster     : optional (N,) int array -- already-reconstructed cluster id
                  (see reconstruct_cluster_id()). If given, by_cluster is computed.
    train_lag24 : optional (M,) array -- TRAIN split's lag_24 values, used only
                  to derive tercile cutoffs. Must be passed together with `lag24`.
    lag24       : optional (N,) array -- this split's own row-level lag_24
                  values. Must be passed together with `train_lag24`.
    n_boot      : number of bootstrap resamples for the station-level MAE CI.
    ci          : confidence level for the MAE CI.
    boot_seed   : seed for the bootstrap CI (fixed by default so CIs are
                  comparable across differently-seeded training runs).

    Returns
    -------
    EvalResult
    """
    if (train_lag24 is None) != (lag24 is None):
        logger.warning(
            "full_evaluation() got only one of train_lag24/lag24 -- "
            "both are required to compute by_volume_tier; skipping it."
        )

    metrics      = compute_metrics(y_true, y_pred, timestamps=timestamps)
    by_station   = evaluate_by_station(y_true, y_pred, station_idx)
    by_hour      = evaluate_by_hour(y_true, y_pred, timestamps)
    by_hour_band = evaluate_by_segment(y_true, y_pred, _assign_hour_band(timestamps), segment_col="hour_band")

    by_cluster = None
    if cluster is not None:
        by_cluster = evaluate_by_segment(y_true, y_pred, cluster, segment_col="cluster")

    by_volume_tier = None
    if train_lag24 is not None and lag24 is not None:
        cutoffs = compute_volume_tier_cutoffs(train_lag24)
        tiers   = assign_volume_tier(lag24, cutoffs)
        tier_labels = np.array(["low", "mid", "high"])[tiers]
        by_volume_tier = evaluate_by_segment(y_true, y_pred, tier_labels, segment_col="volume_tier")

    mae_ci_lo, _, mae_ci_hi = bootstrap_ci(by_station["mae"].values, n_boot=n_boot, ci=ci, seed=boot_seed)

    logger.info(
        "[%s] %s | MAE=%.4f | RMSE=%.4f | WAPE=%.4f | Peak-MAE=%.4f | MAPE=%.2f%% (coverage=%.1f%%)",
        split, model_name,
        metrics.mae, metrics.rmse, metrics.wape, metrics.peak_mae,
        metrics.mape, metrics.mape_coverage * 100,
    )

    return EvalResult(
        metrics        = metrics,
        by_station     = by_station,
        by_hour        = by_hour,
        split          = split,
        model_name     = model_name,
        by_hour_band   = by_hour_band,
        by_cluster     = by_cluster,
        by_volume_tier = by_volume_tier,
        mae_ci_lo      = mae_ci_lo,
        mae_ci_hi      = mae_ci_hi,
        y_true         = np.asarray(y_true),
        y_pred         = np.asarray(y_pred),
        station_idx    = np.asarray(station_idx),
        timestamps     = np.asarray(pd.to_datetime(timestamps)),
    )


# ---------------------------------------------------------------------------
# MLflow logging helper
# ---------------------------------------------------------------------------

def log_metrics_to_mlflow(
    result: EvalResult,
    prefix: str = "",
) -> None:
    """
    Log an EvalResult's scalar metrics to the active MLflow run.

    Parameters
    ----------
    result : EvalResult from full_evaluation()
    prefix : optional prefix e.g. "val_" or "test_"
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed -- skipping metric logging")
        return

    p = f"{prefix}{result.split}_" if prefix else f"{result.split}_"
    mlflow.log_metrics({
        f"{p}mae":           result.metrics.mae,
        f"{p}rmse":          result.metrics.rmse,
        f"{p}mape":          result.metrics.mape,
        f"{p}mape_coverage": result.metrics.mape_coverage,
        f"{p}wape":          result.metrics.wape,
        f"{p}peak_mae":      result.metrics.peak_mae,
        f"{p}mae_ci_lo":     result.mae_ci_lo,
        f"{p}mae_ci_hi":     result.mae_ci_hi,
    })


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(42)
    N   = 10_000

    y_true      = rng.poisson(lam=3.0, size=N).astype(np.float32)
    y_pred      = y_true + rng.normal(0, 1.5, size=N).astype(np.float32)
    station_idx = rng.integers(0, 50, size=N)
    timestamps  = pd.date_range("2024-05-01", periods=N, freq="h")

    print("\n--- compute_metrics ---")
    m = compute_metrics(y_true, y_pred)
    print(f"  MAE={m.mae:.4f}  RMSE={m.rmse:.4f}  MAPE={m.mape:.2f}%  coverage={m.mape_coverage:.2%}")

    print("\n--- evaluate_by_station (top 5 worst) ---")
    by_station = evaluate_by_station(y_true, y_pred, station_idx)
    print(by_station.head())

    print("\n--- evaluate_by_hour ---")
    by_hour = evaluate_by_hour(y_true, y_pred, timestamps)
    print(by_hour.to_string(index=False))

    print("\n--- full_evaluation (with cluster / volume-tier segments) ---")
    cluster_onehot = rng.integers(0, 2, size=(N, 3))  # deliberately includes some all-zero rows
    cluster        = reconstruct_cluster_id(cluster_onehot)
    train_lag24    = rng.poisson(lam=3.0, size=5_000).astype(np.float32)
    lag24          = rng.poisson(lam=3.0, size=N).astype(np.float32)

    result = full_evaluation(
        y_true, y_pred, station_idx, timestamps, model_name="test_model", split="val",
        cluster=cluster, train_lag24=train_lag24, lag24=lag24,
    )
    print(f"  model={result.model_name}  split={result.split}")
    print(f"  WAPE={result.metrics.wape:.4f}  Peak-MAE={result.metrics.peak_mae:.4f}")
    print(f"  worst station: idx={result.by_station.iloc[0]['station_idx']}  mae={result.by_station.iloc[0]['mae']:.4f}")
    print(f"  worst hour:    h={result.by_hour.sort_values('mae', ascending=False).iloc[0]['hour']}  mae={result.by_hour.sort_values('mae', ascending=False).iloc[0]['mae']:.4f}")
    print("\n  by_cluster:")
    print(result.by_cluster.to_string(index=False))
    print("\n  by_volume_tier:")
    print(result.by_volume_tier.to_string(index=False))
    print("\n  by_hour_band:")
    print(result.by_hour_band.to_string(index=False))

    print("\nAll checks passed.")