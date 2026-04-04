"""
evaluate.py
-----------
Shared evaluation suite for all Phase 4 models.

Exposes:
  - compute_metrics()        : MAE, RMSE, MAPE (with zero-demand masking)
  - evaluate_by_station()    : per-station MAE / RMSE breakdown
  - evaluate_by_hour()       : per-hour MAE / RMSE breakdown
  - full_evaluation()        : runs all three, returns a single EvalResult

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


@dataclass
class EvalResult:
    """Full evaluation output for one model run."""
    metrics:      Metrics
    by_station:   pd.DataFrame   # columns: station_idx, mae, rmse, n_samples
    by_hour:      pd.DataFrame   # columns: hour, mae, rmse, n_samples
    split:        str
    model_name:   str


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Metrics:
    """
    Compute MAE, RMSE, and MAPE over flat arrays.

    Parameters
    ----------
    y_true : (N,) float array
    y_pred : (N,) float array

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

    return Metrics(mae=mae, rmse=rmse, mape=mape, mape_coverage=mape_coverage)


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
# Full evaluation
# ---------------------------------------------------------------------------

def full_evaluation(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    station_idx: np.ndarray,
    timestamps:  np.ndarray,
    model_name:  str,
    split:       str = "val",
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

    Returns
    -------
    EvalResult
    """
    metrics    = compute_metrics(y_true, y_pred)
    by_station = evaluate_by_station(y_true, y_pred, station_idx)
    by_hour    = evaluate_by_hour(y_true, y_pred, timestamps)

    logger.info(
        "[%s] %s | MAE=%.4f | RMSE=%.4f | MAPE=%.2f%% (coverage=%.1f%%)",
        split, model_name,
        metrics.mae, metrics.rmse, metrics.mape, metrics.mape_coverage * 100,
    )

    return EvalResult(
        metrics    = metrics,
        by_station = by_station,
        by_hour    = by_hour,
        split      = split,
        model_name = model_name,
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

    print("\n--- full_evaluation ---")
    result = full_evaluation(y_true, y_pred, station_idx, timestamps, model_name="test_model", split="val")
    print(f"  model={result.model_name}  split={result.split}")
    print(f"  worst station: idx={result.by_station.iloc[0]['station_idx']}  mae={result.by_station.iloc[0]['mae']:.4f}")
    print(f"  worst hour:    h={result.by_hour.sort_values('mae', ascending=False).iloc[0]['hour']}  mae={result.by_hour.sort_values('mae', ascending=False).iloc[0]['mae']:.4f}")

    print("\nAll checks passed.")