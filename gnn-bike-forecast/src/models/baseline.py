"""
baseline.py
-----------
Naive and Linear Regression baselines for t+24 bike demand forecasting.

Naive
-----
  Uses lag_24 directly as the prediction. Zero parameters, zero training.
  Sets the floor — every other model must beat this.

Linear Regression
-----------------
  sklearn Ridge regression on all 15 features. Fit on train, evaluated on val.
  Ridge over plain OLS to handle correlated features (lag features are
  highly correlated with each other and with rolling means).

Both runs are logged to the MLflow experiment 'bike-demand-forecasting'.

Usage
-----
  python src/models/baseline.py --data-dir data/processed
  python src/models/baseline.py --data-dir data/processed --model naive
  python src/models/baseline.py --data-dir data/processed --model linear
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import (
    FEATURE_COLS,
    LSTM_SEQ_LEN,
    BikeDataset,
    build_dataloaders,
)
from evaluate import EvalResult, full_evaluation, log_metrics_to_mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike-demand-forecasting"
LAG24_COL       = FEATURE_COLS.index("lag_24")


# ---------------------------------------------------------------------------
# Naive baseline
# ---------------------------------------------------------------------------

def run_naive(
    data_dir: Path,
    log_to_mlflow: bool = True,
) -> EvalResult:
    """
    Naive baseline: predict lag_24 as-is for every station/timestamp.

    No training involved. Evaluated on val split.
    """
    logger.info("Running Naive baseline...")

    train_ds = BikeDataset(data_dir, "train")
    val_ds   = BikeDataset(data_dir, "val")

    # For Naive we only need the val set
    X_val, y_val = val_ds.numpy_xy()

    # Retrieve station_idx and timestamps for breakdown metrics
    station_idx_val = val_ds.station_idx.numpy()

    # Load val parquet directly for timestamps
    import pandas as pd
    val_feat = pd.read_parquet(data_dir / "features_val.parquet")
    timestamps_val = pd.to_datetime(val_feat["timestamp"].values)

    # # Prediction: just use lag_24 column
    # y_pred = X_val[:, LAG24_COL]

    # from feature_metadata.json
    lag24_mean  = 1.6489
    lag24_scale = 3.5264

    y_pred = X_val[:, LAG24_COL] * lag24_scale + lag24_mean
    y_pred = np.clip(y_pred, 0, None)

    result = full_evaluation(
        y_true      = y_val,
        y_pred      = y_pred,
        station_idx = station_idx_val,
        timestamps  = timestamps_val,
        model_name  = "naive",
        split       = "val",
    )

    if log_to_mlflow:
        _log_baseline_to_mlflow(
            model_name = "naive",
            params     = {"model_type": "naive", "prediction": "lag_24"},
            result     = result,
        )

    return result


# ---------------------------------------------------------------------------
# Linear Regression baseline
# ---------------------------------------------------------------------------

def run_linear(
    data_dir: Path,
    alpha:         float = 1.0,
    log_to_mlflow: bool  = True,
) -> EvalResult:
    """
    Ridge regression on all 15 features.

    Fit on train, evaluated on val. Features are already scaled by Phase 3
    but we add a StandardScaler in the pipeline anyway for safety — Ridge is
    sensitive to feature scale and Phase 3 scaling may not be zero-mean.

    Parameters
    ----------
    alpha : Ridge regularisation strength (default 1.0)
    """
    logger.info("Running Linear Regression baseline (alpha=%.2f)...", alpha)

    import pandas as pd

    train_ds = BikeDataset(data_dir, "train")
    val_ds   = BikeDataset(data_dir, "val")

    X_train, y_train = train_ds.numpy_xy()
    X_val,   y_val   = val_ds.numpy_xy()

    station_idx_val = val_ds.station_idx.numpy()

    val_feat       = pd.read_parquet(data_dir / "features_val.parquet")
    timestamps_val = pd.to_datetime(val_feat["timestamp"].values)

    # Build and fit pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha, fit_intercept=True)),
    ])

    logger.info(
        "Fitting Ridge on %d samples, %d features...",
        len(X_train), X_train.shape[1],
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    # Clip negatives — demand cannot be negative
    y_pred = np.clip(y_pred, 0, None)

    result = full_evaluation(
        y_true      = y_val,
        y_pred      = y_pred,
        station_idx = station_idx_val,
        timestamps  = timestamps_val,
        model_name  = "linear",
        split       = "val",
    )

    if log_to_mlflow:
        _log_baseline_to_mlflow(
            model_name  = "linear",
            params      = {
                "model_type":    "ridge",
                "alpha":         alpha,
                "n_features":    X_train.shape[1],
                "feature_cols":  ",".join(FEATURE_COLS),
                "train_samples": len(X_train),
                "val_samples":   len(X_val),
            },
            result      = result,
            sklearn_model = pipeline,
        )

    return result


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def _log_baseline_to_mlflow(
    model_name:    str,
    params:        dict,
    result:        EvalResult,
    sklearn_model  = None,
) -> None:
    """Log a baseline run to MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("phase", "baseline")

        mlflow.log_params(params)
        log_metrics_to_mlflow(result)

        # Save per-station and per-hour breakdowns as CSV artifacts
        result.by_station.to_csv(f"/tmp/{model_name}_by_station.csv", index=False)
        result.by_hour.to_csv(f"/tmp/{model_name}_by_hour.csv",    index=False)
        mlflow.log_artifact(f"/tmp/{model_name}_by_station.csv", artifact_path="eval")
        mlflow.log_artifact(f"/tmp/{model_name}_by_hour.csv",    artifact_path="eval")

        # Log sklearn model artifact
        if sklearn_model is not None:
            mlflow.sklearn.log_model(sklearn_model, artifact_path="model")

        logger.info(
            "MLflow run logged | experiment=%s | run=%s",
            EXPERIMENT_NAME, model_name,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Path to data/processed directory",
    )
    parser.add_argument(
        "--model", choices=["naive", "linear", "all"], default="all",
        help="Which baseline to run (default: all)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Ridge regularisation strength (default: 1.0)",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging (useful for quick tests)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    log = not args.no_mlflow

    if args.model in ("naive", "all"):
        result = run_naive(args.data_dir, log_to_mlflow=log)
        print(f"\nNaive  | MAE={result.metrics.mae:.4f}  RMSE={result.metrics.rmse:.4f}  MAPE={result.metrics.mape:.2f}%")

    if args.model in ("linear", "all"):
        result = run_linear(args.data_dir, alpha=args.alpha, log_to_mlflow=log)
        print(f"Linear | MAE={result.metrics.mae:.4f}  RMSE={result.metrics.rmse:.4f}  MAPE={result.metrics.mape:.2f}%")


if __name__ == "__main__":
    main()