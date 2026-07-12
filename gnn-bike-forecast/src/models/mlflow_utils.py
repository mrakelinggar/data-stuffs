"""
mlflow_utils.py
----------------
MLflow run provenance tagging: git commit, data snapshot, seed, and library
versions, so every run is traceable to the exact code and data that produced
it. Also owns the consolidated eval-artifact-logging helper (per-segment CSVs
and, for val/test splits, a row-level predictions parquet for paired
significance testing).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from evaluate import EvalResult
from utils import get_validated_snapshot_id

_REPO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _REPO_DIR.parent.parent

# Fallback for environments that received a code+data bundle instead of a
# .git checkout (e.g. a remote GPU notebook) -- see write_git_provenance.py.
_PROVENANCE_FALLBACK_FILE = _PROJECT_ROOT / "GIT_PROVENANCE.json"


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_REPO_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _read_provenance_fallback() -> dict | None:
    if not _PROVENANCE_FALLBACK_FILE.exists():
        return None
    with open(_PROVENANCE_FALLBACK_FILE) as f:
        return json.load(f)


def get_git_sha() -> str:
    """HEAD commit sha. Note: this repo is nested inside a larger monorepo,
    so the sha reflects the monorepo's HEAD, not a dedicated-repo commit —
    still valid for traceability, just not scoped to this subdirectory.

    Falls back to GIT_PROVENANCE.json (sha/dirty captured from the real repo
    before packaging) when no .git checkout is present -- raises the
    original git error if that fallback is also absent, rather than
    silently tagging a run with fabricated provenance."""
    try:
        return _run_git("rev-parse", "HEAD")
    except RuntimeError:
        fallback = _read_provenance_fallback()
        if fallback is None:
            raise
        return fallback["sha"]


def _is_git_dirty() -> bool:
    try:
        return bool(_run_git("status", "--porcelain"))
    except RuntimeError:
        fallback = _read_provenance_fallback()
        if fallback is None:
            raise
        return fallback["dirty"]


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}

    import numpy
    import pandas
    import sklearn
    import torch

    versions["torch"] = torch.__version__
    versions["numpy"] = numpy.__version__
    versions["pandas"] = pandas.__version__
    versions["scikit-learn"] = sklearn.__version__
    versions["mlflow"] = mlflow.__version__

    try:
        import torch_geometric
        versions["torch_geometric"] = torch_geometric.__version__
    except ImportError:
        pass

    return versions


def tag_run_provenance(data_dir: Path | str, seed: int) -> None:
    """Tag the active MLflow run with git sha, data snapshot id, seed, and
    library versions. Must be called after `mlflow.start_run(...)`."""
    data_snapshot_id = get_validated_snapshot_id(data_dir)
    git_sha = get_git_sha()
    git_dirty = _is_git_dirty()

    mlflow.set_tag("git_sha", git_sha)
    mlflow.set_tag("git_dirty", git_dirty)
    mlflow.set_tag("data_snapshot_id", data_snapshot_id)
    mlflow.set_tag("seed", seed)

    for package, version in _library_versions().items():
        mlflow.set_tag(f"lib_{package}", version)


def log_predictions_to_mlflow(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    station_idx: np.ndarray,
    timestamps: np.ndarray,
    model_name: str,
    split: str,
) -> None:
    """
    Write row-level (station_idx, timestamp, y_true, y_pred) predictions to a
    parquet file and log it as an MLflow artifact under eval/predictions.

    This is new scope beyond the per-station/per-hour aggregate CSVs logged
    by log_segment_artifacts_to_mlflow below -- aggregates alone can't support
    a true (station, timestamp)-paired significance test across two
    separately trained runs, since aggregation already collapses the join key.

    Filename is `predictions_{split}.parquet` -- no model_name in the name,
    since it lives under this run's own artifact store (one file per split
    per run; model_name is redundant with the run itself).

    Must be called after `mlflow.start_run(...)`.
    """
    df = pd.DataFrame({
        "station_idx": np.asarray(station_idx, dtype=np.int64).ravel(),
        "timestamp":   pd.to_datetime(np.asarray(timestamps).ravel()),
        "y_true":      np.asarray(y_true, dtype=np.float32).ravel(),
        "y_pred":      np.asarray(y_pred, dtype=np.float32).ravel(),
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"predictions_{split}.parquet"
        df.to_parquet(path, index=False)
        mlflow.log_artifact(str(path), artifact_path="eval/predictions")


def log_segment_artifacts_to_mlflow(result: EvalResult, model_name: str) -> None:
    """
    Log an EvalResult's per-segment breakdowns to the active MLflow run as
    CSV artifacts under artifact_path="eval", plus (for val/test splits) a
    row-level predictions parquet under eval/predictions.

    Consolidates the by_station/by_hour(/by_cluster/by_volume_tier/by_hour_band)
    CSV-writing block that was previously duplicated across baseline.py,
    lstm.py, and gcn.py -- this is the single shared call site for both the
    aggregate CSVs and the Phase 3 row-level predictions artifact; add new
    artifact types here, not back in the model scripts.

    Must be called after `mlflow.start_run(...)`.
    """
    frames = {
        "by_station": result.by_station,
        "by_hour": result.by_hour,
        "by_hour_band": result.by_hour_band,
        "by_cluster": result.by_cluster,
        "by_volume_tier": result.by_volume_tier,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for kind, df in frames.items():
            if df is None:
                continue
            csv_path = tmp_path / f"{model_name}_{result.split}_{kind}.csv"
            df.to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path), artifact_path="eval")

    # Row-level predictions for paired significance testing -- val/test only
    # (train predictions are large and out of scope for significance testing,
    # a val/test-only concept per CLAUDE.md's Evaluation contract).
    if result.split in ("val", "test") and result.y_true is not None:
        log_predictions_to_mlflow(
            y_true=result.y_true,
            y_pred=result.y_pred,
            station_idx=result.station_idx,
            timestamps=result.timestamps,
            model_name=model_name,
            split=result.split,
        )
