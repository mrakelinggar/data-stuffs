"""
mlflow_utils.py
----------------
MLflow run provenance tagging: git commit, data snapshot, seed, and library
versions, so every run is traceable to the exact code and data that produced
it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow

from utils import get_validated_snapshot_id

_REPO_DIR = Path(__file__).resolve().parent


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


def get_git_sha() -> str:
    """HEAD commit sha. Note: this repo is nested inside a larger monorepo,
    so the sha reflects the monorepo's HEAD, not a dedicated-repo commit —
    still valid for traceability, just not scoped to this subdirectory."""
    return _run_git("rev-parse", "HEAD")


def _is_git_dirty() -> bool:
    return bool(_run_git("status", "--porcelain"))


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
