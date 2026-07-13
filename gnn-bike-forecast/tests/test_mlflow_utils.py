"""
Tests for src/models/mlflow_utils.py::ensure_portable_artifact_location.

Reproduces the bug found in ROADMAP Phase 6: shipping mlflow.db wholesale to
a remote machine (e.g. a Colab notebook) carries over the *creating*
machine's absolute artifact_location, silently redirecting subsequent
mlflow.log_artifact() calls to a path that only exists on the original
machine. This is exercised against a real temporary sqlite-backed MLflow
store, not mocks -- the bug lives in how MLflow's SqlAlchemyStore persists
and reuses artifact_location, so a mock would hide it.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from mlflow_utils import ensure_portable_artifact_location


def _make_experiment(tmp_path: Path, foreign_root: Path) -> tuple[str, str]:
    """Create an experiment whose artifact_location points at a directory
    that does not belong to `tmp_path` -- simulating a db file copied in
    from a different machine."""
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.create_experiment(
        "bike-demand-forecasting",
        artifact_location=foreign_root.as_uri(),
    )
    return str(db_path), experiment_id


def test_patches_foreign_artifact_location(tmp_path):
    foreign_root = tmp_path / "foreign_machine" / "mlruns" / "1"
    db_path, experiment_id = _make_experiment(tmp_path, foreign_root)

    ensure_portable_artifact_location("bike-demand-forecasting")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment(experiment_id)
    expected = (Path(db_path).parent / "mlruns" / experiment_id).resolve().as_uri()
    assert experiment.artifact_location == expected


def test_leaves_already_correct_location_untouched(tmp_path):
    # Explicitly set artifact_location to what the function itself would
    # compute as "correct" for this db path -- mlflow's own default (when
    # artifact_location is omitted) is derived from process cwd, not the db
    # file's location, so it isn't a reliable stand-in for "already correct"
    # inside a test where cwd and the tmp db live in different places.
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = mlflow.tracking.MlflowClient()
    correct_root = tmp_path / "mlruns" / "1"
    experiment_id = client.create_experiment(
        "bike-demand-forecasting",
        artifact_location=correct_root.as_uri(),
    )
    before = client.get_experiment(experiment_id).artifact_location

    ensure_portable_artifact_location("bike-demand-forecasting")

    after = client.get_experiment(experiment_id).artifact_location
    assert after == before


def test_no_op_when_experiment_does_not_exist(tmp_path):
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    # Must not raise even though the experiment was never created.
    ensure_portable_artifact_location("does-not-exist-yet")


def test_no_op_for_non_sqlite_tracking_uri(tmp_path):
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))

    # File-store backend isn't known to carry this bug -- must be a no-op,
    # not an error, for local ./mlruns-only setups.
    ensure_portable_artifact_location("bike-demand-forecasting")


def test_second_call_is_a_clean_no_op(tmp_path):
    foreign_root = tmp_path / "foreign_machine" / "mlruns" / "1"
    db_path, experiment_id = _make_experiment(tmp_path, foreign_root)

    ensure_portable_artifact_location("bike-demand-forecasting")
    client = mlflow.tracking.MlflowClient()
    after_first = client.get_experiment(experiment_id).artifact_location

    ensure_portable_artifact_location("bike-demand-forecasting")
    after_second = client.get_experiment(experiment_id).artifact_location

    assert after_first == after_second


def test_sqlite_connection_is_closed_after_patching(tmp_path):
    # A leaked connection would leave the db file locked -- a fresh
    # exclusive-mode connection immediately after the call should succeed
    # without a "database is locked" error.
    import sqlite3

    foreign_root = tmp_path / "foreign_machine" / "mlruns" / "1"
    db_path, _ = _make_experiment(tmp_path, foreign_root)

    ensure_portable_artifact_location("bike-demand-forecasting")

    conn = sqlite3.connect(db_path, timeout=0)
    try:
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("SELECT 1")
    finally:
        conn.close()
