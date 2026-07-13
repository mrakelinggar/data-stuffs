"""
Tests for src/models/tune.py.

No dependency on data/processed/ or real model training. Search-space shape
and duration-parsing are checked directly; the pruning/nested-run/sqlite-
persistence plumbing is checked with a trivial stand-in objective instead of
lstm.py/gcn.py's real training loops -- consistent with this repo's
synthetic-data-only test philosophy.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlflow
import optuna
import pytest

from tune import gcn_search_space, lstm_search_space, parse_duration


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "s,expected",
    [("6h", 21600), ("90m", 5400), ("30s", 30), ("120", 120)],
)
def test_parse_duration(s, expected):
    assert parse_duration(s) == expected


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("6x")


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def test_lstm_search_space_bounds():
    trial = optuna.create_study().ask()
    params = lstm_search_space(trial)
    assert params["hidden_size"] in (32, 64, 128, 256)
    assert params["num_layers"] in (1, 2, 3)
    assert 0.0 <= params["dropout"] <= 0.5
    assert 1e-4 <= params["lr"] <= 5e-3
    assert params["seq_len"] in (72, 168, 336)
    assert params["batch_size"] in (256, 512, 1024)


def test_gcn_search_space_excludes_norm_scheme():
    trial = optuna.create_study().ask()
    params = gcn_search_space(trial)
    # norm_scheme is Phase 10 scope -- data_loader.load_adj has no `norm`
    # param yet, so it must not appear in the search space at all.
    assert "norm_scheme" not in params
    assert params["hidden_size"] in (32, 64, 128, 256)
    assert params["num_layers"] in (2, 3)
    assert 0.0 <= params["dropout"] <= 0.5
    assert 1e-4 <= params["lr"] <= 5e-3
    assert params["adj_variant"] in ("knn", "flow", "combined")


# ---------------------------------------------------------------------------
# epoch_callback -> TrialPruned wiring
# ---------------------------------------------------------------------------

def test_callback_raises_pruned_when_trial_says_so():
    trial = MagicMock()
    trial.should_prune.return_value = True

    # Mirrors the closure make_objective builds internally.
    def _cb(epoch, val_mae):
        trial.report(val_mae, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    with pytest.raises(optuna.TrialPruned):
        _cb(1, 5.0)
    trial.report.assert_called_once_with(5.0, step=1)


# ---------------------------------------------------------------------------
# Lazy TRAIN_FNS population
# ---------------------------------------------------------------------------

def test_train_fn_imports_missing_model_even_if_other_already_populated(monkeypatch):
    # If TRAIN_FNS already has "lstm" (e.g. a prior test's monkeypatch), a
    # naive `if not TRAIN_FNS:` guard would skip importing "gcn" entirely
    # and raise a bare KeyError. _train_fn must import per-key instead.
    import tune

    monkeypatch.setitem(tune.TRAIN_FNS, "lstm", lambda **kwargs: 0.0)
    monkeypatch.delitem(tune.TRAIN_FNS, "gcn", raising=False)

    train_fn = tune._train_fn("gcn")
    assert train_fn.__name__ == "run_gcn"
    assert "gcn" in tune.TRAIN_FNS


# ---------------------------------------------------------------------------
# Full-plumbing smoke test: trivial stand-in objective, not real training
# ---------------------------------------------------------------------------

def test_sweep_persists_and_resumes(tmp_path, monkeypatch):
    storage = f"sqlite:///{tmp_path / 'optuna_test.db'}"

    def _fake_train_fn(epoch_callback=None, **kwargs):
        # Stands in for run_lstm/run_gcn: a few fake epochs, same
        # epoch_callback(epoch, val_mae) contract.
        for epoch, val_mae in enumerate([5.0, 4.0, 3.0], start=1):
            if epoch_callback is not None:
                epoch_callback(epoch, val_mae)
        return 3.0

    import tune
    monkeypatch.setitem(tune.TRAIN_FNS, "lstm", _fake_train_fn)

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_test.db'}")
    mlflow.set_experiment("test-sweep")

    objective = tune.make_objective("lstm", tmp_path, fixed_kwargs={})
    study = optuna.create_study(storage=storage, load_if_exists=True, study_name="s1")
    with mlflow.start_run(run_name="parent"):
        study.optimize(objective, n_trials=2)

    assert len(study.trials) == 2
    assert study.best_value == 3.0

    # Resume: re-open same storage/study_name, confirm state persists rather
    # than resetting -- this is the property a multi-hour Colab sweep needs
    # to survive a session disconnect.
    study2 = optuna.create_study(storage=storage, load_if_exists=True, study_name="s1")
    assert len(study2.trials) == 2
