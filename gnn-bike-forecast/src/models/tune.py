"""
tune.py
-------
Optuna hyperparameter sweeps for lstm/gcn. Each trial is logged as a nested
MLflow run under one parent "sweep" run. Pruning (MedianPruner) is driven by
val_mae reported per-epoch via the `epoch_callback` hook already present in
run_lstm()/run_gcn() -- all Optuna-specific logic (trial.report,
should_prune, TrialPruned) lives here; lstm.py/gcn.py have no dependency on
Optuna.

The Optuna study persists to a local sqlite file (`optuna.db`, alongside the
existing `mlflow.db`) so a multi-hour sweep on Colab survives a session
disconnect -- re-running the same --model/--study-name resumes rather than
restarting from trial 0.

Usage
-----
  python src/models/tune.py --model lstm --n-trials 30 --timeout 6h
  python src/models/tune.py --model gcn  --n-trials 30 --timeout 6h
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Callable

import mlflow
import optuna
from optuna.pruners import MedianPruner

from mlflow_utils import ensure_portable_artifact_location

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike-demand-forecasting"
STUDY_STORAGE = "sqlite:///optuna.db"

# ---------------------------------------------------------------------------
# Duration parsing ("6h" / "90m" / "3600s" / plain seconds)
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)([smh]?)$")


def parse_duration(s: str) -> int:
    """Parse '6h' / '90m' / '3600s' / '3600' into seconds."""
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise ValueError(f"Invalid duration: {s!r} (expected e.g. '6h', '90m', '3600')")
    value, unit = m.groups()
    mult = {"": 1, "s": 1, "m": 60, "h": 3600}[unit]
    return int(value) * mult


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

def lstm_search_space(trial: optuna.Trial) -> dict:
    return dict(
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        num_layers  = trial.suggest_int("num_layers", 1, 3),
        dropout     = trial.suggest_float("dropout", 0.0, 0.5),
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        seq_len     = trial.suggest_categorical("seq_len", [72, 168, 336]),
        batch_size  = trial.suggest_categorical("batch_size", [256, 512, 1024]),
    )


def gcn_search_space(trial: optuna.Trial) -> dict:
    # norm_scheme deliberately excluded: data_loader.load_adj has no `norm`
    # param yet -- that's ROADMAP Phase 10 scope, not Phase 7.
    return dict(
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
        num_layers  = trial.suggest_categorical("num_layers", [2, 3]),
        dropout     = trial.suggest_float("dropout", 0.0, 0.5),
        lr          = trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        adj_variant = trial.suggest_categorical("adj_variant", ["knn", "flow", "combined"]),
    )


def hybrid_search_space(trial: optuna.Trial) -> dict:
    # gcn_hidden is separate from lstm_hidden because they play different
    # roles (spatial vs temporal encoder width) -- one might benefit while
    # the other doesn't. num_gcn_layers ∈ {1, 2}: the hybrid does temporal
    # work too, so it doesn't need to be as graph-deep as a standalone GCN.
    return dict(
        gcn_hidden      = trial.suggest_categorical("gcn_hidden", [16, 32, 64]),
        num_gcn_layers  = trial.suggest_categorical("num_gcn_layers", [1, 2]),
        lstm_hidden     = trial.suggest_categorical("lstm_hidden", [32, 64, 128, 256]),
        lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3),
        dropout         = trial.suggest_float("dropout", 0.0, 0.5),
        lr              = trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        adj_variant     = trial.suggest_categorical("adj_variant", ["knn", "flow", "combined"]),
    )


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "lstm":   lstm_search_space,
    "gcn":    gcn_search_space,
    "hybrid": hybrid_search_space,
}

# Populated lazily on first use so importing this module for search-space/
# duration-parsing tests doesn't force-import torch/torch_geometric.
TRAIN_FNS: dict[str, Callable] = {}


def _train_fn(model_name: str) -> Callable:
    # Checked per-key, not per-dict-emptiness -- a caller (e.g. a test) may
    # legitimately pre-populate only one of "lstm"/"gcn"/"hybrid" via
    # monkeypatching, and that must not suppress the real import of the others.
    if model_name not in TRAIN_FNS:
        from lstm import run_lstm
        from gcn import run_gcn
        from hybrid import run_hybrid
        TRAIN_FNS.setdefault("lstm",   run_lstm)
        TRAIN_FNS.setdefault("gcn",    run_gcn)
        TRAIN_FNS.setdefault("hybrid", run_hybrid)
    return TRAIN_FNS[model_name]


# ---------------------------------------------------------------------------
# Shared sweep-runner (one objective builder for both model types)
# ---------------------------------------------------------------------------

def make_objective(
    model_name: str,
    data_dir: Path,
    fixed_kwargs: dict,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective for `model_name` ('lstm' or 'gcn').

    `fixed_kwargs` carries CLI-level constants (max_epochs, patience,
    save_dir, seed, loss_fn) that are NOT part of the search space.
    """
    space_fn = SEARCH_SPACES[model_name]
    train_fn = _train_fn(model_name)

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        run_kwargs = dict(fixed_kwargs, data_dir=data_dir, log_to_mlflow=True, **params)

        def _cb(epoch: int, val_mae: float) -> None:
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with mlflow.start_run(run_name=f"{model_name}_trial{trial.number}", nested=True):
            mlflow.set_tag("optuna_trial_number", trial.number)
            try:
                best_val_mae = train_fn(epoch_callback=_cb, **run_kwargs)
            except optuna.TrialPruned:
                # mlflow.ActiveRun.__exit__ marks any run that exits via an
                # exception as FAILED -- tag it explicitly first so a pruned
                # trial is distinguishable from a genuine crash in the
                # MLflow UI, then let the exception continue propagating to
                # optuna.study.optimize(), which handles TrialPruned itself.
                mlflow.set_tag("optuna_state", "pruned")
                raise
        return best_val_mae

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep")
    parser.add_argument("--model", choices=["lstm", "gcn", "hybrid"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=str, default=None, help="e.g. '6h', '90m'")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--loss-fn", choices=["mse", "poisson"], default="poisson")
    parser.add_argument("--save-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    timeout_s = parse_duration(args.timeout) if args.timeout else None
    study_name = args.study_name or f"{args.model}_sweep"

    ensure_portable_artifact_location(EXPERIMENT_NAME)  # once per sweep, not per trial
    mlflow.set_experiment(EXPERIMENT_NAME)

    fixed_kwargs = dict(
        loss_fn=args.loss_fn,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=STUDY_STORAGE,
        load_if_exists=True,
        direction="minimize",
        pruner=MedianPruner(),
    )
    objective = make_objective(args.model, args.data_dir, fixed_kwargs)

    with mlflow.start_run(run_name=f"{study_name}_parent"):
        mlflow.set_tag("phase", "tune")
        mlflow.log_params({"model": args.model, "n_trials": args.n_trials, **fixed_kwargs})
        study.optimize(objective, n_trials=args.n_trials, timeout=timeout_s)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_mae", study.best_value)

    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
