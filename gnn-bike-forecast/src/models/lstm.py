"""
lstm.py
-------
2-layer LSTM for t+24 bike demand forecasting.

Architecture
------------
  Input  : (batch, seq_len=168, input_size=7)  -- 1-week sequence, 7 features
  LSTM   : 2 layers, hidden_size=64, dropout between layers
  FC head: hidden_size -> 1
  Output : (batch,)  -- predicted demand at t+24

Training
--------
  - Loss      : MSELoss (smooth gradients; MAE used for early stopping / eval)
  - Optimiser : Adam
  - Scheduler : ReduceLROnPlateau on val MAE (patience=3)
  - Early stopping on val MAE (patience=5)
  - Device    : MPS > CUDA > CPU (auto-detected)

MLflow
------
  Logs to experiment 'bike-demand-forecasting', run name 'lstm'.
  Per-epoch val MAE logged as a metric step for learning curve visibility.

Usage
-----
  python src/models/lstm.py --data-dir data/processed
  python src/models/lstm.py --data-dir data/processed --no-mlflow --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_loader import LSTM_SEQ_LEN, LSTMDataset, build_dataloaders
from evaluate import full_evaluation, log_metrics_to_mlflow
from mlflow_utils import log_segment_artifacts_to_mlflow, tag_run_provenance
from utils import get_device, set_seed

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike-demand-forecasting"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BikeDemanLSTM(nn.Module):
    """
    2-layer LSTM + FC head for station-level demand forecasting.

    Parameters
    ----------
    input_size  : number of features per timestep (7)
    hidden_size : LSTM hidden dimension
    num_layers  : number of stacked LSTM layers
    dropout     : dropout probability between LSTM layers
    """

    def __init__(
        self,
        input_size:  int   = 7,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        use_softplus: bool  = False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() if use_softplus else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        (batch,) predicted demand
        """
        # out: (batch, seq_len, hidden_size)
        # we only use the last timestep's hidden state
        out, _ = self.lstm(x)
        last    = out[:, -1, :]          # (batch, hidden_size)
        pred    = self.head(last)        # (batch, 1)
        return pred.squeeze(-1)          # (batch,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model:     BikeDemanLSTM,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """Run one training epoch. Returns mean MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for x_seq, y in pbar:
        x_seq = x_seq.to(device, non_blocking=True)
        y     = y.to(device,     non_blocking=True)

        optimiser.zero_grad()
        pred = model(x_seq)
        loss = criterion(pred, y)
        loss.backward()

        # Gradient clipping -- important for LSTM stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def evaluate_epoch(
    model:  BikeDemanLSTM,
    loader: DataLoader,
    device: torch.device,
    loss_fn: str = "poisson",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader.

    Parameters
    ----------
    loss_fn : "poisson" or "mse" -- determines how the model's raw output is
              turned into a predicted rate. Poisson emits a raw log-rate
              (recovered via exp()); MSE emits a non-negative rate directly
              from its Softplus head (no transform needed).

    Returns
    -------
    y_true, y_pred as numpy arrays
    """
    model.eval()
    all_true = []
    all_pred = []

    for x_seq, y in tqdm(loader, desc="  val  ", leave=False, unit="batch"):
        x_seq = x_seq.to(device, non_blocking=True)
        pred  = model(x_seq)
        all_true.append(y.numpy())
        all_pred.append(pred.cpu().numpy())

    y_true    = np.concatenate(all_true)
    y_pred_raw = np.concatenate(all_pred)

    if loss_fn == "poisson":
        y_pred = np.exp(y_pred_raw)   # recover rate from raw log-rate
    else:
        y_pred = y_pred_raw          # Softplus head already guarantees >= 0

    return y_true, y_pred


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_lstm(
    data_dir:    Path,
    loss_fn:      str   = "poisson",
    hidden_size: int   = 64,
    num_layers:  int   = 2,
    dropout:     float = 0.2,
    lr:          float = 1e-3,
    batch_size:  int   = 512,
    max_epochs:  int   = 50,
    patience:    int   = 5,
    seq_len:     int   = LSTM_SEQ_LEN,
    save_dir:    Path  = Path("models"),
    seed:        int   = 42,
    log_to_mlflow: bool = True,
) -> None:
    """
    Train the LSTM model and log results to MLflow.

    Parameters
    ----------
    data_dir    : path to data/processed/
    hidden_size : LSTM hidden dimension
    num_layers  : number of LSTM layers
    dropout     : dropout between LSTM layers
    lr          : Adam learning rate
    batch_size  : samples per batch
    max_epochs  : maximum training epochs
    patience    : early stopping patience (epochs without val MAE improvement)
    seq_len     : lookback window length
    save_dir    : directory to save best model checkpoint
    seed        : random seed for reproducibility
    log_to_mlflow : whether to log to MLflow
    """
    set_seed(seed)

    device = get_device()
    logger.info("Device: %s", device)

    # --- Data ---
    logger.info("Building LSTM DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir   = data_dir,
        batch_size = batch_size,
        num_workers = 0,      # 0 avoids MPS/multiprocessing conflicts
        model_type = "lstm",
        seq_len    = seq_len,
    )

    # Retrieve train/val/test metadata for full_evaluation (station_idx, timestamps,
    # lag_24, cluster). train_lag24 feeds the volume-tier tercile cutoffs -- computed
    # once here and reused unchanged for train/val/test scoring.
    import pandas as pd
    train_feat     = pd.read_parquet(data_dir / "features_train.parquet")
    val_feat       = pd.read_parquet(data_dir / "features_val.parquet")
    test_feat      = pd.read_parquet(data_dir / "features_test.parquet")
    train_lag24    = train_feat["lag_24"].values
    # LSTM val/test datasets borrow the preceding split's tail as read-only
    # input history (ROADMAP Phase 4), so every own row is scored -- metadata
    # alignment below reads ds.anchor_timestamps rather than re-deriving it.
    val_ds         = LSTMDataset(data_dir, "val", seq_len=seq_len)
    test_ds        = LSTMDataset(data_dir, "test", seq_len=seq_len)
    # Build station_idx and timestamps aligned to LSTM samples
    val_meta  = _build_lstm_metadata(val_feat, val_ds)
    test_meta = _build_lstm_metadata(test_feat, test_ds)

    # --- Model ---
    input_size = len(__import__("data_loader").LSTM_FEATURE_COLS)
    # Poisson emits a raw log-rate (Identity head) for log_input=True; MSE
    # emits a non-negative rate directly (Softplus head) -- non-negativity
    # belongs in the model, not in a post-hoc np.clip.
    model      = BikeDemanLSTM(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
        use_softplus = loss_fn != "poisson",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LSTM params: %d", n_params)

    criterion = nn.PoissonNLLLoss(log_input=True) if loss_fn == "poisson" else nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    # --- Early stopping state ---
    best_val_mae   = float("inf")
    best_epoch     = 0
    patience_count = 0
    save_dir       = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = save_dir / "lstm_best.pt"

    # --- MLflow setup ---
    if log_to_mlflow:
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.start_run(run_name=f"lstm_{loss_fn}")
        mlflow.set_tag("model_name", "lstm")
        mlflow.set_tag("phase", "lstm")
        tag_run_provenance(data_dir, seed)
        mlflow.log_params({
            "model_type":  "lstm",
            "loss_fn": loss_fn,
            "hidden_size": hidden_size,
            "num_layers":  num_layers,
            "dropout":     dropout,
            "lr":          lr,
            "batch_size":  batch_size,
            "seq_len":     seq_len,
            "max_epochs":  max_epochs,
            "patience":    patience,
            "n_params":    n_params,
            "device":      str(device),
            "seed":        seed,
        })

    # --- Training loop ---
    logger.info("Starting training | max_epochs=%d | patience=%d", max_epochs, patience)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
        y_true, y_pred = evaluate_epoch(model, val_loader, device, loss_fn=loss_fn)

        from evaluate import compute_metrics
        val_metrics = compute_metrics(y_true, y_pred)
        val_mae     = val_metrics.mae

        scheduler.step(val_mae)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_MAE=%.4f | val_RMSE=%.4f | %.1fs",
            epoch, max_epochs, train_loss, val_mae, val_metrics.rmse, elapsed,
        )

        if log_to_mlflow:
            mlflow.log_metrics(
                {
                    "train_mse_loss": train_loss,
                    "val_mae":        val_mae,
                    "val_rmse":       val_metrics.rmse,
                    "val_mape":       val_metrics.mape,
                },
                step=epoch,
            )

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae   = val_mae
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
            logger.info("  -> New best val MAE=%.4f saved to %s", best_val_mae, ckpt_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    "Early stopping at epoch %d (best epoch=%d, best val MAE=%.4f)",
                    epoch, best_epoch, best_val_mae,
                )
                break

    # --- Load best checkpoint and run full evaluation (val + test) ---
    logger.info("Loading best checkpoint (epoch %d)...", best_epoch)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_pred = evaluate_epoch(model, val_loader, device, loss_fn=loss_fn)
    result = full_evaluation(
        y_true      = y_true,
        y_pred      = y_pred,
        station_idx = val_meta["station_idx"],
        timestamps  = val_meta["timestamps"],
        model_name  = "lstm",
        split       = "val",
        cluster     = val_meta["cluster"],
        train_lag24 = train_lag24,
        lag24       = val_meta["lag24"],
    )

    y_true_test, y_pred_test = evaluate_epoch(model, test_loader, device, loss_fn=loss_fn)
    test_result = full_evaluation(
        y_true      = y_true_test,
        y_pred      = y_pred_test,
        station_idx = test_meta["station_idx"],
        timestamps  = test_meta["timestamps"],
        model_name  = "lstm",
        split       = "test",
        cluster     = test_meta["cluster"],
        train_lag24 = train_lag24,
        lag24       = test_meta["lag24"],
    )

    train_ds_eval        = LSTMDataset(data_dir, "train", seq_len=seq_len)
    train_eval_loader    = DataLoader(train_ds_eval, batch_size=batch_size, shuffle=False)
    train_meta           = _build_lstm_metadata(train_feat, train_ds_eval)
    y_true_train, y_pred_train = evaluate_epoch(model, train_eval_loader, device, loss_fn=loss_fn)
    train_result = full_evaluation(
        y_true      = y_true_train,
        y_pred      = y_pred_train,
        station_idx = train_meta["station_idx"],
        timestamps  = train_meta["timestamps"],
        model_name  = "lstm",
        split       = "train",
        cluster     = train_meta["cluster"],
        train_lag24 = train_lag24,
        lag24       = train_meta["lag24"],
    )

    if log_to_mlflow:
        mlflow.log_param("best_epoch", best_epoch)

        for r in (train_result, result, test_result):
            log_metrics_to_mlflow(r, prefix="best_")
            log_segment_artifacts_to_mlflow(r, "lstm")

        mlflow.log_artifact(str(ckpt_path), artifact_path="model")
        mlflow.end_run()

    print(
        f"\nLSTM | train MAE={train_result.metrics.mae:.4f}  "
        f"RMSE={train_result.metrics.rmse:.4f}  "
        f"MAPE={train_result.metrics.mape:.2f}%"
    )
    print(
        f"LSTM | val  MAE={result.metrics.mae:.4f}  "
        f"RMSE={result.metrics.rmse:.4f}  "
        f"MAPE={result.metrics.mape:.2f}%  "
        f"(best epoch={best_epoch})"
    )
    print(
        f"LSTM | test MAE={test_result.metrics.mae:.4f}  "
        f"RMSE={test_result.metrics.rmse:.4f}  "
        f"MAPE={test_result.metrics.mape:.2f}%"
    )


# ---------------------------------------------------------------------------
# Metadata alignment helper
# ---------------------------------------------------------------------------

def _build_lstm_metadata(
    feat_df: "pd.DataFrame",
    ds:      LSTMDataset,
) -> dict:
    """
    Build station_idx, timestamps, lag_24, and cluster arrays aligned to
    LSTMDataset samples.

    Reads `ds.anchor_timestamps` (set by LSTMDataset, ROADMAP Phase 4) as
    the source of truth for each anchor's scored target timestamp, rather
    than re-deriving `T - seq_len + 1` independently from `feat_df` -- that
    independent derivation went stale once val/test datasets started
    borrowing a prior split's tail as input-only history, since
    len(LSTMDataset) no longer equals `(T - seq_len + 1) * N` for those
    splits.

    Returns dict with keys 'station_idx', 'timestamps', 'lag24', 'cluster'.
    """
    import pandas as pd
    import numpy as np

    from evaluate import reconstruct_cluster_id

    unique_stations = np.sort(feat_df["station_idx"].unique())
    N = len(unique_stations)
    anchor_timestamps = ds.anchor_timestamps
    n_valid = len(anchor_timestamps)

    if n_valid * N != len(ds):
        raise AssertionError(
            f"metadata anchor count {n_valid}*{N}={n_valid * N} != len(dataset)={len(ds)}"
        )

    timestamps_out  = np.repeat(anchor_timestamps, N)
    station_idx_out = np.tile(unique_stations, n_valid)

    # lag_24 / cluster are genuine per-(timestamp, station) values -- look
    # them up vectorized via a MultiIndex reindex, matching the time-major/
    # station-minor order timestamps_out/station_idx_out already produce.
    idx = pd.MultiIndex.from_product(
        [anchor_timestamps, unique_stations], names=["timestamp", "station_idx"]
    )
    lookup = (
        feat_df.set_index(["timestamp", "station_idx"])
        [["lag_24", "cluster_0", "cluster_1", "cluster_2"]]
        .reindex(idx)
    )
    lag24_out   = lookup["lag_24"].values
    cluster_out = reconstruct_cluster_id(lookup[["cluster_0", "cluster_1", "cluster_2"]].values)

    if len(lag24_out) != n_valid * N:
        raise AssertionError("lag24/cluster lookup length mismatch")

    return {
        "station_idx": station_idx_out,
        "timestamps":  pd.to_datetime(timestamps_out),
        "lag24":       lag24_out,
        "cluster":     cluster_out,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--data-dir",    type=Path,  default=Path("data/processed"))
    parser.add_argument("--hidden-size", type=int,   default=64)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=512)
    parser.add_argument("--max-epochs",  type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--loss-fn", choices=["mse", "poisson"], default="poisson")
    parser.add_argument("--save-dir",    type=Path,  default=Path("models"))
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no-mlflow",   action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    run_lstm(
        data_dir      = args.data_dir,
        loss_fn       = args.loss_fn,
        hidden_size   = args.hidden_size,
        num_layers    = args.num_layers,
        dropout       = args.dropout,
        lr            = args.lr,
        batch_size    = args.batch_size,
        max_epochs    = args.max_epochs,
        patience      = args.patience,
        seed          = args.seed,
        save_dir      = args.save_dir,
        log_to_mlflow = not args.no_mlflow,
    )


if __name__ == "__main__":
    main()