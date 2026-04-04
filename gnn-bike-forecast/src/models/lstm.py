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

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike-demand-forecasting"


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader.

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

    return np.concatenate(all_true), np.concatenate(all_pred)


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
    log_to_mlflow : whether to log to MLflow
    """
    device = get_device()
    logger.info("Device: %s", device)

    # --- Data ---
    logger.info("Building LSTM DataLoaders...")
    train_loader, val_loader, _ = build_dataloaders(
        data_dir   = data_dir,
        batch_size = batch_size,
        num_workers = 0,      # 0 avoids MPS/multiprocessing conflicts
        model_type = "lstm",
        seq_len    = seq_len,
    )

    # Retrieve val metadata for full_evaluation (station_idx, timestamps)
    import pandas as pd
    val_feat       = pd.read_parquet(data_dir / "features_val.parquet")
    # LSTM val dataset skips first seq_len-1 rows per station — align metadata
    val_ds         = LSTMDataset(data_dir, "val", seq_len=seq_len)
    # Build station_idx and timestamps aligned to LSTM samples
    # Each sample (t, n) maps to timestamp at t + seq_len - 1
    val_meta = _build_lstm_metadata(val_feat, val_ds, seq_len)

    # --- Model ---
    input_size = len(__import__("data_loader").LSTM_FEATURE_COLS)
    model      = BikeDemanLSTM(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        dropout     = dropout,
        use_softplus = loss_fn == "poisson",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LSTM params: %d", n_params)

    criterion = nn.PoissonNLLLoss(log_input=False) if loss_fn == "poisson" else nn.MSELoss()
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
        })

    # --- Training loop ---
    logger.info("Starting training | max_epochs=%d | patience=%d", max_epochs, patience)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimiser, device)
        y_true, y_pred = evaluate_epoch(model, val_loader, device)

        # Clip negatives before eval
        y_pred = np.clip(y_pred, 0, None)

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

    # --- Load best checkpoint and run full evaluation ---
    logger.info("Loading best checkpoint (epoch %d)...", best_epoch)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    y_true, y_pred = evaluate_epoch(model, val_loader, device)
    y_pred         = np.clip(y_pred, 0, None)

    result = full_evaluation(
        y_true      = y_true,
        y_pred      = y_pred,
        station_idx = val_meta["station_idx"],
        timestamps  = val_meta["timestamps"],
        model_name  = "lstm",
        split       = "val",
    )

    if log_to_mlflow:
        log_metrics_to_mlflow(result, prefix="best_")
        mlflow.log_param("best_epoch", best_epoch)

        # Save artifacts
        result.by_station.to_csv("/tmp/lstm_by_station.csv", index=False)
        result.by_hour.to_csv("/tmp/lstm_by_hour.csv",       index=False)
        mlflow.log_artifact("/tmp/lstm_by_station.csv", artifact_path="eval")
        mlflow.log_artifact("/tmp/lstm_by_hour.csv",    artifact_path="eval")
        mlflow.log_artifact(str(ckpt_path),             artifact_path="model")
        mlflow.end_run()

    print(
        f"\nLSTM | MAE={result.metrics.mae:.4f}  "
        f"RMSE={result.metrics.rmse:.4f}  "
        f"MAPE={result.metrics.mape:.2f}%  "
        f"(best epoch={best_epoch})"
    )


# ---------------------------------------------------------------------------
# Metadata alignment helper
# ---------------------------------------------------------------------------

def _build_lstm_metadata(
    val_feat: "pd.DataFrame",
    val_ds:   LSTMDataset,
    seq_len:  int,
) -> dict:
    """
    Build station_idx and timestamps arrays aligned to LSTMDataset samples.

    LSTMDataset sample (t, n) corresponds to:
      - timestamp at position t + seq_len - 1 in the sorted time axis
      - station at position n in the sorted station axis

    Returns dict with keys 'station_idx' and 'timestamps'.
    """
    import pandas as pd
    import numpy as np

    unique_times    = np.sort(val_feat["timestamp"].unique())
    unique_stations = np.sort(val_feat["station_idx"].unique())
    T = len(unique_times)
    N = len(unique_stations)
    n_valid = T - seq_len + 1

    station_idx_out = np.empty(n_valid * N, dtype=np.int64)
    timestamps_out  = np.empty(n_valid * N, dtype="datetime64[ns]")

    for t in range(n_valid):
        ts = unique_times[t + seq_len - 1]
        for n in range(N):
            idx = t * N + n
            station_idx_out[idx] = unique_stations[n]
            timestamps_out[idx]  = ts

    return {
        "station_idx": station_idx_out,
        "timestamps":  pd.to_datetime(timestamps_out),
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
        save_dir      = args.save_dir,
        log_to_mlflow = not args.no_mlflow,
    )


if __name__ == "__main__":
    main()