"""
gcn.py
------
2-layer GCN for t+24 bike demand forecasting via PyTorch Geometric.

Architecture
------------
  Input  : (N, 15) node feature matrix per timestamp snapshot
  GCNConv layer 1 : 15 -> hidden_size, ReLU, Dropout
  GCNConv layer 2 : hidden_size -> hidden_size, ReLU
  FC head          : hidden_size -> 32 -> 1 per node
  Output : (N,) predicted demand per station

Three variants (selected via config):
  gcn_knn      -- adj_knn_norm      (KNN spatial graph)
  gcn_flow     -- adj_flow_norm     (flow-based graph)
  gcn_combined -- adj_combined_norm (element-wise average, normalised)

Data loading
------------
  Attempts to load all graph snapshots into memory (faster training).
  Falls back to lazy per-snapshot loading if available RAM < 1.5x required.

Training
--------
  - Loss      : Poisson negative log-likelihood (default) or MSELoss
  - Optimiser : Adam
  - Scheduler : ReduceLROnPlateau on val MAE (patience=3)
  - Early stopping on val MAE (patience=5)
  - One forward pass = one timestamp snapshot (all 1911 nodes simultaneously)
  - Device    : MPS > CUDA > CPU

MLflow
------
  Logs to experiment 'bike-demand-forecasting'.
  Run names: gcn_knn, gcn_flow, gcn_combined.

Usage
-----
  python src/models/gcn.py --data-dir data/processed --adj knn
  python src/models/gcn.py --data-dir data/processed --adj flow
  python src/models/gcn.py --data-dir data/processed --adj combined
  python src/models/gcn.py --data-dir data/processed --adj all
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from data_loader import (
    FEATURE_COLS,
    build_graph_snapshots,
    load_adj,
)
from evaluate import compute_metrics, full_evaluation, log_metrics_to_mlflow

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike-demand-forecasting"
ADJ_VARIANTS    = ("knn", "flow", "combined")


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
# Graph datasets: eager vs lazy
# ---------------------------------------------------------------------------

class EagerGraphDataset(Dataset):
    """
    All snapshots pre-loaded into memory as a list of PyG Data objects.
    Fast training — no disk I/O during epoch.
    """

    def __init__(self, snapshots: list[Data]) -> None:
        self.snapshots = snapshots

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx: int) -> Data:
        return self.snapshots[idx]


class LazyGraphDataset(Dataset):
    """
    Loads one snapshot at a time from the parquet files.
    Used when available RAM < 1.5x required to hold all snapshots.
    Slower (disk I/O per snapshot) but memory-safe.
    """

    def __init__(
        self,
        data_dir:   Path,
        split:      Literal["train", "val", "test"],
        adj_tensor: torch.Tensor,
    ) -> None:
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.adj_tensor = adj_tensor

        # Build index: load metadata only, not features
        import pandas as pd
        feat_df = pd.read_parquet(self.data_dir / f"features_{split}.parquet")
        self.unique_times    = np.sort(feat_df["timestamp"].unique())
        self.unique_stations = np.sort(feat_df["station_idx"].unique())
        self.T = len(self.unique_times)
        self.N = len(self.unique_stations)

        # Pre-convert adjacency to COO once
        edge_idx = adj_tensor.nonzero(as_tuple=False).t().contiguous()
        edge_wt  = adj_tensor[edge_idx[0], edge_idx[1]]
        self.edge_index  = edge_idx
        self.edge_weight = edge_wt

        # Load full feature/target arrays once (cheaper than re-reading parquet per snapshot)
        from data_loader import FEATURE_COLS, _load_split
        features, targets, station_idx, timestamps = _load_split(data_dir, split)

        F = len(FEATURE_COLS)
        time_to_t    = {t: i for i, t in enumerate(self.unique_times)}
        station_to_n = {s: i for i, s in enumerate(self.unique_stations)}

        self._feat_grid   = np.zeros((self.T, self.N, F), dtype=np.float32)
        self._target_grid = np.zeros((self.T, self.N),    dtype=np.float32)

        for row_i in range(len(features)):
            t_idx = time_to_t[timestamps[row_i]]
            n_idx = station_to_n[station_idx[row_i]]
            self._feat_grid[t_idx, n_idx, :]  = features[row_i]
            self._target_grid[t_idx, n_idx]   = targets[row_i]

        logger.info(
            "LazyGraphDataset [%s] | T=%d | N=%d | edges=%d",
            split, self.T, self.N, self.edge_index.shape[1],
        )

    def __len__(self) -> int:
        return self.T

    def __getitem__(self, idx: int) -> Data:
        return Data(
            x           = torch.from_numpy(self._feat_grid[idx]),    # (N, F)
            y           = torch.from_numpy(self._target_grid[idx]),  # (N,)
            edge_index  = self.edge_index,
            edge_weight = self.edge_weight,
            num_nodes   = self.N,
        )


def build_graph_dataset(
    data_dir:   Path,
    split:      Literal["train", "val", "test"],
    adj_tensor: torch.Tensor,
) -> EagerGraphDataset | LazyGraphDataset:
    """
    Build a graph dataset, choosing eager vs lazy based on available RAM.

    Threshold: available RAM must be >= 1.5x the memory required to hold
    all snapshots as float32 tensors.
    """
    import pandas as pd
    feat_df = pd.read_parquet(data_dir / f"features_{split}.parquet")
    T = feat_df["timestamp"].nunique()
    N = feat_df["station_idx"].nunique()
    F = len(FEATURE_COLS)

    required_bytes  = T * N * F * 4        # float32
    available_bytes = psutil.virtual_memory().available
    safety_margin   = 1.5

    logger.info(
        "Memory check [%s] | required=%.2fGB | available=%.2fGB",
        split,
        required_bytes  / 1e9,
        available_bytes / 1e9,
    )

    if available_bytes >= required_bytes * safety_margin:
        logger.info("Using EagerGraphDataset (all snapshots in memory)")
        snapshots = build_graph_snapshots(data_dir, split, adj_tensor)
        return EagerGraphDataset(snapshots)
    else:
        logger.warning(
            "Insufficient RAM for eager loading -- using LazyGraphDataset"
        )
        return LazyGraphDataset(data_dir, split, adj_tensor)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BikeDemanGCN(nn.Module):
    """
    2-layer GCN + FC head for station-level demand forecasting.

    Parameters
    ----------
    in_channels  : node feature dimension (15)
    hidden_size  : GCN hidden dimension
    dropout      : dropout after first GCN layer
    """

    def __init__(
        self,
        in_channels: int   = 15,
        hidden_size: int   = 64,
        dropout:     float = 0.2,
        use_softplus: bool  = False,
    ) -> None:
        super().__init__()

        self.conv1   = GCNConv(in_channels, hidden_size)
        self.conv2   = GCNConv(hidden_size,  hidden_size)
        self.dropout = dropout

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() if use_softplus else nn.Identity(),
        )

    def forward(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (N, in_channels)  node features
        edge_index  : (2, E)            COO edge list
        edge_weight : (E,)              normalised edge weights

        Returns
        -------
        (N,) predicted demand per node
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        out = self.head(x)          # (N, 1)
        return out.squeeze(-1)      # (N,)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model:     BikeDemanGCN,
    dataset:   EagerGraphDataset | LazyGraphDataset,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """One training epoch over all snapshots. Returns mean MSE loss."""
    model.train()
    total_loss = 0.0
    indices    = np.random.permutation(len(dataset))

    pbar = tqdm(indices, desc="  train", leave=False, unit="snap")
    for idx in pbar:
        data = dataset[int(idx)]

        x           = data.x.to(device,           non_blocking=True)
        y           = data.y.to(device,            non_blocking=True)
        edge_index  = data.edge_index.to(device,   non_blocking=True)
        edge_weight = data.edge_weight.to(device,  non_blocking=True)

        optimiser.zero_grad()
        pred = model(x, edge_index, edge_weight)
        loss = criterion(pred, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataset)


@torch.no_grad()
def evaluate_epoch(
    model:   BikeDemanGCN,
    dataset: EagerGraphDataset | LazyGraphDataset,
    device:  torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over all snapshots.

    Returns
    -------
    y_true, y_pred as numpy arrays, both flattened (T * N,)
    """
    model.eval()
    all_true = []
    all_pred = []

    for idx in tqdm(range(len(dataset)), desc="  val  ", leave=False, unit="snap"):
        data = dataset[int(idx)]

        x           = data.x.to(device,          non_blocking=True)
        edge_index  = data.edge_index.to(device,  non_blocking=True)
        edge_weight = data.edge_weight.to(device, non_blocking=True)

        pred = model(x, edge_index, edge_weight)
        all_true.append(data.y.numpy())
        all_pred.append(pred.cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_pred)


# ---------------------------------------------------------------------------
# Metadata helper (station_idx + timestamps aligned to snapshots)
# ---------------------------------------------------------------------------

def _build_gcn_metadata(
    data_dir: Path,
    split:    str,
) -> dict:
    """
    Build flat station_idx and timestamps arrays aligned to snapshot order.
    Each snapshot t contributes N rows (one per station).
    """
    import pandas as pd

    feat_df         = pd.read_parquet(data_dir / f"features_{split}.parquet")
    unique_times    = np.sort(feat_df["timestamp"].unique())
    unique_stations = np.sort(feat_df["station_idx"].unique())
    T = len(unique_times)
    N = len(unique_stations)

    station_idx_out = np.tile(unique_stations, T)                      # (T*N,)
    timestamps_out  = np.repeat(unique_times,  N)                      # (T*N,)

    return {
        "station_idx": station_idx_out.astype(np.int64),
        "timestamps":  pd.to_datetime(timestamps_out),
    }


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_gcn(
    data_dir:      Path,
    adj_variant:   Literal["knn", "flow", "combined"] = "knn",
    loss_fn:       Literal["mse", "poisson"] = "poisson",
    hidden_size:   int   = 64,
    dropout:       float = 0.2,
    lr:            float = 1e-3,
    max_epochs:    int   = 50,
    patience:      int   = 5,
    save_dir:      Path  = Path("models"),
    log_to_mlflow: bool  = True,
) -> None:
    """
    Train a GCN variant and log results to MLflow.

    Parameters
    ----------
    data_dir     : path to data/processed/
    adj_variant  : which adjacency matrix to use ('knn', 'flow', 'combined')
    hidden_size  : GCN hidden dimension
    dropout      : dropout after first GCN layer
    lr           : Adam learning rate
    max_epochs   : maximum training epochs
    patience     : early stopping patience
    save_dir     : directory to save best model checkpoint
    log_to_mlflow: whether to log to MLflow
    """
    run_name = f"gcn_{adj_variant}_{loss_fn}"
    device   = get_device()
    logger.info("Running %s | device=%s", run_name, device)

    # --- Adjacency ---
    adj_knn, adj_flow, adj_combined = load_adj(data_dir)
    adj_map = {"knn": adj_knn, "flow": adj_flow, "combined": adj_combined}
    adj_tensor = adj_map[adj_variant]

    # --- Datasets ---
    logger.info("Building graph datasets...")
    train_ds = build_graph_dataset(data_dir, "train", adj_tensor)
    val_ds   = build_graph_dataset(data_dir, "val",   adj_tensor)

    val_meta = _build_gcn_metadata(data_dir, "val")

    # --- Model ---
    in_channels = len(FEATURE_COLS)
    model = BikeDemanGCN(
        in_channels = in_channels,
        hidden_size = hidden_size,
        dropout     = dropout,
        use_softplus = loss_fn == "poisson",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s params: %d", run_name, n_params)

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
    ckpt_path      = save_dir / f"{run_name}_best.pt"

    # --- MLflow setup ---
    if log_to_mlflow:
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tag("model_name", run_name)
        mlflow.set_tag("phase", "gcn")
        mlflow.log_params({
            "model_type":   "gcn",
            "adj_variant":  adj_variant,
            "loss_fn": loss_fn,
            "hidden_size":  hidden_size,
            "dropout":      dropout,
            "lr":           lr,
            "max_epochs":   max_epochs,
            "patience":     patience,
            "n_params":     n_params,
            "in_channels":  in_channels,
            "device":       str(device),
        })

    # --- Training loop ---
    logger.info(
        "Starting training | max_epochs=%d | patience=%d", max_epochs, patience
    )

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss     = train_epoch(model, train_ds, criterion, optimiser, device)
        y_true, y_pred = evaluate_epoch(model, val_ds, device)
        y_pred         = np.clip(y_pred, 0, None)

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
            logger.info(
                "  -> New best val MAE=%.4f saved to %s", best_val_mae, ckpt_path
            )
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
    y_true, y_pred = evaluate_epoch(model, val_ds, device)
    y_pred         = np.clip(y_pred, 0, None)

    result = full_evaluation(
        y_true      = y_true,
        y_pred      = y_pred,
        station_idx = val_meta["station_idx"],
        timestamps  = val_meta["timestamps"],
        model_name  = run_name,
        split       = "val",
    )

    if log_to_mlflow:
        log_metrics_to_mlflow(result, prefix="best_")
        mlflow.log_param("best_epoch", best_epoch)

        result.by_station.to_csv(f"/tmp/{run_name}_by_station.csv", index=False)
        result.by_hour.to_csv(f"/tmp/{run_name}_by_hour.csv",       index=False)
        mlflow.log_artifact(f"/tmp/{run_name}_by_station.csv", artifact_path="eval")
        mlflow.log_artifact(f"/tmp/{run_name}_by_hour.csv",    artifact_path="eval")
        mlflow.log_artifact(str(ckpt_path),                    artifact_path="model")
        mlflow.end_run()

    print(
        f"\n{run_name} | MAE={result.metrics.mae:.4f}  "
        f"RMSE={result.metrics.rmse:.4f}  "
        f"MAPE={result.metrics.mape:.2f}%  "
        f"(best epoch={best_epoch})"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GCN model")
    parser.add_argument("--data-dir",    type=Path,  default=Path("data/processed"))
    parser.add_argument(
        "--adj", choices=["knn", "flow", "combined", "all"], default="knn",
        help="Adjacency variant to use (default: knn)",
    )
    parser.add_argument("--loss-fn", choices=["mse", "poisson"], default="poisson", help="Loss function to use (default: poisson)")
    parser.add_argument("--hidden-size", type=int,   default=64)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--max-epochs",  type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--save-dir",    type=Path,  default=Path("models"))
    parser.add_argument("--no-mlflow",   action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    variants = ADJ_VARIANTS if args.adj == "all" else (args.adj,)

    for variant in variants:
        run_gcn(
            data_dir      = args.data_dir,
            adj_variant   = variant,
            loss_fn       = args.loss_fn,
            hidden_size   = args.hidden_size,
            dropout       = args.dropout,
            lr            = args.lr,
            max_epochs    = args.max_epochs,
            patience      = args.patience,
            save_dir      = args.save_dir,
            log_to_mlflow = not args.no_mlflow,
        )


if __name__ == "__main__":
    main()