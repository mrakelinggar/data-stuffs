"""
data_loader.py
--------------
Loads Phase 3 feature/target parquets and adjacency matrices, and exposes:

  - BikeDataset         : PyTorch Dataset for Naive / Linear / LSTM
  - build_dataloaders() : returns (train_loader, val_loader, test_loader)
  - build_graph_data()  : returns PyG Data objects for GCN (one per split)
  - load_adj()          : returns the three adjacency tensors (knn, flow, combined)

Sequence length for LSTM: 168 steps (1-week lookback), matching warm_up_hours
from feature_metadata.json.

Adjacency matrices
------------------
  adj_knn     : graph_data.pkl -> adj_knn_norm  (1911x1911, D^{-1}A normalised)
  adj_flow    : graph_data.pkl -> adj_flow_norm (1911x1911, D^{-1}A normalised)
  adj_combined: element-wise average of raw adj_knn + adj_flow, then normalised

All three are returned as dense FloatTensors. Sparse conversion happens inside
the GCN model.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from utils import get_validated_snapshot_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# lat/lon excluded: spatial position is captured by adj_knn/adj_flow
# (GCN) and cluster features (Linear/LSTM)
FEATURE_COLS = [
    "lag_1", "lag_24", "lag_168",
    "rolling_7d_mean",
    "neigh_mean_knn", "neigh_mean_flow",
    "net_flow_pct", "weekday_weekend_ratio",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "cluster_0", "cluster_1", "cluster_2",
]
TARGET_COL = "target_t24"
GCN_FEATURE_COLS = FEATURE_COLS  # same set -- alias for future divergence

# Lag + calendar features used as LSTM sequence input (compact, no spatial leakage)
LSTM_FEATURE_COLS = [
    "lag_1", "lag_24", "lag_168",
    "hour_of_day", "day_of_week", "month", "is_weekend",
]
LSTM_SEQ_LEN = 168  # 1-week lookback, matches warm_up_hours


# ---------------------------------------------------------------------------
# Adjacency helpers
# ---------------------------------------------------------------------------

def _row_normalise(adj: np.ndarray) -> np.ndarray:
    """D^{-1} A row normalisation. Handles zero-degree nodes safely."""
    deg = adj.sum(axis=1, keepdims=True)
    deg = np.where(deg == 0, 1.0, deg)   # avoid division by zero
    return adj / deg


def load_adj(
    data_dir: str | Path,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Load and normalise the three adjacency matrices.

    All matrices come from graph_data.pkl (all 1911x1911, train stations only).
    adj_knn_norm and adj_flow_norm are used directly (verified to match D^{-1}A
    row normalisation). adj_combined is the element-wise average of the two raw
    matrices, normalised with the same scheme for consistency.

    Returns
    -------
    adj_knn_t      : FloatTensor (N, N)
    adj_flow_t     : FloatTensor (N, N)
    adj_combined_t : FloatTensor (N, N)
    """
    data_dir = Path(data_dir)

    with open(data_dir / "graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)

    # Pre-normalised -- verified to match D^{-1}A row normalisation
    adj_knn_norm  = graph_data["adj_knn_norm"].astype(np.float32)
    adj_flow_norm = graph_data["adj_flow_norm"].astype(np.float32)

    # combined: average raw matrices, then normalise
    adj_knn_raw      = graph_data["adj_knn"].astype(np.float32)
    adj_flow_raw     = graph_data["adj_flow"].astype(np.float32)
    adj_combined_raw = (adj_knn_raw + adj_flow_raw) / 2.0
    adj_combined_norm = _row_normalise(adj_combined_raw)

    adj_knn_t      = torch.from_numpy(adj_knn_norm)
    adj_flow_t     = torch.from_numpy(adj_flow_norm)
    adj_combined_t = torch.from_numpy(adj_combined_norm)

    logger.info(
        "Adjacency matrices loaded | shape=%s | "
        "knn_nnz=%d | flow_nnz=%d | combined_nnz=%d",
        adj_knn_t.shape,
        (adj_knn_t > 0).sum().item(),
        (adj_flow_t > 0).sum().item(),
        (adj_combined_t > 0).sum().item(),
    )
    return adj_knn_t, adj_flow_t, adj_combined_t


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_split(
    data_dir: Path,
    split: Literal["train", "val", "test"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load features and targets for a split.

    Returns
    -------
    features : float32 (T * N, F)   — already scaled by Phase 3
    targets  : float32 (T * N,)
    station_idx : int64 (T * N,)    — 0-based station index
    timestamps  : (T * N,)          — pandas Timestamps, used for sequence building
    """
    get_validated_snapshot_id(data_dir)

    feat_df = pd.read_parquet(data_dir / f"features_{split}.parquet")
    tgt_df  = pd.read_parquet(data_dir / f"targets_{split}.parquet")

    # Align on position (both share station_id / timestamp columns)
    assert len(feat_df) == len(tgt_df), "Feature/target row count mismatch"

    features    = feat_df[FEATURE_COLS].values.astype(np.float32)
    targets     = tgt_df[TARGET_COL].values.astype(np.float32)
    station_idx = feat_df["station_idx"].values.astype(np.int64)
    timestamps  = pd.to_datetime(feat_df["timestamp"].values)

    return features, targets, station_idx, timestamps


# ---------------------------------------------------------------------------
# Dataset: Naive / Linear (flat feature vector, no sequencing)
# ---------------------------------------------------------------------------

class BikeDataset(Dataset):
    """
    Flat dataset for Naive baseline and Linear Regression.

    Each sample is one (station, timestamp) pair:
        x : FloatTensor (F,)   — 17 features
        y : FloatTensor ()     — target_t24

    Also exposes:
        lag_24 : FloatTensor (N_total,) — for the Naive model
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"],
    ) -> None:
        data_dir = Path(data_dir)
        features, targets, station_idx, _ = _load_split(data_dir, split)

        self.x           = torch.from_numpy(features)         # (T*N, F)
        self.y           = torch.from_numpy(targets)          # (T*N,)
        self.station_idx = torch.from_numpy(station_idx)      # (T*N,)

        # Lag_24 column index for Naive baseline
        self._lag24_col = FEATURE_COLS.index("lag_24")

        logger.info(
            "BikeDataset [%s] | samples=%d | features=%d",
            split, len(self.x), self.x.shape[1],
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y[idx]

    @property
    def lag24(self) -> Tensor:
        """All lag_24 values — used directly by Naive model."""
        return self.x[:, self._lag24_col]

    def numpy_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) as numpy arrays — for sklearn models."""
        return self.x.numpy(), self.y.numpy()


# ---------------------------------------------------------------------------
# Dataset: LSTM (sequence of LSTM_SEQ_LEN steps per station)
# ---------------------------------------------------------------------------

class LSTMDataset(Dataset):
    """
    Sequence dataset for the LSTM model.

    Data is pivoted into (T, N, F_lstm) tensors, then a sliding window of
    LSTM_SEQ_LEN steps is used to form samples.

    Each sample:
        x_seq : FloatTensor (seq_len, F_lstm)  — sequence of compact features
        y     : FloatTensor ()                  — target at t + seq_len - 1

    The dataset iterates over stations × valid timestamps, so sample count =
    N_stations × (T - seq_len + 1).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"],
        seq_len: int = LSTM_SEQ_LEN,
    ) -> None:
        data_dir = Path(data_dir)
        features, targets, station_idx, timestamps = _load_split(data_dir, split)

        self.seq_len = seq_len
        lstm_cols = [FEATURE_COLS.index(c) for c in LSTM_FEATURE_COLS]

        # Pivot to (T, N, F) tensors
        # Determine unique timestamps and stations in sorted order
        unique_times    = np.sort(np.unique(timestamps))
        unique_stations = np.sort(np.unique(station_idx))
        T = len(unique_times)
        N = len(unique_stations)
        F = len(FEATURE_COLS)
        F_lstm = len(LSTM_FEATURE_COLS)

        logger.info(
            "LSTMDataset [%s] | T=%d timestamps | N=%d stations | seq_len=%d",
            split, T, N, seq_len,
        )

        # Build time→row and station→col lookup
        time_to_t    = {t: i for i, t in enumerate(unique_times)}
        station_to_n = {s: i for i, s in enumerate(unique_stations)}

        feat_grid   = np.zeros((T, N, F),      dtype=np.float32)
        target_grid = np.zeros((T, N),          dtype=np.float32)

        for row_i in range(len(features)):
            t_idx = time_to_t[timestamps[row_i]]
            n_idx = station_to_n[station_idx[row_i]]
            feat_grid[t_idx, n_idx, :]  = features[row_i]
            target_grid[t_idx, n_idx]   = targets[row_i]

        # Extract LSTM feature subset: (T, N, F_lstm)
        lstm_grid = feat_grid[:, :, lstm_cols]

        # Store as tensors
        self.lstm_grid   = torch.from_numpy(lstm_grid)    # (T, N, F_lstm)
        self.target_grid = torch.from_numpy(target_grid)  # (T, N)
        self.T = T
        self.N = N
        self.n_valid = T - seq_len + 1

        if self.n_valid <= 0:
            raise ValueError(
                f"seq_len={seq_len} >= T={T}; not enough timesteps for split '{split}'"
            )

        logger.info(
            "LSTMDataset [%s] | valid windows=%d | total samples=%d",
            split, self.n_valid, self.n_valid * N,
        )

    def __len__(self) -> int:
        return self.n_valid * self.N

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        t_idx = idx // self.N
        n_idx = idx % self.N
        x_seq = self.lstm_grid[t_idx : t_idx + self.seq_len, n_idx, :]  # (seq_len, F_lstm)
        y     = self.target_grid[t_idx + self.seq_len - 1, n_idx]       # scalar
        return x_seq, y


# ---------------------------------------------------------------------------
# Graph data: GCN snapshots (one PyG Data object per timestamp)
# ---------------------------------------------------------------------------

def build_graph_snapshots(
    data_dir: str | Path,
    split: Literal["train", "val", "test"],
    adj_tensor: Tensor,
) -> list[Data]:
    """
    Build a list of PyG Data objects — one per timestamp.

    Each Data object:
        x          : FloatTensor (N, F_gcn)  -- 15-feature node matrix (lat/lon dropped)
        y          : FloatTensor (N,)         -- target_t24 per node
        edge_index : LongTensor (2, E)        -- COO format from adj_tensor
        edge_weight: FloatTensor (E,)         -- normalised edge weights

    Returns
    -------
    List of Data objects, length = T (number of unique timestamps in split).
    """
    data_dir = Path(data_dir)
    features, targets, station_idx, timestamps = _load_split(data_dir, split)

    unique_times    = np.sort(np.unique(timestamps))
    unique_stations = np.sort(np.unique(station_idx))
    T = len(unique_times)
    N = len(unique_stations)
    F = len(FEATURE_COLS)  # same as F_gcn since GCN_FEATURE_COLS == FEATURE_COLS
    F_gcn = F

    # Build feature and target grids (T, N, F) and (T, N)
    time_to_t    = {t: i for i, t in enumerate(unique_times)}
    station_to_n = {s: i for i, s in enumerate(unique_stations)}

    feat_grid   = np.zeros((T, N, F), dtype=np.float32)
    target_grid = np.zeros((T, N),    dtype=np.float32)

    for row_i in range(len(features)):
        t_idx = time_to_t[timestamps[row_i]]
        n_idx = station_to_n[station_idx[row_i]]
        feat_grid[t_idx, n_idx, :]  = features[row_i]
        target_grid[t_idx, n_idx]   = targets[row_i]

    feat_t   = torch.from_numpy(feat_grid)   # (T, N, F_gcn)
    target_t = torch.from_numpy(target_grid)  # (T, N)

    # Convert adjacency to COO edge_index once (shared across all snapshots)
    edge_index, edge_weight = _adj_to_edge_index(adj_tensor)

    snapshots = []
    for t in range(T):
        data = Data(
            x           = feat_t[t],        # (N, F)
            y           = target_t[t],      # (N,)
            edge_index  = edge_index,       # (2, E)
            edge_weight = edge_weight,      # (E,)
            num_nodes   = N,
        )
        snapshots.append(data)

    logger.info(
        "Graph snapshots [%s] | T=%d | N=%d | F=%d | edges=%d",
        split, T, N, F_gcn, edge_index.shape[1],
    )
    return snapshots


def _adj_to_edge_index(adj: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert dense adjacency matrix to COO edge_index and edge_weight."""
    edge_idx = adj.nonzero(as_tuple=False).t().contiguous()   # (2, E)
    edge_wt  = adj[edge_idx[0], edge_idx[1]]                  # (E,)
    return edge_idx, edge_wt


# ---------------------------------------------------------------------------
# Public API: dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 512,
    num_workers: int = 2,
    model_type: Literal["flat", "lstm"] = "flat",
    seq_len: int = LSTM_SEQ_LEN,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    data_dir    : path to data/processed/
    batch_size  : samples per batch
    num_workers : DataLoader workers
    model_type  : 'flat' for Naive/Linear, 'lstm' for LSTM
    seq_len     : sequence length (LSTM only)

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    DatasetClass = LSTMDataset if model_type == "lstm" else BikeDataset

    def _make(split: str, shuffle: bool) -> DataLoader:
        kwargs = {"seq_len": seq_len} if model_type == "lstm" else {}
        ds = DatasetClass(data_dir, split, **kwargs)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    train_loader = _make("train", shuffle=True)
    val_loader   = _make("val",   shuffle=False)
    test_loader  = _make("test",  shuffle=False)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")

    print("\n--- Adjacency matrices ---")
    adj_knn, adj_flow, adj_combined = load_adj(data_dir)
    print(f"  adj_knn      : {adj_knn.shape}  min={adj_knn.min():.4f}  max={adj_knn.max():.4f}")
    print(f"  adj_flow     : {adj_flow.shape}  min={adj_flow.min():.4f}  max={adj_flow.max():.4f}")
    print(f"  adj_combined : {adj_combined.shape}  min={adj_combined.min():.4f}  max={adj_combined.max():.4f}")

    print("\n--- Flat DataLoaders (Naive / Linear) ---")
    train_dl, val_dl, test_dl = build_dataloaders(data_dir, batch_size=1024, model_type="flat")
    xb, yb = next(iter(train_dl))
    print(f"  train batch x={xb.shape}  y={yb.shape}  dtype={xb.dtype}")
    xb, yb = next(iter(val_dl))
    print(f"  val   batch x={xb.shape}  y={yb.shape}")
    xb, yb = next(iter(test_dl))
    print(f"  test  batch x={xb.shape}  y={yb.shape}")

    print("\n--- LSTM DataLoaders ---")
    train_dl, val_dl, test_dl = build_dataloaders(data_dir, batch_size=256, model_type="lstm")
    xb, yb = next(iter(train_dl))
    print(f"  train batch x={xb.shape}  y={yb.shape}  (seq_len={xb.shape[1]}, features={xb.shape[2]})")

    print("\n--- Graph snapshots (train, adj_knn) ---")
    snapshots = build_graph_snapshots(data_dir, "train", adj_knn)
    s0 = snapshots[0]
    print(f"  n_snapshots={len(snapshots)}")
    print(f"  snapshot[0]: x={s0.x.shape}  y={s0.y.shape}  edge_index={s0.edge_index.shape}")

    print("\nAll checks passed.")