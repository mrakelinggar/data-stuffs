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

from utils import get_validated_snapshot_id, release_host_memory

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
    "temp_f", "precip_in", "wind_mph", "humidity_pct",              # ROADMAP Phase 5
    "is_holiday", "is_day_before_holiday", "is_day_after_holiday",  # ROADMAP Phase 5
    "cluster_0", "cluster_1", "cluster_2",
]
TARGET_COL = "target_t24"
GCN_FEATURE_COLS = FEATURE_COLS  # same set -- alias for future divergence

# Lag + calendar features used as LSTM sequence input (compact, no spatial leakage)
LSTM_FEATURE_COLS = [
    "lag_1", "lag_24", "lag_168",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "temp_f", "precip_in", "is_holiday",  # ROADMAP Phase 5 -- vary in time, matter at rush hour
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
    # .to_numpy() (not the bare DatetimeIndex pd.to_datetime returns) so every
    # caller gets a plain ndarray of numpy.datetime64 -- consistent with what
    # np.unique() produces. A DatetimeIndex boxes __getitem__ results as
    # pandas.Timestamp, which can hash differently from the numpy.datetime64
    # keys np.unique(...) yields, causing spurious KeyErrors in any lookup
    # path that skips np.concatenate() (train's LSTMDataset branch has no
    # prior split to concatenate against, so it's the one path this bit).
    timestamps  = pd.to_datetime(feat_df["timestamp"].values).to_numpy()

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

_PRIOR_SPLIT: dict[str, str] = {"val": "train", "test": "val"}


class LSTMDataset(Dataset):
    """
    Sequence dataset for the LSTM model.

    Data is pivoted into (T, N, F_lstm) tensors, then a sliding window of
    LSTM_SEQ_LEN steps is used to form samples.

    Each sample:
        x_seq : FloatTensor (seq_len, F_lstm)  — sequence of compact features
        y     : FloatTensor ()                  — target at t + seq_len - 1

    For val/test, the preceding split's last `seq_len - 1` timestamps are
    borrowed as read-only input history (never a scored target) so every
    sample count equals N_stations × T_own — matching BikeDataset/GCN, which
    score every row in a split. Train has no preceding split to borrow from,
    so it keeps the original N_stations × (T - seq_len + 1) behavior.

    `device`: when a CUDA device is passed, the dense (T,N,F_lstm)/(T,N)
    grids are moved there at construction time instead of staying CPU-resident
    -- on a machine with abundant idle VRAM but a tight system-RAM ceiling
    (e.g. a free-tier Colab T4: 12.7GB system RAM vs 15GB mostly-idle VRAM),
    this shifts the dataset's dominant memory cost off the constrained
    resource. `num_workers=0` is required for this (CUDA tensors can't cross
    a forked DataLoader worker process) -- LSTM training already uses
    num_workers=0 unconditionally, so this is safe. No effect for MPS/CPU
    devices (kept CPU-resident, matching prior behavior) -- MPS's indexing/
    slicing kernel coverage is less mature (see CLAUDE.md's reproducibility
    notes), so this optimization is deliberately CUDA-only for now.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"],
        seq_len: int = LSTM_SEQ_LEN,
        device: torch.device | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        own_features, own_targets, own_station_idx, own_timestamps = _load_split(data_dir, split)

        self.seq_len = seq_len
        lstm_cols = [FEATURE_COLS.index(c) for c in LSTM_FEATURE_COLS]

        borrow_len = seq_len - 1
        prior_split = _PRIOR_SPLIT.get(split)

        if prior_split is not None and borrow_len > 0:
            p_features, _, p_station_idx, p_timestamps = _load_split(data_dir, prior_split)
            p_unique_times = np.sort(np.unique(p_timestamps))
            tail_times = p_unique_times[-borrow_len:]
            tail_mask = np.isin(p_timestamps, tail_times)

            # Two assumptions make the borrowing arithmetic below safe; both
            # are asserted explicitly rather than left implicit, since a
            # silent violation would misalign features/targets rather than
            # raise:
            #   1. Same station universe in both splits -- if it ever
            #      diverges (e.g. a future station-universe change applied
            #      unevenly across splits), N would be derived from the
            #      union rather than the true own-split count, producing
            #      phantom scored samples for a station absent from the
            #      current split.
            own_stations_set = set(np.unique(own_station_idx))
            prior_tail_stations_set = set(np.unique(p_station_idx[tail_mask]))
            if not prior_tail_stations_set <= own_stations_set:
                raise AssertionError(
                    f"LSTMDataset[{split}]: prior split '{prior_split}' has "
                    f"stations not present in '{split}': "
                    f"{sorted(prior_tail_stations_set - own_stations_set)} -- "
                    "borrowing assumes an identical station universe across "
                    "splits (see CLAUDE.md's station-universe-stability note)."
                )
            #   2. The borrowed tail is strictly earlier than the current
            #      split's own timestamps -- true today because splits are
            #      fixed, non-overlapping calendar windows, but not checked
            #      anywhere upstream. A violation would silently corrupt
            #      anchor_timestamps' 1:1 correspondence to real own-split
            #      timestamps (used by _build_lstm_metadata's feat_df
            #      lookup).
            if tail_times.max() >= own_timestamps.min():
                raise AssertionError(
                    f"LSTMDataset[{split}]: borrowed tail from '{prior_split}' "
                    f"(max={tail_times.max()}) is not strictly earlier than "
                    f"'{split}'s own timestamps (min={own_timestamps.min()}) -- "
                    "splits may overlap or be out of chronological order."
                )

            features    = np.concatenate([p_features[tail_mask], own_features])
            station_idx = np.concatenate([p_station_idx[tail_mask], own_station_idx])
            timestamps  = np.concatenate([p_timestamps[tail_mask], own_timestamps])
            # Borrowed rows are read-only input history for the sequence's
            # early timesteps; their targets are never indexed by
            # __getitem__ (see the anchor-count assertion below), so zero
            # placeholders are safe and avoid loading the prior split's
            # targets parquet at all.
            borrowed_targets = np.zeros(int(tail_mask.sum()), dtype=np.float32)
            targets = np.concatenate([borrowed_targets, own_targets])

            # p_features/p_station_idx/p_timestamps are the ENTIRE prior
            # split reloaded from disk (e.g. for val, all of train's ~460MB
            # features array) just to slice out `borrow_len` rows -- their
            # data is now copied into the concatenated arrays above, so
            # release them now rather than holding them through the O(T*N)
            # grid-building loop below.
            del p_features, p_station_idx, p_timestamps, tail_mask
            del p_unique_times, tail_times, borrowed_targets
            release_host_memory()
        else:
            features, targets, station_idx, timestamps = (
                own_features, own_targets, own_station_idx, own_timestamps,
            )

        # own_features/own_targets/own_station_idx are no longer needed past
        # this point (own_timestamps is kept -- still read at the anchor-
        # count assertion below). In the borrow branch they were already
        # copied into a new concatenated array via np.concatenate, so this
        # frees the originals; in the no-borrow (train) branch, features/
        # targets/station_idx are just aliases of the same objects (no copy
        # was made), so deleting the own_* names here is what actually lets
        # those arrays become collectible once `features` etc. are deleted
        # below -- without this, train (the largest split) would silently
        # keep its full features array alive via the own_features alias
        # even after `del features`.
        del own_features, own_targets, own_station_idx

        # Pivot to (T, N, F) tensors
        # Determine unique timestamps and stations in sorted order
        unique_times    = np.sort(np.unique(timestamps))
        unique_stations = np.sort(np.unique(station_idx))
        T = len(unique_times)
        N = len(unique_stations)
        F = len(FEATURE_COLS)
        F_lstm = len(LSTM_FEATURE_COLS)

        logger.info(
            "LSTMDataset [%s] | T=%d timestamps (borrowed=%d) | N=%d stations | seq_len=%d",
            split, T, T - len(np.unique(own_timestamps)), N, seq_len,
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

        # Store as tensors -- moved to `device` immediately when it's CUDA
        # (see docstring), so the CPU-resident numpy arrays above become
        # eligible for GC right after.
        lstm_grid_t   = torch.from_numpy(lstm_grid)    # (T, N, F_lstm)
        target_grid_t = torch.from_numpy(target_grid)  # (T, N)
        if device is not None and device.type == "cuda":
            lstm_grid_t   = lstm_grid_t.to(device)
            target_grid_t = target_grid_t.to(device)
        self.lstm_grid   = lstm_grid_t
        self.target_grid = target_grid_t
        self.T = T
        self.N = N
        self.n_valid = T - seq_len + 1

        # feat_grid/lstm_grid (numpy)/features/targets/station_idx/timestamps
        # are large transient CPU allocations (feat_grid alone is
        # T*N*len(FEATURE_COLS)*4 bytes, ~460MB for the train split) --
        # none are referenced again (unique_times/anchor_timestamps below
        # are already-derived, much smaller arrays). Freeing them here and
        # forcing a real OS-level release (not just a Python-level GC) keeps
        # process RSS from ratcheting up split after split, since glibc
        # otherwise tends to hold onto freed arenas for reuse rather than
        # returning them to the OS.
        del feat_grid, lstm_grid, features, targets, station_idx, timestamps
        release_host_memory()

        if self.n_valid <= 0:
            raise ValueError(
                f"seq_len={seq_len} >= T={T}; not enough timesteps for split '{split}'"
            )

        # When a prior split was borrowed, n_valid must equal the split's own
        # timestamp count exactly: borrowing exactly `seq_len - 1` prior-split
        # timestamps as read-only history makes this hold algebraically
        # (T = borrow_len + T_own => n_valid = T_own). Asserted explicitly so
        # a future edit that breaks the invariant fails loudly instead of
        # silently scoring borrowed rows as if they were the current split's
        # own targets. Train (no prior split) keeps the original formula, so
        # n_valid < T_own there is expected, not a bug.
        own_T = len(np.unique(own_timestamps))
        if prior_split is not None and borrow_len > 0 and self.n_valid != own_T:
            raise AssertionError(
                f"LSTMDataset[{split}] anchor count {self.n_valid} != expected "
                f"{own_T} own timestamps — borrowed prior-split rows may be "
                "leaking into scored anchors."
            )

        # Each anchor's target timestamp, in the same time-major order
        # __getitem__ iterates — the single source of truth for downstream
        # metadata alignment (see _build_lstm_metadata in lstm.py), instead
        # of that function re-deriving the seq_len offset independently.
        self.anchor_timestamps = unique_times[seq_len - 1:]

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
# Dataset: STGNN (spatiotemporal hybrid, ROADMAP Phase 8)
# ---------------------------------------------------------------------------

class STGNNDataset(Dataset):
    """
    Sliding-window dataset for the ROADMAP Phase 8 spatiotemporal hybrid
    (BikeDemandSTGNN: batched-GCN encoder over each of the seq_len timesteps,
    then per-node LSTM). Each sample is one whole-graph window plus the target
    vector at its final timestep.

    Each sample:
        x_window : FloatTensor (seq_len, N, F=22)  -- all-station features
        y        : FloatTensor (N,)                 -- target_t24 per station

    `__len__` returns `n_valid` (one window per anchor timestep), NOT
    `n_valid * N` -- the model already emits one prediction per station per
    window, unlike LSTMDataset where each (station, anchor) pair is its own
    sample.

    For val/test the preceding split's last `seq_len - 1` timestamps are
    borrowed as read-only input history (exactly LSTMDataset's borrow
    contract), so `len(ds) == len(own_unique_timestamps)` for val/test,
    matching BikeDataset/GCN's per-timestep row count exactly. Train has no
    preceding split to borrow from, so `len(ds) = T_train - seq_len + 1`
    (same scoped exception LSTMDataset carries).

    The graph is static within a run: `edge_index` and `edge_weight` are
    built once from `adj_tensor` and exposed as dataset attributes for the
    training loop to read once and pass through the model per batch. They
    never enter `__getitem__` -- no per-window copy.

    `device`: CUDA-only optimization mirroring LSTMDataset. When a CUDA
    device is passed, `feat_grid`, `target_grid`, `edge_index`, and
    `edge_weight` are moved there at construction (requires `num_workers=0`
    on the DataLoader -- see `build_stgnn_dataloaders`). MPS/CPU stay CPU-
    resident.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Literal["train", "val", "test"],
        adj_tensor: Tensor,
        seq_len: int = LSTM_SEQ_LEN,
        device: torch.device | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        own_features, own_targets, own_station_idx, own_timestamps = _load_split(data_dir, split)

        self.seq_len = seq_len

        borrow_len = seq_len - 1
        prior_split = _PRIOR_SPLIT.get(split)

        if prior_split is not None and borrow_len > 0:
            p_features, _, p_station_idx, p_timestamps = _load_split(data_dir, prior_split)
            p_unique_times = np.sort(np.unique(p_timestamps))
            tail_times = p_unique_times[-borrow_len:]
            tail_mask = np.isin(p_timestamps, tail_times)

            # Same station-universe / chronological-order invariants as
            # LSTMDataset. See LSTMDataset.__init__ for the rationale --
            # duplicated (not shared) because the two datasets otherwise have
            # nothing to share (different grid layout, different __len__
            # semantics), and coupling their init through a helper would
            # leak abstraction for no gain.
            own_stations_set = set(np.unique(own_station_idx))
            prior_tail_stations_set = set(np.unique(p_station_idx[tail_mask]))
            if not prior_tail_stations_set <= own_stations_set:
                raise AssertionError(
                    f"STGNNDataset[{split}]: prior split '{prior_split}' has "
                    f"stations not present in '{split}': "
                    f"{sorted(prior_tail_stations_set - own_stations_set)} -- "
                    "borrowing assumes an identical station universe across "
                    "splits (see CLAUDE.md's station-universe-stability note)."
                )
            if tail_times.max() >= own_timestamps.min():
                raise AssertionError(
                    f"STGNNDataset[{split}]: borrowed tail from '{prior_split}' "
                    f"(max={tail_times.max()}) is not strictly earlier than "
                    f"'{split}'s own timestamps (min={own_timestamps.min()}) -- "
                    "splits may overlap or be out of chronological order."
                )

            features    = np.concatenate([p_features[tail_mask], own_features])
            station_idx = np.concatenate([p_station_idx[tail_mask], own_station_idx])
            timestamps  = np.concatenate([p_timestamps[tail_mask], own_timestamps])
            borrowed_targets = np.zeros(int(tail_mask.sum()), dtype=np.float32)
            targets = np.concatenate([borrowed_targets, own_targets])

            del p_features, p_station_idx, p_timestamps, tail_mask
            del p_unique_times, tail_times, borrowed_targets
            release_host_memory()
        else:
            features, targets, station_idx, timestamps = (
                own_features, own_targets, own_station_idx, own_timestamps,
            )

        del own_features, own_targets, own_station_idx

        unique_times    = np.sort(np.unique(timestamps))
        unique_stations = np.sort(np.unique(station_idx))
        T = len(unique_times)
        N = len(unique_stations)
        F = len(FEATURE_COLS)

        logger.info(
            "STGNNDataset [%s] | T=%d timestamps (borrowed=%d) | N=%d stations | seq_len=%d | F=%d",
            split, T, T - len(np.unique(own_timestamps)), N, seq_len, F,
        )

        time_to_t    = {t: i for i, t in enumerate(unique_times)}
        station_to_n = {s: i for i, s in enumerate(unique_stations)}

        feat_grid   = np.zeros((T, N, F), dtype=np.float32)
        target_grid = np.zeros((T, N),    dtype=np.float32)

        for row_i in range(len(features)):
            t_idx = time_to_t[timestamps[row_i]]
            n_idx = station_to_n[station_idx[row_i]]
            feat_grid[t_idx, n_idx, :]  = features[row_i]
            target_grid[t_idx, n_idx]   = targets[row_i]

        feat_grid_t   = torch.from_numpy(feat_grid)     # (T, N, F)
        target_grid_t = torch.from_numpy(target_grid)   # (T, N)

        # Build shared edge_index once from adjacency. Shape (2, E), (E,).
        # These are shared across every window -- the whole hybrid rests on
        # this: the graph is STATIC for the run, so one COO edge_index is
        # correct for all seq_len timesteps and every window.
        edge_index, edge_weight = _adj_to_edge_index(adj_tensor)

        if device is not None and device.type == "cuda":
            feat_grid_t   = feat_grid_t.to(device)
            target_grid_t = target_grid_t.to(device)
            edge_index    = edge_index.to(device)
            edge_weight   = edge_weight.to(device)

        self.feat_grid   = feat_grid_t
        self.target_grid = target_grid_t
        self.edge_index  = edge_index
        self.edge_weight = edge_weight
        self.T = T
        self.N = N
        self.n_valid = T - seq_len + 1

        # Also expose station ordering so downstream code (metadata alignment,
        # ordering-consistency tests) can check it matches the adjacency
        # matrix's node ordering.
        self.station_ids = unique_stations

        del feat_grid, target_grid, features, targets, station_idx, timestamps
        release_host_memory()

        if self.n_valid <= 0:
            raise ValueError(
                f"seq_len={seq_len} >= T={T}; not enough timesteps for split '{split}'"
            )

        own_T = len(np.unique(own_timestamps))
        if prior_split is not None and borrow_len > 0 and self.n_valid != own_T:
            raise AssertionError(
                f"STGNNDataset[{split}] anchor count {self.n_valid} != expected "
                f"{own_T} own timestamps -- borrowed prior-split rows may be "
                "leaking into scored anchors."
            )

        self.anchor_timestamps = unique_times[seq_len - 1:]

        logger.info(
            "STGNNDataset [%s] | valid windows=%d | edges=%d",
            split, self.n_valid, edge_index.shape[1],
        )

    def __len__(self) -> int:
        return self.n_valid

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # x_window (seq_len, N, F); y (N,). edge_index / edge_weight live on
        # `self` -- reader (training loop) reads them once, not per-batch.
        x_window = self.feat_grid[idx : idx + self.seq_len, :, :]
        y        = self.target_grid[idx + self.seq_len - 1, :]
        return x_window, y


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
    device: torch.device | None = None,
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
    device      : LSTM only -- when a CUDA device, LSTMDataset moves its
                  dense grids there at construction (see LSTMDataset
                  docstring). Ignored for 'flat' (BikeDataset stays CPU;
                  Naive/Linear are sklearn-based, no GPU involved).

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    DatasetClass = LSTMDataset if model_type == "lstm" else BikeDataset

    def _make(split: str, shuffle: bool) -> DataLoader:
        kwargs = {"seq_len": seq_len, "device": device} if model_type == "lstm" else {}
        ds = DatasetClass(data_dir, split, **kwargs)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            # pin_memory only pays off when a worker thread/process can copy
            # into page-locked memory while the GPU is busy; with
            # num_workers=0 there's no overlap, just extra host memory
            # pressure held across every batch for no speed benefit.
            pin_memory  = torch.cuda.is_available() and num_workers > 0,
        )

    train_loader = _make("train", shuffle=True)
    val_loader   = _make("val",   shuffle=False)
    test_loader  = _make("test",  shuffle=False)

    return train_loader, val_loader, test_loader


def build_stgnn_dataloaders(
    data_dir: str | Path,
    adj_tensor: Tensor,
    batch_size: int = 4,
    num_workers: int = 0,
    seq_len: int = LSTM_SEQ_LEN,
    device: torch.device | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for the STGNN hybrid (Phase 8).

    Separate from build_dataloaders because STGNN requires an `adj_tensor` --
    that parameter is meaningless for flat/lstm and would just bloat the
    signature. Default `batch_size=4` reflects the T4 VRAM budget described
    in ROADMAP Phase 8; the training script can override.

    `num_workers` is silently forced to 0 whenever the dataset is CUDA-
    resident (via `device`), since CUDA tensors cannot cross a forked
    DataLoader worker process. Caller can pass num_workers > 0 for CPU-
    resident use.
    """
    data_dir = Path(data_dir)

    if device is not None and device.type == "cuda" and num_workers > 0:
        logger.warning(
            "build_stgnn_dataloaders: forcing num_workers=0 -- STGNNDataset "
            "grids are CUDA-resident on this device, and CUDA tensors cannot "
            "cross a forked DataLoader worker."
        )
        num_workers = 0

    def _make(split: str, shuffle: bool) -> DataLoader:
        ds = STGNNDataset(data_dir, split, adj_tensor, seq_len=seq_len, device=device)
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            pin_memory  = torch.cuda.is_available() and num_workers > 0,
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