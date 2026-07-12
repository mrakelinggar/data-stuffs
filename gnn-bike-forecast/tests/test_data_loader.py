"""
Tests for src/models/data_loader.py -- LSTMDataset's val/test windowing fix
(ROADMAP Phase 4).

Before this fix, LSTMDataset silently dropped the first `seq_len - 1`
timestamps of val/test as unscoreable, so LSTM evaluated a smaller, different
set of (station, timestamp) rows than BikeDataset (Naive/Linear) and
build_graph_snapshots (GCN). The fix borrows the preceding split's tail as
read-only input history so every family scores the same rows.

All fixtures are synthetic -- no dependency on data/processed/ or mlflow.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
import torch

from data_loader import FEATURE_COLS, LSTM_FEATURE_COLS, LSTMDataset
from lstm import _build_lstm_metadata
from utils import compute_data_snapshot_id

# Split sizes and split-constant values shared by every test below. Small and
# far from the production LSTM_SEQ_LEN=168 to keep tests fast.
_SEQ_LEN = 10
_SPLIT_SIZES = {"train": 50, "val": 30, "test": 30}
_N_STATIONS = 3
_SPLIT_LSTM_VALUES = {"train": 1.0, "val": 2.0, "test": 3.0}


def _write_synthetic_snapshot(data_dir: Path) -> None:
    """Write a minimal, valid data/processed/-shaped snapshot to data_dir.

    Every LSTM_FEATURE_COLS column is set to a per-split constant (see
    _SPLIT_LSTM_VALUES) so tests can identify which split's rows ended up in
    a given LSTM input window. target_t24 is set to
    `t_own * 1000 + station_idx`, a value unique per (own-split time index,
    station) pair, so target correctness can be checked exactly.
    """
    lstm_col_set = set(LSTM_FEATURE_COLS)

    base_time = pd.Timestamp("2024-01-01")
    offset_hours = 0
    for split in ("train", "val", "test"):
        T = _SPLIT_SIZES[split]
        times = pd.date_range(base_time + pd.Timedelta(hours=offset_hours), periods=T, freq="h")
        offset_hours += T

        feat_rows = []
        tgt_rows = []
        for t_own, ts in enumerate(times):
            for n_idx in range(_N_STATIONS):
                row = {
                    col: (_SPLIT_LSTM_VALUES[split] if col in lstm_col_set else 0.0)
                    for col in FEATURE_COLS
                }
                row["station_idx"] = n_idx
                row["timestamp"] = ts
                feat_rows.append(row)
                tgt_rows.append({"target_t24": float(t_own * 1000 + n_idx)})

        pd.DataFrame(feat_rows).to_parquet(data_dir / f"features_{split}.parquet", index=False)
        pd.DataFrame(tgt_rows).to_parquet(data_dir / f"targets_{split}.parquet", index=False)

    # SNAPSHOT_ARTIFACTS requires graph_data.pkl to exist and be hashable;
    # LSTMDataset never reads its contents.
    with open(data_dir / "graph_data.pkl", "wb") as f:
        pickle.dump({"placeholder": True}, f)

    snapshot_id = compute_data_snapshot_id(data_dir)
    with open(data_dir / "SNAPSHOT.json", "w") as f:
        json.dump({"data_snapshot_id": snapshot_id}, f)


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    _write_synthetic_snapshot(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Sample count / anchor alignment
# ---------------------------------------------------------------------------

def test_lstm_dataset_val_produces_one_sample_per_own_timestamp(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN)
    assert len(ds) == _SPLIT_SIZES["val"] * _N_STATIONS


def test_lstm_dataset_test_produces_one_sample_per_own_timestamp(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "test", seq_len=_SEQ_LEN)
    assert len(ds) == _SPLIT_SIZES["test"] * _N_STATIONS


def test_lstm_dataset_train_split_unchanged_behavior(snapshot_dir):
    # Train has no preceding split to borrow from -- old formula preserved.
    ds = LSTMDataset(snapshot_dir, "train", seq_len=_SEQ_LEN)
    expected = (_SPLIT_SIZES["train"] - _SEQ_LEN + 1) * _N_STATIONS
    assert len(ds) == expected


def test_lstm_dataset_first_anchor_uses_own_first_timestamp(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN)
    val_feat = pd.read_parquet(snapshot_dir / "features_val.parquet")
    expected_first_ts = pd.to_datetime(val_feat["timestamp"]).min()
    assert ds.anchor_timestamps[0] == expected_first_ts


# ---------------------------------------------------------------------------
# Target correctness
# ---------------------------------------------------------------------------

def test_lstm_dataset_targets_match_own_split_exactly(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN)
    for idx in range(len(ds)):
        t_own = idx // ds.N
        n_idx = idx % ds.N
        _, y = ds[idx]
        expected = t_own * 1000 + n_idx
        assert y.item() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Borrowing isolation -- prior split only, never two splits back
# ---------------------------------------------------------------------------

def test_lstm_dataset_test_split_borrows_val_tail_not_train(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "test", seq_len=_SEQ_LEN)

    # First test anchor's window = borrowed val tail (seq_len - 1 rows) +
    # test's own first row.
    x_seq, _ = ds[0]
    borrowed_positions = x_seq[: _SEQ_LEN - 1]

    assert torch.all(borrowed_positions == _SPLIT_LSTM_VALUES["val"])
    assert not torch.any(borrowed_positions == _SPLIT_LSTM_VALUES["train"])


def test_lstm_dataset_val_borrows_train_tail(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN)

    x_seq, _ = ds[0]
    borrowed_positions = x_seq[: _SEQ_LEN - 1]

    assert torch.all(borrowed_positions == _SPLIT_LSTM_VALUES["train"])


def test_lstm_dataset_borrowed_rows_never_scored_as_target(snapshot_dir):
    # Every target across the whole dataset must come from the split's own
    # constant-encoded formula (t_own*1000 + n_idx, t_own in [0, T_own)) --
    # borrowed rows were given zero-filled placeholder targets and must never
    # surface as a `y`.
    ds = LSTMDataset(snapshot_dir, "test", seq_len=_SEQ_LEN)
    max_t_own = _SPLIT_SIZES["test"] - 1
    valid_targets = {t * 1000 + n for t in range(max_t_own + 1) for n in range(_N_STATIONS)}

    for idx in range(len(ds)):
        _, y = ds[idx]
        assert int(y.item()) in valid_targets


# ---------------------------------------------------------------------------
# Guardrail assertions -- station-universe and chronological-order mismatches
# ---------------------------------------------------------------------------

def _write_snapshot_with_extra_train_station(data_dir: Path) -> None:
    """Like _write_synthetic_snapshot, but train has one extra station index
    (3) that val/test do not have -- violates the "identical station universe
    across splits" assumption the borrowing logic depends on.
    """
    lstm_col_set = set(LSTM_FEATURE_COLS)
    stations_by_split = {"train": _N_STATIONS + 1, "val": _N_STATIONS, "test": _N_STATIONS}

    base_time = pd.Timestamp("2024-01-01")
    offset_hours = 0
    for split in ("train", "val", "test"):
        T = _SPLIT_SIZES[split]
        n_stations = stations_by_split[split]
        times = pd.date_range(base_time + pd.Timedelta(hours=offset_hours), periods=T, freq="h")
        offset_hours += T

        feat_rows = []
        tgt_rows = []
        for t_own, ts in enumerate(times):
            for n_idx in range(n_stations):
                row = {
                    col: (_SPLIT_LSTM_VALUES[split] if col in lstm_col_set else 0.0)
                    for col in FEATURE_COLS
                }
                row["station_idx"] = n_idx
                row["timestamp"] = ts
                feat_rows.append(row)
                tgt_rows.append({"target_t24": float(t_own * 1000 + n_idx)})

        pd.DataFrame(feat_rows).to_parquet(data_dir / f"features_{split}.parquet", index=False)
        pd.DataFrame(tgt_rows).to_parquet(data_dir / f"targets_{split}.parquet", index=False)

    with open(data_dir / "graph_data.pkl", "wb") as f:
        pickle.dump({"placeholder": True}, f)

    snapshot_id = compute_data_snapshot_id(data_dir)
    with open(data_dir / "SNAPSHOT.json", "w") as f:
        json.dump({"data_snapshot_id": snapshot_id}, f)


def test_lstm_dataset_raises_on_station_universe_mismatch(tmp_path):
    _write_snapshot_with_extra_train_station(tmp_path)

    with pytest.raises(AssertionError, match="stations not present"):
        LSTMDataset(tmp_path, "val", seq_len=_SEQ_LEN)


def _write_snapshot_with_overlapping_val(data_dir: Path) -> None:
    """Like _write_synthetic_snapshot, but val's own timestamps start before
    train's last (seq_len - 1) timestamps end -- violates the "borrowed tail
    is strictly earlier than own timestamps" assumption.
    """
    lstm_col_set = set(LSTM_FEATURE_COLS)
    base_time = pd.Timestamp("2024-01-01")

    train_times = pd.date_range(base_time, periods=_SPLIT_SIZES["train"], freq="h")
    # Val overlaps train's last 5 hours instead of starting strictly after.
    val_start = train_times[-5]
    val_times = pd.date_range(val_start, periods=_SPLIT_SIZES["val"], freq="h")
    test_start = val_times[-1] + pd.Timedelta(hours=1)
    test_times = pd.date_range(test_start, periods=_SPLIT_SIZES["test"], freq="h")

    for split, times in (("train", train_times), ("val", val_times), ("test", test_times)):
        feat_rows = []
        tgt_rows = []
        for t_own, ts in enumerate(times):
            for n_idx in range(_N_STATIONS):
                row = {
                    col: (_SPLIT_LSTM_VALUES[split] if col in lstm_col_set else 0.0)
                    for col in FEATURE_COLS
                }
                row["station_idx"] = n_idx
                row["timestamp"] = ts
                feat_rows.append(row)
                tgt_rows.append({"target_t24": float(t_own * 1000 + n_idx)})

        pd.DataFrame(feat_rows).to_parquet(data_dir / f"features_{split}.parquet", index=False)
        pd.DataFrame(tgt_rows).to_parquet(data_dir / f"targets_{split}.parquet", index=False)

    with open(data_dir / "graph_data.pkl", "wb") as f:
        pickle.dump({"placeholder": True}, f)

    snapshot_id = compute_data_snapshot_id(data_dir)
    with open(data_dir / "SNAPSHOT.json", "w") as f:
        json.dump({"data_snapshot_id": snapshot_id}, f)


def test_lstm_dataset_raises_on_chronological_overlap(tmp_path):
    _write_snapshot_with_overlapping_val(tmp_path)

    with pytest.raises(AssertionError, match="not strictly earlier"):
        LSTMDataset(tmp_path, "val", seq_len=_SEQ_LEN)


# ---------------------------------------------------------------------------
# _build_lstm_metadata alignment (lstm.py)
# ---------------------------------------------------------------------------

def test_build_lstm_metadata_aligns_with_dataset_length(snapshot_dir):
    val_feat = pd.read_parquet(snapshot_dir / "features_val.parquet")
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN)

    metadata = _build_lstm_metadata(val_feat, ds)

    assert len(metadata["timestamps"]) == len(ds)
    assert len(metadata["station_idx"]) == len(ds)
    assert pd.Timestamp(metadata["timestamps"].min()) == pd.Timestamp(ds.anchor_timestamps.min())


def test_build_lstm_metadata_matches_dataset_for_train_too(snapshot_dir):
    # train_meta is built the same way in lstm.py (line ~400) -- confirm the
    # rewritten _build_lstm_metadata works generically, not just for the
    # borrowing splits.
    train_feat = pd.read_parquet(snapshot_dir / "features_train.parquet")
    ds = LSTMDataset(snapshot_dir, "train", seq_len=_SEQ_LEN)

    metadata = _build_lstm_metadata(train_feat, ds)

    assert len(metadata["timestamps"]) == len(ds)


# ---------------------------------------------------------------------------
# device parameter (Colab OOM mitigation -- move dense grids onto CUDA)
# ---------------------------------------------------------------------------

def test_lstm_dataset_defaults_to_cpu_grids(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN)
    assert ds.lstm_grid.device.type == "cpu"
    assert ds.target_grid.device.type == "cpu"


def test_lstm_dataset_cpu_device_stays_on_cpu(snapshot_dir):
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN, device=torch.device("cpu"))
    assert ds.lstm_grid.device.type == "cpu"
    assert ds.target_grid.device.type == "cpu"


def test_lstm_dataset_non_cuda_device_is_not_moved(snapshot_dir):
    # The move-to-device optimization is deliberately CUDA-only (see
    # LSTMDataset's docstring) -- MPS should behave exactly like the no-device
    # default, not attempt a .to(device) that MPS's slicing kernels may not
    # fully support.
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN, device=torch.device("mps"))
    assert ds.lstm_grid.device.type == "cpu"
    assert ds.target_grid.device.type == "cpu"


def test_lstm_dataset_cuda_device_moves_grids(snapshot_dir):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    ds = LSTMDataset(snapshot_dir, "val", seq_len=_SEQ_LEN, device=torch.device("cuda"))
    assert ds.lstm_grid.device.type == "cuda"
    assert ds.target_grid.device.type == "cuda"
    # Correctness must be unaffected by where the grid physically lives.
    x_seq, y = ds[0]
    assert x_seq.shape == (_SEQ_LEN, len(LSTM_FEATURE_COLS))
    assert x_seq.device.type == "cuda"
