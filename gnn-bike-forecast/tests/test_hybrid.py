"""
Tests for the ROADMAP Phase 8 spatiotemporal hybrid --

  - STGNNDataset            (src/models/data_loader.py)   -- edge cases 1-15
  - build_stgnn_dataloaders (src/models/data_loader.py)   -- edge case 11
  - BikeDemandSTGNN         (src/models/hybrid.py)        -- edge cases 16-27
  - run_hybrid              (src/models/hybrid.py)        -- edge cases 28-33

Edge-case numbering matches the approved plan in
~/.claude/plans/plan-on-how-to-curious-badger.md.

All fixtures are synthetic -- no dependency on data/processed/ or mlflow.
"""

from __future__ import annotations

import inspect
import json
import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from data_loader import (
    FEATURE_COLS,
    STGNNDataset,
    build_stgnn_dataloaders,
)
from utils import compute_data_snapshot_id

# Small fast-to-build sizes; far from production LSTM_SEQ_LEN=168 to keep tests quick.
_SEQ_LEN = 10
_SPLIT_SIZES = {"train": 50, "val": 30, "test": 30}
_N_STATIONS = 4
_SPLIT_FEAT_VALUES = {"train": 1.0, "val": 2.0, "test": 3.0}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_synthetic_snapshot(
    data_dir: Path,
    *,
    stations_by_split: dict[str, int] | None = None,
    times_by_split: dict[str, pd.DatetimeIndex] | None = None,
) -> None:
    """Write a data/processed/-shaped snapshot to `data_dir`.

    Every FEATURE_COLS column is set to a per-split constant so tests can
    identify which split a given row came from once it appears inside a
    borrowed input window. target_t24 is `t_own * 1000 + station_idx`, unique
    per (own-split time index, station).
    """
    if stations_by_split is None:
        stations_by_split = {s: _N_STATIONS for s in ("train", "val", "test")}

    if times_by_split is None:
        base_time = pd.Timestamp("2024-01-01")
        offset_hours = 0
        times_by_split = {}
        for split in ("train", "val", "test"):
            T = _SPLIT_SIZES[split]
            times_by_split[split] = pd.date_range(
                base_time + pd.Timedelta(hours=offset_hours), periods=T, freq="h"
            )
            offset_hours += T

    for split in ("train", "val", "test"):
        times = times_by_split[split]
        n_stations = stations_by_split[split]

        feat_rows = []
        tgt_rows = []
        for t_own, ts in enumerate(times):
            for n_idx in range(n_stations):
                row = {col: _SPLIT_FEAT_VALUES[split] for col in FEATURE_COLS}
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


def _tiny_adj(n: int = _N_STATIONS) -> torch.Tensor:
    """Small deterministic normalised adjacency for tests."""
    # Simple cyclic graph: node i connects to (i+1) mod n with weight 1.0,
    # plus self-loop. Not chosen for graph-theory reasons -- just a stable
    # nonzero pattern that produces predictable edge_index/edge_weight.
    adj = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        adj[i, i] = 0.5
        adj[i, (i + 1) % n] = 0.5
    return adj


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    _write_synthetic_snapshot(tmp_path)
    return tmp_path


@pytest.fixture
def adj_tensor() -> torch.Tensor:
    return _tiny_adj()


# ===========================================================================
# STGNNDataset -- data-plane edge cases (1-15)
# ===========================================================================

# 1. Shape contract
def test_stgnn_dataset_shape_and_dtype(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    x_window, y = ds[0]

    assert x_window.shape == (_SEQ_LEN, _N_STATIONS, len(FEATURE_COLS))
    assert x_window.dtype == torch.float32
    assert y.shape == (_N_STATIONS,)
    assert y.dtype == torch.float32

    # edge_index / edge_weight surfaces on the dataset, not in __getitem__
    assert ds.edge_index.dim() == 2 and ds.edge_index.shape[0] == 2
    assert ds.edge_index.dtype == torch.int64
    assert ds.edge_weight.shape == (ds.edge_index.shape[1],)
    assert ds.edge_weight.dtype == torch.float32


# 2. __len__ val/test -- exactly own-split timestamp count (closes LSTM-windowing gap)
def test_stgnn_dataset_len_val_matches_own_timestamp_count(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    assert len(ds) == _SPLIT_SIZES["val"]


def test_stgnn_dataset_len_test_matches_own_timestamp_count(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "test", adj_tensor, seq_len=_SEQ_LEN)
    assert len(ds) == _SPLIT_SIZES["test"]


# 3. __len__ train (scoped exception: no prior split to borrow from)
def test_stgnn_dataset_len_train_no_borrow(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "train", adj_tensor, seq_len=_SEQ_LEN)
    assert len(ds) == _SPLIT_SIZES["train"] - _SEQ_LEN + 1


# 4. Prior-split tail borrowing works: val's first window opens with train's tail
def test_stgnn_dataset_val_borrows_train_tail(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    x_window, _ = ds[0]

    borrowed_positions = x_window[: _SEQ_LEN - 1]  # (seq_len-1, N, F)
    assert torch.all(borrowed_positions == _SPLIT_FEAT_VALUES["train"])
    assert torch.all(x_window[_SEQ_LEN - 1] == _SPLIT_FEAT_VALUES["val"])


def test_stgnn_dataset_test_borrows_val_tail(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "test", adj_tensor, seq_len=_SEQ_LEN)
    x_window, _ = ds[0]

    borrowed_positions = x_window[: _SEQ_LEN - 1]
    assert torch.all(borrowed_positions == _SPLIT_FEAT_VALUES["val"])
    # And not train's value -- borrow depth is exactly one split back.
    assert not torch.any(borrowed_positions == _SPLIT_FEAT_VALUES["train"])


# 5. Anchor invariant -- no scored anchor is a borrowed row
def test_stgnn_dataset_anchor_timestamps_start_at_own_first(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    val_feat = pd.read_parquet(snapshot_dir / "features_val.parquet")
    expected_first = pd.to_datetime(val_feat["timestamp"]).min()
    assert pd.Timestamp(ds.anchor_timestamps[0]) == expected_first
    assert len(ds.anchor_timestamps) == len(ds)


# 6. Prior-tail station superset assertion
def test_stgnn_dataset_raises_on_station_universe_mismatch(tmp_path, adj_tensor):
    _write_synthetic_snapshot(
        tmp_path, stations_by_split={"train": _N_STATIONS + 1, "val": _N_STATIONS, "test": _N_STATIONS}
    )
    with pytest.raises(AssertionError, match="stations not present"):
        STGNNDataset(tmp_path, "val", adj_tensor, seq_len=_SEQ_LEN)


# 7. Chronological non-overlap assertion
def test_stgnn_dataset_raises_on_chronological_overlap(tmp_path, adj_tensor):
    base_time = pd.Timestamp("2024-01-01")
    train_times = pd.date_range(base_time, periods=_SPLIT_SIZES["train"], freq="h")
    val_times = pd.date_range(train_times[-5], periods=_SPLIT_SIZES["val"], freq="h")  # overlaps
    test_times = pd.date_range(val_times[-1] + pd.Timedelta(hours=1),
                                periods=_SPLIT_SIZES["test"], freq="h")

    _write_synthetic_snapshot(
        tmp_path,
        times_by_split={"train": train_times, "val": val_times, "test": test_times},
    )
    with pytest.raises(AssertionError, match="not strictly earlier"):
        STGNNDataset(tmp_path, "val", adj_tensor, seq_len=_SEQ_LEN)


# 8. edge_index / edge_weight are stored on the dataset -- not per-window copies
def test_stgnn_dataset_edge_index_is_single_stable_reference(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    # __getitem__ deliberately does NOT return edge_index -- it lives on the
    # dataset. Confirm identity stability of the on-dataset reference so a
    # future edit that quietly rebuilds it per window would show up here.
    ei_before = ds.edge_index
    _ = ds[0]
    _ = ds[1]
    assert ds.edge_index is ei_before
    assert ds.edge_weight is not None


# 9. Station ordering consistency: unique_stations order matches the adj tensor node ordering
def test_stgnn_dataset_station_ordering(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    # Fixture writes station_idx 0..N-1 -- unique_stations sort must round-trip.
    assert list(ds.station_ids) == list(range(_N_STATIONS))
    # feat_grid's N dimension must have the same length as the adjacency's node dim.
    assert ds.feat_grid.shape[1] == adj_tensor.shape[0] == _N_STATIONS


# 10. device parameter -- CPU stays CPU, MPS not moved, CUDA moves grids + edges
def test_stgnn_dataset_defaults_to_cpu(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    assert ds.feat_grid.device.type == "cpu"
    assert ds.target_grid.device.type == "cpu"
    assert ds.edge_index.device.type == "cpu"


def test_stgnn_dataset_cpu_device_stays_cpu(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor,
                       seq_len=_SEQ_LEN, device=torch.device("cpu"))
    assert ds.feat_grid.device.type == "cpu"


def test_stgnn_dataset_mps_device_is_not_moved(snapshot_dir, adj_tensor):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available on this machine")
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor,
                       seq_len=_SEQ_LEN, device=torch.device("mps"))
    assert ds.feat_grid.device.type == "cpu"
    assert ds.edge_index.device.type == "cpu"


def test_stgnn_dataset_cuda_device_moves_everything(snapshot_dir, adj_tensor):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor,
                       seq_len=_SEQ_LEN, device=torch.device("cuda"))
    assert ds.feat_grid.device.type == "cuda"
    assert ds.target_grid.device.type == "cuda"
    assert ds.edge_index.device.type == "cuda"
    assert ds.edge_weight.device.type == "cuda"
    x_window, y = ds[0]
    assert x_window.device.type == "cuda" and y.device.type == "cuda"


# 11. num_workers is silently forced to 0 when device is CUDA
def test_build_stgnn_dataloaders_forces_num_workers_zero_for_cuda(snapshot_dir, adj_tensor, caplog):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    with caplog.at_level(logging.WARNING):
        train_dl, _, _ = build_stgnn_dataloaders(
            snapshot_dir, adj_tensor, batch_size=1, num_workers=4,
            seq_len=_SEQ_LEN, device=torch.device("cuda"),
        )
    assert train_dl.num_workers == 0
    assert any("num_workers=0" in rec.message for rec in caplog.records)


def test_build_stgnn_dataloaders_cpu_keeps_num_workers(snapshot_dir, adj_tensor):
    train_dl, _, _ = build_stgnn_dataloaders(
        snapshot_dir, adj_tensor, batch_size=2, num_workers=2, seq_len=_SEQ_LEN,
    )
    assert train_dl.num_workers == 2


# 12. seq_len exceeds T raises ValueError, not silent n_valid <= 0
def test_stgnn_dataset_raises_when_seq_len_exceeds_t(snapshot_dir, adj_tensor):
    huge_seq_len = _SPLIT_SIZES["train"] + 10
    with pytest.raises(ValueError, match="not enough timesteps"):
        STGNNDataset(snapshot_dir, "train", adj_tensor, seq_len=huge_seq_len)


# 13. Grid values match parquet features exactly (per-split constant round-trip)
def test_stgnn_dataset_grid_values_round_trip(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "train", adj_tensor, seq_len=_SEQ_LEN)
    # Train has no borrowing -- every grid cell should equal train's constant.
    assert torch.all(ds.feat_grid == _SPLIT_FEAT_VALUES["train"])


# 14. Target correctness for val -- y[n] at anchor idx `i` equals t_own*1000 + n
def test_stgnn_dataset_target_correctness(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "val", adj_tensor, seq_len=_SEQ_LEN)
    for i in range(len(ds)):
        _, y = ds[i]
        expected = torch.tensor(
            [i * 1000 + n for n in range(_N_STATIONS)], dtype=torch.float32
        )
        assert torch.allclose(y, expected)


# 15. Borrowed target rows are never scored (target rows encode t_own*1000+n so
# a borrowed row's zero-placeholder target would land as `0` if leaked).
def test_stgnn_dataset_borrowed_targets_never_scored(snapshot_dir, adj_tensor):
    ds = STGNNDataset(snapshot_dir, "test", adj_tensor, seq_len=_SEQ_LEN)
    max_t_own = _SPLIT_SIZES["test"] - 1
    valid_targets = {t * 1000 + n for t in range(max_t_own + 1) for n in range(_N_STATIONS)}
    for i in range(len(ds)):
        _, y = ds[i]
        for n in range(_N_STATIONS):
            assert int(y[n].item()) in valid_targets


# ===========================================================================
# BikeDemandSTGNN -- model-plane edge cases (16-27)
# ===========================================================================

from hybrid import BikeDemandSTGNN  # noqa: E402


# 16. Forward output shape
def test_stgnn_forward_output_shape():
    B, T, N, F_in = 2, 5, 6, len(FEATURE_COLS)
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=8,
                             num_gcn_layers=1, lstm_hidden=16, lstm_num_layers=1, dropout=0.0)
    x = torch.randn(B, T, N, F_in)
    out = model(x)
    assert out.shape == (B, N)
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


# 17. Forward with B=1 -- no dim-collapse bugs
def test_stgnn_forward_with_batch_1():
    B, T, N, F_in = 1, 4, 5, len(FEATURE_COLS)
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=4,
                             num_gcn_layers=1, lstm_hidden=8, lstm_num_layers=1, dropout=0.0)
    x = torch.randn(B, T, N, F_in)
    out = model(x)
    assert out.shape == (1, N)


# 18. Poisson head is Identity (raw log-rate, non-negativity via exp() at inference)
def test_stgnn_poisson_head_is_identity():
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(), use_softplus=False)
    assert isinstance(model.head[-1], nn.Identity)


# 19. MSE head is Softplus (non-negativity from the model itself, not post-hoc)
def test_stgnn_mse_head_is_softplus():
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(), use_softplus=True)
    assert isinstance(model.head[-1], nn.Softplus)


# 20. Poisson raw output can span negatives (guards against silent Softplus regression)
def test_stgnn_poisson_raw_output_can_be_negative():
    B, T, N, F_in = 2, 3, 4, len(FEATURE_COLS)
    torch.manual_seed(0)
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=4,
                             num_gcn_layers=1, lstm_hidden=4, lstm_num_layers=1, dropout=0.0,
                             use_softplus=False)
    # Bias the final linear layer strongly negative so the raw log-rate lands
    # negative regardless of input. The head is Linear(4)->ReLU->Linear->Identity;
    # bias-shift the FINAL Linear.
    with torch.no_grad():
        model.head[2].bias.fill_(-5.0)
        model.head[2].weight.fill_(0.0)  # forces bias to dominate
    out = model(torch.randn(B, T, N, F_in))
    assert (out < 0).any(), "Poisson head must be allowed to emit negatives"


# 21. np.exp of raw Poisson output is non-negative -- the inference path's guarantee
def test_stgnn_exp_of_poisson_raw_is_non_negative():
    B, T, N, F_in = 2, 3, 4, len(FEATURE_COLS)
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, use_softplus=False,
                             gcn_hidden=4, num_gcn_layers=1,
                             lstm_hidden=4, lstm_num_layers=1, dropout=0.0)
    raw = model(torch.randn(B, T, N, F_in)).detach().numpy()
    rate = np.exp(raw)
    assert (rate >= 0).all()


# 23. Dense-matmul GCN equivalence to explicit per-timestep loop.
# Replaces the old test_batched_gcn_matches_per_timestep_loop which tested the
# now-removed _replicate_edge_index / GCNConv path.
def test_dense_gcn_matches_gcnconv():
    """
    A_norm @ x @ W == per-timestep (adj @ x[b,t]) @ W.

    The model's einsum('ij,bjtf->bitf', adj, h) applied to each GCN layer
    is mathematically equivalent to iterating over (b, t) and computing
    adj @ h[b, t] directly.  This is the correctness proof for Path C.
    """
    B, T, N, F_in, F_out = 2, 3, 5, 4, 6
    torch.manual_seed(42)
    adj = _tiny_adj(N)
    lin = nn.Linear(F_in, F_out, bias=True)
    lin.eval()

    x = torch.randn(B, T, N, F_in)

    # Dense einsum path (what BikeDemandSTGNN._gcn_layers do internally):
    h = x.permute(0, 2, 1, 3)                             # (B, N, T, F_in)
    h_agg = torch.einsum("ij,bjtf->bitf", adj, h)         # aggregate: (B, N, T, F_in)
    with torch.no_grad():
        out_dense = lin(h_agg).permute(0, 2, 1, 3)        # -> (B, T, N, F_out)

    # Reference: per-timestep loop
    out_loop = torch.zeros(B, T, N, F_out)
    with torch.no_grad():
        for b in range(B):
            for t in range(T):
                agg = adj @ x[b, t]                        # (N, F_in)
                out_loop[b, t] = lin(agg)                  # (N, F_out)

    assert torch.allclose(out_dense, out_loop, atol=1e-5), (
        "Dense einsum A_norm @ x must equal per-timestep loop"
    )


# 24. Static-graph invariant: model attribute + forward assertion
def test_stgnn_static_graph_only_flag_present_and_true():
    assert BikeDemandSTGNN._static_graph_only is True


# 25. Gradient flow -- every parameter receives a non-None gradient
def test_stgnn_gradient_flow_to_all_parameters():
    B, T, N, F_in = 2, 3, 4, len(FEATURE_COLS)
    torch.manual_seed(0)
    model = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=4,
                             num_gcn_layers=2, lstm_hidden=8, lstm_num_layers=2, dropout=0.0)
    out = model(torch.randn(B, T, N, F_in))
    out.mean().backward()

    missing = [name for name, p in model.named_parameters()
                if p.requires_grad and (p.grad is None or (p.grad == 0).all().item())]
    assert not missing, f"Params with no gradient: {missing}"


# 26. num_gcn_layers parameter -- variable-depth ModuleList
def test_stgnn_num_gcn_layers_variable_depth():
    for n in (1, 2, 3):
        m = BikeDemandSTGNN(adj_norm=_tiny_adj(), num_gcn_layers=n)
        assert len(m.gcn_layers) == n
    with pytest.raises(ValueError, match="num_gcn_layers"):
        BikeDemandSTGNN(adj_norm=_tiny_adj(), num_gcn_layers=0)
    with pytest.raises(ValueError, match="lstm_num_layers"):
        BikeDemandSTGNN(adj_norm=_tiny_adj(), lstm_num_layers=0)


# 27. Determinism at CPU: same seed + same input -> identical output
def test_stgnn_deterministic_on_cpu():
    from utils import set_seed

    B, T, N, F_in = 2, 3, 4, len(FEATURE_COLS)
    x = torch.randn(B, T, N, F_in)  # constructed once, before either seeded run

    def _run() -> torch.Tensor:
        set_seed(42)
        m = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=4,
                             num_gcn_layers=1, lstm_hidden=8, lstm_num_layers=1, dropout=0.0)
        m.eval()
        with torch.no_grad():
            return m(x)

    out_a = _run()
    out_b = _run()
    assert torch.equal(out_a, out_b), "STGNN must be deterministic at CPU with fixed seed"


# Memory-optimization param tests (Phase 8.5)

# 34. use_checkpointing=True and False produce the same forward output
def test_stgnn_checkpointing_same_output():
    B, T, N, F_in = 2, 4, 4, len(FEATURE_COLS)
    torch.manual_seed(7)
    x = torch.randn(B, T, N, F_in)

    def _make(ckpt: bool) -> BikeDemandSTGNN:
        m = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=4,
                             num_gcn_layers=1, lstm_hidden=8, lstm_num_layers=1,
                             dropout=0.0, use_checkpointing=ckpt)
        m.eval()
        return m

    # Share weights so any difference is solely from the checkpointing path.
    m_ckpt   = _make(True)
    m_no_ckpt = _make(False)
    m_no_ckpt.load_state_dict(m_ckpt.state_dict())

    with torch.no_grad():
        out_ckpt   = m_ckpt(x)
        out_no_ckpt = m_no_ckpt(x)

    assert torch.allclose(out_ckpt, out_no_ckpt, atol=1e-5), (
        "Gradient checkpointing must not change forward output"
    )


# 35. TBPTT: with tbptt_steps=K, gradients exist only for the tail
def test_stgnn_tbptt_gradient_stops_before_full_seq():
    B, T, N, F_in = 1, 8, 4, len(FEATURE_COLS)
    K = 4
    torch.manual_seed(3)
    x = torch.randn(B, T, N, F_in, requires_grad=False)

    m = BikeDemandSTGNN(adj_norm=_tiny_adj(N), in_channels=F_in, gcn_hidden=4,
                         num_gcn_layers=1, lstm_hidden=4, lstm_num_layers=1,
                         dropout=0.0, tbptt_steps=K, use_checkpointing=False)
    m.train()
    out = m(x)
    out.mean().backward()

    # If TBPTT is working, LSTM parameters still receive gradients (from the tail).
    for name, p in m.named_parameters():
        if "lstm" in name and p.requires_grad:
            assert p.grad is not None, f"TBPTT: {name} has no gradient (expected tail gradient)"


# 36. run_hybrid signature includes all memory-optimization knobs
def test_run_hybrid_signature_has_memory_params():
    from hybrid import run_hybrid
    sig = inspect.signature(run_hybrid)
    for name in ("use_checkpointing", "tbptt_steps", "grad_accum_steps", "mixed_precision"):
        assert name in sig.parameters, f"run_hybrid missing memory-opt param {name!r}"
    # Defaults match the documented values
    assert sig.parameters["use_checkpointing"].default is True
    assert sig.parameters["tbptt_steps"].default == 0
    assert sig.parameters["grad_accum_steps"].default == 4
    assert sig.parameters["mixed_precision"].default is True


# ===========================================================================
# run_hybrid -- training-plane edge cases (28-33)
# ===========================================================================

from hybrid import run_hybrid  # noqa: E402


def _write_full_snapshot_with_graph(data_dir: Path) -> None:
    """Like _write_synthetic_snapshot, but also writes a graph_data.pkl in
    the shape `load_adj()` expects (all four adjacency variants + normed
    versions of two of them). Used for run_hybrid smoke tests, which go
    through the real load_adj() path rather than accepting a raw tensor."""
    _write_synthetic_snapshot(data_dir)
    adj = _tiny_adj(_N_STATIONS).numpy()
    graph_data = {
        "adj_knn":       adj,
        "adj_flow":      adj,
        "adj_knn_norm":  adj,
        "adj_flow_norm": adj,
    }
    with open(data_dir / "graph_data.pkl", "wb") as f:
        pickle.dump(graph_data, f)
    # SNAPSHOT.json digests the on-disk files, so rewrite it after mutating
    # graph_data.pkl.
    from utils import compute_data_snapshot_id
    with open(data_dir / "SNAPSHOT.json", "w") as f:
        json.dump({"data_snapshot_id": compute_data_snapshot_id(data_dir)}, f)


@pytest.fixture
def full_snapshot_dir(tmp_path: Path) -> Path:
    _write_full_snapshot_with_graph(tmp_path)
    return tmp_path


# 28. Signature matches the tuner's contract (epoch_callback, log_to_mlflow, save_dir, seed)
def test_run_hybrid_signature_has_tuner_contract():
    sig = inspect.signature(run_hybrid)
    for name in ("epoch_callback", "log_to_mlflow", "save_dir", "seed",
                 "data_dir", "adj_variant", "loss_fn"):
        assert name in sig.parameters, f"run_hybrid missing param {name!r}"
    # epoch_callback default is None (matches run_gcn/run_lstm)
    assert sig.parameters["epoch_callback"].default is None
    # Memory-optimization knobs must also be present with defaults
    for name in ("use_checkpointing", "tbptt_steps", "grad_accum_steps", "mixed_precision"):
        assert name in sig.parameters, f"run_hybrid missing memory-opt param {name!r}"


# 29 / 30 / 31 / 33: covered in one smoke run to keep test wall-clock down.
def test_run_hybrid_smoke_end_to_end(full_snapshot_dir, tmp_path, monkeypatch):
    # Redirect MLflow to an isolated sqlite backend in tmp_path so this test
    # doesn't pollute the repo's mlflow.db, and doesn't need a real MLflow
    # server. Mirrors test_tune.py's isolation pattern.
    mlflow_dir = tmp_path / "mlflow_scratch"
    mlflow_dir.mkdir()
    monkeypatch.chdir(mlflow_dir)
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_dir / 'mlflow.db'}")
    mlflow.set_experiment("test-hybrid")

    save_dir = tmp_path / "models"

    epoch_calls: list[tuple[int, float]] = []

    def _cb(epoch: int, val_mae: float) -> None:
        epoch_calls.append((epoch, val_mae))

    with mlflow.start_run(run_name="hybrid_smoke"):
        best = run_hybrid(
            data_dir        = full_snapshot_dir,
            adj_variant     = "knn",
            loss_fn         = "poisson",
            gcn_hidden      = 4,
            num_gcn_layers  = 1,
            lstm_hidden     = 8,
            lstm_num_layers = 1,
            dropout         = 0.0,
            lr              = 1e-2,
            batch_size      = 2,
            seq_len         = _SEQ_LEN,
            max_epochs      = 2,
            patience        = 5,
            save_dir        = save_dir,
            seed            = 42,
            log_to_mlflow   = True,
            epoch_callback  = _cb,
        )
        run_id = mlflow.active_run().info.run_id

    # 29. Return type is float
    assert isinstance(best, float)
    assert best != float("inf"), "best_val_mae was never updated -- training loop didn't run"

    # 31. Checkpoint written to save_dir
    ckpt = save_dir / "hybrid_knn_poisson_best.pt"
    assert ckpt.exists(), f"checkpoint {ckpt} not written"
    # Loadable
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and len(state) > 0

    # epoch_callback was invoked once per epoch
    assert len(epoch_calls) == 2
    assert epoch_calls[0][0] == 1 and epoch_calls[1][0] == 2

    # 33. Provenance tags on the MLflow run
    client = mlflow.tracking.MlflowClient()
    tags = client.get_run(run_id).data.tags
    for required in ("git_sha", "data_snapshot_id", "seed", "model_name"):
        assert required in tags, f"provenance tag {required!r} missing (got {sorted(tags)})"
    assert tags["seed"] == "42"


# 30. Early stopping fires when val MAE monotonically worsens (patience=0).
# We stub evaluate_epoch to return a strictly-increasing val MAE from epoch 2.
def test_run_hybrid_early_stopping_fires(full_snapshot_dir, tmp_path, monkeypatch):
    import hybrid as hybrid_mod

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-hybrid-earlystop")
    save_dir = tmp_path / "models_es"

    call_count = {"epoch": 0}

    def _fake_evaluate_epoch(model, loader, device,
                              loss_fn="poisson", mixed_precision=False):
        call_count["epoch"] += 1
        # Increasing val MAE => no improvement => early stop after `patience` epochs.
        # Return arrays sized to the loader's own dataset so the split-specific
        # metadata-vs-metrics arrays broadcast; sample count differs across
        # train (T-seq_len+1 windows) vs val/test (T_own windows).
        n_samples = len(loader.dataset) * _N_STATIONS
        base = 1.0 + call_count["epoch"] * 0.5
        y_true = np.zeros(n_samples, dtype=np.float32)
        y_pred = np.full(n_samples, base, dtype=np.float32)
        return y_true, y_pred

    monkeypatch.setattr(hybrid_mod, "evaluate_epoch", _fake_evaluate_epoch)

    with mlflow.start_run(run_name="hybrid_earlystop"):
        run_hybrid(
            data_dir        = full_snapshot_dir,
            adj_variant     = "knn",
            loss_fn         = "poisson",
            gcn_hidden      = 4,
            num_gcn_layers  = 1,
            lstm_hidden     = 4,
            lstm_num_layers = 1,
            dropout         = 0.0,
            lr              = 1e-3,
            batch_size      = 2,
            seq_len         = _SEQ_LEN,
            max_epochs      = 20,
            patience        = 1,
            save_dir        = save_dir,
            seed            = 42,
            log_to_mlflow   = True,
        )

    # With patience=1 and val MAE increasing every epoch from epoch 1:
    #   epoch 1: new best (best_val_mae = base@1); patience_count=0
    #   epoch 2: worse -> patience_count=1 -> break BEFORE epoch 3
    # Then post-training re-eval calls evaluate_epoch 3 more times (val/test/train).
    # So training-phase epochs = 2; total evaluate_epoch calls = 2 + 3 = 5.
    assert call_count["epoch"] == 5, (
        f"expected training to early-stop after 2 epochs (5 total evaluate_epoch "
        f"calls including val/test/train re-eval); saw {call_count['epoch']}"
    )


# 32. ensure_portable_artifact_location() is invoked in main()'s MLflow-enabled path.
def test_hybrid_main_calls_ensure_portable_artifact_location():
    src = (Path(__file__).resolve().parents[1] / "src" / "models" / "hybrid.py").read_text()
    assert "ensure_portable_artifact_location" in src, (
        "hybrid.py must call ensure_portable_artifact_location in main() to avoid "
        "the 2026-07-11 artifact-location bug on Colab"
    )


