"""
hybrid.py
---------
ROADMAP Phase 8 spatiotemporal hybrid (BikeDemandSTGNN) --

    Input window: (B, seq_len=168, N=1911, F=22)

    Spatial encoding (dense matmul per timestep -- Path C):
        permute         -> (B, N, T, F)          [one copy upfront, avoids copy later]
        for each GCN layer:
            h_agg = einsum('ij,bjtf->bitf', adj_norm, h)   # A_norm @ h
            h     = relu(Linear(h_agg))          # (B, N, T, h_g)

        Why dense matmul instead of GCNConv + edge_index replication:
          GCNConv requires tiling edge_index B*T times (672 copies at B=4, T=168).
          For flow/combined (E≈900K), that is ~10 GB of edge indices alone -- more
          than a T4's VRAM.  At N=1911, adj_norm is (1911,1911) float32 = 14.6 MB;
          a standard matmul is both cheaper and faster than sparse scatter/gather.
          See log entry 2026-07-19 for the full analysis.

        [Static-graph invariant preserved: adj_norm is a registered buffer, fixed
        for the run. If Phase 10's time-of-day adjacency lands, forward() must
        be updated to accept a per-timestep adj and loop over T; the _static_graph_only
        flag guards against silent regressions.]

    Temporal encoding (per-node LSTM, memory optimizations):
        reshape         -> (B*N, T, h_g)    [view, no copy -- h is contiguous]
        TBPTT           -> run first T-K steps under no_grad (dropped from backward)
        Gradient ckpt   -> only K-step tail is checkpointed; saves ~4 GB backward bufs
        LSTM(h_g -> h_lstm, num_layers=2, dropout=0.3)
        take last hidden-> (B*N, h_lstm)
        reshape         -> (B, N, h_lstm)

    Mixed precision + gradient accumulation (in train_epoch):
        torch.autocast("cuda") + GradScaler when mixed_precision=True (CUDA only)
        optimizer.step() every grad_accum_steps batches; effective batch = B*accum

    Prediction head:
        Linear(h_lstm, 32) -> ReLU -> Linear(32, 1) -> {Identity | Softplus}
        (Softplus for --loss-fn mse; Identity for --loss-fn poisson)

Six configs mirror standalone GCN:
    hybrid_{knn,flow,combined}_{mse,poisson}

Training runs on Colab (see notebooks/train_colab.ipynb Section C).

Memory optimizations (Phase 8.5):
  - Path C:            edge_index replication eliminated; adj_norm buffer = 14.6 MB
  - Gradient ckpt:     LSTM backward buffers reduced from ~4 GB to near-zero
  - TBPTT (K=48):      backward only through last K of 168 steps; 3.7x fewer buf rows
  - Mixed precision:   fp16 on CUDA halves activation memory
  - Grad accumulation: effective B=4 with B_mem=1 per step

Usage
-----
  python src/models/hybrid.py --data-dir data/processed --adj knn
  python src/models/hybrid.py --data-dir data/processed --adj flow --loss-fn mse
  python src/models/hybrid.py --data-dir data/processed --adj all --loss-fn poisson
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Literal

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import (
    FEATURE_COLS,
    build_stgnn_dataloaders,
    load_adj,
)
from evaluate import compute_metrics, full_evaluation, log_metrics_to_mlflow
from mlflow_utils import (
    ensure_portable_artifact_location,
    log_segment_artifacts_to_mlflow,
    tag_run_provenance,
)
from utils import get_device, release_host_memory, set_seed

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "bike-demand-forecasting"
ADJ_VARIANTS    = ("knn", "flow", "combined")
DEFAULT_SEQ_LEN = 168


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BikeDemandSTGNN(nn.Module):
    """
    Spatiotemporal hybrid: dense-matmul GCN encoder per timestep, then per-node
    LSTM with gradient checkpointing and TBPTT, then FC head.

    Parameters
    ----------
    adj_norm          : (N, N) row-normalised adjacency tensor. Registered as a
                        buffer (moves with .to(device), not updated by optimizer).
    in_channels       : node feature dim, driven by len(FEATURE_COLS).
    gcn_hidden        : GCN encoder output dim per timestep per node.
    num_gcn_layers    : depth of the spatial encoder (default 1).
    lstm_hidden       : LSTM hidden size (temporal encoder).
    lstm_num_layers   : LSTM depth (default 2).
    dropout           : dropout after each GCN layer except the last, and
                        inside LSTM (only effective when lstm_num_layers > 1).
    use_softplus      : if True, head ends in Softplus (mse loss); if False,
                        head is Identity (poisson -- raw log-rate, inference via exp()).
    use_checkpointing : wrap LSTM call in torch.utils.checkpoint to avoid saving
                        all T gate activations; recomputes forward on backward pass.
    tbptt_steps       : backpropagate only through the last K timesteps of the LSTM
                        sequence; 0 = full BPTT. Composed with use_checkpointing.

    Static-graph invariant: `_static_graph_only = True` -- forward() uses a single
    adj_norm for all seq_len timesteps and all windows in the batch. Switching to
    time-varying edges (Phase 10) requires changing forward() and clearing this flag.
    """

    _static_graph_only: bool = True

    def __init__(
        self,
        adj_norm:         torch.Tensor,
        in_channels:      int   = len(FEATURE_COLS),
        gcn_hidden:       int   = 32,
        num_gcn_layers:   int   = 1,
        lstm_hidden:      int   = 64,
        lstm_num_layers:  int   = 2,
        dropout:          float = 0.3,
        use_softplus:     bool  = False,
        use_checkpointing: bool = True,
        tbptt_steps:      int   = 0,
    ) -> None:
        super().__init__()

        if num_gcn_layers < 1:
            raise ValueError(f"num_gcn_layers must be >= 1, got {num_gcn_layers}")
        if lstm_num_layers < 1:
            raise ValueError(f"lstm_num_layers must be >= 1, got {lstm_num_layers}")
        if tbptt_steps < 0:
            raise ValueError(f"tbptt_steps must be >= 0, got {tbptt_steps}")

        # adj_norm: registered buffer -- moves with .to(device), never updated.
        # Dense (N, N) float32 = 14.6 MB at N=1911, vs ~10 GB replicated edge_index
        # for flow/combined adjacency (see docstring).
        self.register_buffer("adj_norm", adj_norm.float())

        # Dense linear layers replace GCNConv -- A_norm @ X @ W is mathematically
        # equivalent to GCNConv(X, edge_index) when adj_norm already encodes the
        # normalised adjacency (proved in test_dense_gcn_matches_gcnconv).
        self.gcn_layers = nn.ModuleList(
            [
                nn.Linear(in_channels if i == 0 else gcn_hidden, gcn_hidden)
                for i in range(num_gcn_layers)
            ]
        )
        self.dropout          = dropout
        self.use_checkpointing = use_checkpointing
        self.tbptt_steps      = tbptt_steps

        self.lstm = nn.LSTM(
            input_size  = gcn_hidden,
            hidden_size = lstm_hidden,
            num_layers  = lstm_num_layers,
            batch_first = True,
            dropout     = dropout if lstm_num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() if use_softplus else nn.Identity(),
        )

    # -- LSTM helpers for gradient checkpointing --------------------------------

    def _lstm_step(self, seqs: torch.Tensor) -> torch.Tensor:
        """Run full LSTM sequence; extracted for torch.utils.checkpoint."""
        out, _ = self.lstm(seqs)
        return out

    def _lstm_step_with_state(
        self,
        seqs: torch.Tensor,
        h_n:  torch.Tensor,
        c_n:  torch.Tensor,
    ) -> torch.Tensor:
        """Run LSTM from a given hidden state; used with TBPTT tail."""
        out, _ = self.lstm(seqs, (h_n, c_n))
        return out

    # -- Forward ----------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, N, F)
        assert self._static_graph_only, "STGNN currently supports only a static graph"

        B, T, N, _ = x.shape

        # --- Spatial encoding in (B, N, T, F) layout --------------------------
        # Permute once upfront so GCN output is already in LSTM-ready layout,
        # avoiding a second permute+contiguous copy before the reshape.
        h = x.permute(0, 2, 1, 3).contiguous()   # (B, N, T, F)

        for i, lin in enumerate(self.gcn_layers):
            # A_norm @ h: aggregate each node's features from its neighbours.
            # adj_norm (N, N) @ h (B, N, T, F) via einsum:
            #   output[b, i, t, f] = sum_j A[i,j] * h[b, j, t, f]
            h_agg = torch.einsum("ij,bjtf->bitf", self.adj_norm, h)   # (B, N, T, h_g)
            h = lin(h_agg)                                              # (B, N, T, h_g)
            h = torch.relu(h)
            if i < len(self.gcn_layers) - 1:
                h = F_nn.dropout(h, p=self.dropout, training=self.training)

        h_g = h.shape[-1]

        # --- Temporal encoding: per-node LSTM ---------------------------------
        # h is contiguous (B, N, T, h_g) so reshape is a view, zero copy.
        seqs = h.reshape(B * N, T, h_g)                               # (B*N, T, h_g)

        # TBPTT: run first T-K steps under no_grad (dropped from backward graph).
        # Gradient checkpointing on the tail: avoids saving all K*h_l activations.
        K = self.tbptt_steps
        if K > 0 and T > K:
            with torch.no_grad():
                _, (h_n, c_n) = self.lstm(seqs[:, : T - K, :])
            h_n, c_n = h_n.detach(), c_n.detach()
            tail = seqs[:, T - K :, :]
            if self.use_checkpointing:
                lstm_out = torch_checkpoint(
                    self._lstm_step_with_state, tail, h_n, c_n, use_reentrant=False
                )
            else:
                lstm_out, _ = self.lstm(tail, (h_n, c_n))
        else:
            if self.use_checkpointing:
                lstm_out = torch_checkpoint(
                    self._lstm_step, seqs, use_reentrant=False
                )
            else:
                lstm_out, _ = self.lstm(seqs)

        last = lstm_out[:, -1, :]                   # (B*N, h_lstm)

        # --- Head ---
        pred = self.head(last).squeeze(-1)           # (B*N,)
        return pred.reshape(B, N)                    # (B, N)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _amp_ctx(use_amp: bool) -> object:
    """Return an autocast context (CUDA fp16) or a nullcontext."""
    return torch.autocast("cuda") if use_amp else nullcontext()


def train_epoch(
    model:             BikeDemandSTGNN,
    loader:            DataLoader,
    criterion:         nn.Module,
    optimiser:         torch.optim.Optimizer,
    device:            torch.device,
    grad_accum_steps:  int  = 1,
    scaler:            torch.amp.GradScaler | None = None,
    mixed_precision:   bool = False,
) -> float:
    model.train()
    optimiser.zero_grad()
    total_loss = 0.0
    n_batches  = len(loader)
    use_amp    = mixed_precision and device.type == "cuda"

    pbar = tqdm(enumerate(loader), total=n_batches, desc="  train", leave=False, unit="batch")
    for step, (x_batch, y_batch) in pbar:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        with _amp_ctx(use_amp):
            pred = model(x_batch)
            loss = criterion(pred, y_batch) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        is_update = (step + 1) % grad_accum_steps == 0 or (step + 1) == n_batches
        if is_update:
            if scaler is not None:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
            optimiser.zero_grad()

        batch_loss = loss.item() * grad_accum_steps
        total_loss += batch_loss
        pbar.set_postfix(loss=f"{batch_loss:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model:           BikeDemandSTGNN,
    loader:          DataLoader,
    device:          torch.device,
    loss_fn:         str  = "poisson",
    mixed_precision: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred) flattened over all windows × stations.

    For --loss-fn poisson: raw logits -> rates via np.exp().
    For --loss-fn mse: Softplus head already emits non-negative rates.
    Never np.clip (CLAUDE.md Loss/output contract).
    """
    model.eval()
    use_amp   = mixed_precision and device.type == "cuda"
    all_true  = []
    all_pred  = []

    for x_batch, y_batch in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        x_batch = x_batch.to(device, non_blocking=True)

        with _amp_ctx(use_amp):
            pred = model(x_batch)

        all_pred.append(pred.cpu().numpy().reshape(-1))
        all_true.append(y_batch.cpu().numpy().reshape(-1))

    y_true     = np.concatenate(all_true)
    y_pred_raw = np.concatenate(all_pred)

    return y_true, (np.exp(y_pred_raw) if loss_fn == "poisson" else y_pred_raw)


# ---------------------------------------------------------------------------
# Metadata helper -- aligned to STGNNDataset's anchor timestamps + station grid
# ---------------------------------------------------------------------------

def _build_stgnn_metadata(
    data_dir:          Path,
    split:             str,
    anchor_timestamps: np.ndarray,
    station_ids:       np.ndarray,
) -> dict:
    """
    Flat (station_idx, timestamps, lag_24, cluster) arrays aligned to
    evaluate_epoch's output order.

    evaluate_epoch flattens each batch's (B, N) prediction row-major, and
    the loader is unshuffled for val/test, so anchor `i` contributes N rows
    with station_idx = station_ids[0..N-1] at timestamps[i].
    """
    import pandas as pd

    from evaluate import reconstruct_cluster_id

    feat_df = pd.read_parquet(data_dir / f"features_{split}.parquet")

    T = len(anchor_timestamps)
    N = len(station_ids)

    station_idx_out = np.tile(station_ids,         T)   # (T*N,)
    timestamps_out  = np.repeat(anchor_timestamps, N)   # (T*N,)

    idx = pd.MultiIndex.from_product(
        [anchor_timestamps, station_ids], names=["timestamp", "station_idx"]
    )
    lookup = (
        feat_df.set_index(["timestamp", "station_idx"])
        [["lag_24", "cluster_0", "cluster_1", "cluster_2"]]
        .reindex(idx)
    )
    lag24_out   = lookup["lag_24"].values
    cluster_out = reconstruct_cluster_id(
        lookup[["cluster_0", "cluster_1", "cluster_2"]].values
    )

    assert len(lag24_out) == T * N, "lag24/cluster lookup length mismatch"

    return {
        "station_idx": station_idx_out.astype(np.int64),
        "timestamps":  pd.to_datetime(timestamps_out),
        "lag24":       lag24_out,
        "cluster":     cluster_out,
    }


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def run_hybrid(
    data_dir:          Path,
    adj_variant:       Literal["knn", "flow", "combined"] = "knn",
    loss_fn:           Literal["mse", "poisson"] = "poisson",
    gcn_hidden:        int   = 32,
    num_gcn_layers:    int   = 1,
    lstm_hidden:       int   = 64,
    lstm_num_layers:   int   = 2,
    dropout:           float = 0.3,
    lr:                float = 1e-3,
    batch_size:        int   = 4,
    seq_len:           int   = DEFAULT_SEQ_LEN,
    max_epochs:        int   = 50,
    patience:          int   = 5,
    save_dir:          Path  = Path("models"),
    seed:              int   = 42,
    log_to_mlflow:     bool  = True,
    epoch_callback:    Callable[[int, float], None] | None = None,
    # Memory-optimization knobs (Phase 8.5)
    use_checkpointing: bool  = True,
    tbptt_steps:       int   = 0,
    grad_accum_steps:  int   = 4,
    mixed_precision:   bool  = True,
) -> float:
    """
    Train BikeDemandSTGNN and log results to MLflow. Returns best_val_mae.

    `epoch_callback(epoch, val_mae)` is invoked at the end of every epoch
    for optional Optuna trial pruning (see src/models/tune.py). This module
    has no dependency on Optuna; the caller supplies the hook.

    Run lifecycle (mlflow.set_experiment / start_run / end_run) is the
    caller's responsibility -- either main() below (top-level run) or the
    tuner (nested inside mlflow.start_run(nested=True)).

    Memory-optimization knobs:
      use_checkpointing : gradient checkpointing on LSTM (default True)
      tbptt_steps       : backprop through last K steps only (0 = full BPTT)
      grad_accum_steps  : optimizer step every N batches (effective B = batch_size * N)
      mixed_precision   : fp16 autocast + GradScaler on CUDA (default True)
    """
    set_seed(seed)

    run_name = f"hybrid_{adj_variant}_{loss_fn}"
    device   = get_device()
    logger.info("Running %s | device=%s", run_name, device)

    # --- Adjacency ---
    adj_knn, adj_flow, adj_combined = load_adj(data_dir)
    adj_map    = {"knn": adj_knn, "flow": adj_flow, "combined": adj_combined}
    adj_tensor = adj_map[adj_variant]

    # --- Datasets & loaders ---
    logger.info(
        "Building STGNN dataloaders (batch_size=%d, seq_len=%d)...", batch_size, seq_len
    )
    train_loader, val_loader, test_loader = build_stgnn_dataloaders(
        data_dir, adj_tensor,
        batch_size  = batch_size,
        num_workers = 0,
        seq_len     = seq_len,
        device      = device,
    )

    # --- Metadata ---
    train_meta = _build_stgnn_metadata(
        data_dir, "train",
        train_loader.dataset.anchor_timestamps,
        train_loader.dataset.station_ids,
    )
    val_meta = _build_stgnn_metadata(
        data_dir, "val",
        val_loader.dataset.anchor_timestamps,
        val_loader.dataset.station_ids,
    )
    test_meta = _build_stgnn_metadata(
        data_dir, "test",
        test_loader.dataset.anchor_timestamps,
        test_loader.dataset.station_ids,
    )
    train_lag24 = train_meta["lag24"]

    # --- Model ---
    in_channels = len(FEATURE_COLS)
    model = BikeDemandSTGNN(
        adj_norm         = adj_tensor.to(device),
        in_channels      = in_channels,
        gcn_hidden       = gcn_hidden,
        num_gcn_layers   = num_gcn_layers,
        lstm_hidden      = lstm_hidden,
        lstm_num_layers  = lstm_num_layers,
        dropout          = dropout,
        use_softplus     = loss_fn != "poisson",
        use_checkpointing = use_checkpointing,
        tbptt_steps      = tbptt_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s params: %d", run_name, n_params)

    criterion = nn.PoissonNLLLoss(log_input=True) if loss_fn == "poisson" else nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    # Mixed precision: CUDA only.  GradScaler is None on MPS/CPU (no-op path).
    use_amp = mixed_precision and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Early stopping state ---
    best_val_mae   = float("inf")
    best_epoch     = 0
    patience_count = 0
    save_dir       = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = save_dir / f"{run_name}_best.pt"

    # --- MLflow provenance ---
    if log_to_mlflow:
        mlflow.set_tag("model_name", run_name)
        mlflow.set_tag("phase", "hybrid")
        tag_run_provenance(data_dir, seed)
        mlflow.log_params({
            "model_type":        "hybrid",
            "adj_variant":       adj_variant,
            "loss_fn":           loss_fn,
            "gcn_hidden":        gcn_hidden,
            "num_gcn_layers":    num_gcn_layers,
            "lstm_hidden":       lstm_hidden,
            "lstm_num_layers":   lstm_num_layers,
            "dropout":           dropout,
            "lr":                lr,
            "batch_size":        batch_size,
            "grad_accum_steps":  grad_accum_steps,
            "seq_len":           seq_len,
            "max_epochs":        max_epochs,
            "patience":          patience,
            "n_params":          n_params,
            "in_channels":       in_channels,
            "device":            str(device),
            "seed":              seed,
            "use_checkpointing": use_checkpointing,
            "tbptt_steps":       tbptt_steps,
            "mixed_precision":   mixed_precision,
        })

    # --- Training loop ---
    logger.info(
        "Starting training | max_epochs=%d | patience=%d | grad_accum=%d | "
        "checkpointing=%s | tbptt=%d | amp=%s",
        max_epochs, patience, grad_accum_steps, use_checkpointing, tbptt_steps, use_amp,
    )
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimiser, device,
            grad_accum_steps = grad_accum_steps,
            scaler           = scaler,
            mixed_precision  = mixed_precision,
        )
        y_true, y_pred = evaluate_epoch(
            model, val_loader, device, loss_fn=loss_fn, mixed_precision=mixed_precision
        )
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
                    "train_loss": train_loss,
                    "val_mae":    val_mae,
                    "val_rmse":   val_metrics.rmse,
                    "val_mape":   val_metrics.mape,
                },
                step=epoch,
            )

        if epoch_callback is not None:
            epoch_callback(epoch, val_mae)

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

    # --- Full evaluation: val, test, train (log-and-release to keep RAM low) ---
    logger.info("Loading best checkpoint (epoch %d)...", best_epoch)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    split_configs = [
        ("val",   val_loader,   val_meta),
        ("test",  test_loader,  test_meta),
        ("train", train_loader, train_meta),
    ]
    metrics_summary: dict[str, dict[str, float]] = {}
    for split, loader, meta in split_configs:
        y_true, y_pred = evaluate_epoch(
            model, loader, device, loss_fn=loss_fn, mixed_precision=mixed_precision
        )
        r = full_evaluation(
            y_true      = y_true,
            y_pred      = y_pred,
            station_idx = meta["station_idx"],
            timestamps  = meta["timestamps"],
            model_name  = run_name,
            split       = split,
            cluster     = meta["cluster"],
            train_lag24 = train_lag24,
            lag24       = meta["lag24"],
        )
        if log_to_mlflow:
            log_metrics_to_mlflow(r, prefix="best_")
            log_segment_artifacts_to_mlflow(r, run_name)

        metrics_summary[split] = {
            "mae":  r.metrics.mae,
            "rmse": r.metrics.rmse,
            "mape": r.metrics.mape,
        }
        del y_true, y_pred, r
        release_host_memory()

    if log_to_mlflow:
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_artifact(str(ckpt_path), artifact_path="model")

    print(
        f"\n{run_name} | train MAE={metrics_summary['train']['mae']:.4f}  "
        f"RMSE={metrics_summary['train']['rmse']:.4f}  "
        f"MAPE={metrics_summary['train']['mape']:.2f}%"
    )
    print(
        f"{run_name} | val  MAE={metrics_summary['val']['mae']:.4f}  "
        f"RMSE={metrics_summary['val']['rmse']:.4f}  "
        f"MAPE={metrics_summary['val']['mape']:.2f}%  "
        f"(best epoch={best_epoch})"
    )
    print(
        f"{run_name} | test MAE={metrics_summary['test']['mae']:.4f}  "
        f"RMSE={metrics_summary['test']['rmse']:.4f}  "
        f"MAPE={metrics_summary['test']['mape']:.2f}%"
    )

    return best_val_mae


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train BikeDemandSTGNN (Phase 8 hybrid)")
    parser.add_argument("--data-dir",          type=Path,  default=Path("data/processed"))
    parser.add_argument(
        "--adj", choices=["knn", "flow", "combined", "all"], default="knn",
        help="Adjacency variant (default: knn). 'all' loops over knn/flow/combined."
    )
    parser.add_argument("--loss-fn",           choices=["mse", "poisson"], default="poisson")
    parser.add_argument("--gcn-hidden",        type=int,   default=32)
    parser.add_argument("--num-gcn-layers",    type=int,   default=1)
    parser.add_argument("--lstm-hidden",       type=int,   default=64)
    parser.add_argument("--lstm-num-layers",   type=int,   default=2)
    parser.add_argument("--dropout",           type=float, default=0.3)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--seq-len",           type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max-epochs",        type=int,   default=50)
    parser.add_argument("--patience",          type=int,   default=5)
    parser.add_argument("--save-dir",          type=Path,  default=Path("models"))
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--no-mlflow",         action="store_true")
    # Memory-optimization knobs
    parser.add_argument("--grad-accum-steps",  type=int,   default=4)
    parser.add_argument("--tbptt-steps",       type=int,   default=0,
                        help="Backprop through last K LSTM steps (0 = full BPTT)")
    parser.add_argument("--no-checkpointing",  action="store_true",
                        help="Disable gradient checkpointing on LSTM")
    parser.add_argument("--no-mixed-precision",action="store_true",
                        help="Disable fp16 autocast (CUDA only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    variants = ADJ_VARIANTS if args.adj == "all" else (args.adj,)

    if not args.no_mlflow:
        ensure_portable_artifact_location(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

    for variant in variants:
        run_kwargs = dict(
            data_dir          = args.data_dir,
            adj_variant       = variant,
            loss_fn           = args.loss_fn,
            gcn_hidden        = args.gcn_hidden,
            num_gcn_layers    = args.num_gcn_layers,
            lstm_hidden       = args.lstm_hidden,
            lstm_num_layers   = args.lstm_num_layers,
            dropout           = args.dropout,
            lr                = args.lr,
            batch_size        = args.batch_size,
            seq_len           = args.seq_len,
            max_epochs        = args.max_epochs,
            patience          = args.patience,
            save_dir          = args.save_dir,
            seed              = args.seed,
            log_to_mlflow     = not args.no_mlflow,
            use_checkpointing = not args.no_checkpointing,
            tbptt_steps       = args.tbptt_steps,
            grad_accum_steps  = args.grad_accum_steps,
            mixed_precision   = not args.no_mixed_precision,
        )
        if args.no_mlflow:
            run_hybrid(**run_kwargs)
        else:
            with mlflow.start_run(run_name=f"hybrid_{variant}_{args.loss_fn}"):
                run_hybrid(**run_kwargs)


if __name__ == "__main__":
    main()
