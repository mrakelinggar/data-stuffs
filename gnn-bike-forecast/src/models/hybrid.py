"""
hybrid.py
---------
ROADMAP Phase 8 spatiotemporal hybrid (BikeDemandSTGNN) --

    Input window: (B, seq_len=168, N=1911, F=22)

    Spatial encoding (one batched GCN call, not a per-timestep loop):
        reshape         -> (B*seq_len*N, F)
        GCNConv(F -> h_g), with a replicated edge_index
            (single-graph edge_index tiled with +i*N node offsets for
             i in [0, B*seq_len), producing (2, B*seq_len*E) edges)
        ReLU (+ dropout after all but the last GCN layer)
        reshape         -> (B, seq_len, N, h_g)

        [Only correct because the graph is STATIC for the run -- same adj
        for all seq_len timesteps and every window. If Phase 10's time-of-
        day adjacency ever lands, this batched call must revert to a per-
        timestep loop over the distinct edge sets. Guarded by the
        _static_graph_only assertion in forward().]

    Temporal encoding (per-node LSTM):
        permute         -> (B*N, seq_len, h_g)
        LSTM(h_g -> h_lstm, num_layers, dropout)
        take last hidden-> (B*N, h_lstm)
        reshape         -> (B, N, h_lstm)

    Prediction head:
        Linear(h_lstm, 32) -> ReLU -> Linear(32, 1) -> {Identity | Softplus}
        (Softplus for --loss-fn mse; Identity for --loss-fn poisson,
         inference recovers rate via np.exp() on the raw log-rate)

Six configs mirror standalone GCN:
    hybrid_{knn,flow,combined}_{mse,poisson}

Training runs on Colab (see notebooks/train_colab.ipynb Section C). Local
smoke runs (`--max-epochs 1`) exercise the plumbing end to end.

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
from pathlib import Path
from typing import Callable, Literal

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
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

def _replicate_edge_index(
    edge_index:  torch.Tensor,   # (2, E)
    edge_weight: torch.Tensor,   # (E,)
    n_subgraphs: int,
    n_nodes:     int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tile a single graph's (edge_index, edge_weight) across `n_subgraphs`
    disjoint copies with node offsets +i*n_nodes for i in [0, n_subgraphs).

    Correctness: for each i, the sub-block edge_index[:, i*E:(i+1)*E] holds
    edges [src + i*N, dst + i*N] -- so a single GCNConv call on the
    concatenated (n_subgraphs*N, F) node tensor computes exactly the same
    per-sub-block outputs as running GCNConv independently on each sub-block
    with the original edge_index (verified by
    test_batched_gcn_matches_per_timestep_loop).
    """
    device  = edge_index.device
    E       = edge_index.shape[1]
    offsets = (torch.arange(n_subgraphs, device=device) * n_nodes).view(1, -1, 1)  # (1, K, 1)

    ei_rep  = edge_index.unsqueeze(1) + offsets                # (2, K, E)
    ei_rep  = ei_rep.reshape(2, n_subgraphs * E).contiguous()  # (2, K*E)
    ew_rep  = edge_weight.repeat(n_subgraphs)                  # (K*E,)
    return ei_rep, ew_rep


class BikeDemandSTGNN(nn.Module):
    """
    Spatiotemporal hybrid: batched-GCN encoder per timestep, then per-node
    LSTM, then FC head.

    Parameters
    ----------
    in_channels       : node feature dim, driven by len(FEATURE_COLS)
    gcn_hidden        : GCN encoder output dim per timestep per node
    num_gcn_layers    : depth of the spatial encoder (default 1)
    lstm_hidden       : LSTM hidden size (temporal encoder)
    lstm_num_layers   : LSTM depth (default 2)
    dropout           : dropout after each GCN layer except the last, and
                        inside the LSTM (only effective when
                        lstm_num_layers > 1, per PyTorch's LSTM contract).
    use_softplus      : if True, head ends in Softplus (mse loss); if False,
                        head is Identity (poisson loss -- raw log-rate).

    Static-graph invariant: `_static_graph_only = True` -- forward() takes
    a SINGLE (edge_index, edge_weight) per call, valid for ALL seq_len
    timesteps and ALL windows in the batch. Passing time-varying edges
    (Phase 10) would require reverting the batched-GCN call to a per-
    timestep loop; this class deliberately doesn't accept that yet, so a
    future edit that regresses the contract fails obviously.
    """

    _static_graph_only: bool = True

    def __init__(
        self,
        in_channels:     int   = len(FEATURE_COLS),
        gcn_hidden:      int   = 32,
        num_gcn_layers:  int   = 1,
        lstm_hidden:     int   = 64,
        lstm_num_layers: int   = 2,
        dropout:         float = 0.3,
        use_softplus:    bool  = False,
    ) -> None:
        super().__init__()

        if num_gcn_layers < 1:
            raise ValueError(f"num_gcn_layers must be >= 1, got {num_gcn_layers}")
        if lstm_num_layers < 1:
            raise ValueError(f"lstm_num_layers must be >= 1, got {lstm_num_layers}")

        self.gcn_layers = nn.ModuleList(
            [
                GCNConv(in_channels if i == 0 else gcn_hidden, gcn_hidden)
                for i in range(num_gcn_layers)
            ]
        )
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size  = gcn_hidden,
            hidden_size = lstm_hidden,
            num_layers  = lstm_num_layers,
            batch_first = True,
            # PyTorch: recurrent dropout only kicks in between layers when
            # num_layers > 1; setting > 0 on a single-layer LSTM emits a
            # UserWarning and is silently ignored, so gate it here.
            dropout     = dropout if lstm_num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() if use_softplus else nn.Identity(),
        )

    def forward(
        self,
        x:           torch.Tensor,   # (B, T, N, F)
        edge_index:  torch.Tensor,   # (2, E) -- SINGLE-graph edges
        edge_weight: torch.Tensor,   # (E,)
    ) -> torch.Tensor:               # (B, N)
        assert self._static_graph_only, "STGNN currently supports only a static graph"

        B, T, N, F_in = x.shape

        # --- Spatial encoding: one batched GCNConv call across B*T sub-graphs ---
        x_flat = x.reshape(B * T * N, F_in)
        ei_rep, ew_rep = _replicate_edge_index(edge_index, edge_weight, B * T, N)

        n_gcn = len(self.gcn_layers)
        for i, conv in enumerate(self.gcn_layers):
            x_flat = conv(x_flat, ei_rep, ew_rep)
            x_flat = torch.relu(x_flat)
            if i < n_gcn - 1:
                x_flat = torch.nn.functional.dropout(
                    x_flat, p=self.dropout, training=self.training,
                )

        h_g = x_flat.shape[-1]
        spatial = x_flat.reshape(B, T, N, h_g)

        # --- Temporal encoding: per-node LSTM ---
        # (B, T, N, h_g) -> (B, N, T, h_g) -> (B*N, T, h_g)
        seqs = spatial.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, h_g)
        lstm_out, _ = self.lstm(seqs)              # (B*N, T, lstm_hidden)
        last = lstm_out[:, -1, :]                  # (B*N, lstm_hidden)

        # --- Head ---
        pred = self.head(last).squeeze(-1)         # (B*N,)
        return pred.reshape(B, N)                  # (B, N)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model:       BikeDemandSTGNN,
    loader:      DataLoader,
    edge_index:  torch.Tensor,
    edge_weight: torch.Tensor,
    criterion:   nn.Module,
    optimiser:   torch.optim.Optimizer,
    device:      torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for x_batch, y_batch in pbar:
        # STGNNDataset can be CUDA-resident (see docstring); the .to(device)
        # is a no-op in that case.
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimiser.zero_grad()
        pred = model(x_batch, edge_index, edge_weight)   # (B, N)
        loss = criterion(pred, y_batch)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model:       BikeDemandSTGNN,
    loader:      DataLoader,
    edge_index:  torch.Tensor,
    edge_weight: torch.Tensor,
    device:      torch.device,
    loss_fn:     str = "poisson",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred) flattened, both length = len(loader.dataset) * N.

    For --loss-fn poisson: raw logits are recovered to rates via `np.exp()`;
    for --loss-fn mse: the Softplus head already outputs non-negative rates,
    no transform. Never `np.clip` (CLAUDE.md Loss/output contract).
    """
    model.eval()
    all_true = []
    all_pred = []

    for x_batch, y_batch in tqdm(loader, desc="  eval ", leave=False, unit="batch"):
        x_batch = x_batch.to(device, non_blocking=True)

        pred = model(x_batch, edge_index, edge_weight)   # (B, N)
        all_pred.append(pred.cpu().numpy().reshape(-1))
        all_true.append(y_batch.cpu().numpy().reshape(-1))

    y_true     = np.concatenate(all_true)
    y_pred_raw = np.concatenate(all_pred)

    if loss_fn == "poisson":
        y_pred = np.exp(y_pred_raw)
    else:
        y_pred = y_pred_raw

    return y_true, y_pred


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

    # Match evaluate_epoch's flatten order: time-major (each anchor's N
    # station rows contiguous), station order = station_ids.
    station_idx_out = np.tile(station_ids,          T)   # (T*N,)
    timestamps_out  = np.repeat(anchor_timestamps,  N)   # (T*N,)

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
    data_dir:        Path,
    adj_variant:     Literal["knn", "flow", "combined"] = "knn",
    loss_fn:         Literal["mse", "poisson"] = "poisson",
    gcn_hidden:      int   = 32,
    num_gcn_layers:  int   = 1,
    lstm_hidden:     int   = 64,
    lstm_num_layers: int   = 2,
    dropout:         float = 0.3,
    lr:              float = 1e-3,
    batch_size:      int   = 4,
    seq_len:         int   = DEFAULT_SEQ_LEN,
    max_epochs:      int   = 50,
    patience:        int   = 5,
    save_dir:        Path  = Path("models"),
    seed:            int   = 42,
    log_to_mlflow:   bool  = True,
    epoch_callback:  Callable[[int, float], None] | None = None,
) -> float:
    """
    Train BikeDemandSTGNN and log results to MLflow. Returns best_val_mae.

    `epoch_callback(epoch, val_mae)` is invoked at the end of every epoch
    for optional Optuna trial pruning (see src/models/tune.py). This module
    has no dependency on Optuna; the caller supplies the hook.

    Run lifecycle (mlflow.set_experiment / start_run / end_run) is the
    caller's responsibility -- either main() below (top-level run) or the
    tuner (nested inside mlflow.start_run(nested=True)).
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
    logger.info("Building STGNN dataloaders (batch_size=%d, seq_len=%d)...", batch_size, seq_len)
    train_loader, val_loader, test_loader = build_stgnn_dataloaders(
        data_dir, adj_tensor,
        batch_size=batch_size,
        num_workers=0,
        seq_len=seq_len,
        device=device,
    )

    # edge_index / edge_weight are single-graph (built once by STGNNDataset).
    # Model.forward replicates them per-batch internally.
    edge_index  = train_loader.dataset.edge_index.to(device)
    edge_weight = train_loader.dataset.edge_weight.to(device)

    # --- Metadata (aligned to anchor timestamps + station grid) ---
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
        in_channels     = in_channels,
        gcn_hidden      = gcn_hidden,
        num_gcn_layers  = num_gcn_layers,
        lstm_hidden     = lstm_hidden,
        lstm_num_layers = lstm_num_layers,
        dropout         = dropout,
        use_softplus    = loss_fn != "poisson",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s params: %d", run_name, n_params)

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
    ckpt_path      = save_dir / f"{run_name}_best.pt"

    # --- MLflow provenance ---
    if log_to_mlflow:
        mlflow.set_tag("model_name", run_name)
        mlflow.set_tag("phase", "hybrid")
        tag_run_provenance(data_dir, seed)
        mlflow.log_params({
            "model_type":      "hybrid",
            "adj_variant":     adj_variant,
            "loss_fn":         loss_fn,
            "gcn_hidden":      gcn_hidden,
            "num_gcn_layers":  num_gcn_layers,
            "lstm_hidden":     lstm_hidden,
            "lstm_num_layers": lstm_num_layers,
            "dropout":         dropout,
            "lr":              lr,
            "batch_size":      batch_size,
            "seq_len":         seq_len,
            "max_epochs":      max_epochs,
            "patience":        patience,
            "n_params":        n_params,
            "in_channels":     in_channels,
            "device":          str(device),
            "seed":            seed,
        })

    # --- Training loop ---
    logger.info(
        "Starting training | max_epochs=%d | patience=%d", max_epochs, patience
    )
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, edge_index, edge_weight, criterion, optimiser, device
        )
        y_true, y_pred = evaluate_epoch(
            model, val_loader, edge_index, edge_weight, device, loss_fn=loss_fn
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

    # --- Full evaluation on val, test, train (per-split log-and-release
    # to keep peak RAM low, following lstm.py's pattern) ---
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
            model, loader, edge_index, edge_weight, device, loss_fn=loss_fn
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
    parser.add_argument("--data-dir",        type=Path,  default=Path("data/processed"))
    parser.add_argument(
        "--adj", choices=["knn", "flow", "combined", "all"], default="knn",
        help="Adjacency variant (default: knn). 'all' loops over knn/flow/combined."
    )
    parser.add_argument("--loss-fn",         choices=["mse", "poisson"], default="poisson")
    parser.add_argument("--gcn-hidden",      type=int,   default=32)
    parser.add_argument("--num-gcn-layers",  type=int,   default=1)
    parser.add_argument("--lstm-hidden",     type=int,   default=64)
    parser.add_argument("--lstm-num-layers", type=int,   default=2)
    parser.add_argument("--dropout",         type=float, default=0.3)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--batch-size",      type=int,   default=4)
    parser.add_argument("--seq-len",         type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument("--max-epochs",      type=int,   default=50)
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--save-dir",        type=Path,  default=Path("models"))
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--no-mlflow",       action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    variants = ADJ_VARIANTS if args.adj == "all" else (args.adj,)

    if not args.no_mlflow:
        ensure_portable_artifact_location(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)

    for variant in variants:
        run_kwargs = dict(
            data_dir        = args.data_dir,
            adj_variant     = variant,
            loss_fn         = args.loss_fn,
            gcn_hidden      = args.gcn_hidden,
            num_gcn_layers  = args.num_gcn_layers,
            lstm_hidden     = args.lstm_hidden,
            lstm_num_layers = args.lstm_num_layers,
            dropout         = args.dropout,
            lr              = args.lr,
            batch_size      = args.batch_size,
            seq_len         = args.seq_len,
            max_epochs      = args.max_epochs,
            patience        = args.patience,
            save_dir        = args.save_dir,
            seed            = args.seed,
            log_to_mlflow   = not args.no_mlflow,
        )
        if args.no_mlflow:
            run_hybrid(**run_kwargs)
        else:
            with mlflow.start_run(run_name=f"hybrid_{variant}_{args.loss_fn}"):
                run_hybrid(**run_kwargs)


if __name__ == "__main__":
    main()
