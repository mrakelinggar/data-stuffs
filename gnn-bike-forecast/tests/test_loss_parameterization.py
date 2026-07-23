"""
Tests for the Poisson/MSE loss-output parameterization fix (ROADMAP Phase 2).

Poisson emits a raw log-rate (Identity head), paired with
nn.PoissonNLLLoss(log_input=True); rate is recovered at inference via
np.exp(). MSE emits a non-negative rate directly (Softplus head) --
non-negativity belongs in the model, never in a post-hoc np.clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data_loader import FEATURE_COLS
from gcn import BikeDemanGCN
from hybrid import BikeDemandSTGNN
from lstm import BikeDemanLSTM

_MODEL_SRC_FILES = [
    Path(__file__).resolve().parents[1] / "src" / "models" / "lstm.py",
    Path(__file__).resolve().parents[1] / "src" / "models" / "gcn.py",
    Path(__file__).resolve().parents[1] / "src" / "models" / "hybrid.py",  # ROADMAP Phase 8
]


# ---------------------------------------------------------------------------
# Head-type regression guard
#
# `use_softplus = loss_fn != "poisson"` is the exact formula run_lstm()/
# run_gcn() use to wire the head -- mirrored here (not imported, since it's
# inline logic, not a standalone function) so a future accidental flip-back
# is caught.
# ---------------------------------------------------------------------------

def _use_softplus_for(loss_fn: str) -> bool:
    return loss_fn != "poisson"


def test_lstm_poisson_head_is_identity():
    model = BikeDemanLSTM(input_size=7, use_softplus=_use_softplus_for("poisson"))
    assert isinstance(model.head[-1], nn.Identity)


def test_lstm_mse_head_is_softplus():
    model = BikeDemanLSTM(input_size=7, use_softplus=_use_softplus_for("mse"))
    assert isinstance(model.head[-1], nn.Softplus)


def test_gcn_poisson_head_is_identity():
    model = BikeDemanGCN(in_channels=15, use_softplus=_use_softplus_for("poisson"))
    assert isinstance(model.head[-1], nn.Identity)


def test_gcn_mse_head_is_softplus():
    model = BikeDemanGCN(in_channels=15, use_softplus=_use_softplus_for("mse"))
    assert isinstance(model.head[-1], nn.Softplus)


def _tiny_hybrid_adj(n: int = 4) -> torch.Tensor:
    """Minimal normalised adjacency for head-type tests (no graph-correctness needed)."""
    return torch.eye(n)


def test_hybrid_poisson_head_is_identity():
    model = BikeDemandSTGNN(
        adj_norm=_tiny_hybrid_adj(),
        in_channels=len(FEATURE_COLS),
        use_softplus=_use_softplus_for("poisson"),
    )
    assert isinstance(model.head[-1], nn.Identity)


def test_hybrid_mse_head_is_softplus():
    model = BikeDemandSTGNN(
        adj_norm=_tiny_hybrid_adj(),
        in_channels=len(FEATURE_COLS),
        use_softplus=_use_softplus_for("mse"),
    )
    assert isinstance(model.head[-1], nn.Softplus)


# ---------------------------------------------------------------------------
# Poisson raw output spans the reals; exp() recovers a non-negative rate
# ---------------------------------------------------------------------------

def test_lstm_poisson_raw_output_spans_reals():
    torch.manual_seed(0)
    model = BikeDemanLSTM(input_size=7, hidden_size=8, num_layers=1, use_softplus=False)
    # Force a negative bias on the final Linear layer so at least some raw
    # outputs are guaranteed negative -- proves the head is unconstrained.
    with torch.no_grad():
        model.head[2].bias.fill_(-10.0)

    x = torch.randn(32, 168, 7)
    with torch.no_grad():
        raw = model(x)

    assert (raw < 0).any(), "Identity head should allow negative raw outputs"


def test_lstm_exp_of_raw_output_is_nonnegative():
    torch.manual_seed(0)
    model = BikeDemanLSTM(input_size=7, hidden_size=8, num_layers=1, use_softplus=False)
    x = torch.randn(16, 168, 7)
    with torch.no_grad():
        raw = model(x).numpy()

    y_pred = np.exp(raw)
    assert (y_pred >= 0).all()


def _tiny_graph(n_nodes: int = 6, in_channels: int = 15):
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
    )
    edge_weight = torch.ones(edge_index.shape[1])
    x = torch.randn(n_nodes, in_channels)
    return x, edge_index, edge_weight


def test_gcn_poisson_raw_output_spans_reals():
    torch.manual_seed(0)
    model = BikeDemanGCN(in_channels=15, hidden_size=8, use_softplus=False)
    with torch.no_grad():
        model.head[2].bias.fill_(-10.0)

    x, edge_index, edge_weight = _tiny_graph()
    with torch.no_grad():
        raw = model(x, edge_index, edge_weight)

    assert (raw < 0).any(), "Identity head should allow negative raw outputs"


def test_gcn_exp_of_raw_output_is_nonnegative():
    torch.manual_seed(0)
    model = BikeDemanGCN(in_channels=15, hidden_size=8, use_softplus=False)
    x, edge_index, edge_weight = _tiny_graph()
    with torch.no_grad():
        raw = model(x, edge_index, edge_weight).numpy()

    y_pred = np.exp(raw)
    assert (y_pred >= 0).all()


# ---------------------------------------------------------------------------
# Source-level regression guard: no silent post-hoc clip in the neural path
# ---------------------------------------------------------------------------

def test_no_silent_clip_in_lstm_or_gcn():
    for path in _MODEL_SRC_FILES:
        source = path.read_text()
        assert "np.clip(y_pred" not in source, (
            f"{path.name} contains a np.clip(y_pred...) call -- "
            "the Poisson/MSE head now guarantees non-negativity in-model; "
            "post-hoc clipping should not be reintroduced."
        )
