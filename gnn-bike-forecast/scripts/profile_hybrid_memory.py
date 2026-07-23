#!/usr/bin/env python
"""
Profile peak memory for BikeDemandSTGNN.

Run BEFORE and AFTER the Phase 8 memory optimizations:

    .venv/bin/python scripts/profile_hybrid_memory.py

Two sections:
  1. Analytical T4-GPU estimate — pure arithmetic on tensor shapes,
     tells you what a Colab T4 (15 GB) would need for N=1911, B=4, T=168.
  2. Actual local measurement — runs a real forward+backward on scaled-down
     sizes and measures peak MPS / CUDA allocation. Auto-detects whether
     the loaded hybrid.py is the old interface (model(x, ei, ew)) or the
     new one (model(x) with adj baked in), so the same script works both
     before and after the fixes.
"""
from __future__ import annotations

import gc
import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))

# ── Production sizes (Colab T4 target) ───────────────────────────────────────
PROD_B, PROD_T, PROD_N, PROD_F     = 4, 168, 1911, 22
H_GCN, H_LSTM, N_LSTM_LAYERS       = 32, 64, 2
E_KNN, E_FLOW                       = 11_255, 898_842   # from local smoke test
TBPTT_K                             = 48    # backprop only last 48 steps
GRAD_ACCUM                          = 4     # effective B = PROD_B / GRAD_ACCUM

# ── Local sizes (scaled down to fit this machine) ────────────────────────────
LOCAL_B, LOCAL_T, LOCAL_N           = 2, 24, 200

F32, FP16, I64 = 4, 2, 8
GB = 1e9


def _fmt(b: float) -> str:
    return f"{b / GB:6.2f} GB"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Analytical breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _show_components(label: str, components: dict[str, float], limit_gb: float = 15.0) -> float:
    total   = sum(components.values())
    fits    = total / GB <= limit_gb
    status  = "✓ fits" if fits else f"✗ OOM  (over by {_fmt(total - limit_gb * GB)})"
    print(f"\n    [{label}]")
    for name, b in components.items():
        flag = "  ◄◄ DOMINANT" if b > 3 * GB else ""
        print(f"      {name:<42}  {_fmt(b)}{flag}")
    print(f"      {'─' * 52}")
    print(f"      {'PEAK TOTAL':<42}  {_fmt(total)}  {status}")
    return total


def analytical_section() -> None:
    B, T, N, F = PROD_B, PROD_T, PROD_N, PROD_F
    hg, hl, L  = H_GCN, H_LSTM, N_LSTM_LAYERS
    K           = TBPTT_K
    B_eff       = B // GRAD_ACCUM  # = 1 per step after grad accumulation

    print("\n" + "═" * 60)
    print("1. ANALYTICAL T4-GPU ESTIMATE  (15 GB limit)")
    print(f"   Production: N={N}, T={T}, B={B}, F={F}, h_g={hg}, h_l={hl}, L={L}")
    print("═" * 60)

    for adj_label, E in [("KNN  (E=11,255)", E_KNN), ("Flow / Combined  (E=898,842)", E_FLOW)]:
        print(f"\n  ── {adj_label} " + "─" * (48 - len(adj_label)))

        before: dict[str, float] = {
            f"input  ({B}×{T}×{N}×{F}) f32":       B * T * N * F  * F32,
            f"edge_index replicated ({B}×{T}×E)":    2 * B * T * E  * I64,
            f"edge_weight replicated":                B * T * E      * F32,
            "GCN message-pass buffers  (scatter)":    2 * B * T * E  * F32,
            f"spatial features ({B}×{T}×{N}×{hg})": B * T * N * hg * F32,
            "permute+contiguous copy  (extra)":       B * N * T * hg * F32,
            f"LSTM seqs ({B}×{N}×{T}×{hg})":        B * N * T * hg * F32,
            f"LSTM backward (all {T} steps, 6 vals)":6 * L * B * N * T * hl * F32,
            f"LSTM output ({B}×{N}×{T}×{hl})":      B * N * T * hl * F32,
        }

        after: dict[str, float] = {
            f"input  ({B_eff}×{T}×{N}×{F}) fp16  [grad_accum={GRAD_ACCUM}]":
                                                      B_eff * T * N * F  * FP16,
            f"dense adj matrix  ({N}×{N}) f32":       N * N              * F32,
            f"GCN linear output ({B_eff}×{N}×{T}×{hg}) fp16":
                                                      B_eff * N * T * hg * FP16,
            "LSTM input  [view, zero copy]":           0,
            f"LSTM no_grad prefix  ({T - K} steps)":  0,
            f"LSTM backward (ckpt+TBPTT, K={K} steps) fp16":
                                                      6 * L * B_eff * N * K * hl * FP16,
            f"LSTM output ({B_eff}×{N}×{K}×{hl})":  B_eff * N * K * hl * FP16,
        }

        tb = _show_components(f"BEFORE  (B={B}, GCNConv, f32, full BPTT)", before)
        ta = _show_components(
            f"AFTER   (B_eff={B_eff}, dense+ckpt+TBPTT K={K}+fp16)", after
        )
        saving_pct = (tb - ta) / tb * 100
        print(f"\n  → Saving:  {_fmt(tb - ta)}  ({saving_pct:.0f}% reduction)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Actual local measurement
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _mem_before(device: torch.device) -> int:
    if device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        return 0
    return 0


def _mem_peak(device: torch.device, before: int) -> float:
    """Return peak memory in MB since _mem_before() was called."""
    if device.type == "mps":
        current = torch.mps.current_allocated_memory()
        return (current - before) / 1e6
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def _run_forward_backward(device: torch.device) -> float:
    """
    Real forward + backward on local sizes.  Auto-detects old vs new interface:
      old (before fixes): BikeDemandSTGNN(in_channels, ...) + model(x, ei, ew)
      new (after  fixes): BikeDemandSTGNN(adj_norm, ...)    + model(x)
    Returns peak memory delta in MB.
    """
    from data_loader import FEATURE_COLS
    from hybrid import BikeDemandSTGNN

    N   = LOCAL_N
    F   = len(FEATURE_COLS)
    sig = inspect.signature(BikeDemandSTGNN.__init__)
    new_interface = "adj_norm" in sig.parameters

    # Synthetic adj that matches local size
    adj_raw = torch.rand(N, N)

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    before = _mem_before(device)

    if new_interface:
        model = BikeDemandSTGNN(
            adj_norm        = adj_raw.to(device),
            in_channels     = F,
            gcn_hidden      = H_GCN,
            lstm_hidden     = H_LSTM,
            lstm_num_layers = N_LSTM_LAYERS,
            dropout         = 0.0,
        ).to(device)
        model.train()
        x = torch.randn(LOCAL_B, LOCAL_T, N, F, device=device)
        y = torch.rand(LOCAL_B, N, device=device)
        pred = model(x)
    else:
        model = BikeDemandSTGNN(
            in_channels     = F,
            gcn_hidden      = H_GCN,
            lstm_hidden     = H_LSTM,
            lstm_num_layers = N_LSTM_LAYERS,
            dropout         = 0.0,
        ).to(device)
        model.train()
        x  = torch.randn(LOCAL_B, LOCAL_T, N, F, device=device)
        y  = torch.rand(LOCAL_B, N, device=device)
        # Build sparse edge_index from adj
        ei = adj_raw.nonzero(as_tuple=False).t().contiguous().long().to(device)
        ew = adj_raw[ei[0].cpu(), ei[1].cpu()].to(device)
        pred = model(x, ei, ew)

    loss = nn.MSELoss()(pred, y)
    loss.backward()

    peak_mb = _mem_peak(device, before)
    return peak_mb, new_interface


def actual_section() -> None:
    device = _get_device()

    print("\n" + "═" * 60)
    print("2. ACTUAL LOCAL MEASUREMENT")
    print(f"   Local sizes: N={LOCAL_N}, B={LOCAL_B}, T={LOCAL_T}")
    print(f"   Device: {device.type.upper()}")
    print("═" * 60)

    if device.type == "cpu":
        print("\n  [CPU device — memory tracking not supported, skipping]")
        return

    try:
        peak_mb, new_interface = _run_forward_backward(device)
    except Exception as exc:
        print(f"\n  [Measurement failed: {exc}]")
        return

    interface = "new (adj_norm, model(x))" if new_interface else "old (model(x, ei, ew))"
    print(f"\n  Interface detected:  {interface}")
    print(f"  Peak allocation:     {peak_mb:.0f} MB")

    if device.type == "mps":
        print("\n  Note: MPS stats reflect post-backward allocated memory, not true peak.")
        print("  CUDA's max_memory_allocated() (used on Colab T4) is more accurate.")


if __name__ == "__main__":
    analytical_section()
    actual_section()
    print()
