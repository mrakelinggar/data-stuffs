#!/usr/bin/env python
"""
Generate a fully self-contained HTML report comparing all MLflow runs.

Usage
-----
    python src/reports/generate_report.py
    python src/reports/generate_report.py --top-n 10
    python src/reports/generate_report.py --output reports/my_report.html
    python src/reports/generate_report.py --experiment-name bike-demand-forecasting
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from evaluate import paired_bootstrap_test  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLORS = [
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
]

STATUS_COLOR = {
    "FINISHED": "#27ae60",
    "FAILED":   "#c0392b",
    "RUNNING":  "#f39c12",
}

# Baselines log val_mae; neural models log best_val_mae (prefix="best_", split="val")
METRIC_CANDIDATES = [
    ("val_mae",       "best_val_mae"),
    ("val_rmse",      "best_val_rmse"),
    ("val_mape",      "best_val_mape"),
    ("val_wape",      "best_val_wape"),
    ("val_peak_mae",  "best_val_peak_mae"),
    ("val_mae_ci_lo", "best_val_mae_ci_lo"),
    ("val_mae_ci_hi", "best_val_mae_ci_hi"),
    ("train_mae",     "best_train_mae"),
    ("test_mae",      "best_test_mae"),
    ("test_rmse",     "best_test_rmse"),
]
METRIC_LABELS = {
    "val_mae":      "Val MAE",
    "val_rmse":     "Val RMSE",
    "val_mape":     "Val MAPE (%)",
    "val_wape":     "Val WAPE",
    "val_peak_mae": "Val Peak MAE",
    "train_mae":    "Train MAE",
    "test_mae":     "Test MAE",
    "test_rmse":    "Test RMSE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_metric(metrics: dict, *candidates: str) -> float | None:
    for key in candidates:
        if key in metrics:
            return metrics[key]
    return None


def _infer_family(run_name: str) -> str:
    name = run_name.lower()
    if name.startswith("naive"):
        return "naive"
    if name.startswith("linear"):
        return "linear"
    if name.startswith("lstm"):
        return "lstm"
    if name.startswith("gcn"):
        return "gcn"
    return "other"


def _resolve_artifact_dir(run: mlflow.entities.Run) -> Path | None:
    uri = run.info.artifact_uri or ""
    if uri.startswith("file://"):
        return Path(uri[7:])
    if uri.startswith("mlflow-artifacts:"):
        return None  # remote; handled via download
    if uri:
        p = Path(uri)
        if p.exists():
            return p
    return None


def _load_artifact_csv(
    client: mlflow.tracking.MlflowClient,
    run: mlflow.entities.Run,
    artifact_path: str,
) -> pd.DataFrame | None:
    artifact_dir = _resolve_artifact_dir(run)
    if artifact_dir is not None:
        full_path = artifact_dir / artifact_path
        if full_path.exists():
            return pd.read_csv(full_path)
        return None
    # Fallback: download via MLflow artifacts API
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=artifact_path,
                dst_path=tmp,
            )
            return pd.read_csv(local)
    except Exception:
        return None


ARTIFACT_KINDS = ["station", "hour", "cluster", "volume_tier", "hour_band"]


def _load_split_eval_artifacts(
    client: mlflow.tracking.MlflowClient,
    run: mlflow.entities.Run,
    split: str = "val",
) -> dict[str, pd.DataFrame | None]:
    """
    Load all eval CSVs for a given split (station/hour/cluster/volume_tier/
    hour_band), keyed by kind.

    Matches the exact '{model}_{split}_by_{kind}.csv' pattern. Filenames are
    split-suffixed as of Phase 1 -- both a val and a test file (and, as of
    Phase 2, a train file) exist per run, so a plain suffix match (the old
    `_find_eval_csvs` behaviour) would silently pick whichever file
    `list_artifacts()` happened to return last. For split="val" only, also
    falls back to legacy unsuffixed '{model}_by_{kind}.csv' files from
    pre-Phase-1 runs (best-effort -- those runs' split provenance for a given
    artifact isn't reliably recoverable, since the old code overwrote the
    same local filename across splits within a run before logging it).
    """
    result: dict[str, pd.DataFrame | None] = {kind: None for kind in ARTIFACT_KINDS}

    try:
        artifacts = client.list_artifacts(run.info.run_id, "eval")
    except Exception:
        return result

    split_tokens = ("_train_", "_val_", "_test_")
    legacy_candidates: dict[str, str] = {}

    for a in artifacts:
        path = a.path
        for kind in ARTIFACT_KINDS:
            suffix = f"_by_{kind}.csv"
            if not path.endswith(suffix):
                continue
            if path.endswith(f"_{split}{suffix}"):
                result[kind] = _load_artifact_csv(client, run, path)
            elif split == "val" and not any(tok in path for tok in split_tokens):
                legacy_candidates[kind] = path

    for kind, path in legacy_candidates.items():
        if result[kind] is None:
            result[kind] = _load_artifact_csv(client, run, path)

    return result


def _load_predictions_artifact(
    client: mlflow.tracking.MlflowClient,
    run: mlflow.entities.Run,
    split: str = "val",
) -> pd.DataFrame | None:
    """
    Load eval/predictions/predictions_{split}.parquet for a run, if present.

    Filename is deterministic (written by mlflow_utils.log_predictions_to_mlflow),
    so unlike _load_split_eval_artifacts this needs no listing/glob -- just
    attempt the known path and return None on any failure. Historical
    (pre-Phase-3) runs never wrote this artifact and will always resolve to
    None here; callers must treat None as "exclude from significance test,"
    never as an error.
    """
    artifact_path = f"eval/predictions/predictions_{split}.parquet"

    artifact_dir = _resolve_artifact_dir(run)
    if artifact_dir is not None:
        full_path = artifact_dir / artifact_path
        if not full_path.exists():
            return None
        try:
            return pd.read_parquet(full_path)
        except Exception:
            return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id, artifact_path=artifact_path, dst_path=tmp,
            )
            return pd.read_parquet(local)
    except Exception:
        return None


def _load_training_history(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    for metric in ("train_mse_loss", "val_mae", "val_rmse"):
        try:
            records = client.get_metric_history(run_id, metric)
            if records:
                sorted_records = sorted(records, key=lambda x: x.step)
                history[metric] = [r.value for r in sorted_records]
        except Exception:
            pass
    return history


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(experiment_name: str, tracking_uri: str) -> list[dict]:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in {tracking_uri}")

    raw_runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"])
    runs = []

    for r in raw_runs:
        name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
        family = _infer_family(name)
        status = r.info.status  # FINISHED | FAILED | RUNNING | SCHEDULED

        eval_artifacts = _load_split_eval_artifacts(client, r, split="val")
        predictions = _load_predictions_artifact(client, r, split="val")

        training_history: dict[str, list[float]] = {}
        if family in ("lstm", "gcn"):
            training_history = _load_training_history(client, r.info.run_id)

        # Normalise metrics so both baselines and neural runs use the same keys
        metrics = dict(r.data.metrics)
        for primary, fallback in METRIC_CANDIDATES:
            if primary not in metrics and fallback in metrics:
                metrics[primary] = metrics[fallback]

        runs.append({
            "run_id":           r.info.run_id,
            "name":             name,
            "family":           family,
            "status":           status,
            "start_time":       datetime.fromtimestamp(
                                    r.info.start_time / 1000
                                ).strftime("%Y-%m-%d %H:%M"),
            "params":           r.data.params,
            "metrics":          metrics,
            "by_hour":          eval_artifacts["hour"],
            "by_station":       eval_artifacts["station"],
            "by_cluster":       eval_artifacts["cluster"],
            "by_volume_tier":   eval_artifacts["volume_tier"],
            "by_hour_band":     eval_artifacts["hour_band"],
            "predictions":      predictions,
            "training_history": training_history,
        })

    return runs


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_leaderboard_html(runs: list[dict]) -> str:
    col_metrics = [
        "train_mae", "val_mae", "val_wape", "val_peak_mae", "val_rmse", "val_mape",
        "test_mae", "test_rmse",
    ]
    col_labels = [
        "Train MAE", "Val MAE", "Val WAPE", "Val Peak MAE", "Val RMSE", "Val MAPE (%)",
        "Test MAE", "Test RMSE",
    ]
    # "Best" (green) highlighting is a model-selection signal -- restrict it to
    # Val columns. Train MAE is an in-sample fit (always looks best, doesn't mean
    # anything) and Test is touch-once/reference-only (CLAUDE.md: never used to
    # pick between runs), so highlighting either as "best" would visually imply
    # overfitting-prone or test-peeking selection.
    highlightable_labels = {lbl for lbl in col_labels if lbl.startswith("Val ")}

    rows = []
    for r in runs:
        row: dict = {
            "Run":     r["name"],
            "Family":  r["family"],
            "Status":  r["status"],
            "Started": r["start_time"],
        }
        for col, label in zip(col_metrics, col_labels):
            row[label] = r["metrics"].get(col)

        # Val MAE 95% CI -- always available (station-level bootstrap, no
        # predictions-artifact dependency), unlike the Significance figure
        # below. Not wired into best_idx highlighting: a CI range isn't a
        # "lower is better" scalar the highlighting logic is built for.
        ci_lo = r["metrics"].get("val_mae_ci_lo")
        ci_hi = r["metrics"].get("val_mae_ci_hi")
        row["Val MAE 95% CI"] = (
            f"{ci_lo:.3f} – {ci_hi:.3f}" if ci_lo is not None and ci_hi is not None else "—"
        )

        params = r["params"]
        row["Params"] = (
            ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
            if params else "—"
        )
        rows.append((r, row))

    df = pd.DataFrame([row for _, row in rows])

    # Find best (min) row index per metric column -- Val columns only, see above.
    best_idx: dict[str, int] = {}
    for label in col_labels:
        if label not in highlightable_labels:
            continue
        col = df[label].dropna()
        if not col.empty:
            best_idx[label] = int(col.idxmin())

    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    body_rows = []

    for i, (run_obj, _) in enumerate(rows):
        status = run_obj["status"]
        row_cls = ' class="row-failed"' if status == "FAILED" else (
                  ' class="row-running"' if status == "RUNNING" else "")

        cells = []
        for col in df.columns:
            val = df.at[i, col]
            best_style = (
                ' style="background:#d4edda;font-weight:600;"'
                if col in best_idx and best_idx[col] == i else ""
            )
            is_na = val is None or (isinstance(val, float) and pd.isna(val))

            if col == "Status":
                color = STATUS_COLOR.get(str(val), "#555")
                badge = ""
                if str(val) != "FINISHED":
                    badge = ' <span class="badge-warn">⚠ incomplete</span>'
                cells.append(
                    f'<td><span style="color:{color};font-weight:600">{val}</span>{badge}</td>'
                )
            elif is_na:
                cells.append(f"<td{best_style}>—</td>")
            elif isinstance(val, float):
                cells.append(f"<td{best_style}>{val:.4f}</td>")
            else:
                cells.append(f"<td{best_style}>{val}</td>")

        body_rows.append(f"<tr{row_cls}>{''.join(cells)}</tr>")

    return (
        '<table class="leaderboard">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def build_family_comparison_fig(runs: list[dict]) -> go.Figure:
    family_best: dict[str, dict] = {}
    for r in runs:
        if r["status"] != "FINISHED":
            continue
        mae = r["metrics"].get("val_mae")
        if mae is None:
            continue
        fam = r["family"]
        if fam not in family_best or mae < family_best[fam]["metrics"]["val_mae"]:
            family_best[fam] = r

    families = list(family_best.keys())
    fig = go.Figure()

    for metric, label, color in [
        ("val_mae",  "Val MAE",  COLORS[0]),
        ("val_rmse", "Val RMSE", COLORS[1]),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=families,
            y=[family_best[f]["metrics"].get(metric, 0) for f in families],
            text=[f"{family_best[f]['metrics'].get(metric, 0):.3f}" for f in families],
            textposition="outside",
            marker_color=color,
        ))

    fig.update_layout(
        title="Best Run per Model Family — Val Set",
        barmode="group",
        yaxis_title="Error",
        xaxis_title="Model Family",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def build_gcn_comparison_fig(runs: list[dict]) -> go.Figure:
    gcn_runs = [
        r for r in runs
        if r["family"] == "gcn"
        and r["status"] == "FINISHED"
        and r["metrics"].get("val_mae") is not None
    ]

    if not gcn_runs:
        fig = go.Figure()
        fig.add_annotation(
            text="No completed GCN runs found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    records = []
    for r in gcn_runs:
        adj  = r["params"].get("adj_variant", "")
        loss = r["params"].get("loss_fn", "")
        if not adj:
            name = r["name"].lower()
            adj = next((a for a in ("knn", "flow", "combined") if a in name), "unknown")
        if not loss:
            loss = "poisson" if "poisson" in r["name"].lower() else "mse"
        records.append({
            "adj":      adj,
            "loss":     loss,
            "val_mae":  r["metrics"]["val_mae"],
            "val_rmse": r["metrics"].get("val_rmse"),
            "run":      r["name"],
        })

    df = pd.DataFrame(records)
    fig = go.Figure()

    for loss_fn, color in [("mse", COLORS[0]), ("poisson", COLORS[2])]:
        sub = df[df["loss"] == loss_fn]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=f"{loss_fn.upper()} loss",
            x=sub["adj"],
            y=sub["val_mae"],
            text=[f"{v:.3f}" for v in sub["val_mae"]],
            textposition="outside",
            marker_color=color,
            customdata=sub["run"].values,
            hovertemplate="<b>%{customdata}</b><br>Adj: %{x}<br>MAE: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        title="GCN Variants: Adjacency × Loss Function — Val MAE",
        barmode="group",
        yaxis_title="Val MAE",
        xaxis_title="Adjacency Type",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#eee")
    return fig


def build_training_curves_fig(runs: list[dict]) -> go.Figure:
    neural_runs = [r for r in runs if r["family"] in ("lstm", "gcn") and r["training_history"]]

    if not neural_runs:
        fig = go.Figure()
        fig.add_annotation(
            text="No training history found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Validation MAE per Epoch", "Train Loss per Epoch"),
    )

    for i, r in enumerate(neural_runs):
        color = COLORS[i % len(COLORS)]
        hist  = r["training_history"]

        if "val_mae" in hist:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(hist["val_mae"]) + 1)),
                y=hist["val_mae"],
                name=r["name"],
                mode="lines",
                line=dict(color=color),
                legendgroup=r["name"],
            ), row=1, col=1)

        if "train_mse_loss" in hist:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(hist["train_mse_loss"]) + 1)),
                y=hist["train_mse_loss"],
                name=r["name"],
                mode="lines",
                line=dict(color=color),
                legendgroup=r["name"],
                showlegend=False,
            ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Val MAE",        row=1, col=1, gridcolor="#eee")
    fig.update_yaxes(title_text="Train Loss (MSE)", row=1, col=2, gridcolor="#eee")
    fig.update_layout(
        title="Training Curves — LSTM & GCN",
        height=430,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_by_hour_fig(runs: list[dict]) -> go.Figure:
    fig = go.Figure()
    has_data = False

    for i, r in enumerate(runs):
        if r["by_hour"] is None:
            continue
        df = r["by_hour"]
        status_note = "" if r["status"] == "FINISHED" else f" [{r['status']}]"
        fig.add_trace(go.Scatter(
            x=df["hour"],
            y=df["mae"],
            name=f"{r['name']}{status_note}",
            mode="lines+markers",
            line=dict(color=COLORS[i % len(COLORS)]),
            hovertemplate="Hour %{x}:00 — MAE: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
        has_data = True

    if not has_data:
        fig.add_annotation(
            text="No by-hour data found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig.update_layout(
        title="MAE by Hour of Day — Val Set",
        xaxis=dict(title="Hour of Day", tickvals=list(range(0, 24))),
        yaxis=dict(title="MAE", gridcolor="#eee"),
        height=460,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_by_station_fig(runs: list[dict], top_n: int) -> go.Figure:
    """Heatmap: rows = sampled stations, columns = runs, colour = MAE."""
    frames = []
    for r in runs:
        if r["by_station"] is None or r["status"] != "FINISHED":
            continue
        df = r["by_station"][["station_idx", "mae"]].copy()
        df["run"] = r["name"]
        frames.append(df)

    if not frames:
        fig = go.Figure()
        fig.add_annotation(
            text="No by-station data found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        return fig

    combined = pd.concat(frames, ignore_index=True)
    avg_mae   = combined.groupby("station_idx")["mae"].mean().sort_values(ascending=False)

    worst_ids = avg_mae.head(top_n).index.tolist()
    best_ids  = avg_mae.tail(top_n).index.tolist()
    selected  = worst_ids + best_ids

    pivot = (
        combined[combined["station_idx"].isin(selected)]
        .pivot_table(index="station_idx", columns="run", values="mae", aggfunc="first")
        .reindex(selected)
    )

    # Labels: mark worst / best
    y_labels = [
        f"Stn {s} ▲" if s in worst_ids else f"Stn {s} ▼"
        for s in pivot.index
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=y_labels,
        colorscale="RdYlGn_r",
        colorbar=dict(title="MAE"),
        hovertemplate="Station: %{y}<br>Run: %{x}<br>MAE: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Per-Station MAE — Top {top_n} Worst (▲) & Best (▼) Stations",
        xaxis=dict(title="Run", tickangle=30),
        yaxis=dict(title="Station", autorange="reversed"),
        height=max(500, 18 * len(selected)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=100),
    )
    return fig


def build_station_distribution_fig(runs: list[dict]) -> go.Figure:
    """Box plot of per-station MAE per run — shows spread and outliers."""
    fig = go.Figure()
    has_data = False

    for i, r in enumerate(runs):
        if r["by_station"] is None or r["status"] != "FINISHED":
            continue
        fig.add_trace(go.Box(
            y=r["by_station"]["mae"],
            name=r["name"],
            marker_color=COLORS[i % len(COLORS)],
            boxmean="sd",
            hovertemplate="MAE: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
        has_data = True

    if not has_data:
        fig.add_annotation(
            text="No station data found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig.update_layout(
        title="Per-Station MAE Distribution — Val Set",
        yaxis=dict(title="Station MAE", gridcolor="#eee"),
        height=460,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


_HOUR_BANDS = ["off_peak", "am_peak", "mid", "pm_peak"]


def build_peak_offpeak_fig(runs: list[dict]) -> go.Figure:
    """
    Grouped bar of n_samples-weighted MAE per hour band (off-peak / AM peak /
    mid / PM peak), derived from each run's already-logged by_hour DataFrame
    -- no new artifact needed, works for both historical and new runs.
    """
    fig = go.Figure()
    has_data = False

    hour_to_band = {
        **{h: "off_peak" for h in (0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23)},
        **{h: "am_peak" for h in (7, 8, 9)},
        **{h: "mid" for h in (10, 11, 12, 13, 14, 15, 16)},
        **{h: "pm_peak" for h in (17, 18, 19)},
    }

    for i, r in enumerate(runs):
        if r["by_hour"] is None:
            continue
        df = r["by_hour"].copy()
        df["band"] = df["hour"].map(hour_to_band)
        df["weighted_mae"] = df["mae"] * df["n_samples"]
        grouped = df.groupby("band")[["weighted_mae", "n_samples"]].sum()
        band_mae = (grouped["weighted_mae"] / grouped["n_samples"]).reindex(_HOUR_BANDS)

        status_note = "" if r["status"] == "FINISHED" else f" [{r['status']}]"
        fig.add_trace(go.Bar(
            name=f"{r['name']}{status_note}",
            x=_HOUR_BANDS,
            y=band_mae.values,
            text=[f"{v:.3f}" if pd.notna(v) else "" for v in band_mae.values],
            textposition="outside",
            marker_color=COLORS[i % len(COLORS)],
            hovertemplate="Band: %{x}<br>MAE: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
        has_data = True

    if not has_data:
        fig.add_annotation(
            text="No by-hour data found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig.update_layout(
        title="Peak vs. Off-Peak Performance — Val Set",
        barmode="group",
        xaxis=dict(title="Hour Band"),
        yaxis=dict(title="Weighted MAE", gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=460,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_cluster_fig(runs: list[dict]) -> go.Figure:
    """Grouped bar of MAE per cluster id, from each run's by_cluster CSV."""
    fig = go.Figure()
    has_data = False

    for i, r in enumerate(runs):
        if r["by_cluster"] is None:
            continue
        df = r["by_cluster"]
        status_note = "" if r["status"] == "FINISHED" else f" [{r['status']}]"
        fig.add_trace(go.Bar(
            name=f"{r['name']}{status_note}",
            x=df["cluster"].astype(str),
            y=df["mae"],
            text=[f"{v:.3f}" for v in df["mae"]],
            textposition="outside",
            marker_color=COLORS[i % len(COLORS)],
            hovertemplate="Cluster: %{x}<br>MAE: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
        has_data = True

    if not has_data:
        fig.add_annotation(
            text="No by-cluster data found (requires a Phase 2+ run)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig.update_layout(
        title="Performance by Cluster — Val Set",
        barmode="group",
        xaxis=dict(title="Cluster ID"),
        yaxis=dict(title="MAE", gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=460,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_volume_tier_fig(runs: list[dict]) -> go.Figure:
    """Grouped bar of MAE per volume tier (low/mid/high), from by_volume_tier."""
    fig = go.Figure()
    has_data = False

    for i, r in enumerate(runs):
        if r["by_volume_tier"] is None:
            continue
        df = r["by_volume_tier"].set_index("volume_tier").reindex(["low", "mid", "high"])
        status_note = "" if r["status"] == "FINISHED" else f" [{r['status']}]"
        fig.add_trace(go.Bar(
            name=f"{r['name']}{status_note}",
            x=["low", "mid", "high"],
            y=df["mae"].values,
            text=[f"{v:.3f}" if pd.notna(v) else "" for v in df["mae"].values],
            textposition="outside",
            marker_color=COLORS[i % len(COLORS)],
            hovertemplate="Tier: %{x}<br>MAE: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))
        has_data = True

    if not has_data:
        fig.add_annotation(
            text="No by-volume-tier data found (requires a Phase 2+ run)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    fig.update_layout(
        title="Performance by Volume Tier — Val Set",
        barmode="group",
        xaxis=dict(
            title="Volume Tier (train lag_24 terciles)",
            categoryorder="array", categoryarray=["low", "mid", "high"],
        ),
        yaxis=dict(title="MAE", gridcolor="#eee"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=460,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def _find_best_baseline_run(runs: list[dict]) -> dict | None:
    """Lowest val_mae among FINISHED naive/linear runs, or None if none qualify."""
    candidates = [
        r for r in runs
        if r["family"] in ("naive", "linear")
        and r["status"] == "FINISHED"
        and r["metrics"].get("val_mae") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["metrics"]["val_mae"])


def build_significance_fig(runs: list[dict]) -> go.Figure:
    """
    Paired bootstrap significance test of each run's val predictions against
    the best baseline run (lowest val_mae among naive/linear), aligned via
    merge on (station_idx, timestamp).

    Horizontal forest plot: one row per comparable run, x = delta MAE vs.
    baseline, marker + asymmetric error bar = 95% CI. Marker color encodes
    significance (CI excludes zero) and direction. Runs without a predictions
    artifact are fully excluded from the plotted rows (not greyed-in-place --
    grey already means "tested, not significant") and are named in a footnote
    instead.
    """
    fig = go.Figure()
    baseline = _find_best_baseline_run(runs)

    if baseline is None or baseline.get("predictions") is None:
        fig.add_annotation(
            text="No baseline run (naive/linear) with a predictions artifact "
                 "found -- run baseline.py to enable the significance panel.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    base_df = baseline["predictions"]
    comparable = [
        r for r in runs
        if r["run_id"] != baseline["run_id"] and r.get("predictions") is not None
    ]

    rows = []      # (name, delta, ci_lo, ci_hi, p_value, n_stations)
    excluded = [r["name"] for r in runs if r["run_id"] != baseline["run_id"] and r.get("predictions") is None]

    for r in comparable:
        merged = r["predictions"].merge(
            base_df, on=["station_idx", "timestamp"], suffixes=("_r", "_base"),
        )
        if merged.empty:
            excluded.append(f"{r['name']} (no overlapping station/timestamp rows)")
            continue
        if not np.allclose(merged["y_true_r"], merged["y_true_base"], equal_nan=True):
            print(
                f"WARNING: y_true mismatch between {r['name']} and baseline "
                f"{baseline['name']} on overlapping rows -- possible "
                f"data_snapshot_id drift; using {r['name']}'s y_true."
            )
        test = paired_bootstrap_test(
            y_true=merged["y_true_r"].values,
            y_pred_a=merged["y_pred_base"].values,
            y_pred_b=merged["y_pred_r"].values,
            station_idx=merged["station_idx"].values,
        )
        rows.append((r["name"], test["delta"], test["ci_lo"], test["ci_hi"],
                     test["p_value"], test["n_stations"]))

    if not rows:
        fig.add_annotation(
            text="No comparable runs with a predictions artifact found "
                 f"(baseline: {baseline['name']}).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # Best-improvement-first (most negative delta at top), standard forest-plot order.
    rows.sort(key=lambda x: x[1])
    names, deltas, ci_los, ci_his, p_values, n_stations = zip(*rows)

    colors = []
    for delta, lo, hi in zip(deltas, ci_los, ci_his):
        significant = lo > 0 or hi < 0
        colors.append(
            "#c0392b" if significant and delta > 0 else   # worse than baseline
            "#27ae60" if significant and delta < 0 else    # better than baseline
            "#95a5a6"                                        # CI crosses zero
        )

    err_plus = [hi - d for d, hi in zip(deltas, ci_his)]
    err_minus = [d - lo for d, lo in zip(deltas, ci_los)]
    hover = [
        f"{n}<br>&Delta; MAE vs {baseline['name']}: {d:.4f}"
        f"<br>95% CI: [{lo:.4f}, {hi:.4f}]<br>p = {p:.4f}<br>n_stations = {ns}"
        for n, d, lo, hi, p, ns in zip(names, deltas, ci_los, ci_his, p_values, n_stations)
    ]

    fig.add_trace(go.Scatter(
        x=deltas, y=names, mode="markers+text",
        marker=dict(size=12, color=colors, line=dict(width=1, color="#333")),
        error_x=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus,
                     color="#555", thickness=1.5, width=6),
        text=[f"p={p:.3f}" for p in p_values],
        textposition="middle right",
        hovertext=hover, hoverinfo="text",
        name="Δ MAE vs baseline",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#333",
                  annotation_text=f"baseline: {baseline['name']}", annotation_position="top")

    if excluded:
        fig.add_annotation(
            text="Excluded (no predictions artifact / pre-Phase-3 run): " + ", ".join(excluded),
            xref="paper", yref="paper", x=0.0, y=-0.15, showarrow=False,
            font=dict(size=11, color="#888"), align="left", xanchor="left",
        )

    fig.update_layout(
        title=f"Statistical Significance — Δ MAE vs. Baseline ({baseline['name']}), Val Set",
        xaxis=dict(title="Δ MAE (positive = worse than baseline)", gridcolor="#eee", zeroline=False),
        yaxis=dict(title="", categoryorder="array", categoryarray=list(names[::-1])),
        showlegend=False,
        height=max(360, 70 * len(names) + 160),
        margin=dict(b=90),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bike Demand Forecast — Model Comparison</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5;
    color: #2d3436;
    line-height: 1.6;
  }}
  .container {{ max-width: 1440px; margin: 0 auto; padding: 2rem 1.5rem; }}
  header {{ margin-bottom: 2rem; }}
  header h1 {{ font-size: 1.75rem; font-weight: 700; color: #1a1a2e; }}
  .subtitle {{ color: #636e72; font-size: 0.9rem; margin-top: 0.25rem; }}
  .section {{
    background: #fff;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07);
  }}
  .section h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #dfe6e9;
  }}
  .section-meta {{
    font-size: 0.82rem;
    color: #636e72;
    margin-bottom: 1rem;
  }}
  table.leaderboard {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.86rem;
  }}
  table.leaderboard th {{
    background: #2d3436;
    color: #fff;
    padding: 0.55rem 0.85rem;
    text-align: left;
    white-space: nowrap;
  }}
  table.leaderboard td {{
    padding: 0.48rem 0.85rem;
    border-bottom: 1px solid #eee;
    vertical-align: middle;
  }}
  table.leaderboard tr:hover td {{ background: #f8f9fa; }}
  .row-failed td {{ opacity: 0.65; }}
  .badge-warn {{
    display: inline-block;
    background: #fff3cd;
    color: #856404;
    font-size: 0.72rem;
    padding: 1px 7px;
    border-radius: 4px;
    border: 1px solid #ffc107;
    margin-left: 5px;
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>Bike Demand Forecast — Model Comparison Report</h1>
    <p class="subtitle">
      Generated <strong>{generated_at}</strong>
      &nbsp;·&nbsp; Experiment: <strong>{experiment_name}</strong>
      &nbsp;·&nbsp; {n_runs} runs
      (<span style="color:#27ae60">{n_finished} finished</span>,
       <span style="color:#c0392b">{n_other} incomplete/failed</span>)
    </p>
  </header>

  <div class="section">
    <h2>1. Summary Leaderboard</h2>
    <p class="section-meta">
      All runs sorted by start time (newest first).
      Best value per metric column is highlighted in green.
      Incomplete or failed runs are flagged.
    </p>
    {leaderboard}
  </div>

  <div class="section">
    <h2>2. Model Family Comparison</h2>
    <p class="section-meta">Best run per family by Val MAE.</p>
    {fig_family}
  </div>

  <div class="section">
    <h2>3. GCN Adjacency &times; Loss Function Comparison</h2>
    <p class="section-meta">
      Comparing KNN / Flow / Combined adjacency with MSE vs. Poisson loss.
    </p>
    {fig_gcn}
  </div>

  <div class="section">
    <h2>4. Training Curves</h2>
    <p class="section-meta">
      Per-epoch validation MAE and training loss for LSTM and GCN models.
      Only runs with logged training history are shown.
    </p>
    {fig_curves}
  </div>

  <div class="section">
    <h2>5. MAE by Hour of Day</h2>
    <p class="section-meta">
      Error patterns across 24 hours on the validation set.
      Peaks typically correspond to morning and evening rush hours.
    </p>
    {fig_hour}
  </div>

  <div class="section">
    <h2>6. Per-Station Sample — Heatmap</h2>
    <p class="section-meta">
      Top {top_n} worst (▲) and best (▼) stations by average MAE across finished runs.
      Darker red = higher error; darker green = lower error.
    </p>
    {fig_station}
  </div>

  <div class="section">
    <h2>7. Per-Station MAE Distribution</h2>
    <p class="section-meta">
      Box plot of station-level MAE across all stations on the validation set.
      Shows median, IQR, and outlier spread per model.
    </p>
    {fig_dist}
  </div>

  <div class="section">
    <h2>8. Peak vs. Off-Peak Performance</h2>
    <p class="section-meta">
      n_samples-weighted MAE by hour band (off-peak / AM peak / mid / PM peak),
      derived from each run's by-hour breakdown on the validation set.
    </p>
    {fig_peak_offpeak}
  </div>

  <div class="section">
    <h2>9. Performance by Cluster</h2>
    <p class="section-meta">
      MAE by station cluster on the validation set. Requires a Phase 2+ run
      (older runs show as empty).
    </p>
    {fig_cluster}
  </div>

  <div class="section">
    <h2>10. Performance by Volume Tier</h2>
    <p class="section-meta">
      MAE by demand volume tier (low/mid/high, terciles of train lag_24) on
      the validation set. Requires a Phase 2+ run (older runs show as empty).
    </p>
    {fig_volume_tier}
  </div>

  <div class="section">
    <h2>11. Statistical Significance</h2>
    <p class="section-meta">
      Paired bootstrap test (resampled over stations) of each run's validation
      predictions against the best baseline run (lowest Val MAE among naive/linear).
      Green/red when the 95% CI on &Delta; MAE excludes zero; grey when it doesn't.
      Requires a Phase 3+ run with a logged predictions artifact -- older runs are
      listed below the chart, not plotted.
    </p>
    {fig_significance}
  </div>

</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(
    runs: list[dict],
    experiment_name: str,
    top_n: int,
    output_path: Path,
) -> None:
    n_finished = sum(1 for r in runs if r["status"] == "FINISHED")

    print("  Building leaderboard...")
    leaderboard = build_leaderboard_html(runs)

    print("  Building figures...")
    figs = [
        build_family_comparison_fig(runs),
        build_gcn_comparison_fig(runs),
        build_training_curves_fig(runs),
        build_by_hour_fig(runs),
        build_by_station_fig(runs, top_n=top_n),
        build_station_distribution_fig(runs),
        build_peak_offpeak_fig(runs),
        build_cluster_fig(runs),
        build_volume_tier_fig(runs),
        build_significance_fig(runs),
    ]

    # First figure embeds the full plotly.min.js inline; the rest reuse it.
    print("  Serialising figures (first one bundles Plotly JS inline)...")
    divs = [figs[0].to_html(full_html=False, include_plotlyjs=True)]
    divs += [f.to_html(full_html=False, include_plotlyjs=False) for f in figs[1:]]

    html = _HTML.format(  # noqa: E501
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        experiment_name=experiment_name,
        n_runs=len(runs),
        n_finished=n_finished,
        n_other=len(runs) - n_finished,
        leaderboard=leaderboard,
        fig_family=divs[0],
        fig_gcn=divs[1],
        fig_curves=divs[2],
        fig_hour=divs[3],
        fig_station=divs[4],
        fig_dist=divs[5],
        fig_peak_offpeak=divs[6],
        fig_cluster=divs[7],
        fig_volume_tier=divs[8],
        fig_significance=divs[9],
        top_n=top_n,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML model comparison report from MLflow."
    )
    parser.add_argument(
        "--experiment-name",
        default="bike-demand-forecasting",
        help="MLflow experiment name (default: bike-demand-forecasting)",
    )
    parser.add_argument(
        "--tracking-uri",
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of worst + best stations to sample (default: 20 each)",
    )
    parser.add_argument(
        "--output",
        default="reports/report.html",
        help="Output path for the HTML report (default: reports/report.html)",
    )
    args = parser.parse_args()

    print(f"Connecting to MLflow at {args.tracking_uri}...")
    runs = load_runs(args.experiment_name, args.tracking_uri)
    print(f"  Loaded {len(runs)} runs from experiment '{args.experiment_name}'")

    output_path = Path(args.output)
    print(f"Rendering report → {output_path}")
    render_report(
        runs=runs,
        experiment_name=args.experiment_name,
        top_n=args.top_n,
        output_path=output_path,
    )

    size_kb = output_path.stat().st_size // 1024
    print(f"Done. Report written to {output_path} ({size_kb} KB)")


if __name__ == "__main__":
    main()
