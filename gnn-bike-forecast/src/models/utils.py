"""
utils.py
--------
Shared, dependency-light infrastructure for reproducibility: seeding, device
selection, and data-snapshot hashing/validation. No `mlflow` import here —
`data_loader.py` needs the snapshot validator without requiring `mlflow` to
be importable just to load a parquet file.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# The four processed artifacts that define a "data snapshot". targets_*.parquet
# are derived directly from the same rows as features_*.parquet and are
# intentionally excluded.
SNAPSHOT_ARTIFACTS = (
    "features_train.parquet",
    "features_val.parquet",
    "features_test.parquet",
    "graph_data.pkl",
)

_HASH_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed all RNG sources used by this project's training scripts.

    Reproducibility is exact on CPU. On MPS/CUDA it is best-effort:
    `torch.use_deterministic_algorithms(True, warn_only=True)` is set, but
    PyTorch does not currently guarantee deterministic kernels for all ops on
    MPS (e.g. the scatter/gather reductions inside GCNConv), so `warn_only`
    is used rather than a hard failure.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data snapshot hashing
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_data_snapshot_id(data_dir: Path | str) -> str:
    """Sha256 over the four processed artifacts, in a fixed sorted order."""
    data_dir = Path(data_dir)
    file_hashes = {
        name: _sha256_file(data_dir / name) for name in SNAPSHOT_ARTIFACTS
    }
    combined = "".join(file_hashes[name] for name in sorted(SNAPSHOT_ARTIFACTS))
    return hashlib.sha256(combined.encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def get_validated_snapshot_id(data_dir: Path | str) -> str:
    """Recompute the data snapshot id and assert it matches SNAPSHOT.json.

    Memoized per resolved `data_dir` so the ~215MB of hashing runs at most
    once per process, regardless of how many splits/callers request it.
    Raises FileNotFoundError if SNAPSHOT.json is missing, or ValueError if
    the current files on disk have drifted from what it recorded.
    """
    data_dir = Path(data_dir).resolve()
    snapshot_path = data_dir / "SNAPSHOT.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"{snapshot_path} not found. Run notebook 04 (Feature Engineering) "
            "to completion first — it writes SNAPSHOT.json as its final step."
        )

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    expected = snapshot["data_snapshot_id"]
    actual = compute_data_snapshot_id(data_dir)
    if actual != expected:
        raise ValueError(
            f"data_snapshot_id mismatch in {data_dir}: SNAPSHOT.json says "
            f"{expected}, but the current processed files hash to {actual}. "
            "Regenerate processed data by re-running notebook 04 end-to-end."
        )

    logger.info("Data snapshot validated: %s", actual)
    return actual
