"""
write_git_provenance.py
------------------------
Captures the current git HEAD sha and dirty status into GIT_PROVENANCE.json
at the project root, for shipping alongside a code+data bundle to an
environment with no .git checkout (e.g. a Colab/Kaggle GPU notebook).
mlflow_utils.py's tag_run_provenance() falls back to this file when `git`
itself isn't available.

Run before packaging a bundle:
    python scripts/write_git_provenance.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def main() -> None:
    sha = _run_git("rev-parse", "HEAD")
    dirty = bool(_run_git("status", "--porcelain"))

    out_path = _PROJECT_ROOT / "GIT_PROVENANCE.json"
    out_path.write_text(json.dumps({"sha": sha, "dirty": dirty}, indent=2) + "\n")
    print(f"Wrote {out_path} | sha={sha[:8]} dirty={dirty}")


if __name__ == "__main__":
    main()
