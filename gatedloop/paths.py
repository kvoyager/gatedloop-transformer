"""
paths.py
─────────────────────────────────────────────────────────────────────────────
Where relative paths in the configs (``./artifacts/…``, ``./runs_ladder``) are
anchored, for both ways the package is used:

  source checkout   the repo root — the folder holding pyproject.toml next to
                    the package. Covers a clone and ``pip install -e .``, so
                    paths resolve to where the prep scripts write them, no
                    matter which directory training is launched from.
  installed copy    the directory the process was launched from. The package
                    then lives in site-packages, which must never be used as a
                    place to read data from or write runs to.

``GATEDLOOP_ROOT`` overrides both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DIST_NAME = "gatedloop-transformer"

# Captured on first import, i.e. before Hydra could change the working directory.
_LAUNCH_CWD = Path.cwd()


def source_checkout() -> Path | None:
    """The repo root if the package is running from a source checkout, else None."""
    root = PACKAGE_DIR.parent
    return root if (root / "pyproject.toml").is_file() else None


def project_root() -> Path:
    """The directory relative config paths are resolved against."""
    env = os.environ.get("GATEDLOOP_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return source_checkout() or _LAUNCH_CWD


def resolve(path: str | os.PathLike | None) -> str | None:
    """Absolute path for ``path``: unchanged if absolute, else under project_root()."""
    if path is None:
        return None
    p = Path(path).expanduser()
    return str(p if p.is_absolute() else project_root() / p)


def install_info() -> dict | None:
    """Version and source (e.g. the git commit pip installed from) of an installed
    copy, for the run's provenance record. None for a source checkout."""
    if source_checkout() is not None:
        return None
    try:
        from importlib.metadata import PackageNotFoundError, distribution
        dist = distribution(DIST_NAME)
    except (ImportError, PackageNotFoundError):
        return None
    info: dict = {"installed": True, "version": dist.version}
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
    except (OSError, ValueError):
        direct = {}
    if direct.get("url"):
        info["url"] = direct["url"]
    if direct.get("vcs_info", {}).get("commit_id"):
        info["commit"] = direct["vcs_info"]["commit_id"]
    return info
