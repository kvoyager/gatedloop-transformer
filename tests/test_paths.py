"""
test_paths.py
─────────────────────────────────────────────────────────────────────────────
Where relative config paths resolve, and what the code snapshot captures, for
a source checkout vs an installed copy (simulated by pointing the package dir
at a fake site-packages). CPU only, < 1 s.

  python tests/test_paths.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatedloop import paths                                   # noqa: E402
from gatedloop.training import callbacks                      # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _installed_copy():
    """Pretend the package lives in <tmp>/site-packages/gatedloop, launched from <tmp>/work."""
    tmp = Path(tempfile.mkdtemp(prefix="gl_paths_"))
    pkg = tmp / "site-packages" / "gatedloop"
    (pkg / "model").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "model" / "lm.py").write_text("x = 1\n")
    (tmp / "site-packages" / "torch").mkdir()
    (tmp / "site-packages" / "torch" / "huge.py").write_text("y = 2\n")   # must NOT be snapshotted
    work = tmp / "work"
    work.mkdir()
    saved = paths.PACKAGE_DIR, paths._LAUNCH_CWD
    paths.PACKAGE_DIR, paths._LAUNCH_CWD = pkg, work
    try:
        yield tmp, pkg, work
    finally:
        paths.PACKAGE_DIR, paths._LAUNCH_CWD = saved


def test_source_checkout_resolves_to_repo_root():
    with _env(GATEDLOOP_ROOT=None):
        assert paths.source_checkout() == REPO
        assert paths.project_root() == REPO
        assert Path(paths.resolve("./artifacts/mix")) == REPO / "artifacts" / "mix"
    print("  source checkout → repo root — OK")


def test_absolute_and_none_pass_through():
    a = str(Path(tempfile.gettempdir()).resolve() / "x.bin")
    assert paths.resolve(a) == a and paths.resolve(None) is None
    print("  absolute paths and None pass through — OK")


def test_env_override():
    tmp = Path(tempfile.mkdtemp(prefix="gl_root_")).resolve()
    with _env(GATEDLOOP_ROOT=str(tmp)):
        assert paths.project_root() == tmp
        assert Path(paths.resolve("runs")) == tmp / "runs"
    print("  GATEDLOOP_ROOT overrides the root — OK")


def test_installed_copy_resolves_to_launch_dir():
    with _env(GATEDLOOP_ROOT=None), _installed_copy() as (tmp, pkg, work):
        assert paths.source_checkout() is None
        assert paths.project_root() == work
        resolved = Path(paths.resolve("./runs_ladder"))
        assert resolved == work / "runs_ladder"
        assert "site-packages" not in resolved.parts
    print("  installed copy → launch directory, never site-packages — OK")


def test_installed_snapshot_is_package_only():
    with _env(GATEDLOOP_ROOT=None), _installed_copy() as (tmp, pkg, work):
        snap = callbacks.CodeSnapshot()
        assert snap._installed and snap.source_dir == pkg
        data = callbacks._snapshot_code_bytes(snap.source_dir, snap.source_dir.parent)
        names = sorted(zipfile.ZipFile(io.BytesIO(data)).namelist())
        assert names == ["gatedloop/__init__.py", "gatedloop/model/lm.py"], names
    print(f"  installed snapshot holds only the package {names} — OK")


def test_checkout_snapshot_is_repo():
    snap = callbacks.CodeSnapshot()
    assert not snap._installed and snap.source_dir == REPO
    print("  checkout snapshot covers the repo — OK")


def main():
    for fn in [test_source_checkout_resolves_to_repo_root, test_absolute_and_none_pass_through,
               test_env_override, test_installed_copy_resolves_to_launch_dir,
               test_installed_snapshot_is_package_only, test_checkout_snapshot_is_repo]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n✅ all path tests passed")


if __name__ == "__main__":
    main()
