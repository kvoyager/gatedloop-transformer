"""
logfile.py
─────────────────────────────────────────────────────────────────────────────
Mirror the console output of a run into ``<run dir>/train.log``.

Everything printed to stdout/stderr — the resolved config, the model banner, the
per-validation summaries, the periodic samples, warnings and tracebacks — is
duplicated into the run directory next to ``metrics.csv`` and the checkpoints,
so a finished run is self-describing without having to remember how it was
launched.

Two details make the file readable rather than a dump of terminal noise:

  • ANSI escape sequences are stripped on the way to disk (colour codes, cursor
    moves), so the log is plain text.
  • The Rich progress bar is NOT mirrored. It repaints its live region many
    times a second; captured to a file that becomes thousands of duplicated
    frames. ``train.py`` hands the progress bar the *original* stdout, so its
    repaints go to the terminal only while everything else goes to both.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import TextIO

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")


class _Tee:
    """Write-through proxy: everything goes to ``stream``, a de-ANSI'd copy to ``fh``."""

    def __init__(self, stream: TextIO, fh: TextIO):
        self._stream = stream
        self._fh = fh

    def write(self, s: str) -> int:
        n = self._stream.write(s)
        try:
            self._fh.write(_ANSI.sub("", s))
            self._fh.flush()
        except (ValueError, OSError):        # file closed / disk gone: keep the console alive
            pass
        return n

    def flush(self) -> None:
        self._stream.flush()

    # Pass through the bits libraries probe for (rich, tqdm, faulthandler).
    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


class StdoutHandler(logging.StreamHandler):
    """Logging handler that writes to whatever ``sys.stdout`` is at emit time.

    A plain ``StreamHandler(sys.stdout)`` binds the stream when logging is
    configured, i.e. before ``ConsoleLog`` swaps in its tee, so log messages
    would reach the terminal but not ``train.log``. Used by the Hydra logging
    preset ``configs/hydra/job_logging/console.yaml``."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self) -> TextIO:            # type: ignore[override]
        return sys.stdout

    @stream.setter
    def stream(self, value: TextIO) -> None:
        pass                               # always follow sys.stdout


class ConsoleLog:
    """Context manager that tees stdout/stderr into ``path`` for its lifetime.

    ``original_stdout`` is exposed so callers can hand the untee'd stream to the
    progress bar. Restores the real streams on exit, including on exception —
    the traceback lands in the log first.
    """

    def __init__(self, path: str | Path, append: bool = True):
        self.path = Path(path)
        self.mode = "a" if append else "w"
        self._fh: TextIO | None = None
        self.original_stdout: TextIO = sys.stdout
        self.original_stderr: TextIO = sys.stderr

    def __enter__(self) -> "ConsoleLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, self.mode, encoding="utf-8", errors="replace")
        self.original_stdout, self.original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self.original_stdout, self._fh)
        sys.stderr = _Tee(self.original_stderr, self._fh)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        sys.stdout, sys.stderr = self.original_stdout, self.original_stderr
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return False        # never swallow the exception
