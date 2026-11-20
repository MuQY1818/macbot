"""macOS plumbing: PCAN driver loading, running under sudo, and log files.

Why this exists
---------------
* The PCAN-USB adapter is driven by the PCBUSB library (MacCAN). ``motorbridge`` loads it by
  bare name, but under ``sudo`` the dynamic loader ignores ``/usr/local/lib``. We set
  ``DYLD_FALLBACK_LIBRARY_PATH``, re-exec the interpreter once (dyld only reads it at start),
  and preload the library.
* The Orbbec SDK needs root on macOS (``uvc_open -3`` otherwise), so the robot commands run
  under ``sudo -E``. Files written as root are handed back to the invoking user, and ``~``
  resolves to that user's home.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PCBUSB = Path("/usr/local/lib/libPCBUSB.dylib")


def preload_pcan() -> bool:
    """Make the PCBUSB library loadable. May re-exec the current process. Returns True if loaded."""
    if sys.platform != "darwin" or not PCBUSB.exists():
        return False
    if "/usr/local/lib" not in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", ""):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/usr/local/lib:/usr/lib"
        # Preserve interpreter options such as -m on the init-to-collect handoff.
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
    import ctypes

    ctypes.CDLL(str(PCBUSB), mode=ctypes.RTLD_GLOBAL)
    return True


def user_home() -> Path:
    """Home of the real user, also under sudo."""
    u = os.environ.get("SUDO_USER")
    return Path(os.path.expanduser(f"~{u}" if u else "~"))


def expand(p, base: Path | None = None) -> Path:
    """Expand ``~`` to the real user's home; resolve relative paths against ``base``."""
    s = str(p)
    if s == "~" or s.startswith("~/"):
        return user_home() / s[2:]
    q = Path(s)
    return q if q.is_absolute() or base is None else base / q


def give_back(path: Path) -> None:
    """Under sudo, chown a file or folder we created back to the invoking user."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for p in (path, path.parent):
        give_back(p)
    return path


class _Tee:
    """Mirror terminal output into a log file (no shell pipe, so Ctrl-C still works)."""

    def __init__(self, stream, fp):
        self.stream, self.fp = stream, fp

    def write(self, s):
        self.stream.write(s)
        try:
            self.fp.write(s)
            self.fp.flush()
        except ValueError:
            pass
        return len(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()


def start_log(folder: Path, prefix: str) -> Path:
    ensure_dir(folder)
    p = folder / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fp = open(p, "w", buffering=1)
    give_back(p)
    sys.stdout, sys.stderr = _Tee(sys.stdout, fp), _Tee(sys.stderr, fp)
    print(f"log: {p}", flush=True)
    return p
