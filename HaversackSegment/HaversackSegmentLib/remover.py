"""Delete directory trees a little at a time, for a caller on Slicer's main thread. No Qt.

Uninstall removes the environment (tens of thousands of files with torch) and, if asked, uv's
cache and the managed Python. ``shutil.rmtree`` on a Python thread looked like the way to keep
Slicer usable meanwhile, but Slicer's main thread holds the GIL while its event loop idles, so a
Python thread runs only in the moments the main thread runs Python: on Windows (where each delete
is slow) Uninstall and Diagnose took minutes with nothing on the panel (2026-10-28). Instead the
widget calls :meth:`TreeRemover.step` from a timer; each call deletes for a few tens of
milliseconds and returns, and :meth:`TreeRemover.text` / :meth:`TreeRemover.fraction` say where
it is.

Windows refuses to delete a read-only file, so one is made writable and tried again; a file
still in use (a running ``python.exe`` from the environment) is recorded in ``failed`` and the
rest is removed anyway.
"""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

#: failures kept with their messages; the rest are only counted
KEEP_FAILURES = 20


class TreeRemover:
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths if os.path.lexists(p)]
        self.counted = 0           # entries found so far (pass 1)
        self.total = None          # entries to remove, once counted
        self.removed = 0
        self.failed: list = []     # (path, message), the first KEEP_FAILURES
        self.failures = 0
        self.current = None        # the tree being worked on
        self.done = not self.paths
        self._work = self._steps()

    def step(self, budget_s: float = 0.04) -> bool:
        """Work for about ``budget_s``; True while there is more to do."""
        if self.done:
            return False
        end = time.monotonic() + budget_s
        for _ in self._work:
            if time.monotonic() >= end:
                return True
        self.done = True
        return False

    def fraction(self):
        if self.total is None:
            return None
        return self.removed / self.total if self.total else 1.0

    def text(self) -> str:
        where = self.current.name if self.current is not None else (self.paths[0].name if self.paths else "")
        if self.done:
            return f"removed {self.removed:,} files and folders" + (
                f", {self.failures:,} could not be removed" if self.failures else "")
        if self.total is None:
            return f"counting files in {where}: {self.counted:,}"
        return (f"removing {where}: {self.removed:,} of {self.total:,} files and folders"
                + (f" ({self.failures:,} could not be removed)" if self.failures else ""))

    # ----- the work, one entry per yield -----

    def _steps(self):
        for root in self.paths:
            self.current = root
            self.counted += 1
            if root.is_dir() and not root.is_symlink():
                for _dirpath, dirnames, filenames in os.walk(root):
                    self.counted += len(dirnames) + len(filenames)
                    yield
        self.total = self.counted
        for root in self.paths:
            self.current = root
            if not root.is_dir() or root.is_symlink():
                self._remove(str(root), is_dir=False)
                yield
                continue
            # deepest first; symlinked folders are listed in dirnames and not walked into
            for dirpath, dirnames, filenames in os.walk(root, topdown=False):
                for name in filenames:
                    self._remove(os.path.join(dirpath, name), is_dir=False)
                    yield
                for name in dirnames:
                    path = os.path.join(dirpath, name)
                    self._remove(path, is_dir=not os.path.islink(path))
                    yield
            self._remove(str(root), is_dir=True)
            yield

    def _remove(self, path: str, is_dir: bool):
        for attempt in (0, 1):
            try:
                if is_dir:
                    os.rmdir(path)
                else:
                    try:
                        os.unlink(path)
                    except (IsADirectoryError, PermissionError):
                        if not os.path.islink(path):
                            raise
                        os.rmdir(path)           # a folder symlink or junction on Windows
                self.removed += 1
                return
            except FileNotFoundError:
                self.removed += 1
                return
            except OSError as exc:
                if attempt == 0 and isinstance(exc, PermissionError):
                    try:                             # Windows: a read-only file
                        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                        continue
                    except OSError:
                        pass
                self.failures += 1
                if len(self.failed) < KEEP_FAILURES:
                    self.failed.append((path, f"{exc.__class__.__name__}: {exc.strerror or exc}"))
                return
