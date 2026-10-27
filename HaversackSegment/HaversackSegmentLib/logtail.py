"""Read the lines a running program appends to a file, as they arrive. No Qt, no Slicer.

The runner sends a step's stdout and stderr straight to files (QProcess.setStandardOutputFile /
setStandardErrorFile: the operating system does the redirection) and polls them with a timer,
rather than reading QProcess's pipes. On Windows the pipe's readyRead signals reached Slicer's
event loop only when the program ended, so uv's download lines and haversack's per-patch progress
all appeared at once (2026-10-27). A file read from a timer behaves the same on every platform,
and leaves each step's log on disk for a look afterwards.
"""
from __future__ import annotations

import codecs
import os


class LogTail:
    """The complete lines appended to ``path`` since the last :meth:`read`."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._file = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._partial = ""

    def read(self, final: bool = False) -> list[str]:
        """New complete lines (without line endings, blank ones dropped). ``final`` also returns
        an unterminated last line - the program has ended, so nothing more will complete it."""
        if self._file is None:
            try:
                self._file = open(self.path, "rb")
            except OSError:                      # not created yet: the program has not started
                return []
        data = self._file.read()
        text = self._partial + self._decoder.decode(data or b"", final=final)
        lines = text.split("\n")
        self._partial = "" if final else lines.pop()
        return [line.rstrip("\r") for line in lines if line.strip()]

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
