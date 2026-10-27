"""Progress from the lines uv prints when its output is not a terminal. No Qt, no Slicer.

Through a pipe uv shows no progress bars, but it does print, as they happen::

    Resolved 171 packages in 11.06s
    Downloading torch (73.5MiB)              a large download starts
    Downloading cpython-3.12.11-macos-aarch64-none (download) (17.4MiB)
     Downloading torch                       ... and has finished (note the leading space)
    Building haversack @ git+https://...     a package built from source
     Built haversack @ git+https://...
    Prepared 171 packages in 45.2s
    Installed 171 packages in 3.1s

:class:`UvProgress` turns those into a status line and a fraction (bytes of the large downloads
finished over bytes announced so far - the total grows as uv starts more of them, so the fraction
can step back, but it moves while anything is moving).
"""
from __future__ import annotations

import re

_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
_START = re.compile(r"^Downloading (\S+)(?: \((?!\d)[^)]*\))* \(([\d.]+)\s*(B|KiB|MiB|GiB)\)\s*$")
_END = re.compile(r"^ Download(?:ing|ed) (\S+)\s*$")
_COUNT = re.compile(r"^(Resolved|Prepared|Installed|Uninstalled|Audited) (\d+) packages?")
_BUILD = re.compile(r"^(?: (Built)|(Building)) (\S+)")


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit in ("B", "KiB") else f"{n:.1f} {unit}"
        n /= 1024


def human_seconds(s: float) -> str:
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


class UvProgress:
    def __init__(self):
        self.sizes: dict[str, int] = {}
        self.finished: set[str] = set()
        self.building: set[str] = set()
        self.counts: dict[str, int] = {}

    def feed(self, line: str) -> bool:
        """Take one output line; True if it changed the progress."""
        m = _START.match(line)
        if m:
            self.sizes[m.group(1)] = int(float(m.group(2)) * _UNITS[m.group(3)])
            return True
        m = _END.match(line)
        if m and m.group(1) in self.sizes:
            self.finished.add(m.group(1))
            return True
        m = _COUNT.match(line)
        if m:
            self.counts[m.group(1)] = int(m.group(2))
            return True
        m = _BUILD.match(line)
        if m:
            name = m.group(3)
            (self.building.discard if m.group(1) else self.building.add)(name)
            return True
        return False

    def fraction(self) -> float | None:
        total = sum(self.sizes.values())
        if not total:
            return None
        return sum(self.sizes[n] for n in self.finished) / total

    def text(self) -> str:
        parts = []
        if "Resolved" in self.counts:
            parts.append(f"resolved {self.counts['Resolved']} packages")
        if self.sizes:
            done = sum(self.sizes[n] for n in self.finished)
            total = sum(self.sizes.values())
            active = [n for n in self.sizes if n not in self.finished]
            parts.append(f"downloaded {len(self.finished)} of {len(self.sizes)} large files, "
                         f"{human_bytes(done)} of {human_bytes(total)}"
                         + (f" (now: {', '.join(sorted(active, key=lambda n: -self.sizes[n])[:3])})"
                            if active else ""))
        if self.building:
            parts.append("building " + ", ".join(sorted(n.split("@")[0] for n in self.building)))
        for key in ("Prepared", "Installed"):
            if key in self.counts:
                parts.append(f"{key.lower()} {self.counts[key]}")
        return "; ".join(parts)
