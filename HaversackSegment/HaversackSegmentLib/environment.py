"""The Python environment haversack runs in, kept apart from Slicer's own.

haversack needs Python 3.12-3.14 and installs only with uv (several of its dependencies come
from git tags, which uv resolves and pip does not), so it cannot go into Slicer's site-packages.
The module keeps one virtual environment under its home directory instead, built with a uv that
pip installs into Slicer (a single binary wheel), and runs everything there as a subprocess:
haversack for inference, lnq-segmenter for the LNQ models' downloads. The environment brings
its own interpreter, so it works with a Slicer whose Python is older than haversack allows.

No Slicer imports here: the paths and command lines are plain data, testable outside Slicer.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PYTHON_VERSION = "3.12"
DEFAULT_HOME = Path("~/.slicer-haversack")
# Region-based models (PanTS), --folds and --json-progress are in mhalle/haversack#4; until it is
# merged the default installs that branch from Steve's fork. Then: mhalle/haversack, main.
DEFAULT_HAVERSACK = "haversack @ git+https://github.com/pieper/haversack@region-models"
DEFAULT_LNQ = "lnq-segmenter @ git+https://github.com/pieper/lnq-segmenter"
# What this module needs from `haversack segment`; an older install lacks it and is reported as
# out of date rather than failing a run halfway.
REQUIRED_SEGMENT_OPTIONS = ("--json-progress", "--folds")


@dataclass(frozen=True)
class Environment:
    home: Path

    @classmethod
    def at(cls, home=None) -> "Environment":
        return cls(Path(home or DEFAULT_HOME).expanduser())

    @property
    def venv(self) -> Path:
        return self.home / "env"

    @property
    def models(self) -> Path:
        """Where the extra (license-gated) models are downloaded."""
        return self.home / "models"

    @property
    def work(self) -> Path:
        """Scratch space for one run's input and output files."""
        return self.home / "work"

    def _bin(self, name: str) -> Path:
        if sys.platform == "win32":
            return self.venv / "Scripts" / f"{name}.exe"
        return self.venv / "bin" / name

    @property
    def python(self) -> Path:
        if sys.platform == "win32":
            return self.venv / "Scripts" / "python.exe"
        return self.venv / "bin" / "python"

    @property
    def haversack(self) -> Path:
        return self._bin("haversack")

    @property
    def lnq_segmenter(self) -> Path:
        return self._bin("lnq-segmenter")

    def installed(self) -> bool:
        return self.haversack.is_file() and self.lnq_segmenter.is_file()

    # -- install -------------------------------------------------------------------------------

    def create_command(self, uv: str) -> list[str]:
        """Create (or keep) the virtual environment with uv's own Python 3.12, downloaded once
        if the machine has none."""
        return [uv, "venv", "--allow-existing", "--python", PYTHON_VERSION, str(self.venv)]

    def install_command(self, uv: str, haversack: str = DEFAULT_HAVERSACK,
                        lnq: str = DEFAULT_LNQ) -> list[str]:
        """Install or update both CLIs. ``--torch-backend auto`` has uv pick the torch build for
        the machine's GPU (CUDA on Windows needs it; PyPI's Windows torch is CPU-only)."""
        cmd = [uv, "pip", "install", "--python", str(self.python), "--upgrade",
               "--torch-backend", "auto"]
        for spec in (haversack, lnq):
            cmd += _spec_args(spec)
        return cmd

    # -- run -----------------------------------------------------------------------------------

    def tasks_command(self) -> list[str]:
        return [str(self.haversack), "tasks", "--json"]

    def segment_help_command(self) -> list[str]:
        return [str(self.haversack), "segment", "--help"]

    def lnq_list_command(self) -> list[str]:
        return [str(self.lnq_segmenter), "list", "--json"]

    def lnq_download_command(self, name: str, version: str | None = None) -> list[str]:
        spec = f"{name}@{version}" if version else name
        return [str(self.lnq_segmenter), "download", spec, "--json-progress"]

    def segment_command(self, input_path, task, output_path, *, device="auto", folds=None,
                        extra=()) -> list[str]:
        cmd = [str(self.haversack), "segment", str(input_path), "--task", str(task),
               "-o", str(output_path), "--device", device, "--json-progress"]
        if folds:
            cmd += ["--folds", folds if isinstance(folds, str) else ",".join(str(f) for f in folds)]
        return cmd + list(extra)


def _spec_args(spec: str) -> list[str]:
    """A requirement as uv takes it: ``-e <path>`` for an editable local checkout (a path that
    exists, or one written ``-e PATH``), else the requirement string itself."""
    s = spec.strip()
    if s.startswith("-e "):
        return ["-e", os.path.expanduser(s[3:].strip())]
    if os.path.isdir(os.path.expanduser(s)):
        return ["-e", os.path.expanduser(s)]
    return [s]


def missing_segment_options(help_text: str) -> list[str]:
    """The options this module needs that `haversack segment --help` does not list."""
    return [o for o in REQUIRED_SEGMENT_OPTIONS if o not in help_text]
