"""The Python environment haversack runs in, kept apart from Slicer's own.

haversack needs Python 3.12-3.14 and installs only with uv (several of its dependencies come
from git tags, which uv resolves and pip does not), so it cannot go into Slicer's site-packages.
The module keeps one virtual environment under its home directory instead, built with a uv that
pip installs into Slicer (a single binary wheel), and runs haversack there as a subprocess. The
environment brings its own interpreter, so it works with a Slicer whose Python is older than
haversack allows.

No Slicer imports here: the paths and command lines are plain data, testable outside Slicer.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PYTHON_VERSION = "3.12"
DEFAULT_HOME = Path("~/.slicer-haversack")
# haversack 0.16.0 plus --folds/--json-progress (mhalle/haversack#5) and region-based models such
# as PanTS (mhalle/haversack#6). Until both are merged the default installs the branch on Steve's
# fork that carries them; then: mhalle/haversack, main.
DEFAULT_HAVERSACK = "haversack @ git+https://github.com/pieper/haversack@slicerhaversack"
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

    @property
    def logs(self) -> Path:
        """One log file per step the module runs (install, download, segment)."""
        return self.home / "logs"

    @property
    def uv_cache(self) -> Path:
        """uv's download cache for this module, kept here rather than in uv's shared cache so that
        Uninstall can remove exactly what the module downloaded."""
        return self.home / "uv-cache"

    @property
    def python_dir(self) -> Path:
        """Where uv puts the managed Python the environment is built on."""
        return self.home / "python"

    def uv_env(self) -> dict:
        """Environment variables for every uv command the module runs."""
        return {"UV_CACHE_DIR": str(self.uv_cache), "UV_PYTHON_INSTALL_DIR": str(self.python_dir)}

    def uninstall_paths(self, downloads: bool) -> list[Path]:
        """What Uninstall removes: the environment, and with ``downloads`` also uv's cache and the
        managed Python, so a reinstall downloads everything again. Never the model weights
        (haversack's, shared with TotalSegmentator) or the extra models in ``models/``."""
        return [self.venv] + ([self.uv_cache, self.python_dir] if downloads else [])

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

    def installed(self) -> bool:
        return self.haversack.is_file()

    # -- install -------------------------------------------------------------------------------

    def create_command(self, uv: str, host: dict | None = None) -> list[str]:
        """Create (or keep) the virtual environment with a uv-managed Python 3.12 for the
        machine's own architecture (:func:`python_request`), downloaded once."""
        return [uv, "venv", "--allow-existing", "--managed-python",
                "--python", python_request(host if host is not None else host_info()), str(self.venv)]

    def probe_command(self) -> list[str]:
        """What the environment's interpreter really runs as: one JSON line (:data:`PROBE_CODE`)."""
        return [str(self.python), "-c", PROBE_CODE]

    def verify_command(self) -> list[str]:
        """After an install: torch, its GPU, and haversack import (one JSON line)."""
        return [str(self.python), "-c", VERIFY_CODE]

    def install_command(self, uv: str, haversack: str = DEFAULT_HAVERSACK) -> list[str]:
        """Install or update haversack. ``--torch-backend auto`` has uv pick the torch build for
        the machine's GPU (CUDA on Windows needs it; PyPI's Windows torch is CPU-only)."""
        return [uv, "pip", "install", "--python", str(self.python), "--upgrade",
                "--torch-backend", "auto", *_spec_args(haversack)]

    # -- run -----------------------------------------------------------------------------------

    def tasks_command(self) -> list[str]:
        return [str(self.haversack), "tasks", "--json"]

    def segment_help_command(self) -> list[str]:
        return [str(self.haversack), "segment", "--help"]

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


# -- the machine's architecture --------------------------------------------------------------------
#
# On an Apple Silicon Mac, Slicer itself may run under Rosetta (an x86_64 build or launcher). Its
# Python then reports x86_64, the uv pip installs into it is x86_64, and a bare `uv venv --python
# 3.12` makes an x86_64 environment - for which current torch publishes no wheels, so the install
# fails and nothing else works. The environment is therefore built for the HARDWARE, whatever
# Slicer runs as: an arm64 Python is requested by name, and uv (even an x86_64 one) downloads it.

PROBE_CODE = ("import json, platform, sys; print(json.dumps({'machine': platform.machine(), "
              "'version': sys.version.split()[0], 'executable': sys.executable}))")
VERIFY_CODE = (
    "import json, platform, torch, haversack; "
    "print(json.dumps({'machine': platform.machine(), 'torch': torch.__version__, "
    "'cuda': torch.cuda.is_available(), "
    "'mps': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()), "
    "'haversack': getattr(haversack, '__version__', '?')}))")


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def host_info() -> dict:
    """What this process runs as and what the hardware is. ``translated`` is True when the process
    runs under Rosetta; ``hardware_arm64`` holds even then (``hw.optional.arm64``)."""
    info = {"system": platform.system(), "process_machine": platform.machine(),
            "python": sys.version.split()[0], "executable": sys.executable,
            "hardware_arm64": False, "translated": False}
    if sys.platform == "darwin":
        info["hardware_arm64"] = _sysctl("hw.optional.arm64") == "1"
        info["translated"] = _sysctl("sysctl.proc_translated") == "1"
    else:
        info["hardware_arm64"] = platform.machine().lower() in ("arm64", "aarch64")
    return info


def expected_machine(host: dict) -> str | None:
    """The architecture the environment's Python must report, where it matters: arm64 on an
    Apple Silicon Mac. None elsewhere (any interpreter uv picks is the machine's)."""
    if host.get("system") == "Darwin" and host.get("hardware_arm64"):
        return "arm64"
    return None


def python_request(host: dict) -> str:
    """`uv venv --python`: the arm64 build by name on an Apple Silicon Mac, else 3.12."""
    if expected_machine(host) == "arm64":
        return f"cpython-{PYTHON_VERSION}-macos-aarch64-none"
    return PYTHON_VERSION


def check_probe(probe: dict, host: dict) -> str | None:
    """A sentence saying what is wrong with the environment's interpreter, or None if it is fine."""
    want = expected_machine(host)
    got = probe.get("machine")
    if want and got != want:
        return (f"the environment's Python runs as {got}, but this Mac is Apple Silicon ({want}); "
                "torch has no wheels for that. Click Install again: it rebuilds the environment "
                "with an arm64 Python.")
    if not str(probe.get("version", "")).startswith(PYTHON_VERSION):
        return f"the environment's Python is {probe.get('version')}, not {PYTHON_VERSION}"
    return None


def binary_arch(path) -> str:
    """A program file's architecture from its header: arm64, x86_64, universal, ELF, PE or ?."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return "missing"
    if head[:4] in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
        return "universal"
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        cpu = int.from_bytes(head[4:8], "little")
        return {0x01000007: "x86_64", 0x0100000C: "arm64"}.get(cpu, hex(cpu))
    if head[:4] == b"\x7fELF":
        return "ELF"
    if head[:2] == b"MZ":
        return "PE"
    if head[:2] == b"#!":
        return "script"
    return "?"


def parse_json_line(lines) -> dict:
    """The last JSON object among a command's output lines."""
    for line in reversed(list(lines)):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    raise ValueError("no JSON line in the output")
