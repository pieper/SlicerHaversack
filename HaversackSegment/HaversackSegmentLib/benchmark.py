"""Benchmark reports: what the module's Benchmark button measures, and how reports are written and
read back. Standard library only, no Slicer: ``tools/collate_benchmarks.py`` imports this file to
collect reports pasted into a GitHub issue thread.

A benchmark segments Slicer's CTChest sample with ``ts.v2:total_fast`` twice, as two separate
``haversack segment`` processes. The first run may include the weights' download on first use
(haversack reports it as ``weights:*`` in its timings); the second starts from weights on disk,
which is the number to compare across machines. Each run records the wall-clock time the module
measured around the process and haversack's own per-stage timings.

A report is a short Markdown block with one fenced JSON payload, tagged with :data:`SCHEMA`, so it
pastes into an issue comment as it is and :func:`parse_reports` can find it again among anything
else in the thread.
"""
from __future__ import annotations

import json
import re

SCHEMA = "slicer-haversack-benchmark/1"
SAMPLE = "CTChest"
TASK = "ts.v2:total_fast"
RUNS = 2

#: Run with the haversack environment's Python: the machine, as torch and the OS describe it.
#: Prints one JSON line. Standard library plus torch (always in the environment).
SYSTEM_INFO_CODE = r'''
import json, os, platform, subprocess, sys
def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""
def ram_bytes():
    try:
        if sys.platform == "darwin":
            return int(sh("sysctl", "-n", "hw.memsize"))
        if sys.platform.startswith("linux"):
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        if sys.platform == "win32":
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys)
    except Exception:
        pass
    return None
def cpu_name():
    if sys.platform == "darwin":
        return sh("sysctl", "-n", "machdep.cpu.brand_string")
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if sys.platform == "win32":
        return sh("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name") or platform.processor()
    return platform.processor()
info = {"os": platform.platform(), "system": platform.system(), "release": platform.release(),
        "machine": platform.machine(), "cpu": cpu_name(), "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes(), "python": sys.version.split()[0]}
if sys.platform == "darwin":
    info["os_version"] = sh("sw_vers", "-productVersion")
    info["model"] = sh("sysctl", "-n", "hw.model")
try:
    import torch
    info["torch"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_memory_bytes"] = int(p.total_memory)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_runtime"] = torch.version.cuda
        info["gpu_driver"] = sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").splitlines()[0:1]
        info["gpu_driver"] = info["gpu_driver"][0] if info["gpu_driver"] else None
    elif info["mps_available"]:
        # the GPU's own name (Apple M4, or an Intel Mac's AMD card), not the CPU's
        disp = sh("system_profiler", "SPDisplaysDataType")
        chips = [l.split(":", 1)[1].strip() for l in disp.splitlines() if "Chipset Model:" in l]
        cores = [l.split(":", 1)[1].strip() for l in disp.splitlines() if "Total Number of Cores:" in l]
        info["gpu"] = (chips[0] if chips else "Apple GPU") + (f" ({cores[0]} GPU cores)" if cores else "") + " via MPS"
        vram = [l.split(":", 1)[1].strip() for l in disp.splitlines() if "VRAM" in l]
        info["gpu_memory"] = vram[0] if vram else "unified (see ram_bytes)"
except Exception as exc:
    info["torch_error"] = str(exc)
try:
    import haversack
    info["haversack"] = getattr(haversack, "__version__", "?")
except Exception as exc:
    info["haversack_error"] = str(exc)
print(json.dumps(info))
'''


def device_for(system: dict, requested: str = "auto") -> str:
    """The device the benchmark names explicitly, so the report says what ran: the requested one,
    or for "auto" what haversack's own auto picks - CUDA, then MPS, then the CPU."""
    if requested and requested != "auto":
        return requested
    if system.get("cuda_available"):
        return "cuda"
    if system.get("mps_available"):
        return "mps"
    return "cpu"


def build_report(*, system: dict, device: str, runs: list[dict], slicer: dict, module: dict,
                 when: str, note: str = "") -> dict:
    """The payload. ``runs`` holds one dict per run: ``wall_s`` (around the process) and
    ``timings`` (haversack's own, seconds by stage)."""
    return {"schema": SCHEMA, "when": when, "sample": SAMPLE, "task": TASK, "device": device,
            "runs": runs, "system": system, "slicer": slicer, "module": module,
            **({"note": note} if note else {})}


def _gb(n):
    return f"{n / (1 << 30):.0f} GB" if n else "?"


def summary_line(r: dict) -> str:
    s = r.get("system", {})
    runs = r.get("runs") or []
    warm = runs[-1] if runs else {}
    total = (warm.get("timings") or {}).get("total")
    return (f"{r.get('task')} on {r.get('sample')} · {r.get('device')} · "
            f"{s.get('gpu') or s.get('cpu') or '?'} · {_gb(s.get('ram_bytes'))} RAM · "
            f"{s.get('os', '?')} · warm run {total:.1f} s (wall {warm.get('wall_s', 0):.1f} s)"
            if total is not None else f"{r.get('task')} on {r.get('sample')}: incomplete")


def to_markdown(r: dict) -> str:
    """The block to paste into an issue comment: a heading, one summary line, the JSON payload."""
    return ("### SlicerHaversack benchmark\n\n"
            + summary_line(r) + "\n\n"
            + "```json\n" + json.dumps(r, indent=1, sort_keys=True) + "\n```\n")


_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.S)


def parse_reports(text: str) -> list[dict]:
    """Every benchmark payload in ``text`` (an issue thread, a file of pasted comments)."""
    out = []
    for block in _FENCE.findall(text or ""):
        try:
            r = json.loads(block)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("schema") == SCHEMA:
            out.append(r)
    return out


COLUMNS = ("when", "who", "os", "cpu", "ram_gb", "gpu", "gpu_mem_gb", "device", "torch",
           "haversack", "slicer", "first_total_s", "warm_total_s", "warm_wall_s", "warm_predict_s",
           "warm_restore_s", "warm_load_s", "warm_read_s")


def _stage(timings: dict, prefix: str):
    vals = [v for k, v in (timings or {}).items() if k == prefix or k.startswith(prefix + ":")]
    return round(sum(vals), 2) if vals else None


def row(r: dict, who: str = "") -> dict:
    s = r.get("system", {})
    runs = r.get("runs") or [{}]
    first, warm = runs[0], runs[-1]
    wt = warm.get("timings") or {}
    return {"when": r.get("when"), "who": who, "os": s.get("os"), "cpu": s.get("cpu"),
            "ram_gb": round(s["ram_bytes"] / (1 << 30), 1) if s.get("ram_bytes") else None,
            "gpu": s.get("gpu"),
            "gpu_mem_gb": round(s["gpu_memory_bytes"] / (1 << 30), 1) if s.get("gpu_memory_bytes") else None,
            "device": r.get("device"), "torch": s.get("torch"), "haversack": s.get("haversack"),
            "slicer": (r.get("slicer") or {}).get("version"),
            "first_total_s": (first.get("timings") or {}).get("total"),
            "warm_total_s": wt.get("total"), "warm_wall_s": warm.get("wall_s"),
            "warm_predict_s": _stage(wt, "predict"), "warm_restore_s": _stage(wt, "restore"),
            "warm_load_s": _stage(wt, "load"), "warm_read_s": _stage(wt, "read+canonical") or _stage(wt, "read")}
