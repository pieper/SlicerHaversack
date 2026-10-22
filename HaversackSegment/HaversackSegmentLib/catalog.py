"""The models the module offers, from two places, as one list.

- ``haversack``: haversack's own catalog (`haversack tasks --json`) - TotalSegmentator, MOOSE,
  MRSegmentator, CADS, the LNQ lymph-node models (``lnq:*``, haversack 0.16.0) and the rest.
  haversack downloads their weights on first use, and runs a catalog's stated fold ensemble.
  The ``lnq:*`` tasks are listed in a group of their own so they are easy to find.
- ``extra``: models whose terms must be accepted before download (``extra_models.json``),
  fetched by :mod:`fetch_model` and run by haversack as a model folder. Not in haversack's
  catalog by design.

Every entry is a plain dict, so the widget, the logic and the tests share one shape:
``{key, kind, group, title, description, modality, license, installed, data}``.

No Slicer imports here.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import fetch_model

GROUPS = {"haversack": "haversack catalog", "lnq": "LNQ lymph nodes",
          "extra": "Extra models (license acceptance required)"}


def haversack_entries(tasks: list) -> list[dict]:
    out = []
    for t in tasks:
        group = GROUPS["lnq"] if t["name"].startswith("lnq:") else GROUPS["haversack"]
        out.append({"key": f"haversack:{t['name']}", "kind": "haversack", "group": group,
                    "title": t["name"], "description": t.get("description") or "",
                    "modality": t.get("modality") or "", "license": t.get("license") or "",
                    "installed": bool(t.get("installed")), "data": t})
    return out


def extra_entries(manifest: dict, models_dir) -> list[dict]:
    out = []
    for m in manifest["models"]:
        out.append({"key": f"extra:{m['id']}", "kind": "extra", "group": GROUPS["extra"],
                    "title": m["display_name"], "description": m.get("description") or "",
                    "modality": m.get("modality") or "", "license": m["license"]["summary"],
                    "installed": fetch_model.is_installed(m, models_dir), "data": m})
    return out


def parse_json_output(text: str):
    """A command's ``--json`` stdout, tolerating log lines before the document."""
    text = text.strip()
    for opener in ("[", "{"):
        i = text.find(opener)
        if i >= 0:
            try:
                return json.loads(text[i:])
            except ValueError:
                continue
    raise ValueError("no JSON document in the command's output")


def last_path(stdout_lines: list[str]) -> Path:
    """The path a download command prints as its last line (fetch_model)."""
    for line in reversed(stdout_lines):
        if line.strip():
            return Path(line.strip())
    raise ValueError("the command printed no path")


def output_name(entry: dict, volume_name: str) -> str:
    return f"{volume_name} {entry['title'] if entry['kind'] != 'haversack' else entry['data']['name']}"
