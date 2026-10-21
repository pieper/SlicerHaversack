"""The models the module offers, from three places, as one list.

- ``haversack``: haversack's own catalog (`haversack tasks --json`) - TotalSegmentator, MOOSE,
  MRSegmentator, CADS and the rest. haversack downloads their weights on first use.
- ``lnq``: the LNQ lymph-node models (`lnq-segmenter list --json`). lnq-segmenter downloads and
  verifies the bundle; haversack runs the bundle's nnU-Net folder, every fold the registry names.
- ``extra``: models whose terms must be accepted before download (``extra_models.json``),
  fetched by :mod:`fetch_model` and run by haversack as a model folder. Not in haversack's
  catalog by design.

Every entry is a plain dict, so the widget, the logic and the tests share one shape:
``{key, kind, group, title, description, modality, license, installed, data}``.

No Slicer imports here.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import fetch_model

GROUPS = {"haversack": "haversack catalog", "lnq": "LNQ lymph nodes",
          "extra": "Extra models (license acceptance required)"}


def haversack_entries(tasks: list) -> list[dict]:
    out = []
    for t in tasks:
        out.append({"key": f"haversack:{t['name']}", "kind": "haversack", "group": GROUPS["haversack"],
                    "title": t["name"], "description": t.get("description") or "",
                    "modality": t.get("modality") or "", "license": t.get("license") or "",
                    "installed": bool(t.get("installed")), "data": t})
    return out


def lnq_cache_root() -> Path:
    """lnq-segmenter's cache root, by its own rule (``lnq_segmenter.cache.cache_root``):
    ``$LNQ_SEGMENTER_CACHE``, else ``$XDG_CACHE_HOME/lnq-segmenter``, else ``~/.cache/...``."""
    import os
    if os.environ.get("LNQ_SEGMENTER_CACHE"):
        return Path(os.environ["LNQ_SEGMENTER_CACHE"]).absolute()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path("~/.cache").expanduser()) / "lnq-segmenter"


def lnq_cached(m: dict) -> bool:
    """Whether lnq-segmenter has the whole bundle (``lnq_segmenter.cache.is_complete``)."""
    root = lnq_cache_root() / f"{m['name']}-{m['version']}"
    files = ["dataset.json", "plans.json"] + [
        str(Path(m.get("plans_subdir") or "") / f"fold_{f}" / "checkpoint_final.pth")
        for f in m.get("folds") or []]
    return all((root / f).is_file() for f in files)


def lnq_entries(models: list, cached=lnq_cached) -> list[dict]:
    out = []
    for m in models:
        out.append({"key": f"lnq:{m['name']}@{m['version']}", "kind": "lnq", "group": GROUPS["lnq"],
                    "title": m.get("display_name") or m["name"],
                    "description": m.get("recommended_use") or "",
                    "modality": m.get("modality") or "CT", "license": m.get("license") or "",
                    "installed": cached(m), "data": m})
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
    """The path a download command prints as its last line (lnq-segmenter, fetch_model)."""
    for line in reversed(stdout_lines):
        if line.strip():
            return Path(line.strip())
    raise ValueError("the command printed no path")


#: Run with the haversack environment's Python on a fold's checkpoint: what nnU-Net itself reads
#: to rebuild the model (``nnUNetPredictor.initialize_from_trained_model_folder``).
CHECKPOINT_META_CODE = (
    "import json, sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
    "print(json.dumps({'trainer': c.get('trainer_name'), "
    "'configuration': (c.get('init_args') or {}).get('configuration')}))")


def needs_checkpoint_meta(entry: dict) -> bool:
    """Whether the registry leaves the model folder's trainer or configuration unsaid, for a bundle
    whose folder is not named ``<trainer>__<plans>__<config>`` (abdominopelvic-v1, axillary-v1 and
    mediastinal-v1 in lnq-segmenter 0.3.0)."""
    return (Path(entry["plans_subdir"]).name.count("__") != 2
            and not (entry.get("trainer_class") and entry.get("configuration")))


def first_checkpoint(bundle, entry: dict) -> Path:
    folder = Path(bundle) / entry["plans_subdir"]
    for f in [f"fold_{n}" for n in entry.get("folds") or []] + ["fold_all"]:
        for name in ("checkpoint_final.pth", "checkpoint_best.pth"):
            if (folder / f / name).is_file():
                return folder / f / name
    raise FileNotFoundError(f"{folder} has no fold checkpoint")


def prepare_lnq_bundle(bundle, entry: dict, checkpoint_meta: dict | None = None) -> Path:
    """The nnU-Net model folder inside an lnq-segmenter bundle, made runnable as one.

    A bundle holds ``dataset.json`` and ``plans.json`` at its root, beside (not inside) the folder
    the folds unpack into; nnU-Net and haversack both read them from the model folder.
    lnq-segmenter's own predict copies them in the same way (``predict._ensure_nnunet_layout``).

    haversack knows a model folder by its ``<trainer>__<plans>__<config>`` name, and some bundles
    name it after the model (``abdominopelvic-v1/``). Such a folder gets a sibling symlink with
    the stock name. The parts come from the registry, else from ``plans.json``'s ``plans_name``
    and the trainer and configuration the checkpoint itself records (``checkpoint_meta``, from
    :data:`CHECKPOINT_META_CODE`)."""
    bundle = Path(bundle)
    folder = bundle / entry["plans_subdir"]
    if not folder.is_dir():
        raise FileNotFoundError(f"{bundle} has no {entry['plans_subdir']} folder")
    for meta in ("dataset.json", "plans.json"):
        src, dst = bundle / meta, folder / meta
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
        if not dst.is_file():
            raise FileNotFoundError(f"{bundle} has no {meta}")
    if folder.name.count("__") == 2:
        return folder
    meta = checkpoint_meta or {}
    trainer = entry.get("trainer_class") or meta.get("trainer")
    config = entry.get("configuration") or meta.get("configuration")
    plans = entry.get("plans_identifier") or json.loads((folder / "plans.json").read_text()).get("plans_name")
    if not (trainer and plans and config):
        raise ValueError(f"{entry['name']}: cannot name its model folder (trainer {trainer!r}, "
                         f"plans {plans!r}, configuration {config!r})")
    link = bundle / f"{trainer}__{plans}__{config}"
    if not link.exists():
        link.symlink_to(folder.name, target_is_directory=True)
    return link


def lnq_folds(entry: dict) -> str:
    """``--folds`` for an LNQ model: the registry's folds, the ensemble it was released as."""
    folds = entry.get("folds") or []
    return ",".join(str(int(f)) for f in folds) if folds else "all"


def output_name(entry: dict, volume_name: str) -> str:
    return f"{volume_name} {entry['title'] if entry['kind'] != 'haversack' else entry['data']['name']}"
