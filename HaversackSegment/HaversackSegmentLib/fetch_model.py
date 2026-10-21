"""Download one of the module's extra models (``Resources/extra_models.json``) into an nnU-Net
model folder that `haversack segment --task <folder>` runs.

Standard library only, so it runs under PythonSlicer without the haversack environment:

    PythonSlicer fetch_model.py pants-regions --dest ~/.slicer-haversack/models \
        --accept-license '<license id>' --json-progress

An extra model is one whose terms the user has to accept, which is why it is not in haversack's
catalog. Nothing is downloaded unless ``--accept-license`` repeats the manifest's license id
exactly; the acceptance is written into the model folder (``LICENSE_ACCEPTANCE.json``) before the
first byte arrives, so a folder never holds weights nobody accepted. Every file is pinned by
revision, size and sha256, written to ``<name>.part`` and moved into place only once it verifies.
The last line on stdout is the model folder.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

JSON_PROGRESS_PREFIX = "fetch> "
MANIFEST = Path(__file__).resolve().parent.parent / "Resources" / "extra_models.json"
ACCEPTANCE = "LICENSE_ACCEPTANCE.json"
COMPLETE = ".complete.json"
CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


def load_manifest(path=MANIFEST) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_model(model_id: str, manifest: dict | None = None) -> dict:
    manifest = manifest if manifest is not None else load_manifest()
    for m in manifest["models"]:
        if m["id"] == model_id:
            return m
    raise FetchError(f"no extra model {model_id!r}; have "
                     f"{', '.join(m['id'] for m in manifest['models'])}")


def model_folder(model: dict, dest) -> Path:
    """``<dest>/<id>/<revision[:12]>/<dataset_dir>/<model_dir>`` - the folder haversack runs.
    The revision is in the path, so a new pin never mixes files with an old one."""
    rev = model["source"]["revision"][:12]
    return Path(dest).expanduser() / model["id"] / rev / model["dataset_dir"] / model["model_dir"]


def file_url(model: dict, rel: str) -> str:
    src = model["source"]
    if src["kind"] == "huggingface":
        return f"https://huggingface.co/{src['repo']}/resolve/{src['revision']}/{rel}"
    if src["kind"] == "base_url":                  # tests, mirrors: <base>/<rel>
        return src["base"].rstrip("/") + "/" + rel
    raise FetchError(f"{model['id']}: unknown source kind {src['kind']!r}")


def is_installed(model: dict, dest) -> bool:
    """Every pinned file present at its size, the acceptance recorded, and the verification
    record written (sha256 is checked once, at download; size is the cheap re-check)."""
    folder = model_folder(model, dest)
    if not (folder / COMPLETE).is_file() or not (folder / ACCEPTANCE).is_file():
        return False
    return all((folder / f["path"]).is_file() and (folder / f["path"]).stat().st_size == f["size"]
               for f in model["files"])


def accepted_license(model: dict, dest) -> dict | None:
    p = model_folder(model, dest) / ACCEPTANCE
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if rec.get("license_id") == model["license"]["id"] else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, target: Path, size: int, sha256: str, emit, rel: str) -> None:
    part = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "SlicerHaversack-fetch/1"})
    h = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            out.write(block)
            h.update(block)
            done += len(block)
            emit("download", file=rel, bytes=done, total=size)
    if done != size or h.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise FetchError(f"{rel}: got {done} bytes with sha256 {h.hexdigest()}, the pin says "
                         f"{size} bytes with sha256 {sha256}")
    os.replace(part, target)


def fetch(model: dict, dest, *, accept_license: str | None, emit=lambda *a, **k: None) -> Path:
    lic = model["license"]
    if accept_license != lic["id"]:
        raise FetchError(f"{model['id']} needs its license accepted: pass --accept-license "
                         f"{lic['id']!r} after reading its terms ({lic['summary']})")
    folder = model_folder(model, dest)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ACCEPTANCE).write_text(json.dumps({
        "model": model["id"], "license_id": lic["id"], "summary": lic["summary"],
        "terms_sha256": hashlib.sha256(lic["terms"].encode("utf-8")).hexdigest(),
        "accepted_by": getpass.getuser(),
        "accepted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    total = sum(f["size"] for f in model["files"])
    emit("start", model=model["id"], folder=str(folder), total=total)
    for f in model["files"]:
        target = folder / f["path"]
        if target.is_file() and target.stat().st_size == f["size"] and _sha256(target) == f["sha256"]:
            emit("verified", file=f["path"])
            continue
        _download(file_url(model, f["path"]), target, f["size"], f["sha256"], emit, f["path"])
        emit("verified", file=f["path"])
    (folder / COMPLETE).write_text(json.dumps(
        {"revision": model["source"]["revision"],
         "files": {f["path"]: f["sha256"] for f in model["files"]}}, indent=2), encoding="utf-8")
    emit("done", folder=str(folder))
    return folder


def remove(model: dict, dest) -> None:
    """Delete a model's downloaded files and its acceptance record."""
    shutil.rmtree(Path(dest).expanduser() / model["id"], ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="an id from extra_models.json")
    ap.add_argument("--dest", required=True, help="the extra models directory")
    ap.add_argument("--accept-license", help="the model's license id, repeated exactly")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--json-progress", action="store_true")
    args = ap.parse_args(argv)

    def emit(event, **fields):
        if args.json_progress:
            print(JSON_PROGRESS_PREFIX + json.dumps({"event": event, **fields}),
                  file=sys.stderr, flush=True)

    try:
        model = get_model(args.model, load_manifest(args.manifest))
        folder = fetch(model, args.dest, accept_license=args.accept_license, emit=emit)
    except (FetchError, OSError) as e:
        emit("error", kind=type(e).__name__, message=str(e))
        print(f"fetch_model: {e}", file=sys.stderr)
        return 2
    print(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
