"""The module's Slicer-free code, under plain pytest:

    python -m pytest HaversackSegment/Testing/Python

(``runner`` needs Qt and is covered by the module's own self-test in Slicer.)
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from HaversackSegmentLib import catalog, environment, fetch_model  # noqa: E402

FETCH = Path(fetch_model.__file__)


def _model(tmp_path, payloads, *, lic="test-license@1"):
    """A manifest entry served from a local directory through file:// URLs."""
    served = tmp_path / "served"
    files = []
    for rel, data in payloads.items():
        (served / rel).parent.mkdir(parents=True, exist_ok=True)
        (served / rel).write_bytes(data)
        files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})
    return {"id": "toy", "display_name": "Toy", "modality": "CT",
            "source": {"kind": "base_url", "base": served.as_uri(), "revision": "0123456789abcdef"},
            "dataset_dir": "Dataset001_toy", "model_dir": "T__P__3d_fullres", "files": files,
            "license": {"id": lic, "summary": "toy terms", "terms": "be nice"}}


PAYLOADS = {"dataset.json": b'{"labels": {"background": 0, "a": 1}}', "plans.json": b"{}",
            "fold_all/checkpoint_best.pth": b"\x00" * 3000}


class TestFetch:
    def test_nothing_is_fetched_without_the_license(self, tmp_path):
        m = _model(tmp_path, PAYLOADS)
        for wrong in (None, "", "test-license@0"):
            with pytest.raises(fetch_model.FetchError, match="--accept-license"):
                fetch_model.fetch(m, tmp_path / "models", accept_license=wrong)
        assert not (tmp_path / "models").exists()

    def test_a_fetch_lays_out_a_model_folder_and_records_the_acceptance(self, tmp_path):
        m = _model(tmp_path, PAYLOADS)
        events = []
        folder = fetch_model.fetch(m, tmp_path / "models", accept_license="test-license@1",
                                   emit=lambda e, **f: events.append(e))
        assert folder == tmp_path / "models" / "toy" / "0123456789ab" / "Dataset001_toy" / "T__P__3d_fullres"
        for rel, data in PAYLOADS.items():
            assert (folder / rel).read_bytes() == data
        rec = json.loads((folder / fetch_model.ACCEPTANCE).read_text())
        assert rec["license_id"] == "test-license@1"
        assert rec["terms_sha256"] == hashlib.sha256(b"be nice").hexdigest()
        assert fetch_model.is_installed(m, tmp_path / "models")
        assert fetch_model.accepted_license(m, tmp_path / "models")["model"] == "toy"
        assert events[0] == "start" and events[-1] == "done"

    def test_a_file_that_does_not_match_its_pin_is_not_kept(self, tmp_path):
        m = _model(tmp_path, PAYLOADS)
        m["files"][2]["sha256"] = "0" * 64
        with pytest.raises(fetch_model.FetchError, match="sha256"):
            fetch_model.fetch(m, tmp_path / "models", accept_license="test-license@1")
        folder = fetch_model.model_folder(m, tmp_path / "models")
        assert not (folder / "fold_all" / "checkpoint_best.pth").exists()
        assert not (folder / "fold_all" / "checkpoint_best.pth.part").exists()
        assert not fetch_model.is_installed(m, tmp_path / "models")

    def test_a_new_license_id_is_not_an_old_acceptance(self, tmp_path):
        m = _model(tmp_path, PAYLOADS)
        fetch_model.fetch(m, tmp_path / "models", accept_license="test-license@1")
        m["license"]["id"] = "test-license@2"
        assert fetch_model.accepted_license(m, tmp_path / "models") is None

    def test_the_command_line_prints_the_folder_last_and_events_on_stderr(self, tmp_path):
        m = _model(tmp_path, PAYLOADS)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"models": [m]}))
        r = subprocess.run([sys.executable, str(FETCH), "toy", "--dest", str(tmp_path / "models"),
                            "--manifest", str(manifest), "--accept-license", "test-license@1",
                            "--json-progress"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert catalog.last_path(r.stdout.splitlines()) == fetch_model.model_folder(m, tmp_path / "models")
        events = [json.loads(l[len(fetch_model.JSON_PROGRESS_PREFIX):]) for l in r.stderr.splitlines()
                  if l.startswith(fetch_model.JSON_PROGRESS_PREFIX)]
        assert events[0]["event"] == "start" and events[-1]["event"] == "done"

    def test_the_command_line_refuses_without_the_license(self, tmp_path):
        m = _model(tmp_path, PAYLOADS)
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"models": [m]}))
        r = subprocess.run([sys.executable, str(FETCH), "toy", "--dest", str(tmp_path / "models"),
                            "--manifest", str(manifest), "--json-progress"], capture_output=True, text=True)
        assert r.returncode == 2
        assert '"event": "error"' in r.stderr


class TestShippedManifest:
    def test_pants_is_pinned(self):
        m = fetch_model.get_model("pants-regions")
        assert len(m["source"]["revision"]) == 40
        assert {f["path"] for f in m["files"]} == {"dataset.json", "plans.json",
                                                    "fold_all/checkpoint_best.pth"}
        assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in m["files"])
        assert m["segment_renames"] == {"pancreas": "pancreas_other"}
        assert "CC-BY-NC-ND" in m["license"]["id"]
        assert m["model_dir"].count("__") == 2      # haversack runs it as a model folder


class TestEnvironment:
    def test_segment_command(self, tmp_path):
        e = environment.Environment.at(tmp_path)
        cmd = e.segment_command("in.nrrd", "ts.v2:total", "out.seg.nrrd", device="mps",
                                folds="0,1", extra=["--allow-transpose"])
        assert cmd[1:] == ["segment", "in.nrrd", "--task", "ts.v2:total", "-o", "out.seg.nrrd",
                           "--device", "mps", "--json-progress", "--folds", "0,1", "--allow-transpose"]
        assert "--folds" not in e.segment_command("i", "t", "o")

    def test_install_command_takes_a_local_checkout_as_editable(self, tmp_path):
        e = environment.Environment.at(tmp_path / "home")
        cmd = e.install_command("uv", haversack=str(tmp_path), lnq="lnq-segmenter @ git+https://x/y")
        assert cmd[:2] == ["uv", "pip"] and "--torch-backend" in cmd
        assert cmd[-3:] == ["-e", str(tmp_path), "lnq-segmenter @ git+https://x/y"]
        cmd = e.install_command("uv", haversack="-e ~/src/haversack")
        assert "-e" in cmd and str(Path("~/src/haversack").expanduser()) in cmd

    def test_an_old_haversack_is_reported(self):
        assert environment.missing_segment_options("--task --device") == ["--json-progress", "--folds"]
        assert environment.missing_segment_options("--json-progress --folds") == []


TASKS = [{"name": "ts.v2:total_fast", "engine": "nnunetv2", "modality": "CT", "installed": True,
          "license": "Apache-2.0", "description": "117 structures, 3 mm"}]
LNQ = [{"name": "inguinal-v1", "version": "1.0.0", "display_name": "LNQ-inguinal v1",
        "modality": "CT", "folds": [0, 1, 2, 3], "license": "CC BY 4.0",
        "plans_subdir": "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"}]


class TestCatalog:
    def test_entries_share_one_shape(self, tmp_path):
        entries = (catalog.haversack_entries(TASKS) + catalog.lnq_entries(LNQ, cached=lambda m: False)
                   + catalog.extra_entries(fetch_model.load_manifest(), tmp_path))
        assert [e["kind"] for e in entries] == ["haversack", "lnq", "extra"]
        for e in entries:
            assert set(e) == {"key", "kind", "group", "title", "description", "modality", "license",
                              "installed", "data"}
        assert entries[0]["key"] == "haversack:ts.v2:total_fast"
        assert entries[1]["key"] == "lnq:inguinal-v1@1.0.0"

    def test_lnq_runs_the_folds_it_was_released_with(self):
        assert catalog.lnq_folds(LNQ[0]) == "0,1,2,3"
        assert catalog.lnq_folds({}) == "all"

    def test_an_lnq_bundle_becomes_a_model_folder(self, tmp_path):
        bundle = tmp_path / "inguinal-v1-1.0.0"
        folder = bundle / LNQ[0]["plans_subdir"]
        (folder / "fold_0").mkdir(parents=True)
        (bundle / "dataset.json").write_text("{}")
        (bundle / "plans.json").write_text("{}")
        assert catalog.prepare_lnq_bundle(bundle, LNQ[0]) == folder
        assert (folder / "dataset.json").is_file() and (folder / "plans.json").is_file()

    def test_a_bundle_named_after_its_model_gets_a_stock_named_folder(self, tmp_path):
        """abdominopelvic-v1's layout: the folds in `abdominopelvic-v1/`, trainer and configuration
        left blank in the registry, so they come from the checkpoint."""
        m = {"name": "abdominopelvic-v1", "plans_subdir": "abdominopelvic-v1", "folds": [0, 1],
             "trainer_class": "", "plans_identifier": "", "configuration": ""}
        (tmp_path / "abdominopelvic-v1" / "fold_0").mkdir(parents=True)
        (tmp_path / "abdominopelvic-v1" / "fold_0" / "checkpoint_final.pth").write_text("x")
        (tmp_path / "dataset.json").write_text("{}")
        (tmp_path / "plans.json").write_text('{"plans_name": "nnUNetResEncUNetMPlans"}')
        assert catalog.needs_checkpoint_meta(m)
        assert catalog.first_checkpoint(tmp_path, m).name == "checkpoint_final.pth"
        with pytest.raises(ValueError, match="cannot name"):
            catalog.prepare_lnq_bundle(tmp_path, m)
        got = catalog.prepare_lnq_bundle(tmp_path, m, {"trainer": "nnUNetTrainer", "configuration": "3d_fullres"})
        assert got.name == "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
        assert (got / "fold_0" / "checkpoint_final.pth").is_file() and (got / "dataset.json").is_file()
        assert not catalog.needs_checkpoint_meta(LNQ[0])

    def test_an_incomplete_bundle_is_refused(self, tmp_path):
        (tmp_path / LNQ[0]["plans_subdir"]).mkdir()
        with pytest.raises(FileNotFoundError, match="dataset.json"):
            catalog.prepare_lnq_bundle(tmp_path, LNQ[0])

    def test_lnq_cache_is_read_by_lnq_segmenters_rule(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LNQ_SEGMENTER_CACHE", str(tmp_path))
        m = LNQ[0]
        assert not catalog.lnq_cached(m)
        root = tmp_path / "inguinal-v1-1.0.0"
        for f in ["dataset.json", "plans.json"] + [f"{m['plans_subdir']}/fold_{i}/checkpoint_final.pth"
                                                   for i in m["folds"]]:
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            (root / f).write_text("x")
        assert catalog.lnq_cached(m)

    def test_json_output_after_log_lines(self):
        assert catalog.parse_json_output("warning: x\n[1, 2]\n") == [1, 2]
