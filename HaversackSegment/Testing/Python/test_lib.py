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
        cmd = e.install_command("uv", haversack=str(tmp_path))
        assert cmd[:2] == ["uv", "pip"] and "--torch-backend" in cmd
        assert cmd[-2:] == ["-e", str(tmp_path)]
        cmd = e.install_command("uv", haversack="-e ~/src/haversack")
        assert cmd[-2:] == ["-e", str(Path("~/src/haversack").expanduser())]
        assert e.install_command("uv")[-1] == environment.DEFAULT_HAVERSACK

    def test_apple_silicon_gets_an_arm64_python_whatever_slicer_runs_as(self, tmp_path):
        """A Slicer under Rosetta reports x86_64; the environment must still be arm64 (torch has no
        x86_64 macOS wheels), so the Python is requested by architecture."""
        mac = {"system": "Darwin", "process_machine": "x86_64", "hardware_arm64": True, "translated": True}
        intel = {"system": "Darwin", "process_machine": "x86_64", "hardware_arm64": False}
        linux = {"system": "Linux", "process_machine": "x86_64", "hardware_arm64": False}
        assert environment.python_request(mac) == "cpython-3.12-macos-aarch64-none"
        assert environment.python_request(intel) == environment.python_request(linux) == "3.12"
        cmd = environment.Environment.at(tmp_path).create_command("uv", mac)
        assert cmd[:4] == ["uv", "venv", "--allow-existing", "--managed-python"]
        assert cmd[-2] == "cpython-3.12-macos-aarch64-none"
        assert "Apple Silicon" in environment.check_probe({"machine": "x86_64", "version": "3.12.11"}, mac)
        assert environment.check_probe({"machine": "arm64", "version": "3.12.11"}, mac) is None
        assert environment.check_probe({"machine": "x86_64", "version": "3.12.3"}, linux) is None
        assert "3.12" in environment.check_probe({"machine": "x86_64", "version": "3.9.6"}, linux)

    def test_binary_arch_reads_headers(self, tmp_path):
        cases = {"arm64": b"\xcf\xfa\xed\xfe\x0c\x00\x00\x01", "x86_64": b"\xcf\xfa\xed\xfe\x07\x00\x00\x01",
                 "universal": b"\xca\xfe\xba\xbe\x00\x00\x00\x02", "ELF": b"\x7fELF\x02\x01\x01\x00",
                 "script": b"#!/bin/sh\n"}
        for want, head in cases.items():
            (tmp_path / want).write_bytes(head)
            assert environment.binary_arch(tmp_path / want) == want
        assert environment.binary_arch(tmp_path / "nope") == "missing"

    def test_probe_and_verify_print_one_json_line(self):
        out = subprocess.run([sys.executable, "-c", environment.PROBE_CODE], capture_output=True, text=True)
        probe = environment.parse_json_line(["noise", out.stdout.strip()])
        assert set(probe) == {"machine", "version", "executable"}
        with pytest.raises(ValueError):
            environment.parse_json_line(["no json here"])

    def test_an_old_haversack_is_reported(self):
        assert environment.missing_segment_options("--task --device") == ["--json-progress", "--folds"]
        assert environment.missing_segment_options("--json-progress --folds") == []


TASKS = [{"name": "ts.v2:total_fast", "engine": "nnunetv2", "modality": "CT", "installed": True,
          "license": "Apache-2.0", "description": "117 structures, 3 mm"},
         {"name": "lnq:abdominopelvic-v1", "engine": "nnunetv2", "modality": "CT", "installed": False,
          "license": "CC-BY-4.0", "description": "abdominopelvic lymph nodes"}]


class TestCatalog:
    def test_entries_share_one_shape(self, tmp_path):
        entries = catalog.haversack_entries(TASKS) + catalog.extra_entries(fetch_model.load_manifest(), tmp_path)
        assert [e["kind"] for e in entries] == ["haversack", "haversack", "extra"]
        for e in entries:
            assert set(e) == {"key", "kind", "group", "title", "description", "modality", "license",
                              "installed", "data"}
        assert entries[0]["key"] == "haversack:ts.v2:total_fast"

    def test_lnq_catalog_tasks_get_their_own_group(self):
        groups = {e["title"]: e["group"] for e in catalog.haversack_entries(TASKS)}
        assert groups == {"ts.v2:total_fast": catalog.GROUPS["haversack"],
                          "lnq:abdominopelvic-v1": catalog.GROUPS["lnq"]}

    def test_json_output_after_log_lines(self):
        assert catalog.parse_json_output("warning: x\n[1, 2]\n") == [1, 2]


# --- progress plumbing, uninstall, benchmark (2026-10-27) ---------------------------------------

from HaversackSegmentLib import benchmark, logtail, stepprogress  # noqa: E402


class TestLogTail:
    def test_lines_arrive_as_they_are_appended(self, tmp_path):
        p = tmp_path / "step.log"
        t = logtail.LogTail(p)
        assert t.read() == []                                   # not created yet
        with open(p, "wb") as f:
            f.write(b"first\nsecond, half")
            f.flush()
            assert t.read() == ["first"]                        # a partial line waits
            f.write(" done\r\n\n".encode() + "caf\xc3".encode("latin-1"))
            f.flush()
            assert t.read() == ["second, half done"]            # \r dropped, blank skipped
            f.write(b"\xa9 ok\nlast without newline")           # a UTF-8 character split across reads
            f.flush()
            assert t.read() == ["café ok"]
        assert t.read(final=True) == ["last without newline"]
        t.close()


class TestUvProgress:
    LINES = ["Resolved 171 packages in 11.06s", "Downloading torch (73.5MiB)",
             "Downloading cpython-3.12.11-macos-aarch64-none (download) (17.4MiB)",
             "Downloading numpy (6.5MiB)", " Downloading numpy",
             "Building haversack @ git+https://github.com/pieper/haversack@x"]

    def test_downloads_drive_the_fraction(self):
        u = stepprogress.UvProgress()
        assert u.fraction() is None
        assert all(u.feed(line) for line in self.LINES)
        assert not u.feed("anything else")
        assert set(u.sizes) == {"torch", "cpython-3.12.11-macos-aarch64-none", "numpy"}
        assert u.finished == {"numpy"}
        assert abs(u.fraction() - 6.5 / (73.5 + 17.4 + 6.5)) < 1e-6
        text = u.text()
        assert "resolved 171 packages" in text and "1 of 3 large files" in text
        assert "now: torch" in text and "building haversack" in text
        u.feed(" Built haversack @ git+https://github.com/pieper/haversack@x")
        assert "building" not in u.text()

    def test_human_units(self):
        assert stepprogress.human_bytes(512) == "512 B"
        assert stepprogress.human_bytes(3 << 30) == "3.0 GiB"
        assert stepprogress.human_seconds(59) == "59s" and stepprogress.human_seconds(125) == "2m 05s"


class TestUninstall:
    def test_paths(self, tmp_path):
        e = environment.Environment.at(tmp_path)
        assert e.uninstall_paths(downloads=False) == [e.venv]
        assert e.uninstall_paths(downloads=True) == [e.venv, e.uv_cache, e.python_dir]
        assert e.models not in e.uninstall_paths(downloads=True)          # weights are kept
        assert e.uv_env() == {"UV_CACHE_DIR": str(e.uv_cache), "UV_PYTHON_INSTALL_DIR": str(e.python_dir)}


def _report(total=12.5, gpu="NVIDIA GeForce RTX 4090"):
    return benchmark.build_report(
        system={"os": "Linux-6.8", "cpu": "AMD EPYC", "ram_bytes": 64 << 30, "gpu": gpu,
                "gpu_memory_bytes": 24 << 30, "torch": "2.14.1", "haversack": "0.16.0",
                "cuda_available": True},
        device="cuda", slicer={"version": "5.12.4"}, module={}, when="2026-10-27T12:00:00Z",
        runs=[{"run": 1, "wall_s": 40.0, "timings": {"total": 35.0, "weights:ts.v2:total_fast": 20.0}},
              {"run": 2, "wall_s": total + 3, "timings": {"total": total, "predict:x": 4.0, "load:x": 1.5}}])


class TestBenchmark:
    def test_device_for(self):
        assert benchmark.device_for({"cuda_available": True, "mps_available": True}) == "cuda"
        assert benchmark.device_for({"mps_available": True}) == "mps"
        assert benchmark.device_for({}) == "cpu"
        assert benchmark.device_for({"cuda_available": True}, "cpu") == "cpu"

    def test_a_pasted_report_reads_back_among_other_text(self):
        md = benchmark.to_markdown(_report())
        assert md.startswith("### SlicerHaversack benchmark") and "warm run 12.5 s" in md
        thread = "Here is mine:\n\n" + md + "\nand some ```python\nprint(1)\n``` code\n" + benchmark.to_markdown(_report(9.0))
        got = benchmark.parse_reports(thread)
        assert [r["runs"][-1]["timings"]["total"] for r in got] == [12.5, 9.0]
        row = benchmark.row(got[0], who="someone")
        assert set(row) == set(benchmark.COLUMNS)
        assert row["warm_total_s"] == 12.5 and row["first_total_s"] == 35.0
        assert row["warm_predict_s"] == 4.0 and row["warm_load_s"] == 1.5
        assert row["ram_gb"] == 64.0 and row["gpu_mem_gb"] == 24.0 and row["who"] == "someone"

    def test_the_collate_tool_tabulates_files(self, tmp_path):
        f = tmp_path / "pasted.md"
        f.write_text(benchmark.to_markdown(_report(20.0)) + benchmark.to_markdown(_report(10.0, gpu="Apple M4")))
        tool = Path(__file__).resolve().parents[3] / "tools" / "collate_benchmarks.py"
        out = subprocess.run([sys.executable, str(tool), str(f)], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        lines = out.stdout.strip().splitlines()
        assert lines[0].startswith("| when |") and len(lines) == 4
        assert "Apple M4" in lines[2]                          # fastest warm run first
        out = subprocess.run([sys.executable, str(tool), "--csv", str(f)], capture_output=True, text=True)
        assert out.stdout.splitlines()[0].split(",") == list(benchmark.COLUMNS)
