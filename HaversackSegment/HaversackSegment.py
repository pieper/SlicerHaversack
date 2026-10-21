"""Haversack Segment - run haversack's segmentation models on a volume in the scene.

One module for every model haversack runs: its own catalog (TotalSegmentator, MOOSE,
MRSegmentator, CADS, ...), the LNQ lymph-node models (downloaded by lnq-segmenter, run by
haversack), and extra models whose license the user accepts before the weights are fetched
(PanTS). Inference runs as a subprocess - `haversack segment ... --json-progress` - in an
environment the module keeps apart from Slicer's Python (HaversackSegmentLib.environment), with
progress streamed into the panel and a cancel that kills the process.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

from HaversackSegmentLib import catalog, environment, fetch_model
from HaversackSegmentLib.runner import CommandRunner

SETTINGS = "HaversackSegment"


# =============================================================================
# Module
# =============================================================================

class HaversackSegment(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Haversack Segment"
        parent.categories = ["Segmentation"]
        parent.dependencies = []
        parent.contributors = ["Steve Pieper (Isomics)"]
        parent.helpText = (
            "Segment a volume with haversack: TotalSegmentator, MOOSE, MRSegmentator, CADS and "
            "the rest of haversack's catalog, the LNQ lymph-node models, and extra models such as "
            "PanTS whose license you accept before download. Models run in a separate Python "
            "environment the module installs (button at the top), on the GPU when there is one."
        )
        parent.acknowledgementText = (
            "haversack: https://github.com/mhalle/haversack. LNQ: https://lnqproject.org. "
            "Each model's own license and citation apply to its results."
        )


# =============================================================================
# Logic
# =============================================================================

class HaversackSegmentLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    # ----- settings -----

    @staticmethod
    def setting(key, default=""):
        return str(qt.QSettings().value(f"{SETTINGS}/{key}", default) or default)

    @staticmethod
    def setSetting(key, value):
        qt.QSettings().setValue(f"{SETTINGS}/{key}", value)

    @property
    def env(self) -> environment.Environment:
        return environment.Environment.at(self.setting("Home", str(environment.DEFAULT_HOME)))

    def haversackSource(self) -> str:
        return self.setting("HaversackSource", environment.DEFAULT_HAVERSACK)

    def lnqSource(self) -> str:
        return self.setting("LNQSource", environment.DEFAULT_LNQ)

    # ----- environment -----

    @staticmethod
    def uvBinary() -> str:
        """uv, pip-installed into Slicer's Python on first use (one binary wheel)."""
        try:
            import uv
        except ImportError:
            slicer.util.pip_install("uv")
            import uv
        return uv.find_uv_bin()

    def installCommands(self) -> list:
        uv = self.uvBinary()
        e = self.env
        return [e.create_command(uv), e.install_command(uv, self.haversackSource(), self.lnqSource())]

    def _run(self, command, timeout=120) -> str:
        """A short command in the haversack environment, its stdout returned."""
        r = subprocess.run([str(c) for c in command], capture_output=True, text=True,
                           timeout=timeout, env=_clean_env())
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(map(str, command))} failed:\n{r.stderr.strip()[-2000:]}")
        return r.stdout

    def missingSegmentOptions(self) -> list:
        return environment.missing_segment_options(self._run(self.env.segment_help_command()))

    # ----- models -----

    @staticmethod
    def extraManifest() -> dict:
        return fetch_model.load_manifest()

    def listModels(self) -> list:
        """Every model the module offers, haversack's catalog first. A source that cannot be
        listed is skipped with a warning rather than hiding the others."""
        entries = []
        try:
            entries += catalog.haversack_entries(
                catalog.parse_json_output(self._run(self.env.tasks_command())))
        except Exception as exc:
            logging.warning(f"haversack tasks: {exc}")
        try:
            entries += catalog.lnq_entries(
                catalog.parse_json_output(self._run(self.env.lnq_list_command())))
        except Exception as exc:
            logging.warning(f"lnq-segmenter list: {exc}")
        entries += catalog.extra_entries(self.extraManifest(), self.env.models)
        return entries

    # ----- one run's files -----

    def newWorkDir(self) -> Path:
        d = self.env.work / time.strftime("run-%Y%m%d-%H%M%S")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def saveInput(volumeNode, path) -> str:
        if not slicer.util.saveNode(volumeNode, str(path), {"useCompression": False}):
            raise RuntimeError(f"failed to save {volumeNode.GetName()} to {path}")
        return str(path)

    @staticmethod
    def loadResult(path, entry: dict, volumeNode):
        """Load a haversack ``.seg.nrrd`` as a segmentation on ``volumeNode``'s geometry, with
        the entry's renames (PanTS's leftover ``pancreas`` is ``pancreas_other``) and, for an
        LNQ model, the registry's segment name, color and terminology."""
        seg = slicer.util.loadSegmentation(str(path))
        if seg is None:
            raise RuntimeError(f"failed to load segmentation from {path}")
        seg.SetName(slicer.mrmlScene.GenerateUniqueName(catalog.output_name(entry, volumeNode.GetName())))
        seg.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
        segmentation = seg.GetSegmentation()
        renames = (entry["data"].get("segment_renames") or {}) if entry["kind"] == "extra" else {}
        for i in range(segmentation.GetNumberOfSegments()):
            segment = segmentation.GetNthSegment(i)
            if segment.GetName() in renames:
                segment.SetName(renames[segment.GetName()])
        if entry["kind"] == "lnq" and segmentation.GetNumberOfSegments() == 1:
            m = entry["data"]
            segment = segmentation.GetNthSegment(0)
            if m.get("segment_name"):
                segment.SetName(m["segment_name"])
            if m.get("color"):
                segment.SetColor(*[c / 255.0 for c in m["color"][:3]])
            if m.get("snomed_code"):
                segment.SetTag("SNOMED-CT.code", m["snomed_code"])
            if m.get("snomed_term"):
                segment.SetTag("SNOMED-CT.term", m["snomed_term"])
        seg.SetAttribute("Haversack.model", entry["key"])
        return seg


def _clean_env() -> dict:
    env = dict(slicer.util.startupEnvironment())
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE"):
        env.pop(key, None)
    return env


# =============================================================================
# Widget
# =============================================================================

class HaversackSegmentWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = HaversackSegmentLogic()
        self._runner: Optional[CommandRunner] = None
        self._steps: list = []          # (label, command, clean_env, on_ok) still to run
        self._entries: list = []
        self._workDir: Optional[Path] = None
        self._task = None               # what `haversack segment --task` gets for this run
        self._lastDone: Optional[dict] = None

    # ----- UI construction -----

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.addWidget(self._buildSetupSection())
        self.layout.addWidget(self._buildInputSection())
        self.layout.addWidget(self._buildModelSection())
        self.layout.addWidget(self._buildRunSection())
        self.layout.addWidget(self._buildProgressSection(), 1)
        self._runner = CommandRunner(self.parent)
        self._runner.progressEvent.connect(self._onEvent)
        self._runner.logLine.connect(self._log)
        self._runner.finished.connect(self._onStepFinished)
        self._refreshEnvironmentState()

    def _buildSetupSection(self):
        box = ctk_collapsible("Setup")
        form = qt.QFormLayout(box)
        row = qt.QHBoxLayout()
        self._envStatus = qt.QLabel("checking…")
        self._envStatus.wordWrap = True
        self._installButton = qt.QPushButton("Install")
        self._installButton.toolTip = ("Create the haversack environment (Python 3.12, uv) and "
                                       "install haversack and lnq-segmenter into it; again to update.")
        self._installButton.connect("clicked()", self._onInstallClicked)
        row.addWidget(self._envStatus, 1)
        row.addWidget(self._installButton, 0)
        form.addRow(row)

        advanced = ctk_collapsible("Advanced", collapsed=True)
        aform = qt.QFormLayout(advanced)
        self._homeEdit = qt.QLineEdit(self.logic.setting("Home", str(environment.DEFAULT_HOME)))
        self._homeEdit.toolTip = "Holds the environment (env/), extra models (models/) and run files (work/)."
        self._haversackEdit = qt.QLineEdit(self.logic.haversackSource())
        self._haversackEdit.toolTip = ("What `uv pip install` installs for haversack: a requirement "
                                       "(git URL, @branch) or a local checkout path (installed editable).")
        self._lnqEdit = qt.QLineEdit(self.logic.lnqSource())
        for edit, key in ((self._homeEdit, "Home"), (self._haversackEdit, "HaversackSource"),
                          (self._lnqEdit, "LNQSource")):
            edit.connect("editingFinished()", lambda e=edit, k=key: self._onSettingEdited(k, e))
        aform.addRow("Home:", self._homeEdit)
        aform.addRow("haversack:", self._haversackEdit)
        aform.addRow("lnq-segmenter:", self._lnqEdit)
        form.addRow(advanced)
        return box

    def _buildInputSection(self):
        box = ctk_collapsible("Input")
        form = qt.QFormLayout(box)
        self._inputSelector = slicer.qMRMLNodeComboBox()
        self._inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self._inputSelector.addEnabled = False
        self._inputSelector.removeEnabled = False
        self._inputSelector.noneEnabled = True
        self._inputSelector.setMRMLScene(slicer.mrmlScene)
        self._inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda _n: self._refreshRunButton())
        form.addRow("Volume:", self._inputSelector)
        self._deviceCombo = qt.QComboBox()
        self._deviceCombo.addItems(["auto", "mps", "cuda", "cpu"])
        self._deviceCombo.toolTip = "auto: CUDA, then Apple Silicon (MPS), then CPU."
        form.addRow("Device:", self._deviceCombo)
        return box

    def _buildModelSection(self):
        box = ctk_collapsible("Model")
        lay = qt.QVBoxLayout(box)
        row = qt.QHBoxLayout()
        self._filterEdit = qt.QLineEdit()
        self._filterEdit.placeholderText = "filter: name, modality, description…"
        self._filterEdit.connect("textChanged(QString)", self._applyFilter)
        refresh = qt.QPushButton("Refresh")
        refresh.connect("clicked()", self._populateModels)
        row.addWidget(self._filterEdit, 1)
        row.addWidget(refresh, 0)
        lay.addLayout(row)
        self._modelTree = qt.QTreeWidget()
        self._modelTree.setHeaderLabels(["Model", "Modality", "Weights", "License"])
        self._modelTree.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self._modelTree.setMinimumHeight(220)
        self._modelTree.connect("itemSelectionChanged()", self._onModelSelected)
        lay.addWidget(self._modelTree)
        self._modelInfo = qt.QLabel("")
        self._modelInfo.wordWrap = True
        self._modelInfo.setStyleSheet("color: gray;")
        lay.addWidget(self._modelInfo)
        self._transposeCheck = qt.QCheckBox("Allow transposed plans (--allow-transpose)")
        self._transposeCheck.toolTip = ("Some models (dentalsegmentator:base, totalvibe:vibe_sagittal, "
                                        "totalvibe:pancreas) permute axes; haversack refuses them unless asked.")
        lay.addWidget(self._transposeCheck)
        return box

    def _buildRunSection(self):
        box = qt.QWidget()
        row = qt.QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        self._runButton = qt.QPushButton("Segment")
        self._runButton.setDefault(True)
        self._runButton.connect("clicked()", self._onRunClicked)
        self._cancelButton = qt.QPushButton("Cancel")
        self._cancelButton.enabled = False
        self._cancelButton.connect("clicked()", self._onCancelClicked)
        row.addWidget(self._runButton, 1)
        row.addWidget(self._cancelButton, 0)
        return box

    def _buildProgressSection(self):
        box = ctk_collapsible("Progress")
        lay = qt.QVBoxLayout(box)
        self._statusLabel = qt.QLabel("idle")
        self._statusLabel.wordWrap = True
        self._progressBar = qt.QProgressBar()
        self._progressBar.setRange(0, 100)
        self._logView = qt.QPlainTextEdit()
        self._logView.setReadOnly(True)
        self._logView.setMaximumBlockCount(5000)
        self._logView.setFont(qt.QFontDatabase.systemFont(qt.QFontDatabase.FixedFont))
        lay.addWidget(self._statusLabel)
        lay.addWidget(self._progressBar)
        lay.addWidget(self._logView, 1)
        return box

    # ----- state -----

    def _refreshEnvironmentState(self):
        env = self.logic.env
        if not env.installed():
            self._envStatus.text = f"Not installed. Install creates {env.venv} (about 2 GB with torch)."
            self._installButton.text = "Install"
        else:
            try:
                missing = self.logic.missingSegmentOptions()
            except Exception as exc:
                missing = None
                self._envStatus.text = f"Installed at {env.venv}, but haversack does not start: {exc}"
            if missing:
                self._envStatus.text = (f"haversack in {env.venv} is too old for this module (no "
                                        f"{', '.join(missing)}). Update, or set its source under Advanced.")
            elif missing is not None:
                self._envStatus.text = f"Ready: {env.venv}"
            self._installButton.text = "Update"
        self._populateModels()

    def _populateModels(self):
        self._entries = self.logic.listModels() if self.logic.env.installed() else \
            catalog.extra_entries(self.logic.extraManifest(), self.logic.env.models)
        selected = self._selectedEntry()
        self._modelTree.clear()
        groups = {}
        for e in self._entries:
            parent = groups.get(e["group"])
            if parent is None:
                parent = groups[e["group"]] = qt.QTreeWidgetItem(self._modelTree, [e["group"]])
                parent.setFlags(qt.Qt.ItemIsEnabled)
                parent.setExpanded(True)
            weights = {True: "installed", False: "download", None: ""}[e["installed"]]
            item = qt.QTreeWidgetItem(parent, [e["title"], e["modality"], weights, e["license"]])
            item.setToolTip(0, e["description"])
            item.setData(0, qt.Qt.UserRole, e["key"])
            if selected and selected["key"] == e["key"]:
                item.setSelected(True)
        self._modelTree.resizeColumnToContents(0)
        self._applyFilter(self._filterEdit.text)
        self._refreshRunButton()

    def _applyFilter(self, text):
        words = str(text).lower().split()
        root = self._modelTree.invisibleRootItem()
        for g in range(root.childCount()):
            group = root.child(g)
            for i in range(group.childCount()):
                item = group.child(i)
                e = self._entry(item.data(0, qt.Qt.UserRole))
                hay = " ".join([e["title"], e["modality"], e["description"], e["key"]]).lower()
                item.setHidden(not all(w in hay for w in words))

    def _entry(self, key):
        return next((e for e in self._entries if e["key"] == key), None)

    def _selectedEntry(self):
        items = self._modelTree.selectedItems() if hasattr(self, "_modelTree") else []
        return self._entry(items[0].data(0, qt.Qt.UserRole)) if items else None

    def _onModelSelected(self):
        e = self._selectedEntry()
        self._modelInfo.text = "" if e is None else f"{e['key']} — {e['description']}"
        self._refreshRunButton()

    def _refreshRunButton(self):
        busy = self._runner is not None and (self._runner.isRunning() or bool(self._steps))
        e = self._selectedEntry()
        self._runButton.enabled = (not busy and self.logic.env.installed() and e is not None
                                   and self._inputSelector.currentNode() is not None)
        self._installButton.enabled = not busy
        self._cancelButton.enabled = busy

    def _onSettingEdited(self, key, edit):
        self.logic.setSetting(key, edit.text.strip())
        self._refreshEnvironmentState()

    # ----- actions -----

    def _onInstallClicked(self):
        self._logView.clear()
        self._status("Installing uv into Slicer…")
        try:
            commands = self.logic.installCommands()
        except Exception as exc:
            logging.exception("uv install failed")
            slicer.util.errorDisplay(f"Could not install uv into Slicer's Python: {exc}")
            return
        self._steps = [("Creating the environment", commands[0], True, None),
                       ("Installing haversack and lnq-segmenter (a few minutes the first time)",
                        commands[1], True, None)]
        self._onAllDone = self._refreshEnvironmentState
        self._runNext()

    def _onRunClicked(self):
        e = self._selectedEntry()
        volume = self._inputSelector.currentNode()
        if e is None or volume is None:
            return
        env = self.logic.env
        self._logView.clear()
        self._lastDone = None
        self._workDir = self.logic.newWorkDir()
        inPath = self._workDir / "input.nrrd"
        outPath = self._workDir / "labels.seg.nrrd"
        try:
            self._status(f"Saving {volume.GetName()}…")
            self.logic.saveInput(volume, inPath)
        except Exception as exc:
            slicer.util.errorDisplay(str(exc))
            return
        extra = ["--allow-transpose"] if self._transposeCheck.checked else []
        device = self._deviceCombo.currentText
        steps = []
        folds = None
        if e["kind"] == "haversack":
            self._task = e["data"]["name"]
        elif e["kind"] == "lnq":
            m = e["data"]
            folds = catalog.lnq_folds(m)

            def bundleReady():
                bundle = catalog.last_path(self._runner.stdoutLines)
                meta = None
                if catalog.needs_checkpoint_meta(m):
                    self._status("Reading the model's trainer and configuration from its checkpoint…")
                    meta = json.loads(self.logic._run([env.python, "-c", catalog.CHECKPOINT_META_CODE,
                                                       catalog.first_checkpoint(bundle, m)]).strip().splitlines()[-1])
                self._task = catalog.prepare_lnq_bundle(bundle, m, meta)
            steps.append((f"Fetching {m['name']}@{m['version']} (lnq-segmenter)",
                          env.lnq_download_command(m["name"], m["version"]), True, bundleReady))
        else:
            m = e["data"]
            if not fetch_model.is_installed(m, env.models):
                if not self._askLicense(m):
                    self._status("Cancelled: the license was not accepted.")
                    return
                # the environment's Python, not PythonSlicer: a quarantined Slicer bundle's
                # launcher refuses to start python-real (seen with 5.10 and 5.12 on macOS)
                cmd = [str(env.python), str(Path(fetch_model.__file__)), m["id"], "--dest", str(env.models),
                       "--accept-license", m["license"]["id"], "--json-progress"]
                steps.append((f"Downloading {m['display_name']}", cmd, True, None))
            extra += m.get("haversack_args") or []
            self._task = fetch_model.model_folder(m, env.models)

        def segmentStep():
            return env.segment_command(inPath, self._task, outPath, device=device, folds=folds, extra=extra)

        steps.append((f"Segmenting {volume.GetName()} with {e['title']}", segmentStep, True,
                      lambda: self._loadResult(outPath, e, volume)))
        self._steps = steps
        self._onAllDone = self._finishRun
        self._runNext()

    def _askLicense(self, m) -> bool:
        lic = m["license"]
        links = "".join(f'<li><a href="{l["url"]}">{l["title"]}</a></li>' for l in lic.get("links", []))
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle(f"License: {m['display_name']}")
        box.setTextFormat(qt.Qt.RichText)
        box.setText(f"<b>{lic['summary']}</b>")
        box.setInformativeText(
            "<p>" + lic["terms"].replace("\n\n", "</p><p>") + "</p>"
            + (f"<ul>{links}</ul>" if links else "")
            + (f"<p><i>Cite:</i> {lic['citation']}</p>" if lic.get("citation") else "")
            + f"<p>Download: {sum(f['size'] for f in m['files']) / 1e6:.0f} MB from {m['source']['url']}</p>")
        accept = box.addButton("I accept", qt.QMessageBox.AcceptRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(qt.QMessageBox.Cancel)
        box.exec_()
        return box.clickedButton() == accept

    def _onCancelClicked(self):
        self._steps = []
        if self._runner.isRunning():
            self._runner.cancel()
        self._refreshRunButton()

    # ----- the step queue -----

    def _runNext(self):
        if not self._steps:
            done, self._onAllDone = getattr(self, "_onAllDone", None), None
            self._refreshRunButton()
            if done:
                done()
            return
        label, command, clean, _onOk = self._steps[0]
        if callable(command):
            command = command()
        self._status(label + "…")
        self._progressBar.setValue(0)
        self._runner.start(command, clean_environment=clean)
        self._refreshRunButton()

    def _onStepFinished(self, code, status):
        if not self._steps:                                  # cancelled: the queue was cleared
            self._stopRun(f"Cancelled ({status}).")
            return
        label, _command, _clean, onOk = self._steps.pop(0)
        if status != "ok":
            self._stopRun(f"{label} {status} (exit {code}); see the log below.")
            return
        try:
            if onOk:
                onOk()
        except Exception as exc:
            logging.exception(label)
            self._stopRun(f"{label}: {exc}")
            return
        self._runNext()

    def _stopRun(self, message):
        """End the queue early. A failed run's files stay, for a look at what it was given."""
        self._steps = []
        self._onAllDone = None
        self._status(message)
        if self._workDir is not None:
            self._log(f"run files kept in {self._workDir}")
            self._workDir = None
        self._refreshRunButton()

    def _onEvent(self, text):
        ev = json.loads(text)
        kind = ev.get("event")
        if ev["source"] == "haversack":
            if kind == "progress":
                self._progressBar.setValue(int(100 * float(ev.get("fraction") or 0)))
                steps = f" {ev['step']}/{ev['n_steps']}" if ev.get("n_steps") else ""
                parts = f" [{ev['part'] + 1}/{ev['n_parts']}]" if ev.get("n_parts", 1) > 1 else ""
                self._statusLabel.text = f"{ev.get('stage')}{parts}{steps} {ev.get('detail') or ''}".strip()
            elif kind == "done":
                self._lastDone = ev
            elif kind == "error":
                self._log(f"haversack error ({ev.get('kind')}): {ev.get('message')}")
                self._statusLabel.text = ev.get("message") or "haversack failed"
            elif kind == "log":
                self._log(ev.get("message", ""))
            return
        done = ev.get("bytes") if "bytes" in ev else ev.get("bytes_done")
        total = ev.get("total") if "total" in ev else ev.get("bytes_total")
        if done is not None and total:
            self._progressBar.setValue(int(100 * done / total))
            self._statusLabel.text = (f"downloading {ev.get('file') or ev.get('asset') or ''} "
                                      f"{done / 1e6:.0f} / {total / 1e6:.0f} MB")
        elif kind == "error":
            self._log(f"{ev['source']} error: {ev.get('message')}")
        else:
            self._log(f"[{ev['source']}] {kind}")

    # ----- results -----

    def _loadResult(self, outPath, entry, volume):
        seg = self.logic.loadResult(outPath, entry, volume)
        present = len((self._lastDone or {}).get("present") or [])
        took = ((self._lastDone or {}).get("timings") or {}).get("total")
        self._status(f"Loaded {seg.GetName()}: {present} structures"
                     + (f" in {took:.0f} s" if took else "") + ".")
        self._progressBar.setValue(100)

    def _finishRun(self):
        if self._workDir is not None and not self.logic.setting("KeepRunFiles", ""):
            shutil.rmtree(self._workDir, ignore_errors=True)
        self._workDir = None
        self._populateModels()                      # weights may now be installed

    def _status(self, text):
        self._statusLabel.text = text
        self._log(text)

    def _log(self, line):
        self._logView.appendPlainText(line)

    def cleanup(self):
        if self._runner is not None and self._runner.isRunning():
            self._steps = []
            self._runner.cancel()


def ctk_collapsible(title, collapsed=False):
    import ctk
    box = ctk.ctkCollapsibleButton()
    box.text = title
    box.collapsed = collapsed
    return box


# =============================================================================
# Test
# =============================================================================

class HaversackSegmentTest(ScriptedLoadableModuleTest):
    """Runs without the haversack environment: the catalog of extra models, the license gate, the
    LNQ bundle layout, and the runner's event parsing against a stand-in command."""

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_extra_models_are_listed()
        self.test_fetch_refuses_without_the_license()
        self.test_runner_parses_events()

    def test_extra_models_are_listed(self):
        manifest = fetch_model.load_manifest()
        entries = catalog.extra_entries(manifest, Path(slicer.app.temporaryPath) / "hs-models")
        self.assertIn("extra:pants-regions", [e["key"] for e in entries])
        self.assertFalse(entries[0]["installed"])

    def test_fetch_refuses_without_the_license(self):
        m = fetch_model.get_model("pants-regions")
        dest = Path(slicer.app.temporaryPath) / "hs-models"
        with self.assertRaises(fetch_model.FetchError):
            fetch_model.fetch(m, dest, accept_license=None)
        self.assertFalse(fetch_model.model_folder(m, dest).exists())

    def test_runner_parses_events(self):
        runner = CommandRunner()
        events, finished = [], []
        runner.progressEvent.connect(lambda t: events.append(json.loads(t)))
        runner.finished.connect(lambda c, s: finished.append((c, s)))
        # a shell, not PythonSlicer, as the stand-in: the test must not depend on the launcher
        lines = ['haversack> {"event": "progress", "fraction": 0.5}', 'lnq> {"event": "download_done"}',
                 "plain line"]
        if sys.platform == "win32":
            script = "echo /some/folder& " + "& ".join(f"echo {l} 1>&2" for l in lines)
            runner.start(["cmd", "/c", script], clean_environment=True)
        else:
            script = "echo /some/folder; " + "; ".join(f"echo '{l}' >&2" for l in lines)
            runner.start(["/bin/sh", "-c", script], clean_environment=True)
        deadline = time.time() + 30
        while not finished and time.time() < deadline:
            slicer.app.processEvents()
            time.sleep(0.02)
        self.assertEqual(finished, [(0, "ok")])
        self.assertEqual([(e["source"], e["event"]) for e in events],
                         [("haversack", "progress"), ("lnq", "download_done")])
        self.assertEqual(runner.stdoutLines, ["/some/folder"])
