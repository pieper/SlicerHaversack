"""Haversack Segment - run haversack's segmentation models on a volume in the scene.

One module for every model haversack runs: its own catalog (TotalSegmentator, MOOSE,
MRSegmentator, CADS, the LNQ lymph-node models, ...), and extra models whose license the user
accepts before the weights are fetched (PanTS). Inference runs as a subprocess - `haversack segment ... --json-progress` - in an
environment the module keeps apart from Slicer's Python (HaversackSegmentLib.environment), with
progress streamed into the panel and a cancel that kills the process.
"""
from __future__ import annotations

import json
import logging
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

    def probeEnvironment(self) -> dict:
        """What the environment's Python runs as (machine, version), or raises."""
        return environment.parse_json_line(self._run(self.env.probe_command(), timeout=60).splitlines())

    def removeEnvironment(self):
        shutil.rmtree(self.env.venv, ignore_errors=True)

    def diagnostics(self, verify=False) -> list:
        """Lines describing the setup, for the log: this process and the hardware, uv, the
        environment's Python, and (``verify``) torch and haversack in it."""
        host = environment.host_info()
        lines = [f"Slicer {slicer.app.applicationVersion}; its Python {host['python']} runs as "
                 f"{host['process_machine']}"
                 + (" under Rosetta" if host["translated"] else "")
                 + ("; hardware: Apple Silicon" if host["hardware_arm64"] and host["system"] == "Darwin" else ""),
                 f"environment wanted: Python {environment.python_request(host)}"
                 + (f" (must run as {environment.expected_machine(host)})" if environment.expected_machine(host) else "")]
        try:
            import uv
            ub = uv.find_uv_bin()
            ver = subprocess.run([ub, "--version"], capture_output=True, text=True, timeout=30,
                                 env=_clean_env()).stdout.strip()
            lines.append(f"uv: {ub} ({environment.binary_arch(ub)}) {ver}")
        except Exception as exc:
            lines.append(f"uv: not available in Slicer's Python yet ({exc.__class__.__name__}); Install adds it")
        e = self.env
        if not e.python.exists():
            lines.append(f"environment: none at {e.venv}")
            return lines
        try:
            probe = self.probeEnvironment()
            problem = environment.check_probe(probe, host)
            lines.append(f"environment: {e.venv}: Python {probe['version']} runs as {probe['machine']}"
                         + (f" - PROBLEM: {problem}" if problem else " - ok"))
        except Exception as exc:
            lines.append(f"environment: {e.venv}: its Python does not run: {exc}")
        lines.append(f"haversack: {e.haversack} {'present' if e.haversack.is_file() else 'MISSING'}")
        if verify and e.haversack.is_file():
            try:
                v = environment.parse_json_line(self._run(e.verify_command(), timeout=300).splitlines())
                lines.append(f"torch {v['torch']} (cuda {v['cuda']}, mps {v['mps']}), haversack {v['haversack']}")
                lines.append("haversack segment options: " + (
                    "ok" if not self.missingSegmentOptions() else
                    "MISSING " + ", ".join(self.missingSegmentOptions()) + " - its source is too old"))
            except Exception as exc:
                lines.append(f"torch/haversack import check failed: {exc}")
        lines.append(f"haversack source: {self.haversackSource()}")
        return lines

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
        listed is skipped rather than hiding the others; why is kept in ``listProblems``."""
        entries, self.listProblems = [], []
        try:
            entries += catalog.haversack_entries(
                catalog.parse_json_output(self._run(self.env.tasks_command())))
        except Exception as exc:
            self.listProblems.append(f"haversack's catalog could not be listed: {exc}")
        for problem in self.listProblems:
            logging.warning(problem)
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
        the entry's renames (PanTS's leftover ``pancreas`` is ``pancreas_other``)."""
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
                                       "install haversack into it; again to update.")
        self._installButton.connect("clicked()", self._onInstallClicked)
        self._diagnoseButton = qt.QPushButton("Diagnose")
        self._diagnoseButton.toolTip = ("Write what this setup is into the log below: Slicer's Python and "
                                        "its architecture, uv, the environment's Python, torch and haversack.")
        self._diagnoseButton.connect("clicked()", self._onDiagnoseClicked)
        row.addWidget(self._envStatus, 1)
        row.addWidget(self._installButton, 0)
        row.addWidget(self._diagnoseButton, 0)
        form.addRow(row)

        advanced = ctk_collapsible("Advanced", collapsed=True)
        aform = qt.QFormLayout(advanced)
        self._homeEdit = qt.QLineEdit(self.logic.setting("Home", str(environment.DEFAULT_HOME)))
        self._homeEdit.toolTip = "Holds the environment (env/), extra models (models/) and run files (work/)."
        self._haversackEdit = qt.QLineEdit(self.logic.haversackSource())
        self._haversackEdit.toolTip = ("What `uv pip install` installs for haversack: a requirement "
                                       "(git URL, @branch) or a local checkout path (installed editable).")
        for edit, key in ((self._homeEdit, "Home"), (self._haversackEdit, "HaversackSource")):
            edit.connect("editingFinished()", lambda e=edit, k=key: self._onSettingEdited(k, e))
        aform.addRow("Home:", self._homeEdit)
        aform.addRow("haversack:", self._haversackEdit)
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
        col = qt.QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        row = qt.QHBoxLayout()
        col.addLayout(row)
        self._runHint = qt.QLabel("")
        self._runHint.wordWrap = True
        self._runHint.setStyleSheet("color: gray;")
        col.addWidget(self._runHint)
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
            if env.python.exists():
                self._envStatus.text = (f"{env.venv} exists, but haversack is not installed in it: the last "
                                        "Install did not finish. Click Install again; Diagnose and the log "
                                        "below say why it stopped.")
            else:
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
        for problem in getattr(self.logic, "listProblems", []) if self.logic.env.installed() else []:
            self._log(problem)
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
        why = ("running…" if busy else
               "Segment needs the environment: click Install under Setup." if not self.logic.env.installed() else
               "Choose a model in the list above." if e is None else
               "Choose an input volume." if self._inputSelector.currentNode() is None else "")
        self._runButton.enabled = not why
        self._runHint.text = "" if busy else why
        self._installButton.enabled = not busy
        self._diagnoseButton.enabled = not busy
        self._cancelButton.enabled = busy

    def _onSettingEdited(self, key, edit):
        self.logic.setSetting(key, edit.text.strip())
        self._refreshEnvironmentState()

    # ----- actions -----

    def _onDiagnoseClicked(self):
        self._status("Diagnosing (checks torch and haversack, which takes a few seconds)…")
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            for line in self.logic.diagnostics(verify=True):
                self._log("  " + line)
        finally:
            slicer.app.restoreOverrideCursor()
        self._status("Diagnosis above.")

    def _onInstallClicked(self):
        self._logView.clear()
        env = self.logic.env
        host = environment.host_info()
        self._status("Installing uv into Slicer's Python…")
        try:
            uv = self.logic.uvBinary()
        except Exception as exc:
            logging.exception("uv install failed")
            self._status(f"Could not install uv into Slicer's Python: {exc}")
            return
        for line in self.logic.diagnostics():
            self._log("  " + line)
        # --allow-existing keeps an environment as it is, so one built for the wrong architecture
        # (an x86_64 Python on Apple Silicon, from a Slicer under Rosetta) is removed first
        if env.python.exists():
            try:
                problem = environment.check_probe(self.logic.probeEnvironment(), host)
            except Exception as exc:
                problem = f"its Python does not run ({exc})"
            if problem:
                self._log(f"Removing {env.venv}: {problem}")
                self.logic.removeEnvironment()

        def checkPython():
            probe = environment.parse_json_line(self._runner.stdoutLines)
            self._log(f"  environment Python {probe['version']} runs as {probe['machine']}")
            problem = environment.check_probe(probe, host)
            if problem:
                raise RuntimeError(problem)

        def checkInstall():
            v = environment.parse_json_line(self._runner.stdoutLines)
            self._log(f"  torch {v['torch']} (cuda {v['cuda']}, mps {v['mps']}), haversack {v['haversack']}, "
                      f"running as {v['machine']}")
            if environment.expected_machine(host) == "arm64" and not v["mps"]:
                self._log("  note: torch does not see the Apple GPU (MPS); segmentation will use the CPU")

        self._steps = [
            (f"Creating the environment (Python {environment.python_request(host)})",
             env.create_command(uv, host), True, None),
            ("Checking the environment's Python", env.probe_command(), True, checkPython),
            ("Installing haversack (several minutes and about 2 GB the first time)",
             env.install_command(uv, self.logic.haversackSource()), True, None),
            ("Checking torch and haversack", env.verify_command(), True, checkInstall),
        ]
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
        folds = None              # a catalog task runs the ensemble its catalog states (lnq:*)
        if e["kind"] == "haversack":
            self._task = e["data"]["name"]
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
            self._stopRun(f"{label} {status} (exit {code}){self._lastError()}; the log below has the rest.")
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
        self._recent = (getattr(self, "_recent", []) + [str(line)])[-40:]
        self._logView.appendPlainText(line)

    def _lastError(self) -> str:
        """The failed command's most telling line, for the status: uv's and haversack's errors."""
        for line in reversed(getattr(self, "_recent", [])):
            low = line.lower()
            if low.startswith(("error", "haversack error", "×", "  ×")) or "error:" in low or "no solution" in low:
                return ": " + line.strip()[:300]
        return ""

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
    """Runs without the haversack environment: the catalog of extra models, the license gate, and
    the runner's event parsing against a stand-in command."""

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
        lines = ['haversack> {"event": "progress", "fraction": 0.5}', 'fetch> {"event": "done"}',
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
                         [("haversack", "progress"), ("fetch", "done")])
        self.assertEqual(runner.stdoutLines, ["/some/folder"])
