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

from HaversackSegmentLib import benchmark, catalog, environment, fetch_model, stepprogress
from HaversackSegmentLib.remover import TreeRemover
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

    @staticmethod
    def uvBinaryIfPresent():
        """uv if Slicer's Python has it already, else None (installs nothing)."""
        try:
            import uv
            return uv.find_uv_bin()
        except Exception:
            return None

    def diagnosticsHeader(self) -> list:
        """The lines of the diagnosis that need no subprocess (instant): this process and the
        hardware, uv's location, the paths. The rest - uv's version, the environment's Python,
        torch and haversack - are background steps (the widget's Diagnose)."""
        e = self.env
        host = environment.host_info()
        lines = [f"Slicer {slicer.app.applicationVersion}; its Python {host['python']} runs as "
                 f"{host['process_machine']}"
                 + (" under Rosetta" if host["translated"] else "")
                 + ("; hardware: Apple Silicon" if host["hardware_arm64"] and host["system"] == "Darwin" else ""),
                 f"environment wanted: Python {environment.python_request(host)}"
                 + (f" (must run as {environment.expected_machine(host)})" if environment.expected_machine(host) else "")]
        ub = self.uvBinaryIfPresent()
        lines.append(f"uv: {ub} ({environment.binary_arch(ub)})" if ub else
                     "uv: not in Slicer's Python yet; Install adds it")
        lines.append(f"uv cache: {e.uv_cache}; managed Python: {e.python_dir}; step logs: {e.logs}")
        lines.append(f"environment: {e.venv}" + ("" if e.python.exists() else " - none"))
        if e.python.exists():
            lines.append(f"haversack: {e.haversack} {'present' if e.haversack.is_file() else 'MISSING'}")
        lines.append(f"haversack source: {self.haversackSource()}")
        return lines

    def _run(self, command, timeout=120, env=None) -> str:
        """A short command in the haversack environment, its stdout returned."""
        r = subprocess.run([str(c) for c in command], capture_output=True, text=True,
                           timeout=timeout, env=env if env is not None else _clean_env())
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
    """The panel. Nothing slow runs on Slicer's main thread: installs, downloads, segmentations,
    Diagnose's checks and the model listing are QProcess steps (:class:`CommandRunner`), Uninstall
    deletes a little at a time from a timer (:class:`TreeRemover`), and a one-second heartbeat
    keeps the status line moving while any of them works. No Python threads: Slicer's main thread
    holds the GIL while it idles, so a thread hardly runs."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = HaversackSegmentLogic()
        self._runner: Optional[CommandRunner] = None
        self._steps: list = []          # (label, command, clean_env, on_ok[, on_fail[, echo]]) still to run
        self._entries: list = []
        self._workDir: Optional[Path] = None
        self._task = None               # what `haversack segment --task` gets for this run
        self._lastDone: Optional[dict] = None
        self._uv = None                 # stepprogress.UvProgress while a uv step runs
        self._stepLabel = ""
        self._stepText = ""
        self._stepStarted = 0.0
        self._stepCount = (0, 0)        # (this step, steps in the job)
        self._setupJob = None           # "Installing" / "Diagnosing" / "Uninstalling": shown under Setup too
        self._remover = None            # TreeRemover while Uninstall deletes
        self._removerStarted = 0.0
        self._onRemoved = None
        self._quiet: list = []          # runners listing models, alive until they finish
        self._bench = None
        self._recent: list = []

    # ----- UI construction -----

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.layout.addWidget(self._buildSetupSection())
        self.layout.addWidget(self._buildInputSection())
        self.layout.addWidget(self._buildModelSection())
        self.layout.addWidget(self._buildRunSection())
        self.layout.addWidget(self._buildProgressSection(), 1)
        self._runner = self._newRunner()
        self._runner.progressEvent.connect(self._onEvent)
        self._runner.logLine.connect(self._onLogLine)
        self._runner.finished.connect(self._onStepFinished)
        self._heartbeat = qt.QTimer(self.parent)
        self._heartbeat.setInterval(1000)
        self._heartbeat.connect("timeout()", self._onHeartbeat)
        self._heartbeat.start()
        self._refreshEnvironmentState()

    def _newRunner(self) -> CommandRunner:
        return CommandRunner(self.parent, log_dir=str(self.logic.env.logs),
                             output_mode=self.logic.setting("OutputMode", "pipe"))

    def _buildSetupSection(self):
        box = ctk_collapsible("Setup")
        form = qt.QFormLayout(box)
        self._envStatus = qt.QLabel("checking…")
        self._envStatus.wordWrap = True
        form.addRow(self._envStatus)
        row = qt.QHBoxLayout()
        self._installButton = qt.QPushButton("Install")
        self._installButton.toolTip = ("Create the haversack environment (Python 3.12, uv) and "
                                       "install haversack into it; again to update.")
        self._installButton.connect("clicked()", self._onInstallClicked)
        self._uninstallButton = qt.QPushButton("Uninstall…")
        self._uninstallButton.toolTip = ("Remove the environment, and optionally what it downloaded, so that "
                                         "Install starts from nothing. Model weights are kept.")
        self._uninstallButton.connect("clicked()", self._onUninstallClicked)
        self._diagnoseButton = qt.QPushButton("Diagnose")
        self._diagnoseButton.toolTip = ("Write what this setup is into the log below: Slicer's Python and "
                                        "its architecture, uv, the environment's Python, torch and haversack.")
        self._diagnoseButton.connect("clicked()", self._onDiagnoseClicked)
        for b in (self._installButton, self._uninstallButton, self._diagnoseButton):
            row.addWidget(b)
        form.addRow(row)
        # Install, Uninstall and Diagnose show their progress here as well as under Progress,
        # which may be scrolled out of view
        self._setupBar = qt.QProgressBar()
        self._setupBar.setRange(0, 100)
        self._setupBar.visible = False
        form.addRow(self._setupBar)

        advanced = ctk_collapsible("Advanced", collapsed=True)
        aform = qt.QFormLayout(advanced)
        self._homeEdit = qt.QLineEdit(self.logic.setting("Home", str(environment.DEFAULT_HOME)))
        self._homeEdit.toolTip = ("Holds the environment (env/), its downloads (uv-cache/, python/), extra "
                                  "models (models/), step logs (logs/) and run files (work/).")
        self._haversackEdit = qt.QLineEdit(self.logic.haversackSource())
        self._haversackEdit.toolTip = ("What `uv pip install` installs for haversack: a requirement "
                                       "(git URL, @branch) or a local checkout path (installed editable).")
        for edit, key in ((self._homeEdit, "Home"), (self._haversackEdit, "HaversackSource")):
            edit.connect("editingFinished()", lambda e=edit, k=key: self._onSettingEdited(k, e))
        self._outputCombo = qt.QComboBox()
        self._outputCombo.addItems(["pipe", "file"])
        self._outputCombo.setCurrentIndex(max(0, self._outputCombo.findText(self.logic.setting("OutputMode", "pipe"))))
        self._outputCombo.toolTip = ("How a step's output reaches this panel. pipe: the operating system "
                                     "signals each line (with a fallback check when it goes quiet). file: "
                                     "output goes to a log file that is read four times a second - try it "
                                     "if progress only appears when a step ends.")
        self._outputCombo.connect("currentIndexChanged(int)", self._onOutputModeChanged)
        aform.addRow("Home:", self._homeEdit)
        aform.addRow("haversack:", self._haversackEdit)
        aform.addRow("Output:", self._outputCombo)
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
        refresh.connect("clicked()", self._refreshEnvironmentState)
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
        self._benchButton = qt.QPushButton("Benchmark")
        self._benchButton.toolTip = (f"Segment Slicer's {benchmark.SAMPLE} sample with {benchmark.TASK} twice and "
                                     "report the timings and this machine (OS, CPU, RAM, GPU) as a block to "
                                     "paste into a GitHub issue; it is copied to the clipboard.")
        self._benchButton.connect("clicked()", self._onBenchmarkClicked)
        self._cancelButton = qt.QPushButton("Cancel")
        self._cancelButton.enabled = False
        self._cancelButton.connect("clicked()", self._onCancelClicked)
        row.addWidget(self._runButton, 1)
        row.addWidget(self._benchButton, 0)
        row.addWidget(self._cancelButton, 0)
        return box

    def _buildProgressSection(self):
        box = self._progressBox = ctk_collapsible("Progress")
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

    def _busy(self) -> bool:
        return (self._runner is not None and (self._runner.isRunning() or bool(self._steps))) \
            or self._remover is not None

    def _refreshEnvironmentState(self):
        """The environment's status and the model list. Checking haversack and listing its catalog
        start Python in the environment, which can take seconds, so both run as background steps
        and fill the panel in when they answer."""
        if self._busy():
            return                                   # re-run when the job finishes
        env = self.logic.env
        extra = catalog.extra_entries(self.logic.extraManifest(), env.models)
        if not env.installed():
            if env.python.exists():
                self._envStatus.text = (f"{env.venv} exists, but haversack is not installed in it: the last "
                                        "Install did not finish. Click Install again; Diagnose and the log "
                                        "below say why it stopped.")
            else:
                self._envStatus.text = f"Not installed. Install creates {env.venv} (about 2 GB with torch)."
            self._installButton.text = "Install"
            self._showModels(extra)
            return
        self._installButton.text = "Update"
        self._envStatus.text = f"Checking haversack in {env.venv}…"
        self._showModels(self._entries if any(e["kind"] == "haversack" for e in self._entries) else extra)

        def helpDone(code, status, r):
            if status == "cancelled":                # Uninstall stopped it
                return
            if status != "ok":
                self._envStatus.text = (f"Installed at {env.venv}, but haversack does not start; "
                                        "Diagnose says more. Its output:")
                for line in r.lines[-15:]:
                    self._log("  " + line)
                self._showModels(extra)
                return
            missing = environment.missing_segment_options("\n".join(r.lines))
            self._envStatus.text = (f"haversack in {env.venv} is too old for this module (no "
                                    f"{', '.join(missing)}). Update, or set its source under Advanced."
                                    if missing else f"Ready: {env.venv}")
            self._envStatus.text += " Listing models…"
            self._runQuiet(env.tasks_command(), tasksDone)

        def tasksDone(code, status, r):
            if status == "cancelled":
                return
            self._envStatus.text = self._envStatus.text.replace(" Listing models…", "")
            entries = []
            try:
                if status != "ok":
                    raise RuntimeError("\n".join(r.lines[-10:]))
                entries = catalog.haversack_entries(catalog.parse_json_output("\n".join(r.stdoutLines)))
            except Exception as exc:
                self._log(f"haversack's catalog could not be listed: {exc}")
            self._showModels(entries + extra)

        self._runQuiet(env.segment_help_command(), helpDone)

    def _runQuiet(self, command, done):
        """A short background command whose output is collected rather than shown."""
        r = self._newRunner()
        r.lines = []
        r.logLine.connect(lambda line: r.lines.append(line))
        self._quiet.append(r)

        def finished(code, status):
            self._quiet.remove(r)
            done(code, status, r)
        r.finished.connect(finished)
        r.start(command, clean_environment=True)

    def _showModels(self, entries):
        self._entries = list(entries)
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
        if not hasattr(self, "_cancelButton"):       # the selector signals while the panel is built
            return
        busy = self._busy()
        e = self._selectedEntry()
        installed = self.logic.env.installed()
        why = ("" if busy else
               "Segment needs the environment: click Install under Setup." if not installed else
               "Choose a model in the list above." if e is None else
               "Choose an input volume." if self._inputSelector.currentNode() is None else "")
        self._runButton.enabled = not busy and not why
        self._runHint.text = why
        for b in (self._installButton, self._uninstallButton, self._diagnoseButton):
            b.enabled = not busy
        self._benchButton.enabled = not busy and installed
        self._cancelButton.enabled = busy

    def _onSettingEdited(self, key, edit):
        self.logic.setSetting(key, edit.text.strip())
        self._runner.log_dir = str(self.logic.env.logs)
        self._refreshEnvironmentState()

    def _onOutputModeChanged(self, _index):
        mode = self._outputCombo.currentText
        self.logic.setSetting("OutputMode", mode)
        self._runner.output_mode = mode

    # ----- progress -----

    def _setFraction(self, fraction):
        """A real fraction moves the bar; None makes it a busy indicator (no fraction known yet).
        During Install, Uninstall and Diagnose the bar under Setup does the same."""
        bars = [self._progressBar] + ([self._setupBar] if self._setupJob else [])
        for bar in bars:
            if fraction is None:
                if bar.maximum != 0:
                    bar.setRange(0, 0)
            else:
                if bar.maximum == 0:
                    bar.setRange(0, 100)
                bar.setValue(int(100 * min(max(fraction, 0.0), 1.0)))

    def _beginSetupJob(self, name):
        """Install, Uninstall or Diagnose starts: its progress shows under Setup, and the
        Progress section opens."""
        self._setupJob = name
        self._setupBar.visible = True
        self._progressBox.collapsed = False
        self._setFraction(None)

    def _endSetupJob(self):
        self._setupJob = None
        self._setupBar.visible = False

    def _render(self):
        """The status line: the current step's own words, how long it has run, and a note when it
        has been quiet for a while (a big download or a long model load says nothing)."""
        if self._runner is not None and self._runner.isRunning():
            n, total = self._stepCount
            where = f"[{n}/{total}] " if total > 1 else ""
            elapsed = stepprogress.human_seconds(time.monotonic() - self._stepStarted)
            quiet = time.monotonic() - self._runner.lastOutput
            note = f" · no output for {stepprogress.human_seconds(quiet)}, still running" if quiet > 20 else ""
            text = f"{where}{self._stepText} · {elapsed}{note}"
        elif self._remover is not None:
            elapsed = stepprogress.human_seconds(time.monotonic() - self._removerStarted)
            text = f"Uninstalling: {self._remover.text()} · {elapsed}"
        else:
            return
        self._statusLabel.text = text
        if self._setupJob:
            self._envStatus.text = text

    def _onHeartbeat(self):
        self._render()

    def _onLogLine(self, line):
        if getattr(self, "_echo", True) or line.startswith(("$ ", "output delivery")):
            self._log(line)
        if self._uv is not None and self._uv.feed(line):
            self._stepText = f"{self._stepLabel}: {self._uv.text()}"
            frac = self._uv.fraction()
            if frac is not None:
                self._setFraction(frac)
            self._render()

    def _onEvent(self, text):
        ev = json.loads(text)
        kind = ev.get("event")
        if ev["source"] == "haversack":
            if kind == "progress":
                stage, detail = ev.get("stage") or "", ev.get("detail") or ""
                if stage == "weights" and ev.get("n_steps"):
                    # a first use's weights install: its own bytes, not the run's fraction
                    self._setFraction(ev["step"] / ev["n_steps"])
                    self._stepText = (f"{self._stepLabel}: downloading weights {detail} "
                                      f"{stepprogress.human_bytes(ev['step'])} of "
                                      f"{stepprogress.human_bytes(ev['n_steps'])}")
                else:
                    self._setFraction(float(ev.get("fraction") or 0))
                    steps = f" {ev['step']}/{ev['n_steps']}" if ev.get("n_steps") else ""
                    parts = f" [part {ev['part'] + 1}/{ev['n_parts']}]" if ev.get("n_parts", 1) > 1 else ""
                    self._stepText = f"{self._stepLabel}: {stage}{parts}{steps} {detail}".strip()
                self._render()
            elif kind == "done":
                self._lastDone = ev
            elif kind == "error":
                self._log(f"haversack error ({ev.get('kind')}): {ev.get('message')}")
            elif kind == "log":
                self._log(ev.get("message", ""))
            return
        done = ev.get("bytes") if "bytes" in ev else ev.get("bytes_done")
        total = ev.get("total") if "total" in ev else ev.get("bytes_total")
        if done is not None and total:
            self._setFraction(done / total)
            self._stepText = (f"{self._stepLabel}: {ev.get('file') or ev.get('asset') or ''} "
                              f"{stepprogress.human_bytes(done)} of {stepprogress.human_bytes(total)}")
            self._render()
        elif kind == "error":
            self._log(f"{ev['source']} error: {ev.get('message')}")
        else:
            self._log(f"[{ev['source']}] {kind}")

    # ----- Uninstall's deleting, a little at a time from a timer -----

    def _removeInSteps(self, paths, done):
        """Delete ``paths`` with :class:`TreeRemover`, about 40 ms of work per timer tick so Slicer
        keeps answering; ``done(remover)`` when it ends (or is cancelled)."""
        self._remover = TreeRemover(paths)
        self._removerStarted = time.monotonic()
        self._onRemoved = done
        self._setFraction(None)
        self._refreshRunButton()
        self._render()
        qt.QTimer.singleShot(0, self._removeSome)

    def _removeSome(self):
        r = self._remover
        if r is None:                                    # cancelled
            return
        more = r.step(0.04)
        self._setFraction(r.fraction())
        if more:
            qt.QTimer.singleShot(10, self._removeSome)   # let Slicer handle its events between
            return
        self._render()
        self._finishRemoving()

    def _finishRemoving(self):
        r, done = self._remover, self._onRemoved
        self._remover, self._onRemoved = None, None
        self._refreshRunButton()
        if done:
            done(r)

    # ----- actions -----

    def _onDiagnoseClicked(self):
        """The instant part first, then uv, the environment's Python, torch and haversack as
        background steps; a check that fails is reported and the next one still runs."""
        env = self.logic.env
        self._beginSetupJob("Diagnosing")
        self._log("Diagnosis:")
        for line in self.logic.diagnosticsHeader():
            self._log("  " + line)
        host = environment.host_info()

        def out():
            return "\n".join(self._runner.stdoutLines)

        def failed(what):
            return lambda: self._log(f"  {what} - FAILED (its output is above)")

        def probeOk():
            probe = environment.parse_json_line(self._runner.stdoutLines)
            problem = environment.check_probe(probe, host)
            self._log(f"  environment Python {probe['version']} runs as {probe['machine']}"
                      + (f" - PROBLEM: {problem}" if problem else " - ok"))

        def verifyOk():
            v = environment.parse_json_line(self._runner.stdoutLines)
            self._log(f"  torch {v['torch']} (cuda {v['cuda']}, mps {v['mps']}), haversack {v['haversack']}")

        def helpOk():
            missing = environment.missing_segment_options(out())
            self._log("  haversack segment options: " + (
                "ok" if not missing else "MISSING " + ", ".join(missing) + " - its source is too old"))

        steps = []
        uv = self.logic.uvBinaryIfPresent()
        if uv:
            steps.append(("Diagnosing: uv's version", [uv, "--version"], True,
                          lambda: self._log("  " + out().strip()), failed("uv --version")))
        if env.python.exists():
            steps.append(("Diagnosing: the environment's Python", env.probe_command(), True, probeOk,
                          failed("the environment's Python does not run")))
            if env.haversack.is_file():
                steps.append(("Diagnosing: importing torch and haversack (can take a minute the first time)",
                              env.verify_command(), True, verifyOk, failed("importing torch and haversack")))
                steps.append(("Diagnosing: haversack segment --help", env.segment_help_command(), True, helpOk,
                              failed("haversack segment --help"), False))
        self._steps = steps

        def diagnosed():
            self._endSetupJob()
            self._setFraction(1.0)
            self._status("Diagnosis is in the log below. (Select it and copy it to share.)")
            self._refreshEnvironmentState()
        self._onAllDone = diagnosed
        self._startJob()

    def _onUninstallClicked(self):
        env = self.logic.env
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle("Uninstall haversack's environment")
        box.setText("Remove the environment so that Install starts again?")
        box.setInformativeText(
            f"<p><b>Environment only</b> removes {env.venv}; a reinstall reuses the downloads it "
            f"kept, so it is quick.</p><p><b>Environment and downloads</b> also removes {env.uv_cache} "
            f"and {env.python_dir}, so a reinstall downloads everything again (about 2 GB) - the way to "
            "see Install's progress from the start.</p><p>Model weights are kept either way.</p>")
        envOnly = box.addButton("Environment only", qt.QMessageBox.AcceptRole)
        everything = box.addButton("Environment and downloads", qt.QMessageBox.DestructiveRole)
        box.addButton(qt.QMessageBox.Cancel)
        box.setDefaultButton(qt.QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked not in (envOnly, everything):
            return
        # the model listing runs the environment's Python, which Windows will not let us delete
        for r in list(self._quiet):
            r.cancel()
        paths = env.uninstall_paths(downloads=clicked == everything)
        self._logView.clear()
        self._beginSetupJob("Uninstalling")
        self._log("Uninstalling: removing " + ", ".join(str(p) for p in paths))

        def done(r):
            self._endSetupJob()
            left = [p for p in paths if p.exists()]
            if r.done and not r.failures and not left:
                self._setFraction(1.0)
                self._status(f"Uninstalled ({r.text()}). Install sets it up again.")
            else:
                for path, message in r.failed:
                    self._log(f"  could not remove {path}: {message}")
                why = ("stopped" if not r.done else
                       f"{r.failures:,} files or folders could not be removed (in use? a program from the "
                       "environment may still run - close it, or restart Slicer, and Uninstall again)")
                self._setFraction(0)
                self._status(f"Uninstall {why}; still present: {', '.join(map(str, left)) or 'nothing'}.")
            self._refreshEnvironmentState()
        self._removeInSteps(paths, done)

    def _onInstallClicked(self):
        self._logView.clear()
        env = self.logic.env
        host = environment.host_info()
        self._beginSetupJob("Installing")
        self._status("Installing: adding uv to Slicer's Python…")
        slicer.app.processEvents()                       # show that before pip_install blocks
        try:
            uv = self.logic.uvBinary()
        except Exception as exc:
            logging.exception("uv install failed")
            self._endSetupJob()
            self._status(f"Could not install uv into Slicer's Python: {exc}")
            self._refreshEnvironmentState()
            return
        for line in self.logic.diagnosticsHeader():
            self._log("  " + line)
        # --allow-existing keeps an environment as it is, so one built for the wrong architecture
        # (an x86_64 Python on Apple Silicon, from a Slicer under Rosetta), or whose Python does
        # not start, is replaced (--clear) by the create step
        replace = {"yes": False}

        def existingOk():
            problem = environment.check_probe(environment.parse_json_line(self._runner.stdoutLines), host)
            if problem:
                self._log(f"  replacing {env.venv}: {problem}")
                replace["yes"] = True

        def existingFailed():
            self._log(f"  replacing {env.venv}: its Python does not run")
            replace["yes"] = True

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

        steps = [("Installing: checking the existing environment", env.probe_command(), True,
                  existingOk, existingFailed)] if env.python.exists() else []
        self._steps = steps + [
            (f"Installing: creating the environment (Python {environment.python_request(host)})",
             lambda: env.create_command(uv, host, clear=replace["yes"]), True, None),
            ("Installing: checking the environment's Python", env.probe_command(), True, checkPython),
            ("Installing haversack (several minutes and about 2 GB the first time)",
             env.install_command(uv, self.logic.haversackSource()), True, None),
            ("Installing: checking torch and haversack", env.verify_command(), True, checkInstall),
        ]

        def installed():
            self._endSetupJob()
            self._setFraction(1.0)
            self._status("Installed.")
            self._refreshEnvironmentState()
        self._onAllDone = installed
        self._startJob()

    def _prepareInput(self, volume) -> tuple:
        self._workDir = self.logic.newWorkDir()
        inPath = self._workDir / "input.nrrd"
        self._status(f"Saving {volume.GetName()}…")
        self.logic.saveInput(volume, inPath)
        return inPath

    def _onRunClicked(self):
        e = self._selectedEntry()
        volume = self._inputSelector.currentNode()
        if e is None or volume is None:
            return
        env = self.logic.env
        self._logView.clear()
        try:
            inPath = self._prepareInput(volume)
        except Exception as exc:
            self._status(str(exc))
            return
        outPath = self._workDir / "labels.seg.nrrd"
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
        self._startJob()

    def _onBenchmarkClicked(self):
        """CTChest + ts.v2:total_fast twice, timed, with this machine described; the report goes to
        the log, the clipboard and a file under the home directory."""
        env = self.logic.env
        self._logView.clear()
        self._status(f"Benchmark: loading the {benchmark.SAMPLE} sample…")
        try:
            import SampleData
            volume = slicer.mrmlScene.GetFirstNodeByName(benchmark.SAMPLE) or SampleData.downloadSample(benchmark.SAMPLE)
            inPath = self._prepareInput(volume)
        except Exception as exc:
            self._status(f"Benchmark: could not prepare the sample: {exc}")
            return
        self._bench = {"system": None, "device": None, "runs": [], "requested": self._deviceCombo.currentText}

        def systemRead():
            self._bench["system"] = environment.parse_json_line(self._runner.stdoutLines)
            self._bench["device"] = benchmark.device_for(self._bench["system"], self._bench["requested"])
            s = self._bench["system"]
            self._log(f"  {s.get('os')} · {s.get('cpu')} · {s.get('gpu') or 'no GPU'} · device {self._bench['device']}")

        def runStep(k):
            return lambda: env.segment_command(inPath, benchmark.TASK, self._workDir / f"bench-{k}.seg.nrrd",
                                               device=self._bench["device"])

        def runDone(k):
            def done():
                ev = self._lastDone or {}
                run = {"run": k + 1, "wall_s": round(time.monotonic() - self._stepStarted, 3),
                       "timings": {key: round(v, 3) for key, v in (ev.get("timings") or {}).items()},
                       "structures_present": len(ev.get("present") or [])}
                self._bench["runs"].append(run)
                self._log(f"  run {k + 1}: haversack total {run['timings'].get('total', 0):.2f} s, "
                          f"wall {run['wall_s']:.2f} s, {run['structures_present']} structures")
            return done

        self._steps = [("Benchmark: reading this machine", [str(env.python), "-c", benchmark.SYSTEM_INFO_CODE],
                        True, systemRead)]
        for k in range(benchmark.RUNS):
            what = "first run (includes the weights' download on first use)" if k == 0 else "warm run"
            self._steps.append((f"Benchmark: {benchmark.TASK} on {benchmark.SAMPLE}, {what}", runStep(k), True,
                                runDone(k)))
        self._onAllDone = self._finishBenchmark
        self._startJob()

    def _finishBenchmark(self):
        b, self._bench = self._bench, None
        report = benchmark.build_report(
            system=b["system"], device=b["device"], runs=b["runs"],
            slicer={"version": slicer.app.applicationVersion, "revision": slicer.app.revision,
                    "python": sys.version.split()[0]},
            module={"haversack_source": self.logic.haversackSource(),
                    "output_mode": self._runner.output_mode, "device_requested": b["requested"]},
            when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        md = benchmark.to_markdown(report)
        out = self.logic.env.home / "benchmarks" / time.strftime("benchmark-%Y%m%d-%H%M%S.md")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(md, encoding="utf-8")
        except OSError as exc:
            self._log(f"could not save the report: {exc}")
        qt.QApplication.clipboard().setText(md)
        self._log("")
        for line in md.splitlines():
            self._log(line)
        self._status(f"Benchmark done: {benchmark.summary_line(report)}. The report is on the clipboard "
                     f"(paste it into the GitHub issue) and in {out}.")
        if self._workDir is not None:
            shutil.rmtree(self._workDir, ignore_errors=True)
            self._workDir = None
        self._refreshEnvironmentState()

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
        self._bench = None
        if self._runner.isRunning():
            self._runner.cancel()
        if self._remover is not None:
            self._finishRemoving()                       # done() reports what is left
        self._refreshRunButton()

    # ----- the step queue -----

    def _startJob(self):
        self._stepCount = (0, len(self._steps))
        self._runNext()

    def _runNext(self):
        if not self._steps:
            done, self._onAllDone = getattr(self, "_onAllDone", None), None
            self._uv = None
            self._refreshRunButton()
            if done:
                done()
            return
        label, command, clean = self._steps[0][:3]
        self._echo = len(self._steps[0]) < 6 or self._steps[0][5]   # False: output only in the step log
        if callable(command):
            command = command()
        n, total = self._stepCount
        self._stepCount = (n + 1, total)
        self._stepLabel = self._stepText = label
        self._stepStarted = time.monotonic()
        self._lastDone = None
        self._uv = stepprogress.UvProgress() if Path(str(command[0])).name.lower().startswith("uv") else None
        self._log(label + "…")
        self._setFraction(None)
        self._runner.start(command, clean_environment=clean, extra_env=self.logic.env.uv_env())
        self._refreshRunButton()
        self._render()

    def _onStepFinished(self, code, status):
        if not self._steps:                                  # cancelled: the queue was cleared
            self._stopRun(f"Cancelled ({status}).")
            return
        step = self._steps.pop(0)
        label, onOk = step[0], step[3]
        onFail = step[4] if len(step) > 4 else None      # a check whose failure is only reported
        if status != "ok" and onFail is not None and status != "cancelled":
            onFail()
            self._runNext()
            return
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
        self._bench = None
        self._uv = None
        self._setFraction(0)
        self._endSetupJob()
        self._status(message)
        if self._runner.logFile:
            self._log(f"step log: {self._runner.logFile}")
        if self._workDir is not None:
            self._log(f"run files kept in {self._workDir}")
            self._workDir = None
        self._refreshRunButton()
        self._refreshEnvironmentState()

    # ----- results -----

    def _loadResult(self, outPath, entry, volume):
        seg = self.logic.loadResult(outPath, entry, volume)
        present = len((self._lastDone or {}).get("present") or [])
        took = ((self._lastDone or {}).get("timings") or {}).get("total")
        self._setFraction(1.0)
        self._status(f"Loaded {seg.GetName()}: {present} structures"
                     + (f" in {took:.0f} s" if took else "") + ".")

    def _finishRun(self):
        if self._workDir is not None and not self.logic.setting("KeepRunFiles", ""):
            shutil.rmtree(self._workDir, ignore_errors=True)
        self._workDir = None
        self._refreshEnvironmentState()              # weights may now be installed

    def _status(self, text):
        self._statusLabel.text = text
        self._log(text)

    def _log(self, line):
        self._recent = (self._recent + [str(line)])[-40:]
        self._logView.appendPlainText(line)

    def _lastError(self) -> str:
        """The failed command's most telling line, for the status: uv's and haversack's errors."""
        for line in reversed(self._recent):
            low = line.lower()
            if low.startswith(("error", "haversack error", "×", "  ×")) or "error:" in low or "no solution" in low:
                return ": " + line.strip()[:300]
        return ""

    def cleanup(self):
        if getattr(self, "_heartbeat", None) is not None:
            self._heartbeat.stop()
        if self._runner is not None and self._runner.isRunning():
            self._steps = []
            self._runner.cancel()
        self._remover = None


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
        """Both output modes deliver the events, the plain lines and stdout, and say how."""
        lines = ['haversack> {"event": "progress", "fraction": 0.5}', 'fetch> {"event": "done"}',
                 "plain line"]
        for mode in ("pipe", "file"):
            runner = CommandRunner(output_mode=mode)
            events, finished = [], []
            runner.progressEvent.connect(lambda t: events.append(json.loads(t)))
            runner.finished.connect(lambda c, s: finished.append((c, s)))
            # a shell, not PythonSlicer, as the stand-in: the test must not depend on the launcher
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
            self.assertEqual(finished, [(0, "ok")], mode)
            self.assertEqual([(e["source"], e["event"]) for e in events],
                             [("haversack", "progress"), ("fetch", "done")], mode)
            self.assertEqual(runner.stdoutLines, ["/some/folder"], mode)
            self.assertEqual(sum(v for k, v in runner.delivery.items() if k != "mode"), 4, mode)
            self.assertTrue(Path(runner.logFile).is_file(), mode)
