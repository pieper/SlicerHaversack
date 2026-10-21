"""QProcess wrapper that runs one command and re-emits its JSON progress lines as Qt signals.

Adapted from SlicerLNQ's LymphNodeQuantifierLib/runner.py: pure Qt, no Python ``subprocess`` and
no threads - the QProcess plugs into Slicer's event loop, so output arrives on the GUI thread, and
cancel is QProcess.kill (SIGKILL on POSIX, TerminateProcess on Windows). Generalized to any
program and any set of event prefixes: haversack writes ``haversack> {json}``, lnq-segmenter
``lnq> {json}``, fetch_model ``fetch> {json}``, all on stderr.

A program from the haversack environment must not see Slicer's own Python settings
(``PYTHONHOME``, ``PYTHONPATH``, which the Slicer launcher sets for PythonSlicer): its interpreter
would load Slicer's standard library. Such commands run with ``clean_environment=True``, the
environment Slicer was started with (``slicer.util.startupEnvironment``).
"""
from __future__ import annotations

import json

import qt
import slicer

PREFIXES = ("haversack> ", "lnq> ", "fetch> ")


class CommandRunner(qt.QObject):
    """Runs one command at a time.

    Signals:
      progressEvent(str)     one event, JSON text, with the prefix it came under as "source"
      logLine(str)           any other output line
      finished(int, str)     exit code and status: "ok" | "failed" | "cancelled" | "crashed"
    ``stdoutLines`` holds the run's stdout, for the commands that print a result path.
    """

    progressEvent = qt.Signal(str)
    logLine = qt.Signal(str)
    finished = qt.Signal(int, str)

    def __init__(self, parent=None):
        qt.QObject.__init__(self, parent)
        self._process = qt.QProcess(self)
        self._process.setProcessChannelMode(qt.QProcess.SeparateChannels)
        self._buffers = {"stderr": "", "stdout": ""}
        self._cancelled = False
        self.stdoutLines: list[str] = []
        self._process.readyReadStandardError.connect(self._onStderr)
        self._process.readyReadStandardOutput.connect(self._onStdout)
        self._process.finished.connect(self._onFinished)
        self._process.errorOccurred.connect(self._onErrorOccurred)

    def start(self, command: list, clean_environment: bool = False):
        program, args = str(command[0]), [str(a) for a in command[1:]]
        if clean_environment:
            env = qt.QProcessEnvironment()
            for key, value in slicer.util.startupEnvironment().items():
                env.insert(key, value)
            for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE"):
                env.remove(key)
            self._process.setProcessEnvironment(env)
        else:
            self._process.setProcessEnvironment(qt.QProcessEnvironment.systemEnvironment())
        self._cancelled = False
        self.stdoutLines = []
        self._buffers = {"stderr": "", "stdout": ""}
        self.logLine.emit("$ " + " ".join(_quote(a) for a in [program, *args]))
        self._process.start(program, args)

    def cancel(self):
        if self.isRunning():
            self._cancelled = True
            self._process.kill()

    def isRunning(self) -> bool:
        return self._process.state() != qt.QProcess.NotRunning

    # ----- stream parsing -----

    def _onStderr(self):
        self._read("stderr", self._process.readAllStandardError())

    def _onStdout(self):
        self._read("stdout", self._process.readAllStandardOutput())

    def _read(self, channel, data):
        text = bytes(data.data()).decode("utf-8", errors="replace")
        self._buffers[channel] = self._dispatch(channel, self._buffers[channel] + text)

    def _dispatch(self, channel, buf):
        while "\n" in buf:
            line, _, buf = buf.partition("\n")
            line = line.rstrip("\r")
            if not line:
                continue
            if channel == "stdout":
                self.stdoutLines.append(line)
            event = _event(line)
            if event is not None:
                self.progressEvent.emit(json.dumps(event))
            else:
                self.logLine.emit(line)
        return buf

    # ----- termination -----

    def _onErrorOccurred(self, err):
        # FailedToStart fires instead of finished; without this the UI waits forever
        if err == qt.QProcess.FailedToStart:
            self.logLine.emit(f"failed to start {self._process.program()}: {self._process.errorString()}")
            self.finished.emit(-1, "crashed")

    def _onFinished(self, exit_code, exit_status=None):
        # PythonQt routes the (int) overload, so the status may arrive as None
        if exit_status is None:
            exit_status = self._process.exitStatus()
        for channel in ("stderr", "stdout"):
            tail, self._buffers[channel] = self._buffers[channel], ""
            if tail:
                self._dispatch(channel, tail + "\n")
        if self._cancelled:
            status = "cancelled"
        elif exit_status == qt.QProcess.CrashExit:
            status = "crashed"
        else:
            status = "ok" if exit_code == 0 else "failed"
        self.finished.emit(int(exit_code), status)


def _event(line: str):
    for prefix in PREFIXES:
        bare = prefix.strip()
        if line.startswith(bare):
            try:
                event = json.loads(line[len(bare):].strip())
            except ValueError:
                return None
            if isinstance(event, dict):
                event["source"] = bare.rstrip(">")
                return event
    return None


def _quote(arg: str) -> str:
    return f"'{arg}'" if (" " in arg or not arg) else arg
