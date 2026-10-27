"""QProcess wrapper that runs one command and re-emits its output, line by line, as Qt signals.

Adapted from SlicerLNQ's LymphNodeQuantifierLib/runner.py: the QProcess plugs into Slicer's event
loop, so Slicer stays usable while a step runs, and cancel is QProcess.kill (SIGKILL on POSIX,
TerminateProcess on Windows). JSON progress lines are recognized by prefix: haversack writes
``haversack> {json}`` and fetch_model ``fetch> {json}``, both on stderr.

How output reaches Slicer, in order of preference:

1. **The OS's notification**: QProcess's readyReadStandardOutput / readyReadStandardError signals.
   Every command runs with ``PYTHONUNBUFFERED=1`` so a Python program writes each line as it
   prints it.
2. **A fallback, only while the signals are silent**: a watchdog that, when the process is
   running and no readyRead has arrived for a second, asks QProcess to look at the pipes now
   (``waitForReadyRead(0)``, which emits readyRead itself if data is waiting). On Windows an
   install showed nothing for minutes and a segmentation sat at 27 % and then printed everything
   at once (2026-10-27); the watchdog is what makes that impossible, whatever the cause.
3. **File output** (``output_mode="file"``, the module's Advanced setting): the OS writes the
   program's output straight to files and a timer reads them
   (:class:`~HaversackSegmentLib.logtail.LogTail`) - for a machine where the pipes cannot be
   made to deliver at all.

Each step's output is also written to a log file in ``log_dir``, and the step ends with a line
saying how its output arrived (``delivery``), which is what to look at if progress stalls.

A program from the haversack environment must not see Slicer's own Python settings
(``PYTHONHOME``, ``PYTHONPATH``, which the Slicer launcher sets for PythonSlicer): its interpreter
would load Slicer's standard library. Such commands run with ``clean_environment=True``, the
environment Slicer was started with (``slicer.util.startupEnvironment``).
"""
from __future__ import annotations

import codecs
import json
import os
import tempfile
import time

import qt
import slicer

from .logtail import LogTail

PREFIXES = ("haversack> ", "fetch> ")
WATCHDOG_MS = 500          # how often the fallback looks
SILENT_S = 1.0             # how long without a readyRead before it does
FILE_POLL_MS = 250         # file mode's read interval


class CommandRunner(qt.QObject):
    """Runs one command at a time.

    Signals:
      progressEvent(str)     one event, JSON text, with the prefix it came under as "source"
      logLine(str)           any other output line
      finished(int, str)     exit code and status: "ok" | "failed" | "cancelled" | "crashed"
    ``stdoutLines`` holds the run's stdout, for the commands that print a result path;
    ``logFile`` the current (or last) step's log; ``delivery`` how its lines arrived.
    """

    progressEvent = qt.Signal(str)
    logLine = qt.Signal(str)
    finished = qt.Signal(int, str)

    def __init__(self, parent=None, log_dir=None, output_mode="pipe"):
        qt.QObject.__init__(self, parent)
        self.log_dir = log_dir
        self.output_mode = output_mode
        self._process = qt.QProcess(self)
        self._process.setProcessChannelMode(qt.QProcess.SeparateChannels)
        self._process.readyReadStandardError.connect(self._onStderr)
        self._process.readyReadStandardOutput.connect(self._onStdout)
        self._process.finished.connect(self._onFinished)
        self._process.errorOccurred.connect(self._onErrorOccurred)
        self._timer = qt.QTimer(self)
        self._timer.connect("timeout()", self._onTimer)
        self._decoders, self._buffers, self._tails = {}, {}, {}
        self._log = None
        self._cancelled = False
        self._done = False
        self._inFallback = False
        self.stdoutLines: list[str] = []
        self.logFile = None
        self.delivery = {}
        self.lastOutput = 0.0          # time.monotonic() of the last line
        self._lastSignal = 0.0

    def start(self, command: list, clean_environment: bool = False, extra_env: dict | None = None):
        program, args = str(command[0]), [str(a) for a in command[1:]]
        source = (slicer.util.startupEnvironment().items() if clean_environment
                  else os.environ.items())
        env = qt.QProcessEnvironment()
        for key, value in source:
            env.insert(key, value)
        if clean_environment:
            for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE"):
                env.remove(key)
        env.insert("PYTHONUNBUFFERED", "1")
        for key, value in (extra_env or {}).items():
            env.insert(str(key), str(value))
        self._process.setProcessEnvironment(env)

        log_dir = self.log_dir or os.path.join(tempfile.gettempdir(), "slicer-haversack-logs")
        os.makedirs(log_dir, exist_ok=True)
        stem = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S-") + os.path.basename(program).split(".")[0])
        n, self.logFile = 1, stem + ".log"
        while os.path.exists(self.logFile):                      # two steps in one second
            n += 1
            self.logFile = f"{stem}-{n}.log"
        self._log = open(self.logFile, "w", encoding="utf-8")
        self._log.write("$ " + " ".join(_quote(a) for a in [program, *args]) + "\n")

        self._decoders = {c: codecs.getincrementaldecoder("utf-8")(errors="replace") for c in ("stdout", "stderr")}
        self._buffers = {"stdout": "", "stderr": ""}
        self._tails = {}
        if self.output_mode == "file":
            # the OS redirects; QProcess opens no pipes, so no readyRead will come
            self._process.setStandardOutputFile(self.logFile + ".stdout")
            self._process.setStandardErrorFile(self.logFile + ".stderr")
            self._tails = {"stdout": LogTail(self.logFile + ".stdout"),
                           "stderr": LogTail(self.logFile + ".stderr")}
        else:
            # an empty name restores the pipe, after a file-mode run on this same QProcess
            self._process.setStandardOutputFile("")
            self._process.setStandardErrorFile("")
        self._cancelled = False
        self._done = False
        self.stdoutLines = []
        self.delivery = {"mode": self.output_mode, "signal": 0, "fallback": 0, "file": 0, "final": 0}
        self.lastOutput = self._lastSignal = time.monotonic()
        self.logLine.emit("$ " + " ".join(_quote(a) for a in [program, *args]))
        self._process.start(program, args)
        self._timer.start(FILE_POLL_MS if self.output_mode == "file" else WATCHDOG_MS)

    def cancel(self):
        if self.isRunning():
            self._cancelled = True
            self._process.kill()

    def isRunning(self) -> bool:
        return self._process.state() != qt.QProcess.NotRunning

    # ----- output: signals first -----

    def _onStderr(self):
        self._read("stderr")

    def _onStdout(self):
        self._read("stdout")

    def _read(self, channel, final=False):
        data = (self._process.readAllStandardError() if channel == "stderr"
                else self._process.readAllStandardOutput())
        raw = bytes(data.data())
        if not raw and not final:
            return
        how = "fallback" if self._inFallback else ("final" if final else "signal")
        if not self._inFallback and not final:
            self._lastSignal = time.monotonic()
        text = self._buffers[channel] + self._decoders[channel].decode(raw, final=final)
        lines = text.split("\n")
        self._buffers[channel] = "" if final else lines.pop()
        self._emit(channel, (l.rstrip("\r") for l in lines if l.strip()), how)

    def _emit(self, channel, lines, how):
        for line in lines:
            self.delivery[how] += 1
            self.lastOutput = time.monotonic()
            if self._log is not None:
                self._log.write(line + "\n")
            if channel == "stdout":
                self.stdoutLines.append(line)
            event = _event(line)
            if event is not None:
                self.progressEvent.emit(json.dumps(event))
            else:
                self.logLine.emit(line)
        if self._log is not None:
            self._log.flush()

    # ----- output: the fallbacks -----

    def _onTimer(self):
        if self.output_mode == "file":
            for channel, tail in self._tails.items():
                self._emit(channel, tail.read(), "file")
            return
        if not self.isRunning() or time.monotonic() - self._lastSignal < SILENT_S:
            return
        # silent for a second while running: have QProcess look at the pipes now. It emits
        # readyRead itself when there is data, which lands in _read, counted as "fallback".
        self._inFallback = True
        try:
            self._process.waitForReadyRead(0)
        finally:
            self._inFallback = False

    # ----- termination -----

    def _finish(self, code, status):
        if self._done:
            return
        self._done = True
        self._timer.stop()
        if self.output_mode == "file":
            for channel, tail in self._tails.items():
                self._emit(channel, tail.read(final=True), "file")
                tail.close()
        else:
            for channel in ("stdout", "stderr"):
                self._read(channel, final=True)
        d = self.delivery
        summary = (f"output delivery ({d['mode']}): {d['signal']} lines by readyRead, {d['fallback']} by "
                   f"the fallback, {d['file']} from files, {d['final']} at exit; log: {self.logFile}")
        if self._log is not None:
            self._log.write(summary + "\n")
            self._log.close()
            self._log = None
        self.logLine.emit(summary)
        self.finished.emit(int(code), status)

    def _onErrorOccurred(self, err):
        # FailedToStart fires instead of finished; without this the UI waits forever
        if err == qt.QProcess.FailedToStart:
            self.logLine.emit(f"failed to start {self._process.program()}: {self._process.errorString()}")
            self._finish(-1, "crashed")

    def _onFinished(self, exit_code, exit_status=None):
        # PythonQt routes the (int) overload, so the status may arrive as None
        if exit_status is None:
            exit_status = self._process.exitStatus()
        if self._cancelled:
            status = "cancelled"
        elif exit_status == qt.QProcess.CrashExit:
            status = "crashed"
        else:
            status = "ok" if exit_code == 0 else "failed"
        self._finish(exit_code, status)


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
