# SlicerHaversack

A 3D Slicer extension with one module, **Haversack Segment**, that runs
[haversack](https://github.com/mhalle/haversack)'s segmentation models on a volume in the scene:

- **haversack's catalog**: TotalSegmentator (v2, v3), MOOSE, MRSegmentator, DentalSegmentator,
  TotalVibeSegmentator, CADS, FastSurfer, SynthStrip. Weights download on first use.
- **LNQ lymph-node models**: haversack's `lnq:*` tasks (haversack 0.16.0), listed in a group of
  their own. haversack downloads them from the
  [lnq-segmenter](https://github.com/pieper/lnq-segmenter) releases and runs each model's full
  fold ensemble.
- **Extra models that need a license accepted**, listed in
  `HaversackSegment/Resources/extra_models.json`. They are kept out of haversack's catalog on
  purpose. The first one is the PanTS nnU-Net (pancreatic lesion, pancreas head/body/tail/duct
  and 24 other structures). Its weights are MIT but its training data is CC BY-NC-ND 4.0.

## How it runs

Inference doesn't run in Slicer's Python. haversack needs Python 3.12–3.14 and installs only
with uv, so the module keeps its own environment at `~/.slicer-haversack/env`. **Install**
pip-installs `uv` into Slicer, creates the environment with uv's Python 3.12 (the arm64 build on
an Apple Silicon Mac, even when Slicer itself runs under Rosetta), and installs haversack into it.
`--torch-backend auto` picks the right torch build for a CUDA card. uv keeps its downloads and the
managed Python under the module's home (`uv-cache/`, `python/`), not in its shared cache, so
**Uninstall…** can remove exactly what the module downloaded. A run is three steps:

1. The selected volume is saved to `~/.slicer-haversack/work/run-*/input.nrrd`.
2. `haversack segment input.nrrd --task <task> -o labels.seg.nrrd --json-progress` runs as a
   `QProcess`. Progress lines (`haversack> {...}`) feed the progress bar, and **Cancel** kills
   the process.
3. `labels.seg.nrrd` is loaded as a segmentation on the volume's geometry.

For an extra model, `fetch_model.py` downloads the weights first.

Slicer stays usable throughout. Every install, download and segmentation step is a background
process, and so are the model listing, the version check and Diagnose's checks. Uninstall deletes
a few hundred files at a time between Slicer's own events. (No Python threads: Slicer's main thread
holds the GIL while it idles, so a background thread barely runs.) Install, Uninstall and Diagnose
show their progress under *Setup* as well as under *Progress*.
- **The status line** shows the step, its elapsed time, and a note when the step has printed
  nothing for a while.
- **The progress bar** follows what the step reports: uv's downloads (bytes finished of bytes
  announced), haversack's per-patch progress, or a weights download's bytes. Until a step reports
  a fraction, it shows as busy.
- **Each step's output** goes to its own log under `~/.slicer-haversack/logs/`.
- **How output reaches the panel:** each step ends with a line saying how its output arrived
  (readyRead signals, the fallback check, files). Lines normally arrive through QProcess's
  readyRead signals. If those go quiet for a second while the step runs, the module asks
  QProcess to check the pipes. *Advanced → Output: file* sends output to files instead, read
  four times a second.

A license is accepted once per model and revision, in a dialog that shows its terms, links and
citation. `fetch_model.py` downloads nothing unless the exact license id is passed. It records
the acceptance in `LICENSE_ACCEPTANCE.json` in the model folder before the first byte
arrives, and checks every file's size and sha256 against the manifest.

Supported machines are haversack's: Apple Silicon (MPS), a CUDA card, or the CPU on Linux and
Windows. Intel Macs can't run it, because current torch publishes no wheels for them.

## Try it on an Apple Silicon Mac

1. **Get Slicer and the module.**
   - You need macOS 14 (Sonoma) or later: the oldest the current torch publishes Apple
     Silicon wheels for.
   - Install a current [3D Slicer](https://download.slicer.org) (5.10 or later) for macOS
     arm64.
   - `git clone https://github.com/pieper/SlicerHaversack`.
2. **Point Slicer at the module.** In Slicer, open *Edit → Application Settings → Modules*,
   add `<clone>/HaversackSegment` under *Additional module paths*, and restart Slicer.
3. **Install.** Open *Segmentation → Haversack Segment*, expand *Setup*, and click
   **Install**. It builds `~/.slicer-haversack/env` with Python 3.12, torch and haversack
   (about 2 GB). It takes a few minutes the first time, and progress shows in the log at the
   bottom of the panel.
4. **Segment.**
   - Load a CT. *Download Sample Data → CTA abdomen (Panoramix)* is a good first case.
   - Pick a model, such as `ts.v2:total_fast`, an LNQ model, or PanTS (which asks you to accept
     its license first).
   - Set *Device* to `mps` (or leave it on `auto`) and click **Segment**.
5. **Troubleshooting.**
   - **Start over.** **Uninstall…** removes the environment only (a reinstall is quick), or the
     environment and its downloads (a reinstall downloads everything again). Model weights are
     kept either way.
   - **Progress appears only when a step ends.** Look at the delivery line in the log. If it
     says the lines came "at exit", switch *Advanced → Output* to `file`.
   - **Install or Segment doesn't work.** Click **Diagnose** under *Setup*. It writes the setup
     into the log: Slicer's Python and its architecture, uv, the environment's Python, torch and
     whether it sees the GPU. A line under **Segment** says why it is disabled.
   - **Install fails with "Application is NOT executable … python-real".** macOS has
     quarantined the Slicer bundle, so Slicer's own Python launcher refuses to start, and
     `pip_install` goes through that launcher. Clear the flag and retry:
     `xattr -dr com.apple.quarantine /Applications/Slicer.app` (use your Slicer's path).
   - **Slicer hangs at startup on a "reopen windows?" alert after a crash.** Launch it once
     with `-ApplePersistenceIgnoreState YES`.

Until [mhalle/haversack#5](https://github.com/mhalle/haversack/pull/5) (`--folds`,
`--json-progress`) and [#6](https://github.com/mhalle/haversack/pull/6) (region-based models such
as PanTS) are merged, Install takes haversack from `pieper/haversack@slicerhaversack`: haversack
0.16.0 with both. Set *Advanced → haversack* to use another source.

## Benchmark

**Benchmark** (next to Segment) measures this machine:
- It segments Slicer's CTChest sample with `ts.v2:total_fast` twice. The first run includes the
  weights' download on first use; the second is the warm number to compare.
- It records the OS, CPU, RAM, GPU and its memory, driver/CUDA, and the versions of torch,
  haversack and Slicer.
- The report is a short Markdown block with one JSON payload. It is copied to the clipboard and
  saved under `~/.slicer-haversack/benchmarks/`. Paste it into the benchmark issue as a comment.

To collect every report in a thread into a table:

```bash
python tools/collate_benchmarks.py --issue pieper/SlicerHaversack#<n>          # Markdown table
python tools/collate_benchmarks.py --issue pieper/SlicerHaversack#<n> --csv    # CSV
```

## Settings (Setup → Advanced)

- **Home**: where the environment (`env/`), its downloads (`uv-cache/`, `python/`), the extra
  models, step logs, benchmark reports and run files live.
- **Output**: `pipe` (default) or `file`, as described under *How it runs*.
- **haversack**: what `uv pip install` installs. This is a requirement (`haversack @
  git+https://github.com/mhalle/haversack@<branch>`) or a local checkout path, which is
  installed editable. The module needs `haversack segment --json-progress` and `--folds`. If
  the installed haversack lacks them, Setup says so.

## Tests

- The Slicer-free parts (the fetcher, the license gate, the environment commands, the catalog,
  log tailing, uv progress, benchmark reports and the collate script):
  `python -m pytest HaversackSegment/Testing/Python`.
- In Slicer, the module's self-test also covers the QProcess runner in both output modes.

## License

Apache-2.0. Each model's own license and citation apply to its weights and its results.
