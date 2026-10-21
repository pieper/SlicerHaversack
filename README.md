# SlicerHaversack

A 3D Slicer extension with one module, **Haversack Segment**, that runs
[haversack](https://github.com/mhalle/haversack)'s segmentation models on a volume in the scene:

- **haversack's catalog**: TotalSegmentator (v2, v3), MOOSE, MRSegmentator, DentalSegmentator,
  TotalVibeSegmentator, CADS, FastSurfer, SynthStrip. Weights download on first use.
- **LNQ lymph-node models** ([lnq-segmenter](https://github.com/pieper/lnq-segmenter)).
  lnq-segmenter downloads and verifies the bundle, and haversack runs every fold the registry
  names.
- **Extra models that need a license accepted**, listed in
  `HaversackSegment/Resources/extra_models.json`. They are kept out of haversack's catalog on
  purpose. The first one is the PanTS nnU-Net (pancreatic lesion, pancreas head/body/tail/duct
  and 24 other structures). Its weights are MIT but its training data is CC BY-NC-ND 4.0.

## How it runs

Inference doesn't run in Slicer's Python. haversack needs Python 3.12–3.14 and installs only
with uv, so the module keeps its own environment at `~/.slicer-haversack/env`. **Install**
pip-installs `uv` into Slicer, creates the environment with uv's Python 3.12, and installs
haversack and lnq-segmenter into it. `--torch-backend auto` picks the right torch build for a
CUDA card. A run is three steps:

1. The selected volume is saved to `~/.slicer-haversack/work/run-*/input.nrrd`.
2. `haversack segment input.nrrd --task <task> -o labels.seg.nrrd --json-progress` runs as a
   `QProcess`. Progress lines (`haversack> {...}`) feed the progress bar, and **Cancel** kills
   the process.
3. `labels.seg.nrrd` is loaded as a segmentation on the volume's geometry.

For an LNQ model, `lnq-segmenter download` runs first. For an extra model,
`fetch_model.py` runs first.

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
   - **Install fails with "Application is NOT executable … python-real".** macOS has
     quarantined the Slicer bundle, so Slicer's own Python launcher refuses to start, and
     `pip_install` goes through that launcher. Clear the flag and retry:
     `xattr -dr com.apple.quarantine /Applications/Slicer.app` (use your Slicer's path).
   - **Slicer hangs at startup on a "reopen windows?" alert after a crash.** Launch it once
     with `-ApplePersistenceIgnoreState YES`.

Until [mhalle/haversack#4](https://github.com/mhalle/haversack/pull/4) is merged, Install takes
haversack from `pieper/haversack@region-models`. That branch adds region-based models (PanTS),
`--folds` and `--json-progress`. Set *Advanced → haversack* to use another source.

## Settings (Setup → Advanced)

- **Home**: where the environment, the extra models and the run files live.
- **haversack**: what `uv pip install` installs. This is a requirement (`haversack @
  git+https://github.com/mhalle/haversack@<branch>`) or a local checkout path, which is
  installed editable. The module needs `haversack segment --json-progress` and `--folds`. If
  the installed haversack lacks them, Setup says so.
- **lnq-segmenter**: the same, for lnq-segmenter.

## Tests

- The Slicer-free parts (the fetcher, the license gate, the environment commands, the
  catalog): `python -m pytest HaversackSegment/Testing/Python`.
- In Slicer, the module's self-test also covers the QProcess runner.

## License

Apache-2.0. Each model's own license and citation apply to its weights and its results.
