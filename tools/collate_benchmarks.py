#!/usr/bin/env python3
"""Collect SlicerHaversack benchmark reports into one table.

The module's Benchmark button produces a Markdown block with a fenced JSON payload
(``slicer-haversack-benchmark/1``); people paste it into a GitHub issue. This finds every such
payload in an issue thread (its body and every comment, through the ``gh`` CLI) or in files, and
prints one row per report:

    python tools/collate_benchmarks.py --issue pieper/SlicerHaversack#1
    python tools/collate_benchmarks.py --issue pieper/SlicerHaversack#1 --csv > benchmarks.csv
    python tools/collate_benchmarks.py pasted.md more.md

Standard library only (plus ``gh`` for --issue).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).resolve().parents[1] / "HaversackSegment" / "HaversackSegmentLib" / "benchmark.py")
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def issue_texts(ref: str) -> list[tuple[str, str]]:
    """``[(author, text)]`` for an issue's body and comments; ``ref`` is OWNER/REPO#N."""
    repo, _, number = ref.partition("#")
    if not (repo and number.isdigit()):
        raise SystemExit(f"--issue takes OWNER/REPO#NUMBER, not {ref!r}")

    def gh(path, pages=False):
        cmd = ["gh", "api", path] + (["--paginate", "--slurp"] if pages else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"gh api {path}: {out.stderr.strip()}")
        return json.loads(out.stdout)

    issue = gh(f"repos/{repo}/issues/{number}")
    comments = [c for page in gh(f"repos/{repo}/issues/{number}/comments", pages=True) for c in page]
    texts = [(issue["user"]["login"], issue.get("body") or "")]
    texts += [(c["user"]["login"], c.get("body") or "") for c in comments]
    return texts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="files holding pasted reports (default: stdin, unless --issue)")
    ap.add_argument("--issue", help="OWNER/REPO#NUMBER: read the issue's body and comments with gh")
    ap.add_argument("--csv", action="store_true", help="CSV instead of a Markdown table")
    args = ap.parse_args(argv)

    sources = issue_texts(args.issue) if args.issue else []
    for f in args.files:
        sources.append((Path(f).stem, Path(f).read_text(encoding="utf-8")))
    if not args.issue and not args.files:
        sources.append(("stdin", sys.stdin.read()))
    rows = [benchmark.row(r, who) for who, text in sources for r in benchmark.parse_reports(text)]
    if not rows:
        print("no benchmark reports found", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: (r["warm_total_s"] is None, r["warm_total_s"] or 0))
    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=benchmark.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    else:
        print("| " + " | ".join(benchmark.COLUMNS) + " |")
        print("|" + "---|" * len(benchmark.COLUMNS))
        for r in rows:
            print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in benchmark.COLUMNS) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
