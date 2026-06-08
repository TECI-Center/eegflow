"""
Batch process all Muse NDJSON files in a directory tree.

For each .ndjson file found:
  results/<stem>/
      <stem>_timeseries.json
      <stem>_aggregate.json     {"baseline": {...}, "task": {...}}
      plots/
          <stem>_eeg_metrics.png      (all 4 metrics combined)
          <stem>_focus_index.png
          <stem>_engagement_index.png
          <stem>_FAA_index.png
          <stem>_TLX.png

Also writes:
  results/metrics_summary.csv   (one row per participant)

Usage:
    python3.11 batch_process_ndjson.py --input_dir /path/to/day_2_app/
    python3.11 batch_process_ndjson.py --input_dir /path/to/ --xlim "11:17 11:25"
    python3.11 batch_process_ndjson.py --input_dir /path/to/ --output_dir my_results/
"""

import argparse
import csv
import glob
import json
import os
import pathlib
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from musereader_ndjson import MuseNDJSONReader
from plot_eeg_metrics import (
    load_timeseries,
    plot_combined,
    plot_individual,
    parse_xlim,
    _EASTERN,
)

_HERE    = pathlib.Path(__file__).parent
_DEFAULT_OUTPUT = str(_HERE / "results")

_METRICS = ["focus_index", "engagement_index", "FAA_index", "TLX"]
_PHASES  = ["baseline", "task"]


def process_one(filepath: str, output_dir: str, xlim_str: str | None) -> dict:
    stem = os.path.splitext(os.path.basename(filepath))[0]
    part_dir  = os.path.join(output_dir, stem)
    plots_dir = os.path.join(part_dir, "plots")
    os.makedirs(part_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {os.path.basename(filepath)}")
    print(f"  → {part_dir}")
    print(f"{'='*60}")

    # ── load & filter ──────────────────────────────────────────────────
    reader = MuseNDJSONReader(filepath)
    print(f"  Loaded {len(reader.data):,} samples")
    if "phase" in reader.data.columns:
        counts = reader.data["phase"].value_counts().to_dict()
        print(f"  Phase counts: {counts}")

    # ── timeseries (full session) ──────────────────────────────────────
    reader.compute_metrics(phase=None)
    ts_payload = {}
    for metric, (times, vals) in reader.metric_time_series.items():
        ts_payload[metric] = {
            "time":   [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in times],
            "values": [float(v) for v in vals],
        }
    ts_path = os.path.join(part_dir, f"{stem}_timeseries.json")
    with open(ts_path, "w") as f:
        json.dump(ts_payload, f, indent=2)
    print(f"  Timeseries  → {os.path.relpath(ts_path, output_dir)}")

    # ── aggregate (per phase) ──────────────────────────────────────────
    agg_payload = {}
    row = {"participant": stem}
    for phase in _PHASES:
        try:
            reader.compute_metrics(phase=phase)
            agg = reader.aggregate_metrics
            agg_payload[phase] = {m: round(agg.get(m, float("nan")), 6) for m in _METRICS}
            for m in _METRICS:
                row[f"{phase}_{m}"] = agg_payload[phase][m]
            print(f"  {phase:10s}: " + "  ".join(f"{m}={agg_payload[phase][m]:.4f}" for m in _METRICS))
        except ValueError as e:
            print(f"  SKIP {phase} ({e})")
            agg_payload[phase] = {m: None for m in _METRICS}
            for m in _METRICS:
                row[f"{phase}_{m}"] = ""

    agg_path = os.path.join(part_dir, f"{stem}_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg_payload, f, indent=2)
    print(f"  Aggregate   → {os.path.relpath(agg_path, output_dir)}")

    # ── plots ──────────────────────────────────────────────────────────
    timeseries = load_timeseries(ts_path)
    first_time = list(timeseries.values())[0][0][0]
    ref_date   = first_time.astimezone(_EASTERN).date()
    xlim = parse_xlim(xlim_str, ref_date) if xlim_str else None

    plot_combined(timeseries,  [], plots_dir, xlim=xlim, pid=stem)
    plot_individual(timeseries, [], plots_dir, xlim=xlim, pid=stem)

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Batch compute metrics + plots for all Muse NDJSON files."
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Root directory to search for .ndjson files.")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT,
                        help=f"Output root directory (default: {_DEFAULT_OUTPUT}).")
    parser.add_argument("--xlim",       default=None,
                        help="Optional plot x-axis window as 'HH:MM HH:MM' (e.g. '11:17 11:25').")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.ndjson"), recursive=True))
    # Skip OS duplicate copies (e.g. "file - Copy.ndjson")
    files = [f for f in files if " - Copy" not in os.path.basename(f)]
    if not files:
        print(f"No .ndjson files found under {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) under {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    errors = []
    for fp in files:
        try:
            rows.append(process_one(fp, args.output_dir, args.xlim))
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((fp, str(e)))

    # ── summary CSV ────────────────────────────────────────────────────
    if rows:
        fieldnames = ["participant"] + [f"{ph}_{m}" for ph in _PHASES for m in _METRICS]
        csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary CSV → {csv_path}")
        print(f"  {len(rows)} participant(s)  ×  {len(fieldnames)-1} metric columns")

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for fp, e in errors:
            print(f"  {os.path.basename(fp)}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
