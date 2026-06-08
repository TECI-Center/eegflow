"""
Batch compute EEG metrics from Muse NDJSON files.

Produces a single summary CSV with one row per participant and columns for
each metric in both the baseline and task phases.

Usage:
    python3.11 compute_ndjson_metrics.py --input_dir /path/to/ndjson_files/
    python3.11 compute_ndjson_metrics.py --input_dir /path/to/ --output_dir results/
    python3.11 compute_ndjson_metrics.py --file single.ndjson
"""

import argparse
import csv
import json
import os
import sys
import pathlib
import glob

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from musereader_ndjson import MuseNDJSONReader

_HERE = pathlib.Path(__file__).parent
_DEFAULT_OUTPUT = str(_HERE / "results")

_METRICS  = ["focus_index", "engagement_index", "FAA_index", "TLX"]
_PHASES   = ["baseline", "task"]


def process_file(filepath: str, output_dir: str) -> dict:
    """
    For one ndjson file writes:
      {stem}_timeseries.json  – all 4 metrics, full session
      {stem}_aggregate.json   – {"baseline": {...}, "task": {...}}
    Returns a flat dict for the summary CSV.
    """
    stem = os.path.splitext(os.path.basename(filepath))[0]
    row  = {"participant": stem}

    print(f"\n{'='*60}")
    print(f"  {os.path.basename(filepath)}")
    print(f"{'='*60}")

    reader = MuseNDJSONReader(filepath)
    print(f"  Loaded {len(reader.data):,} samples")
    if "phase" in reader.data.columns:
        counts = reader.data["phase"].value_counts().to_dict()
        print(f"  Phase counts: {counts}")

    os.makedirs(output_dir, exist_ok=True)

    # ── timeseries: full session, all metrics in one file ──────────────
    reader.compute_metrics(phase=None)
    ts_payload = {}
    for metric, (times, vals) in reader.metric_time_series.items():
        ts_payload[metric] = {
            "time":   [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in times],
            "values": [float(v) for v in vals],
        }
    ts_path = os.path.join(output_dir, f"{stem}_timeseries.json")
    with open(ts_path, "w") as f:
        json.dump(ts_payload, f, indent=2)
    print(f"  Timeseries JSON → {os.path.basename(ts_path)}")

    # ── aggregate: per-phase ────────────────────────────────────────────
    agg_payload = {}
    for phase in _PHASES:
        try:
            reader.compute_metrics(phase=phase)
            agg = reader.aggregate_metrics
            agg_payload[phase] = {m: round(agg.get(m, float("nan")), 6) for m in _METRICS}
            for m in _METRICS:
                row[f"{phase}_{m}"] = agg_payload[phase][m]
            print(f"  {phase}: " + "  ".join(f"{m}={agg_payload[phase][m]:.4f}" for m in _METRICS))
        except ValueError as e:
            print(f"  SKIP {phase} ({e})")
            agg_payload[phase] = {m: None for m in _METRICS}
            for m in _METRICS:
                row[f"{phase}_{m}"] = ""

    agg_path = os.path.join(output_dir, f"{stem}_aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg_payload, f, indent=2)
    print(f"  Aggregate JSON  → {os.path.basename(agg_path)}")

    return row


def main():
    parser = argparse.ArgumentParser(
        description="Compute EEG metrics from Muse NDJSON files and write summary CSV."
    )
    parser.add_argument("--input_dir", default=None,
                        help="Directory containing .ndjson files.")
    parser.add_argument("--file", default=None,
                        help="Process a single .ndjson file.")
    parser.add_argument("--output_dir", default=_DEFAULT_OUTPUT,
                        help=f"Output directory (default: {_DEFAULT_OUTPUT}).")
    args = parser.parse_args()

    if not args.input_dir and not args.file:
        parser.error("Provide --input_dir or --file.")

    if args.file:
        files = [args.file]
    else:
        pattern = os.path.join(args.input_dir, "**", "*.ndjson")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            print(f"No .ndjson files found under {args.input_dir}")
            sys.exit(1)
        print(f"Found {len(files)} .ndjson file(s)")

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for fp in files:
        try:
            rows.append(process_file(fp, args.output_dir))
        except Exception as e:
            print(f"  ERROR processing {fp}: {e}")

    if not rows:
        print("No results to write.")
        sys.exit(1)

    # CSV columns: participant, then baseline_* then task_* for each metric
    fieldnames = ["participant"] + \
                 [f"{ph}_{m}" for ph in _PHASES for m in _METRICS]

    out_csv = os.path.join(args.output_dir, "metrics_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary CSV → {out_csv}")
    print(f"  {len(rows)} participant(s)  ×  {len(fieldnames)-1} metric columns")


if __name__ == "__main__":
    main()
