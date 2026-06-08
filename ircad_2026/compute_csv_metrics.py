"""
Batch script to compute EEG metrics from Muse CSV export files.

Usage:
    python compute_csv_metrics.py --input_dir  /path/to/csv/folder \
                                  --output_dir results/

    # Single file:
    python compute_csv_metrics.py --input_dir  /path/to/folder \
                                  --output_dir results/ \
                                  --file F_01_faculty_MuseS-951F_2026-05-27-09-33-08_1.csv
"""

import argparse
import os
import sys
import glob
import traceback
import pathlib

# Ensure the parent eegflow directory is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from musereader_csv import MuseCSVReader


def process_file(filepath: str, output_dir: str) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    reader = MuseCSVReader(filepath)
    print(f"  EEG samples loaded : {len(reader.data):,}")
    print(f"  Time range         : {reader.data['datetime'].iloc[0]}  →  {reader.data['datetime'].iloc[-1]}")

    reader.compute_metrics()

    print("  Aggregate metrics:")
    for metric, value in reader.aggregate_metrics.items():
        print(f"    {metric:<20} {value:.4f}")

    reader.save(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Compute EEG metrics from Muse CSV files.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing Muse CSV files (*.csv).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where JSON results will be written.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Process a single named file inside input_dir instead of all CSVs.",
    )
    args = parser.parse_args()

    if args.file:
        files = [os.path.join(args.input_dir, args.file)]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))

    if not files:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(files)} file(s) to process.")

    failed = []
    for fp in files:
        try:
            process_file(fp, args.output_dir)
        except Exception as e:
            print(f"  ERROR processing {fp}: {e}")
            traceback.print_exc()
            failed.append(fp)

    print(f"\nDone. {len(files) - len(failed)}/{len(files)} file(s) processed successfully.")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()
