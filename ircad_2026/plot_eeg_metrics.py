"""
Plot the four EEG metric time-series and annotate with events from ann.json.

Usage:
    python plot_eeg_metrics.py \
        --timeseries results/<pid>_timeseries.json \
        --annotations /path/to/ann.json \
        --output_dir  results/plots/

    # Optionally restrict the x-axis window:
    --xlim "09:30 10:30"
"""

import argparse
import json
import os
import sys
import pathlib
from datetime import datetime, date
from zoneinfo import ZoneInfo

# Ensure the parent eegflow directory is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

_EASTERN = ZoneInfo("America/New_York")

METRIC_LABELS = {
    "focus_index":      "Focus Index (β/α)",
    "engagement_index": "Engagement Index (β/(α+θ))",
    "FAA_index":        "FAA Index (ln α_R − ln α_L)",
    "TLX":              "Cognitive Load / TLX (θ/α)",
}

METRIC_COLORS = {
    "focus_index":      "#2196F3",
    "engagement_index": "#4CAF50",
    "FAA_index":        "#FF9800",
    "TLX":              "#E91E63",
}


def load_timeseries(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    parsed = {}
    for metric, payload in raw.items():
        times = [datetime.fromisoformat(t) for t in payload["time"]]
        values = np.array(payload["values"], dtype=float)
        parsed[metric] = (times, values)
    return parsed


def load_annotations(path: str, ref_date: date, tz=_EASTERN) -> list[dict]:
    """
    Parse ann.json.  Times are HH:MM:SS strings with no date; combine with
    ref_date (inferred from the timeseries) and localise to Eastern time.
    """
    with open(path) as f:
        raw = json.load(f)
    result = []
    for entry in raw:
        if not entry.get("time", "").strip():
            continue
        t = datetime.strptime(entry["time"].strip(), "%H:%M:%S").replace(
            year=ref_date.year,
            month=ref_date.month,
            day=ref_date.day,
            tzinfo=tz,
        )
        result.append({"time": t, "annotation": entry["annotation"]})
    return result


def smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    """Simple moving-average smoother."""
    if window < 2 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def _draw_metric(ax, times, values, label, color, annotations, xlim=None):
    """Draw one metric panel onto *ax* with annotations labelled in-plot."""
    ann_colors = plt.cm.tab10.colors

    # Smoothed line only
    ax.plot(times, smooth(values), color=color, linewidth=1.6, label=label)

    # Mean line
    mean_val = np.nanmean(values)
    ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"mean = {mean_val:.3f}")

    # Vertical annotation lines + in-plot text labels
    y_min, y_max = np.nanmin(smooth(values)), np.nanmax(smooth(values))
    y_range = y_max - y_min or 1.0

    # Alternate label height to avoid overlap
    label_y_offsets = [0.92, 0.78, 0.64, 0.50]

    for i, ann in enumerate(annotations):
        c = ann_colors[i % len(ann_colors)]
        ax.axvline(ann["time"], color=c, linestyle=":", linewidth=1.5, alpha=0.9)

        # Text label inside the plot
        label_y = y_min + label_y_offsets[i % len(label_y_offsets)] * y_range
        t_str = ann["time"].strftime("%H:%M:%S")
        ax.text(
            ann["time"], label_y,
            f" {t_str}\n {ann['annotation']}",
            color=c, fontsize=7, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.75),
            clip_on=True,
        )

    ax.set_ylabel(label, fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=_EASTERN))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))

    if xlim:
        ax.set_xlim(xlim)


def plot_combined(timeseries: dict, annotations: list, output_dir: str,
                  xlim: tuple = None, pid: str = "EEG") -> None:
    """4-panel combined figure."""
    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(18, 4 * len(metrics)),
                             sharex=True)
    fig.suptitle(f"EEG Metrics – {pid}", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        times, values = timeseries[metric]
        _draw_metric(ax, times, values, METRIC_LABELS[metric],
                     METRIC_COLORS[metric], annotations, xlim)

    axes[-1].set_xlabel("Time (Eastern)", fontsize=10)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{pid}_eeg_metrics.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined    → {out_path}")


def plot_individual(timeseries: dict, annotations: list, output_dir: str,
                    xlim: tuple = None, pid: str = "EEG") -> None:
    """One figure per metric."""
    os.makedirs(output_dir, exist_ok=True)
    for metric in METRIC_LABELS:
        times, values = timeseries[metric]
        fig, ax = plt.subplots(figsize=(18, 4))
        fig.suptitle(f"{METRIC_LABELS[metric]} – {pid}", fontsize=12,
                     fontweight="bold")
        _draw_metric(ax, times, values, METRIC_LABELS[metric],
                     METRIC_COLORS[metric], annotations, xlim)
        ax.set_xlabel("Time (Eastern)", fontsize=10)
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()

        out_path = os.path.join(output_dir, f"{pid}_{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved individual  → {out_path}")


def parse_xlim(xlim_str: str, ref_date: date) -> tuple:
    parts = xlim_str.strip().split()
    result = []
    for p in parts:
        dt = datetime.strptime(p, "%H:%M").replace(
            year=ref_date.year, month=ref_date.month, day=ref_date.day,
            tzinfo=_EASTERN
        )
        result.append(dt)
    return tuple(result)


def main():
    parser = argparse.ArgumentParser(description="Plot EEG metric time-series with annotations.")
    parser.add_argument("--timeseries",  required=True, help="Path to *_timeseries.json")
    parser.add_argument("--annotations", default=None, help="Path to ann.json (optional)")
    parser.add_argument("--output_dir",  required=True, help="Directory to save the plot")
    parser.add_argument("--xlim", default=None,
                        help="Optional x-axis range as 'HH:MM HH:MM' (e.g. '09:30 10:30')")
    args = parser.parse_args()

    timeseries = load_timeseries(args.timeseries)

    # Infer recording date from the first timestamp in the timeseries
    first_time = list(timeseries.values())[0][0][0]
    ref_date = first_time.astimezone(_EASTERN).date()

    annotations = load_annotations(args.annotations, ref_date) if args.annotations else []

    xlim = parse_xlim(args.xlim, ref_date) if args.xlim else None

    pid = os.path.splitext(os.path.basename(args.timeseries))[0].replace("_timeseries", "")

    plot_combined(timeseries, annotations, args.output_dir, xlim=xlim, pid=pid)
    plot_individual(timeseries, annotations, args.output_dir, xlim=xlim, pid=pid)


if __name__ == "__main__":
    main()
