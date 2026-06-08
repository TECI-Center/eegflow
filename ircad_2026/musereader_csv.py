"""
Reader and metric computer for Muse EEG data exported in the CSV format:
    Timestamp (microseconds), PacketType, Data (comma-separated channel values)

Usage:
    reader = MuseCSVReader("path/to/file.csv")
    reader.compute_metrics()
    reader.save("output_dir/")        # saves JSON time-series + aggregate CSV
"""

import os
import sys
import json
import importlib.util
import pathlib

# Ensure the parent eegflow directory is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from eeg_functions import (
    filter_raw_data,
    compute_focus_index,
    compute_engagement_index,
    compute_FAA_index,
    compute_TLX,
)

# Load config from the parent eegflow directory
_cfg_path = pathlib.Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("eegflow_config", _cfg_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
config = _mod.config


class MuseCSVReader:
    """
    Reads and processes a Muse EEG CSV export file.

    The CSV is expected to have three columns:
        Timestamp  – microsecond epoch integer
        PacketType – string, e.g. "EEG"
        Data       – comma-separated floats: ch1, ch2, ch3, ch4, [extras…]

    After construction, ``self.data`` holds a filtered DataFrame with columns:
        ts, datetime, ch1, ch2, ch3, ch4
    """

    def __init__(self, filepath: str, cfg: dict = config):
        self.filepath = filepath
        self.config = cfg
        self.data: pd.DataFrame = pd.DataFrame()
        self.metric_time_series: dict = {}
        self.aggregate_metrics: dict = {}

        self._read(filepath)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _read(self, filepath: str) -> None:
        """Parse the CSV and build the filtered EEG DataFrame."""
        df = pd.read_csv(filepath)

        eeg_df = df[df["PacketType"] == "EEG"].copy()
        if eeg_df.empty:
            raise ValueError(f"No EEG rows found in {filepath}")

        # Timestamp is in microseconds → convert to epoch seconds (float)
        eeg_df["ts"] = eeg_df["Timestamp"].astype(float) / 1e6

        # Split the Data column into individual channel columns
        data_cols = eeg_df["Data"].str.split(",", expand=True)
        for i, ch in enumerate(["ch1", "ch2", "ch3", "ch4"]):
            eeg_df[ch] = pd.to_numeric(data_cols[i], errors="coerce")

        # Human-readable datetime in Eastern Time
        eeg_df["datetime"] = (
            pd.to_datetime(eeg_df["ts"], unit="s", utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        eeg_data = eeg_df[["ts", "datetime", "ch1", "ch2", "ch3", "ch4"]].reset_index(
            drop=True
        )

        # Bandpass + notch filter
        self.data = pd.DataFrame(
            filter_raw_data(
                eeg_data,
                self.config["fs"],
                self.config["lowcut"],
                self.config["highcut"],
            )
        )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self) -> None:
        """
        Compute focus, engagement, FAA, and TLX time-series from the EEG data.
        Results are stored in ``self.metric_time_series`` as
        ``{metric_name: (time_array, value_array)}``.
        """
        fs = self.config["fs"]
        bb = self.config["beta_band"]
        ab = self.config["alpha_band"]
        tb = self.config["theta_band"]
        start = self.data["ts"]

        t_focus, focus = compute_focus_index(self.data, fs, bb, ab, start)
        t_eng, eng = compute_engagement_index(self.data, fs, bb, ab, tb, start)
        t_faa, faa = compute_FAA_index(self.data, fs, ab, start)
        t_tlx, tlx = compute_TLX(self.data, fs, ab, tb, start)

        self.metric_time_series = {
            "focus_index": (t_focus, focus),
            "engagement_index": (t_eng, eng),
            "FAA_index": (t_faa, faa),
            "TLX": (t_tlx, tlx),
        }

        # Aggregate (mean over entire session)
        self.aggregate_metrics = {
            metric: float(np.nanmean(vals))
            for metric, (_, vals) in self.metric_time_series.items()
        }

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """
        Save results to *output_dir*:
          - <pid>_timeseries.json  – per-metric time-series arrays
          - <pid>_aggregate.json   – scalar mean for each metric

        ``compute_metrics()`` must be called before saving.
        """
        if not self.metric_time_series:
            raise RuntimeError("Call compute_metrics() before save().")

        os.makedirs(output_dir, exist_ok=True)

        pid = os.path.splitext(os.path.basename(self.filepath))[0]

        # --- time-series JSON ---
        ts_payload = {}
        for metric, (times, vals) in self.metric_time_series.items():
            # datetime objects → ISO strings for JSON serialisation
            time_strs = [
                t.isoformat() if hasattr(t, "isoformat") else str(t) for t in times
            ]
            ts_payload[metric] = {
                "time": time_strs,
                "values": [float(v) for v in vals],
            }

        ts_path = os.path.join(output_dir, f"{pid}_timeseries.json")
        with open(ts_path, "w") as f:
            json.dump(ts_payload, f, indent=2)

        # --- aggregate JSON ---
        agg_path = os.path.join(output_dir, f"{pid}_aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(self.aggregate_metrics, f, indent=2)

        print(f"Saved time-series  → {ts_path}")
        print(f"Saved aggregate    → {agg_path}")
