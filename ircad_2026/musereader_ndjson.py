"""
Reader and metric computer for Muse EEG data exported in the NDJSON format
produced by the companion app (one JSON object per line).

Each relevant record has the form:
    {
      "kind":       "sample",
      "sensorId":   "muse",
      "stream":     "eeg:AF7" | "eeg:AF8" | "eeg:TP9" | "eeg:TP10",
      "hostTime":   <milliseconds since epoch, float>,
      "deviceTime": <device-relative int>,
      "values":     [<12 floats – one packet at fs=256 Hz>],
      "phase":      "baseline" | "task"
    }

Channel mapping (consistent with eeg_functions.py):
    ch1 = TP9   (left temporal)
    ch2 = AF7   (left frontal  – used for FAA left)
    ch3 = AF8   (right frontal – used for FAA right)
    ch4 = TP10  (right temporal)

Usage:
    reader = MuseNDJSONReader("path/to/file.ndjson")
    reader.compute_metrics()                    # whole session
    reader.compute_metrics(phase="task")        # only "task" phase
    reader.save("output_dir/")
"""

import json
import os
import sys
import importlib.util
import pathlib

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

_cfg_path = pathlib.Path(__file__).parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("eegflow_config", _cfg_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
config = _mod.config

# Muse stream name → DataFrame column name
_STREAM_TO_COL = {
    "eeg:TP9":  "ch1",
    "eeg:AF7":  "ch2",
    "eeg:AF8":  "ch3",
    "eeg:TP10": "ch4",
}
_EEG_STREAMS = set(_STREAM_TO_COL.keys())


class MuseNDJSONReader:
    """
    Reads and processes a Muse EEG NDJSON file.

    After construction, ``self.data`` is a filtered DataFrame with columns:
        ts, datetime, phase, ch1, ch2, ch3, ch4

    The ``phase`` column contains the original "baseline"/"task" labels.
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
        """Parse the NDJSON and build the filtered 4-channel EEG DataFrame."""
        fs = self.config["fs"]         # samples per second (256)
        samples_per_packet = None      # inferred from first record

        # Accumulate per-channel lists of (timestamp_s, value) pairs
        channel_data: dict[str, list[tuple[float, float]]] = {
            col: [] for col in _STREAM_TO_COL.values()
        }
        # Keep phase aligned with AF7 (reference channel)
        ref_stream = "eeg:AF7"
        ref_col    = _STREAM_TO_COL[ref_stream]
        phase_map: list[tuple[float, str]] = []  # (ts_s, phase)

        with open(filepath, "r") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                stream = rec.get("stream", "")
                if stream not in _EEG_STREAMS:
                    continue

                values = rec.get("values", [])
                if not values:
                    continue

                if samples_per_packet is None:
                    samples_per_packet = len(values)

                # hostTime is milliseconds → epoch seconds
                t0_s = rec["hostTime"] / 1000.0
                phase = rec.get("phase", "")
                col   = _STREAM_TO_COL[stream]

                for k, v in enumerate(values):
                    t_s = t0_s + k / fs
                    channel_data[col].append((t_s, v))
                    if stream == ref_stream:
                        phase_map.append((t_s, phase))

        if not channel_data[ref_col]:
            raise ValueError(f"No EEG records found in {filepath}")

        # Sort each channel by timestamp
        for col in channel_data:
            channel_data[col].sort(key=lambda x: x[0])

        # Build reference time axis from AF7 (ch2)
        ref_pairs = channel_data[ref_col]
        ts_arr   = np.array([p[0] for p in ref_pairs])

        # Assemble channel arrays, then trim all to the shortest one
        ch_arrays = {}
        for col in ("ch1", "ch2", "ch3", "ch4"):
            pairs = channel_data[col]
            pairs.sort(key=lambda x: x[0])
            ch_arrays[col] = np.array([p[1] for p in pairs])

        n = min(len(ts_arr), *(len(v) for v in ch_arrays.values()))
        ts_arr = ts_arr[:n]
        for col in ch_arrays:
            ch_arrays[col] = ch_arrays[col][:n]

        # Align phase labels to reference timestamps
        phase_map.sort(key=lambda x: x[0])
        phase_arr = np.array([p[1] for p in phase_map])[:n]

        # Human-readable datetime in Eastern Time
        dt_arr = (
            pd.to_datetime(ts_arr, unit="s", utc=True)
            .tz_convert("America/New_York")
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        raw_df = pd.DataFrame({
            "ts":       ts_arr,
            "datetime": dt_arr,
            "phase":    phase_arr,
            "ch1":      ch_arrays["ch1"],
            "ch2":      ch_arrays["ch2"],
            "ch3":      ch_arrays["ch3"],
            "ch4":      ch_arrays["ch4"],
        })

        # Bandpass + notch filter (operates on ch1-ch4)
        filtered = filter_raw_data(
            raw_df,
            self.config["fs"],
            self.config["lowcut"],
            self.config["highcut"],
        )
        self.data = pd.DataFrame(filtered)
        # Re-attach phase (filter_raw_data may not preserve it)
        if "phase" not in self.data.columns:
            self.data["phase"] = phase_arr[:len(self.data)]

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_metrics(self, phase: str | None = None) -> None:
        """
        Compute focus, engagement, FAA, and TLX time-series.

        Parameters
        ----------
        phase : str or None
            If provided (e.g. ``"task"`` or ``"baseline"``), restrict
            computation to rows where ``data["phase"] == phase``.
            If None, use all rows.
        """
        if phase is not None:
            eeg = self.data[self.data["phase"] == phase].copy()
            if eeg.empty:
                raise ValueError(
                    f"No data for phase={phase!r}. "
                    f"Available: {self.data['phase'].unique().tolist()}"
                )
        else:
            eeg = self.data

        fs = self.config["fs"]
        bb = self.config["beta_band"]
        ab = self.config["alpha_band"]
        tb = self.config["theta_band"]
        start = eeg["ts"]

        t_focus, focus = compute_focus_index(eeg, fs, bb, ab, start)
        t_eng,   eng   = compute_engagement_index(eeg, fs, bb, ab, tb, start)
        t_faa,   faa   = compute_FAA_index(eeg, fs, ab, start)
        t_tlx,   tlx   = compute_TLX(eeg, fs, ab, tb, start)

        self.metric_time_series = {
            "focus_index":       (t_focus, focus),
            "engagement_index":  (t_eng,   eng),
            "FAA_index":         (t_faa,   faa),
            "TLX":               (t_tlx,   tlx),
        }
        self.aggregate_metrics = {
            metric: float(np.nanmean(vals))
            for metric, (_, vals) in self.metric_time_series.items()
        }
        self._computed_phase = phase  # remember for save() naming

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """
        Save results to *output_dir* — one file per metric:
          - <stem>[_<phase>]_<metric>_timeseries.json
          - <stem>[_<phase>]_<metric>_aggregate.json
        """
        if not self.metric_time_series:
            raise RuntimeError("Call compute_metrics() before save().")

        os.makedirs(output_dir, exist_ok=True)

        stem  = os.path.splitext(os.path.basename(self.filepath))[0]
        phase = getattr(self, "_computed_phase", None)
        tag   = f"_{phase}" if phase else ""

        for metric, (times, vals) in self.metric_time_series.items():
            time_strs = [
                t.isoformat() if hasattr(t, "isoformat") else str(t)
                for t in times
            ]

            # per-metric time-series
            ts_path = os.path.join(output_dir, f"{stem}{tag}_{metric}_timeseries.json")
            with open(ts_path, "w") as f:
                json.dump({"time": time_strs, "values": [float(v) for v in vals]}, f, indent=2)

            # per-metric aggregate (scalar)
            agg_path = os.path.join(output_dir, f"{stem}{tag}_{metric}_aggregate.json")
            with open(agg_path, "w") as f:
                json.dump({"metric": metric, "mean": float(np.nanmean(vals))}, f, indent=2)

            print(f"  {metric:20s}  ts → {os.path.basename(ts_path)}")
            print(f"  {metric:20s} agg → {os.path.basename(agg_path)}")
