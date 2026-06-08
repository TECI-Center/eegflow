"""
Create a synchronized video: camera footage stacked above the EEG focus metric
with a moving time cursor.

Usage:
    # 3-second test clip (one or both videos):
    python create_eeg_video.py --test

    # Full video for a single file:
    python create_eeg_video.py --video lecturing.mp4

    # Full video for all entries in video_start_times.json:
    python create_eeg_video.py
"""

import argparse
import json
import os
import subprocess
import sys
import pathlib
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

_EASTERN = ZoneInfo("America/New_York")

# ---------- defaults (override via CLI) ----------
_HERE = pathlib.Path(__file__).parent   # ircad_2026/
DEFAULT_TIMESERIES   = str(_HERE / "results/F_01_faculty_MuseS-951F_2026-05-27-09-33-08_1_timeseries.json")
DEFAULT_VIDEO_TIMES  = "/Users/calvinperumalla/git/ircad_26/video_start_times.json"
DEFAULT_VIDEO_DIR    = "/Users/calvinperumalla/git/ircad_26/video_clips"
DEFAULT_ANN_PATH     = "/Users/calvinperumalla/git/ircad_26/annotations/ann.json"
DEFAULT_OUTPUT_DIR   = str(_HERE / "results/videos")
DEFAULT_PLOT_START   = "09:45:00"   # wider context window shown in EEG panel
DEFAULT_PLOT_STOP    = "10:10:00"
EEG_PANEL_HEIGHT_PX  = 220   # height of the EEG strip appended below the video


# ── helpers ──────────────────────────────────────────────────────────────────

def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    if window < 2 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="same")


def load_focus(timeseries_path: str):
    with open(timeseries_path) as f:
        raw = json.load(f)
    times  = [datetime.fromisoformat(t) for t in raw["focus_index"]["time"]]
    values = np.array(raw["focus_index"]["values"], dtype=float)
    return times, values


def load_video_times(json_path: str, ref_date_str: str) -> dict:
    """Return {filename: (start_dt, stop_dt)} with Eastern-aware datetimes."""
    ref = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    with open(json_path) as f:
        entries = json.load(f)
    result = {}
    for entry in entries:
        for name, window in entry.items():
            def _parse(s):
                return datetime.strptime(s.strip(), "%H:%M:%S").replace(
                    year=ref.year, month=ref.month, day=ref.day, tzinfo=_EASTERN)
            result[name] = (_parse(window["start"]), _parse(window["stop"]))
    return result


def load_annotations(json_path: str, ref_date_str: str) -> list:
    """Return list of {time: datetime, annotation: str}, skipping blank entries."""
    ref = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    with open(json_path) as f:
        raw = json.load(f)
    result = []
    for entry in raw:
        t_str = entry.get("time", "").strip()
        if not t_str:
            continue
        # Accept both H:MM:SS and HH:MM:SS
        fmt = "%H:%M:%S" if t_str.count(":") == 2 else "%H:%M"
        t = datetime.strptime(t_str, fmt).replace(
            year=ref.year, month=ref.month, day=ref.day, tzinfo=_EASTERN
        )
        result.append({"time": t, "annotation": entry.get("annotation", "")})
    return result


# ── EEG panel rendering ───────────────────────────────────────────────────────

def build_eeg_background(
    eeg_times, eeg_values,
    plot_start, plot_stop,
    vid_start, vid_stop,
    width_px: int, height_px: int,
    annotations: list = None,
    dpi: int = 100
):
    """
    Render the static EEG panel (no cursor) as an RGB numpy array.
    The x-axis spans plot_start → plot_stop for context.
    The active video window is shaded.  Returns the bg image plus the pixel
    x-coordinates of plot_start/plot_stop for cursor mapping.
    """
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # Slice EEG to the wider plot window
    mask = np.array([(t >= plot_start) & (t <= plot_stop) for t in eeg_times])
    t_win = [t for t, m in zip(eeg_times, mask) if m]
    v_win = eeg_values[mask]

    if len(t_win) >= 2:
        sm = _smooth(v_win)
        ax.fill_between(t_win, sm, alpha=0.20, color="#2196F3")
        ax.plot(t_win, sm, color="#64B5F6", linewidth=1.8, label="Focus Index (β/α)")
        mean_val = float(np.nanmean(v_win))
        ax.axhline(mean_val, color="#90CAF9", linestyle="--", linewidth=1.0,
                   alpha=0.8, label=f"mean = {mean_val:.2f}")
        ax.set_ylim(bottom=max(0, float(np.nanmin(sm)) * 0.8),
                    top=float(np.nanmax(sm)) * 1.15)

    # Shade the active video window
    ax.axvspan(vid_start, vid_stop, alpha=0.12, color="#FFD700", zorder=0)
    ax.axvline(vid_start, color="#FFD700", linewidth=1.0, alpha=0.6, linestyle="--")
    ax.axvline(vid_stop,  color="#FFD700", linewidth=1.0, alpha=0.6, linestyle="--")

    # Annotation lines + labels
    ann_palette = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
                   "#FF8B94", "#B8B8FF", "#FFA07A", "#98FB98"]
    label_y_fracs = [0.90, 0.72, 0.54, 0.36]   # alternate heights to avoid overlap
    if annotations:
        visible = [a for a in annotations
                   if plot_start <= a["time"] <= plot_stop]
        for idx, ann in enumerate(visible):
            c = ann_palette[idx % len(ann_palette)]
            ax.axvline(ann["time"], color=c, linewidth=1.3, alpha=0.9, linestyle=":")
            # Short label inside the panel
            if len(t_win) >= 2:
                y_bot = max(0, float(np.nanmin(_smooth(v_win))) * 0.8)
                y_top = float(np.nanmax(_smooth(v_win))) * 1.15
            else:
                y_bot, y_top = 0, 1
            y_pos = y_bot + label_y_fracs[idx % len(label_y_fracs)] * (y_top - y_bot)
            t_str = ann["time"].strftime("%H:%M:%S")
            ax.text(
                ann["time"], y_pos,
                f" {t_str}\n {ann['annotation']}",
                color=c, fontsize=6.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec=c, alpha=0.80),
                clip_on=True,
            )

    ax.set_xlim(plot_start, plot_stop)
    ax.set_ylabel("Focus Index", color="white", fontsize=8)
    ax.set_xlabel("Time (ET)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=_EASTERN))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=2))
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(fontsize=7, framealpha=0.4, facecolor="#161b22",
              labelcolor="white", loc="upper right")
    ax.grid(True, alpha=0.15, color="white")

    fig.tight_layout(pad=0.4)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Pixel x-coordinates for plot_start and plot_stop (used for cursor mapping)
    trans   = ax.transData
    x_left  = trans.transform((mdates.date2num(plot_start), 0))[0]
    x_right = trans.transform((mdates.date2num(plot_stop),  0))[0]

    bg = np.array(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)

    return bg, x_left, x_right


def _cursor_x(cursor_dt, plot_start, plot_stop, x_left, x_right) -> int:
    total = (plot_stop - plot_start).total_seconds()
    frac  = (cursor_dt - plot_start).total_seconds() / total
    return int(x_left + frac * (x_right - x_left))


def stamp_cursor(eeg_bg: np.ndarray, cursor_dt, plot_start, plot_stop,
                 x_left, x_right) -> np.ndarray:
    """Return a copy of eeg_bg with a bright red cursor at the current time."""
    frame = eeg_bg.copy()
    x = _cursor_x(cursor_dt, plot_start, plot_stop, x_left, x_right)
    x = max(0, min(frame.shape[1] - 1, x))
    frame[:, max(0, x - 1):x + 2, :] = [255, 70, 70]
    return frame


def mux_audio(video_no_audio: str, source_video: str, out_path: str,
             duration: float | None = None) -> None:
    """
    Use ffmpeg to combine the rendered (silent) video with the audio track
    from the original source video.  The result is written to out_path.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio,   # rendered composite (video only)
        "-i", source_video,     # original clip (audio source)
        "-map", "0:v:0",        # video from rendered file
        "-map", "1:a:0",        # audio from original
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",            # stop at the shorter stream (handles test clips)
    ]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd.append(out_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error:\n{result.stderr[-800:]}")
        raise RuntimeError("ffmpeg mux failed")
    print(f"  Audio muxed → {out_path}")


# ── main rendering loop ───────────────────────────────────────────────────────

def process_video(
    video_name: str,
    vid_start: datetime,
    vid_stop: datetime,
    eeg_times,
    eeg_values,
    video_dir: str,
    output_dir: str,
    plot_start: datetime = None,
    plot_stop: datetime = None,
    annotations: list = None,
    test_seconds: float | None = None,
    include_audio: bool = False,
):
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return

    fps         = cap.get(cv2.CAP_PROP_FPS)
    vid_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = vid_w
    out_h = vid_h + EEG_PANEL_HEIGHT_PX

    os.makedirs(output_dir, exist_ok=True)
    stem   = os.path.splitext(video_name)[0]
    suffix = f"_test{int(test_seconds)}s" if test_seconds else "_full"

    # If audio requested, render to a temp file then mux
    if include_audio:
        tmp_fd, render_path = tempfile.mkstemp(suffix="_noaudio.mp4", dir=output_dir)
        os.close(tmp_fd)
        out_path = os.path.join(output_dir, f"{stem}{suffix}_audio.mp4")
    else:
        render_path = os.path.join(output_dir, f"{stem}{suffix}.mp4")
        out_path = render_path

    writer = cv2.VideoWriter(
        render_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )

    # Use the wider plot window if provided, else fall back to video window
    p_start = plot_start or vid_start
    p_stop  = plot_stop  or vid_stop

    # Pre-render static EEG background once
    eeg_bg, x_left, x_right = build_eeg_background(
        eeg_times, eeg_values,
        p_start, p_stop,
        vid_start, vid_stop,
        out_w, EEG_PANEL_HEIGHT_PX,
        annotations=annotations,
    )

    n_frames = int(test_seconds * fps) if test_seconds else total_frames
    print(f"  Rendering {n_frames} frames ({n_frames / fps:.1f}s) → {render_path}")

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        cursor_dt = datetime.fromtimestamp(
            vid_start.timestamp() + i / fps, tz=_EASTERN
        )
        eeg_frame = stamp_cursor(eeg_bg, cursor_dt, p_start, p_stop,
                                 x_left, x_right)

        video_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        composite = np.vstack([video_rgb, eeg_frame])
        writer.write(cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        if (i + 1) % 150 == 0:
            print(f"    {i + 1}/{n_frames}")

    cap.release()
    writer.release()

    if include_audio:
        duration = test_seconds  # None = full length, ffmpeg -shortest handles it
        mux_audio(render_path, video_path, out_path, duration=duration)
        os.remove(render_path)
    else:
        print(f"  Saved → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Render synchronized EEG + video clips."
    )
    parser.add_argument("--timeseries",  default=DEFAULT_TIMESERIES,
                        help="Path to *_timeseries.json")
    parser.add_argument("--video_times", default=DEFAULT_VIDEO_TIMES,
                        help="Path to video_start_times.json")
    parser.add_argument("--video_dir",   default=DEFAULT_VIDEO_DIR,
                        help="Directory containing the source MP4 files")
    parser.add_argument("--output_dir",  default=DEFAULT_OUTPUT_DIR,
                        help="Where to write output MP4s")
    parser.add_argument("--video", default=None,
                        help="Process only this filename (e.g. lecturing.mp4)")
    parser.add_argument("--test", action="store_true",
                        help="Render a 3-second preview instead of the full clip")
    parser.add_argument("--test_seconds", type=float, default=3.0,
                        help="Duration of the test clip in seconds (default: 3)")
    parser.add_argument("--audio", action="store_true",
                        help="Include original audio track in the output (uses ffmpeg)")
    parser.add_argument("--annotations", default=DEFAULT_ANN_PATH,
                        help="Path to ann.json (set to empty string to disable)")
    parser.add_argument("--plot_start", default=DEFAULT_PLOT_START,
                        help="EEG panel x-axis start time HH:MM:SS (default: 09:45:00)")
    parser.add_argument("--plot_stop",  default=DEFAULT_PLOT_STOP,
                        help="EEG panel x-axis stop  time HH:MM:SS (default: 10:10:00)")
    args = parser.parse_args()

    eeg_times, eeg_values = load_focus(args.timeseries)
    ref_date_str = eeg_times[0].astimezone(_EASTERN).strftime("%Y-%m-%d")
    video_times  = load_video_times(args.video_times, ref_date_str)

    ref = datetime.strptime(ref_date_str, "%Y-%m-%d").date()
    def _parse_time(s):
        return datetime.strptime(s.strip(), "%H:%M:%S").replace(
            year=ref.year, month=ref.month, day=ref.day, tzinfo=_EASTERN)

    plot_start = _parse_time(args.plot_start)
    plot_stop  = _parse_time(args.plot_stop)

    annotations = []
    if args.annotations and os.path.isfile(args.annotations):
        annotations = load_annotations(args.annotations, ref_date_str)
        print(f"Loaded {len(annotations)} annotation(s) from {args.annotations}")

    test_sec = args.test_seconds if args.test else None

    videos = [args.video] if args.video else list(video_times.keys())

    for video_name in videos:
        if video_name not in video_times:
            print(f"No timing entry for '{video_name}' — skipping.")
            continue
        vid_start, vid_stop = video_times[video_name]
        print(f"\n{'='*60}")
        print(f"  {video_name}  {vid_start.strftime('%H:%M:%S')} → {vid_stop.strftime('%H:%M:%S')} ET"
              + (" [TEST]" if test_sec else ""))
        print(f"  EEG context: {plot_start.strftime('%H:%M:%S')} → {plot_stop.strftime('%H:%M:%S')} ET")
        print(f"{'='*60}")
        process_video(
            video_name, vid_start, vid_stop,
            eeg_times, eeg_values,
            args.video_dir, args.output_dir,
            plot_start=plot_start, plot_stop=plot_stop,
            annotations=annotations,
            test_seconds=test_sec,
            include_audio=args.audio,
        )


if __name__ == "__main__":
    main()
