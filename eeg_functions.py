import os
from os import path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.signal import welch, butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from datetime import timedelta
import sys
from io import StringIO
import math
from zoneinfo import ZoneInfo

_EASTERN = ZoneInfo("America/New_York")


def read_eeg(fp):
    df = pd.read_csv(fp, 
                 sep=" ",          # split on any whitespace
                 header=None,         # no header in file
                 engine="python", 
                 names=range(7))
    df.columns = ["ts", "path", "dtype", "val1", "val2", "val3", "val4"]

    # Filter for only /eeg/ rows
    eeg_df = df[df["path"] == "/eeg/"].copy()
    
    # Rename channels
    eeg_df.rename(columns={
        "val1": "ch1",
        "val2": "ch2",
        "val3": "ch3",
        "val4": "ch4"
    }, inplace=True)
    
    # Convert timestamp to float epoch seconds
    eeg_df["ts"] = eeg_df["ts"].astype(float)
    
    # (Optional) Convert to datetime
    eeg_df['datetime'] = (
    pd.to_datetime(eeg_df['ts'], unit='s', utc=True)
      .dt.tz_convert('America/Los_Angeles')
      .dt.strftime('%Y-%m-%d %H:%M:%S')
)
    # eeg_df["datetime"] = pd.to_datetime(eeg_df["ts"], unit="s").dt.tz_localize('America/Los_Angeles')
    
    # Reorder columns for convenience
    eeg_data_cleaned = pd.DataFrame(eeg_df[["ts", "datetime", "ch1", "ch2", "ch3", "ch4"]])
    eeg_data_cleaned = pd.DataFrame(filter_raw_data(eeg_data_cleaned, config['fs'], config['lowcut'], config['highcut']))
    
    return eeg_data_cleaned


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def notch_filter(data, fs, notch_freq=60, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data)

def filter_raw_data(eeg_data_cleaned, fs, lowcut, highcut):
    for channel in ['ch1', 'ch2', 'ch3', 'ch4']:
        if channel in eeg_data_cleaned.columns:
            data = eeg_data_cleaned[channel].interpolate().fillna(0)
            data_filtered = notch_filter(data, fs)
            data_filtered = bandpass_filter(data_filtered, lowcut, highcut, fs)
            eeg_data_cleaned[channel] = data_filtered
    return eeg_data_cleaned

def calculate_band_power(data, fs, band):
    f, Pxx = welch(data, fs=fs, nperseg=256)
    return np.trapz(Pxx[(f >= band[0]) & (f <= band[1])], f[(f >= band[0]) & (f <= band[1])])

def compute_power_over_time(data, fs, band, start, window_sec=2):
    window_samples = int(window_sec * fs)
    num_windows = len(data) // window_samples

    # FORCE scalar datetime (UTC-aware)
    start = pd.to_datetime(
        start.iloc[0] if hasattr(start, "__len__") else start,
        unit="s",
        utc=True
    ).to_pydatetime()

    time_series = []
    power_series = []

    for i in range(num_windows):
        segment = data[i * window_samples:(i + 1) * window_samples]
        power = calculate_band_power(segment, fs, band)
        power_series.append(power)

        center_sec = (i + 0.5) * float(window_sec)
        ts = (start + timedelta(seconds=center_sec)).astimezone(_EASTERN)

        time_series.append(ts)

    return np.array(time_series), np.array(power_series)



# INDICES ############################
# ALSO CONSIDERED AN ENGAGEMENT INDEX, BUT LESS OFTEN USED. 
def compute_focus_index(eeg_data, fs, beta_band, alpha_band, start_time, window_sec=2): ## CHANGE THIS FOR SAMPLING RATE
        focus_time_series = None
        focus_index_series = []

        for channel in ['ch1', 'ch2', 'ch3', 'ch4']:
            if channel not in eeg_data.columns or eeg_data[channel].isna().all():
                continue
            time_beta, beta_power = compute_power_over_time(eeg_data[channel].dropna(), fs, beta_band, start_time, window_sec)
            time_alpha, alpha_power = compute_power_over_time(eeg_data[channel].dropna(), fs, alpha_band, start_time, window_sec)
            if focus_time_series is None:
                focus_time_series = time_beta
                print("here", time_beta[0])
            else:
                min_len = min(len(focus_time_series), len(time_beta), len(time_alpha))
                focus_time_series = focus_time_series[:min_len]
                beta_power = beta_power[:min_len]
                alpha_power = alpha_power[:min_len]
            focus_index = beta_power / (alpha_power + 1e-6)
            focus_index_series.append(focus_index)

        focus_index_series = [series[:len(focus_time_series)] for series in focus_index_series]
        focus_index_avg = np.mean(focus_index_series, axis=0)
        # print(len(focus_time_series), len(eeg_data['ts']))
        return focus_time_series, focus_index_avg

# MAIN ENGAGEMENT INDEX
def compute_engagement_index(eeg_data, fs, beta_band, alpha_band, theta_band, start_time, window_sec=2): ## CHANGE THIS FOR SAMPLING RATE
        engagement_time_series = None
        engagement_index_series = []

        for channel in ['ch1', 'ch2', 'ch3', 'ch4']:
            if channel not in eeg_data.columns or eeg_data[channel].isna().all():
                continue
            time_beta, beta_power = compute_power_over_time(eeg_data[channel].dropna(), fs, beta_band, start_time, window_sec)
            time_alpha, alpha_power = compute_power_over_time(eeg_data[channel].dropna(), fs, alpha_band, start_time, window_sec)
            time_theta, theta_power = compute_power_over_time(eeg_data[channel].dropna(), fs, theta_band, start_time, window_sec)
            if engagement_time_series is None:
                engagement_time_series = time_beta
            else:
                min_len = min(len(engagement_time_series), len(time_beta), len(time_alpha), len(time_theta))
                engagement_time_series = engagement_time_series[:min_len]
                beta_power = beta_power[:min_len]
                alpha_power = alpha_power[:min_len]
                theta_power = theta_power[:min_len]

            engagement_index = beta_power / (alpha_power + theta_power)
            engagement_index_series.append(engagement_index)

        engagement_index_series = [series[:len(engagement_time_series)] for series in engagement_index_series]
        engagement_index_avg = np.mean(engagement_index_series, axis=0)
        return engagement_time_series, engagement_index_avg

def compute_TLX(eeg_data, fs, alpha_band, theta_band, start_time, window_sec=2): ## CHANGE THIS FOR SAMPLING RATE
        tlx_time_series = None
        tlx_index_series = []

        for channel in ['ch1', 'ch2', 'ch3', 'ch4']:
            if channel not in eeg_data.columns or eeg_data[channel].isna().all():
                continue
            time_alpha, alpha_power = compute_power_over_time(eeg_data[channel].dropna(), fs, alpha_band, start_time, window_sec)
            time_theta, theta_power = compute_power_over_time(eeg_data[channel].dropna(), fs, theta_band, start_time, window_sec)
            if tlx_time_series is None:
                tlx_time_series = time_alpha
            else:
                min_len = min(len(tlx_time_series), len(time_theta), len(time_alpha))
                tlx_time_series = tlx_time_series[:min_len]
                alpha_power = alpha_power[:min_len]
                theta_power = theta_power[:min_len]

            tlx = theta_power / (alpha_power + 1e-6)
            tlx_index_series.append(tlx)

        # Final alignment: trim all to the true shortest length
        min_len = min([len(tlx_time_series)] + [len(series) for series in tlx_index_series])
        tlx_time_series = tlx_time_series[:min_len]
        tlx_index_series = [series[:min_len] for series in tlx_index_series]

        tlx_avg = np.mean(tlx_index_series, axis=0) # this takes the average of all 4 channels 
        # print("TLX", len(tlx_index_series))
        return tlx_time_series, tlx_avg
# Frontal Alpha Asymmetry (FAA) Index - associated with emotional valence and approach/avoidance motivation
def compute_FAA_index(eeg_data, fs, alpha_band, start_time, window_sec=2): ## CHANGE THIS FOR SAMPLING RATE
        faa_time_series = None
        faa_index_series = []

        time_alpha_L, alpha_L_power = compute_power_over_time(eeg_data['ch2'].dropna(), fs, alpha_band, start_time, window_sec)
        time_alpha_R, alpha_R_power = compute_power_over_time(eeg_data['ch3'].dropna(), fs, alpha_band, start_time, window_sec)
        if faa_time_series is None:
            faa_time_series = time_alpha_L
        else:
            min_len = min(len(faa_time_series), len(time_alpha_L), len(time_alpha_R))
            faa_time_series = faa_time_series[:min_len]
            alpha_L_power = alpha_L_power[:min_len]
            alpha_R_power = alpha_R_power[:min_len]
        faa_index_series = []
        for i in range(len(alpha_L_power)): # LOG OF MEANS OR MEAN OF LOGS?
            idx = math.log(alpha_R_power[i]) - math.log(alpha_L_power[i])
            faa_index_series.append(idx)
            
        # Find the minimum length among all series (including time series)
        min_len = min(len(faa_time_series), len(faa_index_series))
        # Trim all to that length
        faa_time_series = faa_time_series[:min_len]
        faa_index_series = faa_index_series[:min_len]
        return faa_time_series, np.array(faa_index_series)

def plot_bandpower_function(title, xlabel, ylabel, eeg_data_cleaned, fs, bands, titles, colors, xlim, ylim, start_time, scale=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    print(xlim)
    ylim_range = 0
    # x_axis_time = 0
    for band, title, color in zip(bands, titles, colors):
        time, power = compute_power_over_time(eeg_data_cleaned['ch2'].dropna(), fs, band, start_time)
        # time = pd.to_datetime(time, unit='s')
        # time = [x+25200 for x in time]
        # time = [start_time+timedelta(seconds=x) for x in time]
        # print(time)
        ax.plot(time, power, label=f"{title} Power", color=color)
        if np.max(power) > ylim_range:
            ylim_range = np.max(power)
        # x_axis_time = time[0]
        
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    if scale == 'log':
        ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True)
    # ax.tight_layout()
    plt.show()
    return fig, ax


def plot_index_function(title, xlabel, ylabel, xdata, ydata, xlim, ylim, scale=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    # ylim = [0, 18]
    ax.set_title(title)
    ax.plot(xdata, ydata)
    ax.axhline(y=np.mean(ydata), color='blue', linestyle='--', linewidth=2,
                label=ylabel + f': {np.mean(ydata):.2f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if scale == 'log':
        ax.set_yscale('log')
    ax.grid(True)
    ax.legend(loc='upper left')
    fig.tight_layout()
    # plt.show()

    # Return figure and axis so caller can modify or add more plots
    return fig, ax

def FilterByAnnos(timestamps, values, startend):
            df = pd.DataFrame({
                "timestamp": timestamps,  # epoch seconds OR datetime strings 
                "value": values
            })
            start = startend[0]
            end   = startend[1]
            # Filter
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            filtered_values = df.loc[mask, "value"]
            filtered_timestamps = df.loc[mask, "timestamp"]
            return filtered_timestamps, filtered_values