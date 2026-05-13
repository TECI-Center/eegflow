#!/usr/bin/env python3
import json
import numpy as np
from datetime import datetime

# Load P209's time series data
with open("metrics/eeg_metrics_results.json") as f:
    hugo = json.load(f)

p209_metrics = [item for item in hugo if item['sid'] == 'P209'][0]
ts_data = p209_metrics.get('time_series', {})

# Get the corrected annotation times
with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data_NORMALIZED.json") as f:
    ann_data = json.load(f)

p209_ann = [item for item in ann_data if item['pid'] == 'P209'][0]
fac_times = p209_ann['annotations']['Fourth Arm Cutting']
fac_start = fac_times[0]
fac_end = fac_times[1]

print("P209 Fourth Arm Cutting Analysis:")
print("="*60)
print(f"Phase time window: {fac_start} to {fac_end}")
print(f"Duration: {fac_end - fac_start} seconds")

# Get time series
ts_times = ts_data.get('time', [])
ts_focus = ts_data.get('focus_index', [])
ts_engagement = ts_data.get('engagement_index', [])

print(f"\nTotal time series points: {len(ts_times)}")

if ts_times:
    first_time = datetime.fromisoformat(ts_times[0])
    last_time = datetime.fromisoformat(ts_times[-1])
    first_unix = int(first_time.timestamp())
    last_unix = int(last_time.timestamp())
    
    print(f"Time span: {first_unix} to {last_unix}")
    
    # Find indices of Fourth Arm Cutting phase
    fac_indices = []
    for i, ts_str in enumerate(ts_times):
        ts_dt = datetime.fromisoformat(ts_str)
        ts_unix = int(ts_dt.timestamp())
        if fac_start <= ts_unix <= fac_end:
            fac_indices.append(i)
    
    print(f"\nIndices in Fourth Arm Cutting window: {len(fac_indices)}")
    if fac_indices:
        print(f"  First index: {fac_indices[0]}")
        print(f"  Last index: {fac_indices[-1]}")
        
        # Check the focus_index values in that window
        fac_focus_values = [ts_focus[i] for i in fac_indices]
        fac_engagement_values = [ts_engagement[i] for i in fac_indices]
        
        focus_arr = np.array(fac_focus_values)
        engagement_arr = np.array(fac_engagement_values)
        
        print(f"\nFocus Index values in window:")
        print(f"  Count: {len(fac_focus_values)}")
        print(f"  Valid (non-NaN): {np.sum(~np.isnan(focus_arr))}")
        print(f"  NaN: {np.sum(np.isnan(focus_arr))}")
        if np.sum(~np.isnan(focus_arr)) > 0:
            print(f"  Mean (valid): {np.nanmean(focus_arr):.4f}")
        
        print(f"\nEngagement Index values in window:")
        print(f"  Count: {len(fac_engagement_values)}")
        print(f"  Valid (non-NaN): {np.sum(~np.isnan(engagement_arr))}")
        print(f"  NaN: {np.sum(np.isnan(engagement_arr))}")
        if np.sum(~np.isnan(engagement_arr)) > 0:
            print(f"  Mean (valid): {np.nanmean(engagement_arr):.4f}")
    else:
        print("✗ NO time points found in Fourth Arm Cutting window!")
        print(f"  Searched for unix timestamps between {fac_start} and {fac_end}")
        print(f"  Available data: {first_unix} to {last_unix}")
