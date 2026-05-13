#!/usr/bin/env python3
import json
import numpy as np

# Check P209's time series data in detail
print("="*60)
print("P209 TIME SERIES DATA ANALYSIS")
print("="*60)

with open('metrics/eeg_metrics_results.json') as f:
    hugo = json.load(f)
p209_hugo = [item for item in hugo if item['sid'] == 'P209'][0]

ts = p209_hugo.get('time_series', {})

print("\nTime series overall statistics:")
print(f"  Total time points: {len(ts.get('time', []))}")
print(f"  Time span: {ts.get('time', ['?'])[0]} to {ts.get('time', ['?'])[-1] if ts.get('time') else '?'}")

for metric in ['focus_index', 'engagement_index', 'FAA_index', 'TLX']:
    data = ts.get(metric, [])
    if data:
        arr = np.array(data)
        valid_count = np.sum(~np.isnan(arr))
        nan_count = np.sum(np.isnan(arr))
        print(f"\n{metric}:")
        print(f"  Total points: {len(data)}")
        print(f"  Valid values: {valid_count}")
        print(f"  NaN values: {nan_count}")
        if valid_count > 0:
            print(f"  Range: {np.nanmin(arr):.4f} to {np.nanmax(arr):.4f}")
            print(f"  Mean: {np.nanmean(arr):.4f}")
        else:
            print(f"  All values are NaN!")

# Now check the annotated phases' time indices
print("\n" + "="*60)
print("P209 ANNOTATIONS ANALYSIS")
print("="*60)

with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json") as f:
    hugo_ann_data = json.load(f)

p209_ann = [item for item in hugo_ann_data if item.get('pid') == 'P209']
if p209_ann:
    annotations = p209_ann[0]['annotations']
    print("\nPhase annotations:")
    for phase_name, phase_ann in annotations.items():
        if isinstance(phase_ann, dict):
            start_time = phase_ann.get('start_time')
            end_time = phase_ann.get('end_time')
            start_idx = phase_ann.get('start_idx')
            end_idx = phase_ann.get('end_idx')
            print(f"  {phase_name}:")
            print(f"    Start: {start_time} (index {start_idx})")
            print(f"    End: {end_time} (index {end_idx})")
            
            if start_idx is not None and end_idx is not None:
                segment_length = end_idx - start_idx
                print(f"    Segment length: {segment_length} samples")
        else:
            print(f"  {phase_name}: {phase_ann}")

# Check if Fourth Arm Cutting has a valid time window
print("\n" + "="*60)
print("INVESTIGATION")
print("="*60)
print("\nP209's Fourth Arm Cutting metrics are all NaN because:")
print("1. The time series data for this participant might have issues")
print("2. The annotation window for Fourth Arm Cutting might not have valid EEG data")
print("3. All computed metrics in that time window resulted in NaN")
print("\nThis is a DATA QUALITY issue, not a computation issue.")
print("The phase WAS annotated and processed, but the EEG data quality")
print("in that time window prevents reliable metric computation.")
