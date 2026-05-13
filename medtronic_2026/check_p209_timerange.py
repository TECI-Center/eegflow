#!/usr/bin/env python3
import json
from datetime import datetime

# Load corrected annotations
with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data_NORMALIZED.json") as f:
    data = json.load(f)

p209_entry = [item for item in data if item['pid'] == 'P209'][0]
annotations = p209_entry['annotations']

print("P209 Annotation Summary:")
print("="*60)
for phase, ts_pair in annotations.items():
    dt_start = datetime.utcfromtimestamp(ts_pair[0])
    dt_end = datetime.utcfromtimestamp(ts_pair[1])
    duration = ts_pair[1] - ts_pair[0]
    print(f"{phase:<30} {ts_pair[0]:<15} {ts_pair[1]:<15} {duration:>5}s")

# Now check what's in the metrics file (time series)
with open("metrics/eeg_metrics_results.json") as f:
    hugo = json.load(f)

p209_metrics = [item for item in hugo if item['sid'] == 'P209'][0]
ts_data = p209_metrics.get('time_series', {})
ts_times = ts_data.get('time', [])

print("\nEEG Time Series Data for P209:")
print("="*60)
if ts_times:
    first_time_str = ts_times[0]
    last_time_str = ts_times[-1]
    print(f"First timestamp: {first_time_str}")
    print(f"Last timestamp: {last_time_str}")
    print(f"Total size: {len(ts_times)} points")
    
    # Parse timestamps
    from datetime import datetime
    first_dt = datetime.fromisoformat(first_time_str)
    last_dt = datetime.fromisoformat(last_time_str)
    first_unix = int(first_dt.timestamp())
    last_unix = int(last_dt.timestamp())
    
    print(f"First (unix): {first_unix}")
    print(f"Last (unix): {last_unix}")
    
    # Check each phase's coverage
    print("\nPhase Coverage Analysis:")
    print("="*60)
    for phase, ts_pair in annotations.items():
        start_unix = ts_pair[0]
        end_unix = ts_pair[1]
        
        in_range = (start_unix >= (first_unix - 2)) and (end_unix <= (last_unix + 2))
        print(f"{phase:<30} In Range? {'✓ YES' if in_range else '✗ NO'}")
        if not in_range:
            print(f"  Phase: {start_unix}-{end_unix}")
            print(f"  Data:  {first_unix}-{last_unix}")
