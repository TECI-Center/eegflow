#!/usr/bin/env python3
import json
from datetime import datetime

# Load the updated annotations
with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data_NORMALIZED.json") as f:
    data = json.load(f)

p209_entry = [item for item in data if item['pid'] == 'P209'][0]

# Get the indices
annotations = p209_entry['annotations']
fac_ts = annotations['Fourth Arm Cutting']

print(f"Fourth Arm Cutting timestamps: {fac_ts}")
print(f"Start: {datetime.utcfromtimestamp(fac_ts[0])}")
print(f"End: {datetime.utcfromtimestamp(fac_ts[1])}")
print(f"Duration: {fac_ts[1] - fac_ts[0]} seconds")

# Load the EEG data file to see timestamps
import os
eeg_file = "/Users/calvinperumalla/datasets/medtronic/eeg/ACS_2024_database/P209/eeg/P209.txt"
print(f"\nEEG file exists: {os.path.exists(eeg_file)}")

if os.path.exists(eeg_file):
    with open(eeg_file) as f:
        lines = f.readlines()
    print(f"Total lines in EEG file: {len(lines)}")
    # Check first and last timestamps
    if len(lines) > 1:
        first_ts = lines[1].split(',')[0] if ',' in lines[1] else "?"
        last_ts = lines[-1].split(',')[0] if ',' in lines[-1] else "?"
        print(f"First timestamp: {first_ts}")
        print(f"Last timestamp: {last_ts}")
        
        # Convert to unix timestamps
        from musereader import musereader
        muse = musereader(eeg_file)
        print(f"\nEEG data timespan from musereader:")
        print(f"  Data shape: {muse.data.shape if hasattr(muse, 'data') else 'N/A'}")
