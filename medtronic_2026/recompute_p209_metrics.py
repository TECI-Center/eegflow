#!/usr/bin/env python3
"""
Recompute metrics for P209 only to include corrected Fourth Arm Cutting annotations
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_all_metrics import process_surgeon
from config import config


ROOT = Path(__file__).parent
# Load annotations
print("Loading annotations...")
with open('/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json') as f:
    all_annotations = json.load(f)

annotations = {}
for entry in all_annotations:
    if 'sid' in entry:
        annotations[entry['sid']] = entry['annotations']
    elif 'pid' in entry:
        annotations[entry['pid']] = entry['annotations']

# Find P209 EEG file
surgeon_dir = Path('/Users/calvinperumalla/datasets/MEDTRONIC_ACS2024/eeg/P209')
p209_file = surgeon_dir / 'eeg' / 'P209.txt'

if not p209_file.exists():
    # Try alternative path
    for variant_dir in Path('/Users/calvinperumalla/datasets/medtronic').glob('*/P209'):
        p209_file = variant_dir / 'eeg' / 'P209.txt'
        if p209_file.exists():
            break

if not p209_file:
    print("❌ P209 EEG file not found")
    sys.exit(1)

print(f"Found P209 file: {p209_file}")

# Process P209
print("\nProcessing P209...")
try:
    result = process_surgeon('P209', p209_file, annotations['P209'])
    print(f"\n✅ Successfully processed P209")
    print(f"Phases: {result['phases']}")
    
    # Load current metrics and update with P209
    with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
        hugo_scores = json.load(f)
    
    # Get the new P209 data from results
    with open(f'{ROOT}/metrics/eeg_metrics_results.json') as f:
        all_metrics = json.load(f)
    
    p209_metrics = None
    for entry in all_metrics:
        if entry.get('sid') == 'P209':
            p209_metrics = entry
            break
    
    if p209_metrics and 'Fourth Arm Cutting' in p209_metrics.get('aggregate_metrics', {}):
        # Extract normalized scores for P209
        fourth_arm_data = p209_metrics['aggregate_metrics']['Fourth Arm Cutting']
        hugo_scores['P209']['Fourth Arm Cutting'] = {
            'duration': int(fourth_arm_data.get('duration_median', 0)),
            'duration percentile': round(fourth_arm_data.get('duration_percentile', 0), 2)
        }
        
        # Save updated metrics
        with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json', 'w') as f:
            json.dump(hugo_scores, f, indent=4)
        
        print(f"\n✅ Updated metrics with P209 Fourth Arm Cutting data:")
        print(f"   Duration: {hugo_scores['P209']['Fourth Arm Cutting']['duration']} seconds")
        print(f"   Percentile: {hugo_scores['P209']['Fourth Arm Cutting']['duration percentile']}")
    else:
        print("\n⚠️  P209 Fourth Arm Cutting metrics were not computed")
        
except Exception as e:
    print(f"❌ Error processing P209: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
