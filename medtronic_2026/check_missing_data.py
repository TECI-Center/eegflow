#!/usr/bin/env python3
import json
import numpy as np

# Load data
with open('metrics/eeg_metrics_results.json') as f:
    hugo_metrics = json.load(f)
with open('scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
    hugo_scores = json.load(f)
with open('metrics/flexvr_metrics_results.json') as f:
    flexvr_metrics = json.load(f)
with open('scores/flexvr_data_using_annotations_scores.json') as f:
    flexvr_scores = json.load(f)

# Build dicts
hugo_dict = {}
for item in hugo_metrics:
    sid = item['sid']
    if 'aggregate_metrics' in item and 'Fourth Arm Cutting' in item['aggregate_metrics']:
        hugo_dict[sid] = item['aggregate_metrics']['Fourth Arm Cutting']

flexvr_dict = {}
for item in flexvr_metrics:
    pid = item['pid']
    if 'aggregate_metrics' in item and 'Fourth Arm Cutting' in item['aggregate_metrics']:
        flexvr_dict[pid] = item['aggregate_metrics']['Fourth Arm Cutting']

# Find common
common_pids = set(hugo_dict.keys()) & set(flexvr_dict.keys())
common_with_scores = common_pids & set(hugo_scores.keys()) & set(flexvr_scores.keys())

print(f"HUGO with Fourth Arm Cutting metrics: {len(hugo_dict)}")
print(f"FlexVR with Fourth Arm Cutting metrics: {len(flexvr_dict)}")
print(f"Common in both metrics: {len(common_pids)}")
print(f"Common with scores: {len(common_with_scores)}")

# Check for NaN values
filtered_out = []
for pid in common_with_scores:
    for metric in ['focus_index', 'engagement_index', 'FAA_index', 'TLX']:
        hugo_val = hugo_dict[pid].get(metric)
        flexvr_val = flexvr_dict[pid].get(metric)
        if hugo_val is None or flexvr_val is None:
            filtered_out.append((pid, metric, "Missing"))
            break
        if np.isnan(hugo_val) or np.isnan(flexvr_val):
            filtered_out.append((pid, metric, f"NaN (HUGO: {hugo_val}, FlexVR: {flexvr_val})"))
            break

print(f"\nFiltered out due to missing/NaN metrics: {len(filtered_out)}")
for pid, metric, reason in filtered_out:
    print(f"  {pid}: {metric} - {reason}")

print(f"\nFinal plotted points: {len(common_with_scores) - len(filtered_out)}")
