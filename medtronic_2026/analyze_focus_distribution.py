"""
Script to analyze focus index distribution
"""

import json
import numpy as np
from pathlib import Path



ROOT = Path(__file__).parent
# Load metrics
with open(f'{ROOT}/metrics/eeg_metrics_results.json', 'r') as f:
    hugo_metrics = json.load(f)

with open(f'{ROOT}/metrics/flexvr_metrics_results.json', 'r') as f:
    flexvr_metrics = json.load(f)

# Extract focus indices for Fourth Arm Cutting
hugo_focus = {}
for p in hugo_metrics:
    pid = p.get('sid')
    if pid:
        focus = p.get('aggregate_metrics', {}).get('Fourth Arm Cutting', {}).get('focus_index')
        if focus is not None and not np.isnan(focus):
            hugo_focus[pid] = focus

flexvr_focus = {}
for p in flexvr_metrics:
    pid = p.get('pid')
    if pid:
        focus = p.get('aggregate_metrics', {}).get('Fourth Arm Cutting', {}).get('focus_index')
        if focus is not None and not np.isnan(focus):
            flexvr_focus[pid] = focus

# Find common participants
common = set(hugo_focus.keys()) & set(flexvr_focus.keys())

# Analyze
print("FOCUS INDEX DISTRIBUTION (Fourth Arm Cutting)\n")
print(f"Total with both platforms: {len(common)}")
print(f"\nHugo focus statistics:")
hugo_vals = [hugo_focus[pid] for pid in common]
print(f"  Mean: {np.mean(hugo_vals):.2f}")
print(f"  Median: {np.median(hugo_vals):.2f}")
print(f"  Std: {np.std(hugo_vals):.2f}")
print(f"  Min: {min(hugo_vals):.2f}")
print(f"  Max: {max(hugo_vals):.2f}")
print(f"  Count >= 17: {sum(1 for v in hugo_vals if v >= 17)}")

print(f"\nFlexVR focus statistics:")
flexvr_vals = [flexvr_focus[pid] for pid in common]
print(f"  Mean: {np.mean(flexvr_vals):.2f}")
print(f"  Median: {np.median(flexvr_vals):.2f}")
print(f"  Std: {np.std(flexvr_vals):.2f}")
print(f"  Min: {min(flexvr_vals):.2f}")
print(f"  Max: {max(flexvr_vals):.2f}")
print(f"  Count >= 17: {sum(1 for v in flexvr_vals if v >= 17)}")

print(f"\nCross-platform analysis:")
both_high = sum(1 for pid in common if hugo_focus[pid] >= 17 and flexvr_focus[pid] >= 17)
both_low = sum(1 for pid in common if hugo_focus[pid] < 17 and flexvr_focus[pid] < 17)
mixed = sum(1 for pid in common if (hugo_focus[pid] >= 17) != (flexvr_focus[pid] >= 17))

print(f"  Both >= 17: {both_high}")
print(f"  Both < 17: {both_low}")
print(f"  Mixed (one high, one low): {mixed}")

# Show high focus participants
print(f"\nParticipants with focus >= 17 on either platform:")
for pid in sorted(common):
    hugo_v = hugo_focus[pid]
    flexvr_v = flexvr_focus[pid]
    if hugo_v >= 17 or flexvr_v >= 17:
        print(f"  {pid}: Hugo={hugo_v:.2f}, FlexVR={flexvr_v:.2f}")
