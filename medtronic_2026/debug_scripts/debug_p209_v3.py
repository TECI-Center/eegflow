#!/usr/bin/env python3
import json
import numpy as np

# Check P209's Fourth Arm Cutting metrics in detail
print("="*60)
print("P209 FOURTH ARM CUTTING METRICS (HUGO)")
print("="*60)

with open('metrics/eeg_metrics_results.json') as f:
    hugo = json.load(f)
p209_hugo = [item for item in hugo if item['sid'] == 'P209'][0]

fac = p209_hugo.get('aggregate_metrics', {}).get('Fourth Arm Cutting', {})
print("\nMetric values for Fourth Arm Cutting:")
for metric in ['focus_index', 'engagement_index', 'FAA_index', 'TLX']:
    value = fac.get(metric)
    is_nan = value != value if isinstance(value, float) else (value is None)
    print(f"  {metric}: {value} (NaN: {is_nan})")

# Check all phases to see which have NaN values
print(f"\n{'Phase':<30} {'Focus':<15} {'Engagement':<15} {'FAA':<15} {'TLX':<15}")
print("-"*90)
for phase in ['Fourth Arm Cutting', 'Knot Tying', 'Ring Tower Transfer', 'Suturing (Railroad Track)', 'calibration', 'full']:
    phase_metrics = p209_hugo.get('aggregate_metrics', {}).get(phase, {})
    focus = phase_metrics.get('focus_index')
    eng = phase_metrics.get('engagement_index')
    faa = phase_metrics.get('FAA_index')
    tlx = phase_metrics.get('TLX')
    
    focus_nan = focus != focus if isinstance(focus, float) else (focus is None)
    eng_nan = eng != eng if isinstance(eng, float) else (eng is None)
    faa_nan = faa != faa if isinstance(faa, float) else (faa is None)
    tlx_nan = tlx != tlx if isinstance(tlx, float) else (tlx is None)
    
    print(f"{phase:<30} {str(focus)[:12]:<15} {str(eng)[:12]:<15} {str(faa)[:12]:<15} {str(tlx)[:12]:<15}")

# Check annotations to understand why Fourth Arm Cutting might have NaN
print("\n" + "="*60)
print("P209 ANNOTATIONS (HUGO)")
print("="*60)

with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json") as f:
    hugo_ann_data = json.load(f)

p209_ann = [item for item in hugo_ann_data if item.get('pid') == 'P209']
if p209_ann:
    ann_list = p209_ann[0]['annotations']
    print(f"\nType of annotations: {type(ann_list)}")
    print(f"Number of annotations: {len(ann_list)}")
    print("\nAnnotations:")
    for i, ann in enumerate(ann_list):
        print(f"  [{i}] {ann}")

print("\n" + "="*60)
print("WHY IS FOURTH ARM CUTTING NaN?")
print("="*60)
print("\nPossible reasons:")
print("1. No valid time series data for Fourth Arm Cutting phase")
print("2. Time series data is all NaN or invalid")
print("3. Annotations exist but time window is empty")
print("4. EEG data issues during Fourth Arm Cutting task")
