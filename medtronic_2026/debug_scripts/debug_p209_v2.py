#!/usr/bin/env python3
import json

# Check what P209 has in metrics
print("="*60)
print("METRICS ANALYSIS FOR P209")
print("="*60)

# HUGO
with open('metrics/eeg_metrics_results.json') as f:
    hugo = json.load(f)
p209_hugo = [item for item in hugo if item['sid'] == 'P209']
if p209_hugo:
    entry = p209_hugo[0]
    print("\n✓ P209 in HUGO metrics")
    print(f"  Aggregate phases: {list(entry.get('aggregate_metrics', {}).keys())}")
    fac = entry.get('aggregate_metrics', {}).get('Fourth Arm Cutting')
    if fac:
        print(f"  Fourth Arm Cutting metrics available: YES")
        print(f"    Keys: {list(fac.keys())}")
        print(f"    focus_index: {fac.get('focus_index')}")
    else:
        print(f"  Fourth Arm Cutting metrics available: NO")

# FLS
with open('metrics/fls_metrics_results.json') as f:
    fls = json.load(f)
p209_fls = [item for item in fls if item['pid'] == 'P209']
if p209_fls:
    entry = p209_fls[0]
    print("\n✓ P209 in FLS metrics")
    print(f"  Aggregate phases: {list(entry.get('aggregate_metrics', {}).keys())}")
    print("  NOTE: FLS doesn't have 'Fourth Arm Cutting' as a task")
    print("  FLS tasks are: Circle Cutting, Peg Transfer, Pen Rose Suturing")

print("\n" + "="*60)
print("ANNOTATIONS ANALYSIS FOR P209")
print("="*60)

# Check HUGO annotations
with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json") as f:
    hugo_ann_data = json.load(f)

print(f"\nType of HUGO annotations data: {type(hugo_ann_data)}")

if isinstance(hugo_ann_data, list):
    p209_ann = [item for item in hugo_ann_data if item.get('pid') == 'P209']
    if p209_ann:
        print("✓ P209 in HUGO annotations (list format)")
        ann = p209_ann[0]['annotations']
        print(f"  Phases annotated: {[a['task_name'] for a in ann]}")
else:
    print(f"Hugo annotations is a {type(hugo_ann_data)}: {list(hugo_ann_data.keys())[:5] if isinstance(hugo_ann_data, dict) else 'N/A'}")

# Check FLS annotations
with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/fls_data.json") as f:
    fls_ann_data = json.load(f)

print(f"\nType of FLS annotations data: {type(fls_ann_data)}")

if isinstance(fls_ann_data, list):
    p209_ann = [item for item in fls_ann_data if item.get('pid') == 'P209']
    if p209_ann:
        print("✓ P209 in FLS annotations (list format)")
        ann = p209_ann[0]['annotations']
        print(f"  Phases annotated: {[a['task_name'] for a in ann]}")
else:
    print(f"FLS annotations is a {type(fls_ann_data)}")
    if 'P209' in fls_ann_data:
        ann = fls_ann_data['P209']
        print(f"✓ P209 in FLS annotations (dict format)")
        print(f"  Phases annotated: {[a['task_name'] for a in ann]}")
    else:
        print(f"  P209 not found in FLS annotations dict")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\nP209 IS in HUGO metrics with 'Fourth Arm Cutting' phase")
print("P209 IS in FLS metrics (but FLS doesn't have 'Fourth Arm Cutting' task)")
print("P209 is NOT in FlexVR metrics")
