#!/usr/bin/env python3
import json

# Check HUGO
try:
    with open('metrics/eeg_metrics_results.json') as f:
        hugo = json.load(f)
    p209_hugo = [item for item in hugo if item['sid'] == 'P209']
    if p209_hugo:
        print("P209 in HUGO metrics:")
        entry = p209_hugo[0]
        print(f"  Aggregate phases: {list(entry.get('aggregate_metrics', {}).keys())}")
        print(f"  Time series keys: {list(entry.get('time_series', {}).keys())}")
    else:
        print("P209 not found in HUGO metrics")
except Exception as e:
    print(f"Error checking HUGO: {e}")

# Check FLS
try:
    with open('metrics/fls_metrics_results.json') as f:
        fls = json.load(f)
    p209_fls = [item for item in fls if item['pid'] == 'P209']
    if p209_fls:
        print("\nP209 in FLS metrics:")
        entry = p209_fls[0]
        print(f"  Aggregate phases: {list(entry.get('aggregate_metrics', {}).keys())}")
    else:
        print("\nP209 not found in FLS metrics")
except Exception as e:
    print(f"Error checking FLS: {e}")

# Check FlexVR
try:
    with open('metrics/flexvr_metrics_results.json') as f:
        flexvr = json.load(f)
    p209_flexvr = [item for item in flexvr if item['pid'] == 'P209']
    if p209_flexvr:
        print("\nP209 in FlexVR metrics:")
        entry = p209_flexvr[0]
        print(f"  Aggregate phases: {list(entry.get('aggregate_metrics', {}).keys())}")
    else:
        print("\nP209 not found in FlexVR metrics")
except Exception as e:
    print(f"Error checking FlexVR: {e}")

# If P209 is found, check annotations
print("\n" + "="*60)
print("Checking annotations...")

# Check HUGO annotations
try:
    with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json") as f:
        hugo_ann = json.load(f)
    p209_ann = [item for item in hugo_ann if item.get('pid') == 'P209']
    if p209_ann:
        print("\nP209 in HUGO annotations:")
        ann = p209_ann[0]['annotations']
        print(f"  Phases annotated: {[a['task_name'] for a in ann]}")
    else:
        print("\nP209 not found in HUGO annotations")
except Exception as e:
    print(f"Error checking HUGO annotations: {e}")

# Check FLS annotations
try:
    with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/fls_data.json") as f:
        fls_ann = json.load(f)
    p209_ann = [item for item in fls_ann if item.get('pid') == 'P209']
    if p209_ann:
        print("\nP209 in FLS annotations:")
        ann = p209_ann[0]['annotations']
        print(f"  Phases annotated: {[a['task_name'] for a in ann]}")
    else:
        print("\nP209 not found in FLS annotations")
except Exception as e:
    print(f"Error checking FLS annotations: {e}")

# Check FlexVR annotations
try:
    with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/flexvr_data.json") as f:
        flexvr_ann = json.load(f)
    p209_ann = [item for item in flexvr_ann if item.get('pid') == 'P209']
    if p209_ann:
        print("\nP209 in FlexVR annotations:")
        ann = p209_ann[0]['annotations']
        print(f"  Phases annotated: {[a['task_name'] for a in ann]}")
    else:
        print("\nP209 not found in FlexVR annotations")
except Exception as e:
    print(f"Error checking FlexVR annotations: {e}")
