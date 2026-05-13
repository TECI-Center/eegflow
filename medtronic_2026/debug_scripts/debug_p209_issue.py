#!/usr/bin/env python3
import json

# Found the issue! Check the annotation indices
print("="*70)
print("P209 ANNOTATION INDICES - FOUND THE ISSUE!")
print("="*70)

with open("/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json") as f:
    hugo_ann_data = json.load(f)

p209_ann = [item for item in hugo_ann_data if item.get('pid') == 'P209']
if p209_ann:
    annotations = p209_ann[0]['annotations']
    print("\nPhase annotation indices (Unix timestamps):")
    print(f"{'Phase':<30} {'Start':<15} {'End':<15} {'Duration':<15} {'Valid?':<10}")
    print("-"*70)
    
    for phase_name, phase_idx in annotations.items():
        if isinstance(phase_idx, list) and len(phase_idx) == 2:
            start = phase_idx[0]
            end = phase_idx[1]
            duration = end - start
            is_valid = "✓" if end > start else "✗"
            
            print(f"{phase_name:<30} {start:<15} {end:<15} {duration:<15} {is_valid:<10}")
        else:
            print(f"{phase_name:<30} {phase_idx}")

print("\n" + "="*70)
print("PROBLEM IDENTIFIED!")
print("="*70)
print("\nFourth Arm Cutting has INVERTED indices:")
print("  Start:  1729546901")
print("  End:    1729546873")
print("  Duration: -28 seconds")
print("\n✗ Start > End means the time window is BACKWARDS!")
print("✗ This causes NaN metrics because there's no valid segment")
print("\nSOLUTION:")
print("  The annotation indices for 'Fourth Arm Cutting' need to be corrected.")
print("  They should be: [1729546873, 1729546901] (start before end)")
print("\nThis is an ANNOTATION ERROR in medtronic_hugo_data.json")
