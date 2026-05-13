import json

# Load the data
with open('eeg_metrics_results.json', 'r') as f:
    eeg_data = json.load(f)

with open('medtronic_hugo_metrics_NORMALIZED_scores.json', 'r') as f:
    hugo_scores = json.load(f)

print(f"Total surgeons in eeg_metrics: {len(eeg_data)}")
print(f"Total surgeons in hugo_scores: {len(hugo_scores)}")

# Check how many surgeons have each phase
phases_count = {}
for entry in eeg_data:
    sid = entry['sid']
    agg_metrics = entry.get('aggregate_metrics', {})
    for phase in agg_metrics.keys():
        if phase not in phases_count:
            phases_count[phase] = 0
        phases_count[phase] += 1

print("\nSurgeons with data per phase:")
for phase, count in sorted(phases_count.items()):
    print(f"  {phase}: {count}")

# Check which surgeons are in each dataset
eeg_sids = set(entry['sid'] for entry in eeg_data)
hugo_sids = set(hugo_scores.keys())

print(f"\nSurgeons only in EEG: {eeg_sids - hugo_sids}")
print(f"Surgeons only in HUGO: {hugo_sids - eeg_sids}")
print(f"\nTotal overlap: {len(eeg_sids & hugo_sids)}")
