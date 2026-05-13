import json
from scipy import stats
from pathlib import Path



ROOT = Path(__file__).parent
# Load P209 annotations
with open('/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json') as f:
    all_annotations = json.load(f)

p209_annotations = None
for entry in all_annotations:
    if entry.get('pid') == 'P209':
        p209_annotations = entry['annotations']
        break

if p209_annotations and 'Fourth Arm Cutting' in p209_annotations:
    indices = p209_annotations['Fourth Arm Cutting']
    p209_duration = indices[1] - indices[0]
    print(f"P209 Fourth Arm Cutting duration: {p209_duration} seconds")
    
    # Load normalized scores to get all durations
    with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
        scores = json.load(f)
    
    # Collect all Fourth Arm Cutting durations
    all_durations = []
    for pid in scores:
        if 'Fourth Arm Cutting' in scores[pid]:
            all_durations.append(scores[pid]['Fourth Arm Cutting']['duration'])
    
    all_durations.sort()
    print(f"Total participants with Fourth Arm Cutting: {len(all_durations)}")
    print(f"Duration range: {min(all_durations)} - {max(all_durations)} seconds")
    
    # Calculate percentile for P209
    percentile = stats.percentileofscore(all_durations, p209_duration)
    print(f"P209 duration percentile: {percentile:.2f}")
    
    # Update scores
    scores['P209']['Fourth Arm Cutting'] = {
        'duration': p209_duration,
        'duration percentile': round(percentile, 2)
    }
    
    # Save updated scores
    with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json', 'w') as f:
        json.dump(scores, f, indent=4)
    
    print(f"\n✅ Updated P209 with Fourth Arm Cutting duration: {p209_duration} seconds (percentile: {percentile:.2f})")

else:
    print("P209 annotations not found")
