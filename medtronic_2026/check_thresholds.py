import json
import numpy as np

# Check FLS scores
with open('scores/fls_metrics_scores.json') as f:
    fls_scores = json.load(f)
fls_values = [v['score'] for v in fls_scores.values() if 'score' in v and not np.isnan(v.get('score', float('nan')))]
print("FLS SCORES:")
print(f"  Count: {len(fls_values)}")
print(f"  Min: {np.min(fls_values):.2f}, Max: {np.max(fls_values):.2f}")
print(f"  Mean: {np.mean(fls_values):.2f}, Median: {np.median(fls_values):.2f}")
print(f"  Q1: {np.percentile(fls_values, 25):.2f}, Q3: {np.percentile(fls_values, 75):.2f}")

# Check FlexVR scores
with open('scores/flexvr_data_using_annotations_scores.json') as f:
    flexvr_scores = json.load(f)
flexvr_values = [v['score'] for v in flexvr_scores.values() if 'score' in v and not np.isnan(v.get('score', float('nan')))]
print("\nFLEXVR SCORES:")
print(f"  Count: {len(flexvr_values)}")
print(f"  Min: {np.min(flexvr_values):.2f}, Max: {np.max(flexvr_values):.2f}")
print(f"  Mean: {np.mean(flexvr_values):.2f}, Median: {np.median(flexvr_values):.2f}")
print(f"  Q1: {np.percentile(flexvr_values, 25):.2f}, Q3: {np.percentile(flexvr_values, 75):.2f}")

# Check HUGO scores
with open('scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
    hugo_scores = json.load(f)
hugo_values = [v['score'] for v in hugo_scores.values() if 'score' in v and not np.isnan(v.get('score', float('nan')))]
print("\nHUGO SCORES:")
print(f"  Count: {len(hugo_values)}")
print(f"  Min: {np.min(hugo_values):.2f}, Max: {np.max(hugo_values):.2f}")
print(f"  Mean: {np.mean(hugo_values):.2f}, Median: {np.median(hugo_values):.2f}")
print(f"  Q1: {np.percentile(hugo_values, 25):.2f}, Q3: {np.percentile(hugo_values, 75):.2f}")

print("\nRECOMMENDED THRESHOLDS (Bottom 25% vs Top 25%):")
print(f"  FLS: Low < {np.percentile(fls_values, 25):.2f}, High > {np.percentile(fls_values, 75):.2f}")
print(f"  FlexVR: Low < {np.percentile(flexvr_values, 25):.2f}, High > {np.percentile(flexvr_values, 75):.2f}")
print(f"  HUGO: Low < {np.percentile(hugo_values, 25):.2f}, High > {np.percentile(hugo_values, 75):.2f}")
