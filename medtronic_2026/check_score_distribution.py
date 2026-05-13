import json
import numpy as np
from pathlib import Path



ROOT = Path(__file__).parent
# Check Hugo scores
with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json', 'r') as f:
    hugo_scores = json.load(f)
hugo_vals = [v['score'] for v in hugo_scores.values() if 'score' in v and not np.isnan(v['score'])]
print("HUGO Scores:")
print(f"  Count: {len(hugo_vals)}, Min: {min(hugo_vals):.0f}, Max: {max(hugo_vals):.0f}")
print(f"  Mean: {np.mean(hugo_vals):.0f}, Median: {np.median(hugo_vals):.0f}")
print(f"  Q1: {np.percentile(hugo_vals, 25):.0f}, Q3: {np.percentile(hugo_vals, 75):.0f}")

# Check FLS scores
with open(f'{ROOT}/scores/fls_metrics_scores.json', 'r') as f:
    fls_scores = json.load(f)
fls_vals = [v['score'] for v in fls_scores.values() if 'score' in v and not np.isnan(v['score'])]
print("\nFLS Scores:")
print(f"  Count: {len(fls_vals)}, Min: {min(fls_vals):.0f}, Max: {max(fls_vals):.0f}")
print(f"  Mean: {np.mean(fls_vals):.0f}, Median: {np.median(fls_vals):.0f}")
print(f"  Q1: {np.percentile(fls_vals, 25):.0f}, Q3: {np.percentile(fls_vals, 75):.0f}")

# Check FlexVR scores
with open(f'{ROOT}/scores/flexvr_data_using_annotations_scores.json', 'r') as f:
    flexvr_scores = json.load(f)
flexvr_vals = [v['score'] for v in flexvr_scores.values() if 'score' in v and not np.isnan(v['score'])]
print("\nFlexVR Scores:")
print(f"  Count: {len(flexvr_vals)}, Min: {min(flexvr_vals):.0f}, Max: {max(flexvr_vals):.0f}")
print(f"  Mean: {np.mean(flexvr_vals):.0f}, Median: {np.median(flexvr_vals):.0f}")
print(f"  Q1: {np.percentile(flexvr_vals, 25):.0f}, Q3: {np.percentile(flexvr_vals, 75):.0f}")
