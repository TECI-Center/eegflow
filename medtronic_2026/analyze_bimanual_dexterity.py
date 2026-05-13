import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Define the two groups
group_highest = ['P106', 'P218', 'P140', 'P219', 'P347', 'P180', 'P101']
group_high = ['P121', 'P146', 'P150', 'P177', 'P188', 'P187', 'P111', 'P173', 'P203', 'P115', 'P202', 'P216', 'P224', 'P228', 'P239', 'P240', 'P227', 'P181', 'P114', 'P126']

# Load Hugo metrics
with open(f'{ROOT}/metrics/eeg_metrics_results.json') as f:
    all_metrics = json.load(f)

# Load normalized scores for fourth arm cutting focus index
with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
    hugo_scores = json.load(f)

# Load presurvey for FLS participation
with open('/Users/calvinperumalla/git/inert_pipe/presurvey_responses.json') as f:
    presurvey_data = json.load(f)

presurvey_dict = {entry['participant_id']: entry for entry in presurvey_data}

# Load FLS metrics to check participation
with open(f'{ROOT}/scores/fls_metrics_scores.json') as f:
    fls_data = json.load(f)

# Create metrics lookup
metrics_dict = {}
for entry in all_metrics:
    sid = entry.get('sid')
    if sid:
        metrics_dict[sid] = entry

# Collect metrics for both groups
def extract_metrics_for_group(group_pids, metrics_dict, phase='Fourth Arm Cutting'):
    """Extract metrics for a group of participants"""
    data = {}
    for pid in group_pids:
        if pid in metrics_dict:
            entry = metrics_dict[pid]
            if 'aggregate_metrics' in entry and phase in entry['aggregate_metrics']:
                phase_data = entry['aggregate_metrics'][phase]
                data[pid] = phase_data
    return data

# Extract Fourth Arm Cutting metrics
highest_metrics = extract_metrics_for_group(group_highest, metrics_dict, 'Fourth Arm Cutting')
high_metrics = extract_metrics_for_group(group_high, metrics_dict, 'Fourth Arm Cutting')

print("=" * 80)
print("BIMANUAL DEXTERITY GROUPS - EEG METRICS ANALYSIS")
print("=" * 80)
print(f"\nGroup 1 (Highest Min Angular Velocity): {len(group_highest)} participants")
print(f"  {group_highest}")
print(f"\nGroup 2 (High Min Angular Velocity): {len(group_high)} participants")
print(f"  {group_high}")

print(f"\nMetrics available:")
print(f"  Group 1: {len(highest_metrics)} participants with Fourth Arm Cutting data")
print(f"  Group 2: {len(high_metrics)} participants with Fourth Arm Cutting data")

# Extract individual metrics for comparison
metrics_keys = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']

# Collect data for each metric
results = []

for metric in metrics_keys:
    group1_values = []
    group2_values = []
    
    # Extract from Group 1 (Highest)
    for pid in group_highest:
        if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
            if 'Fourth Arm Cutting' in metrics_dict[pid]['aggregate_metrics']:
                data = metrics_dict[pid]['aggregate_metrics']['Fourth Arm Cutting']
                if metric in data and data[metric] is not None:
                    val = data[metric]
                    # Handle NaN values
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        group1_values.append(val)
    
    # Extract from Group 2 (High)
    for pid in group_high:
        if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
            if 'Fourth Arm Cutting' in metrics_dict[pid]['aggregate_metrics']:
                data = metrics_dict[pid]['aggregate_metrics']['Fourth Arm Cutting']
                if metric in data and data[metric] is not None:
                    val = data[metric]
                    # Handle NaN values
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        group2_values.append(val)
    
    if len(group1_values) > 0 and len(group2_values) > 0:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_values, group2_values)
        
        # Calculate Cohen's d
        n1, n2 = len(group1_values), len(group2_values)
        mean1, mean2 = np.mean(group1_values), np.mean(group2_values)
        std1, std2 = np.std(group1_values, ddof=1), np.std(group2_values, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'Metric': metric,
            'μ (H)': mean1,
            'N (H)': n1,
            'NaN% (H)': 0,
            'μ (L)': mean2,
            'N (L)': n2,
            'NaN% (L)': 0,
            't-stat': t_stat,
            'p-value': p_value,
            "Cohen's d": cohens_d
        })

# Create DataFrame
df_results = pd.DataFrame(results)

print("\n" + "=" * 120)
print("FOURTH ARM CUTTING - STATISTICAL COMPARISON")
print("=" * 120)
print(df_results.to_string(index=False))

# Save table to CSV
output_dir = ROOT / 'results/arm_cutting_analysis'
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / 'bimanual_dexterity_statistics.csv'
df_results.to_csv(csv_path, index=False)
print(f"\n✅ Saved statistics table to: {csv_path}")

# Now create the bar plot
print("\n" + "=" * 80)
print("GENERATING FOCUS INDEX BAR PLOT")
print("=" * 80)

# Get all participants from hugo scores
all_pids = sorted(list(hugo_scores.keys()), key=lambda x: int(x[1:]))

# Collect focus index data and group assignments
focus_indices = []
colors = []
labels_for_legend = []
fls_indicators = []

for pid in all_pids:
    if pid in hugo_scores and 'Fourth Arm Cutting' in hugo_scores[pid]:
        # Get focus index from metrics
        focus_idx = None
        if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
            if 'Fourth Arm Cutting' in metrics_dict[pid]['aggregate_metrics']:
                val = metrics_dict[pid]['aggregate_metrics']['Fourth Arm Cutting'].get('focus_index')
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    focus_idx = val
        
        if focus_idx is not None:
            focus_indices.append(focus_idx)
            
            # Determine color and group
            if pid in group_highest:
                colors.append('#2ca02c')  # Green for Group 1 (Highest)
            elif pid in group_high:
                colors.append('#000000')  # Black for Group 2 (High)
            else:
                colors.append('#cccccc')  # Light gray for others
            
            # Check if participated in FLS
            has_fls = pid in fls_data
            fls_indicators.append('*' if has_fls else '')

# Filter to only include participants with valid data
valid_pids_with_data = []
for i, pid in enumerate(all_pids):
    if len(valid_pids_with_data) < len(focus_indices):
        if pid in hugo_scores and 'Fourth Arm Cutting' in hugo_scores[pid]:
            val = None
            if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
                if 'Fourth Arm Cutting' in metrics_dict[pid]['aggregate_metrics']:
                    v = metrics_dict[pid]['aggregate_metrics']['Fourth Arm Cutting'].get('focus_index')
                    if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                        val = v
            if val is not None:
                valid_pids_with_data.append(pid)

# Sort by focus index (increasing order)
sorted_data = sorted(zip(focus_indices, colors, valid_pids_with_data, fls_indicators), key=lambda x: x[0])
focus_indices_sorted, colors_sorted, valid_pids_sorted, fls_indicators_sorted = zip(*sorted_data) if sorted_data else ([], [], [], [])
focus_indices_sorted = list(focus_indices_sorted)
colors_sorted = list(colors_sorted)
valid_pids_sorted = list(valid_pids_sorted)
fls_indicators_sorted = list(fls_indicators_sorted)

# Create bar plot
fig, ax = plt.subplots(figsize=(16, 6))

x_pos = np.arange(len(focus_indices_sorted))

bars = ax.bar(x_pos, focus_indices_sorted, color=colors_sorted, edgecolor='black', linewidth=1.2, alpha=0.8)

# Add FLS indicators
for i, (bar, indicator) in enumerate(zip(bars, fls_indicators_sorted)):
    if indicator:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                indicator, ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Focus Index Score', fontsize=12, fontweight='bold')
ax.set_title('Focus Index Scores for Fourth Arm Cutting (HUGO)\nGrouped by Bimanual Dexterity (Sorted by Focus Index)', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(valid_pids_sorted, rotation=45, ha='right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Create custom legend
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
legend_elements = [
    Patch(facecolor='#2ca02c', edgecolor='black', label='Group 1: Highest Min Angular Velocity (n=7)'),
    Patch(facecolor='#000000', edgecolor='black', label='Group 2: High Min Angular Velocity (n=20)'),
    Patch(facecolor='#cccccc', edgecolor='black', label='Not in groups'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

# Add note about FLS
ax.text(0.98, 0.02, '* = Participant completed FLS', transform=ax.transAxes, 
        fontsize=10, ha='right', va='bottom', style='italic')

fig.subplots_adjust(bottom=0.2)
plt.savefig(output_dir / 'focus_index_bimanual_dexterity_groups.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved focus index plot to: {output_dir / 'focus_index_bimanual_dexterity_groups.png'}")
plt.close()

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nFocus Index - Fourth Arm Cutting:")
print(f"  Group 1 (Highest): {len(group_highest)} total")
group1_focus = []
for pid in group_highest:
    if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
        if 'Fourth Arm Cutting' in metrics_dict[pid]['aggregate_metrics']:
            v = metrics_dict[pid]['aggregate_metrics']['Fourth Arm Cutting'].get('focus_index')
            if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                group1_focus.append(v)

if group1_focus:
    print(f"    With data: {len(group1_focus)}")
    print(f"    Mean: {np.mean(group1_focus):.3f}, SD: {np.std(group1_focus):.3f}")

print(f"  Group 2 (High): {len(group_high)} total")
group2_focus = []
for pid in group_high:
    if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
        if 'Fourth Arm Cutting' in metrics_dict[pid]['aggregate_metrics']:
            v = metrics_dict[pid]['aggregate_metrics']['Fourth Arm Cutting'].get('focus_index')
            if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                group2_focus.append(v)

if group2_focus:
    print(f"    With data: {len(group2_focus)}")
    print(f"    Mean: {np.mean(group2_focus):.3f}, SD: {np.std(group2_focus):.3f}")

print(f"\n✅ Analysis complete!")
