import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load all necessary data
with open(f'{ROOT}/results/group_comparisons/group_definitions.json') as f:
    groups = json.load(f)

with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
    hugo_data = json.load(f)

with open(f'{ROOT}/scores/fls_metrics_scores.json') as f:
    fls_data = json.load(f)

with open(f'{ROOT}/scores/flexvr_metrics_scores.json') as f:
    flexvr_data = json.load(f)

# Create output directory
output_dir = ROOT / 'results' / 'arm_cutting_analysis'
output_dir.mkdir(exist_ok=True)

# Determine study participation for each PID
def get_study_label(pid):
    """Determine if participant did FLS or FLEXVR before HUGO"""
    fls_completed = pid in fls_data
    flexvr_completed = pid in flexvr_data
    
    # If both completed, choose FLS as instructed
    if fls_completed and flexvr_completed:
        return 'FLS'
    elif fls_completed:
        return 'FLS'
    elif flexvr_completed:
        return 'FLEXVR'
    else:
        return 'None'

# Collect data for each group
group_data = {}
for group_name, pids in groups.items():
    times = []
    valid_pids = []
    study_labels = []
    
    for pid in pids:
        if pid in hugo_data and 'Fourth Arm Cutting' in hugo_data[pid]:
            duration = hugo_data[pid]['Fourth Arm Cutting']['duration']
            times.append(duration)
            valid_pids.append(pid)
            study_labels.append(get_study_label(pid))
    
    # Sort by time (ascending)
    sorted_indices = np.argsort(times)
    times = [times[i] for i in sorted_indices]
    valid_pids = [valid_pids[i] for i in sorted_indices]
    study_labels = [study_labels[i] for i in sorted_indices]
    
    group_data[group_name] = {
        'pids': valid_pids,
        'times': times,
        'labels': study_labels
    }
    
    print(f"\n{group_name}:")
    print(f"PIDs: {valid_pids}")
    for i, (pid, time, label) in enumerate(zip(valid_pids, times, study_labels)):
        print(f"  {pid}: {time} seconds ({label})")

# === PLOT 1: Three Subplots (one per group) ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Fourth Arm Cutting Task Completion Time by Group (HUGO)', fontsize=14, fontweight='bold')

# Find global min/max for consistent y-axis scaling
all_times = []
for data in group_data.values():
    all_times.extend(data['times'])
y_max = max(all_times) * 1.1

colors_by_study = {'FLS': '#1f77b4', 'FLEXVR': '#ff7f0e', 'None': '#7f7f7f'}

group_names = sorted(group_data.keys())
for idx, group_name in enumerate(group_names):
    ax = axes[idx]
    data = group_data[group_name]
    
    bar_colors = [colors_by_study[label] for label in data['labels']]
    
    bars = ax.bar(range(len(data['pids'])), data['times'], color=bar_colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Participant ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title(f'{group_name} (n={len(data["pids"])})', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(data['pids'])))
    ax.set_xticklabels(data['pids'], rotation=45, ha='right')
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

# Add legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors_by_study['FLS'], edgecolor='black', label='FLS'),
                   plt.Rectangle((0,0),1,1, facecolor=colors_by_study['FLEXVR'], edgecolor='black', label='FLEXVR'),
                   plt.Rectangle((0,0),1,1, facecolor=colors_by_study['None'], edgecolor='black', label='None')]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(output_dir / 'plot1_group_subplots.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: plot1_group_subplots.png")
plt.close()

# === PLOT 2: Combined Bar Plot with Groups Clustered ===
fig, ax = plt.subplots(figsize=(14, 6))

x_pos = 0
group_positions = {}
colors_by_group = {'Group1': '#1f77b4', 'Group2': '#ff7f0e', 'Group3': '#2ca02c'}
tick_positions = []
tick_labels = []

for group_name in group_names:
    data = group_data[group_name]
    group_color = colors_by_group[group_name]
    
    for pid, time, label in zip(data['pids'], data['times'], data['labels']):
        # Create a darker or lighter version based on study label for visual distinction
        if label == 'FLS':
            bar_color = group_color
            hatch = ''
        elif label == 'FLEXVR':
            bar_color = group_color
            hatch = '//'
        else:
            bar_color = group_color
            hatch = 'xx'
        
        ax.bar(x_pos, time, color=bar_color, edgecolor='black', linewidth=1.5, hatch=hatch)
        ax.text(x_pos, time + 5, f'{int(time)}', ha='center', va='bottom', fontsize=9)
        
        tick_positions.append(x_pos)
        tick_labels.append(f'{pid}\n({label[:3]})')
        
        x_pos += 1
    
    # Add vertical separator between groups
    if group_name != group_names[-1]:
        ax.axvline(x=x_pos - 0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        x_pos += 0.5

ax.set_xlabel('Participant ID (Study)', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Fourth Arm Cutting Task Completion Time - All Participants by Group', fontsize=13, fontweight='bold')
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.set_ylim(0, y_max)
ax.grid(axis='y', alpha=0.3)

# Create custom legend
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
legend_elements = [
    Patch(facecolor=colors_by_group['Group1'], edgecolor='black', label='Group1'),
    Patch(facecolor=colors_by_group['Group2'], edgecolor='black', label='Group2'),
    Patch(facecolor=colors_by_group['Group3'], edgecolor='black', label='Group3'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'plot2_combined_grouped_barplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot2_combined_grouped_barplot.png")
plt.close()

# === PLOT 3: Box Plot of Completion Times by Group ===
fig, ax = plt.subplots(figsize=(10, 6))

box_data = [group_data[group_name]['times'] for group_name in group_names]
positions = [1, 2, 3]
colors = [colors_by_group[name] for name in group_names]

bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                 medianprops=dict(color='red', linewidth=2),
                 boxprops=dict(linewidth=1.5, edgecolor='black'),
                 whiskerprops=dict(linewidth=1.5, color='black'),
                 capprops=dict(linewidth=1.5, color='black'),
                 flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Group', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Fourth Arm Cutting Task Completion Time Distribution by Group (HUGO)', fontsize=13, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(group_names)
ax.grid(axis='y', alpha=0.3)

# Add sample size
for i, group_name in enumerate(group_names):
    n = len(group_data[group_name]['times'])
    ax.text(positions[i], ax.get_ylim()[1] * 0.95, f'n={n}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'plot3_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Saved: plot3_boxplot.png")
plt.close()

print(f"\n✅ All visualizations saved to: {output_dir}")
