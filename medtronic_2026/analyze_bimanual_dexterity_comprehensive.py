import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).parent
# Define the two groups (with corrected naming - Group 1 is HIGHEST dexterity, Group 2 is HIGH dexterity)
group_1_highest = ['P106', 'P218', 'P140', 'P219', 'P347', 'P180', 'P101']
group_2_high = ['P121', 'P146', 'P150', 'P188', 'P187', 'P111', 'P173', 'P203', 'P115', 'P202', 'P216', 'P224', 'P228', 'P239', 'P240', 'P227', 'P181', 'P114', 'P126']

# Load data sources
with open(f'{ROOT}/metrics/eeg_metrics_results.json') as f:
    all_metrics = json.load(f)

with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
    hugo_scores = json.load(f)

with open(f'{ROOT}/scores/fls_metrics_scores.json') as f:
    fls_data = json.load(f)

# Create metrics lookup
metrics_dict = {}
for entry in all_metrics:
    sid = entry.get('sid')
    if sid:
        metrics_dict[sid] = entry

# Create output directory
output_dir = ROOT / 'results/bimanual_dexterity_analysis'
output_dir.mkdir(exist_ok=True, parents=True)

print("\n" + "="*100)
print("BIMANUAL DEXTERITY ANALYSIS - RING TOWER TRANSFER & KNOT TYING")
print("="*100)

print(f"\nGroup 1 (Highest Min Angular Velocity): {group_1_highest}")
print(f"  Total: {len(group_1_highest)} participants")
print(f"\nGroup 2 (High Min Angular Velocity): {group_2_high}")
print(f"  Total: {len(group_2_high)} participants")

# Save groupings to JSON
groupings = {
    'Group 1 (Highest Min Angular Velocity)': {
        'description': 'Highest minimum angular velocity indicating excellent bimanual dexterity',
        'participants': group_1_highest,
        'count': len(group_1_highest)
    },
    'Group 2 (High Min Angular Velocity)': {
        'description': 'High minimum angular velocity indicating good bimanual dexterity',
        'participants': group_2_high,
        'count': len(group_2_high)
    }
}

with open(output_dir / 'bimanual_dexterity_groupings.json', 'w') as f:
    json.dump(groupings, f, indent=2)
print(f"\n✅ Saved groupings to: {output_dir / 'bimanual_dexterity_groupings.json'}")

# Get all participants from hugo scores, sorted by PID number
all_pids = sorted(list(hugo_scores.keys()), key=lambda x: int(x[1:]))

# Function to extract data and create plots
def create_analysis_plots(task_name, metric_type='focus_index', exclude_pids=None):
    """
    Create bar plots for a task showing either focus index (metric) or completion time
    metric_type: 'focus_index' or 'duration'
    exclude_pids: list of participant IDs to exclude from the plot
    """
    if exclude_pids is None:
        exclude_pids = []
    
    data_list = []
    
    for pid in all_pids:
        # Skip excluded PIDs
        if pid in exclude_pids:
            continue
            
        # Check if task data exists
        if pid in hugo_scores and task_name in hugo_scores[pid]:
            data_dict = {'PID': pid}
            
            # Get the data based on metric type
            if metric_type == 'focus_index':
                # Get focus index from EEG metrics
                if pid in metrics_dict and 'aggregate_metrics' in metrics_dict[pid]:
                    if task_name in metrics_dict[pid]['aggregate_metrics']:
                        val = metrics_dict[pid]['aggregate_metrics'][task_name].get('focus_index')
                        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                            data_dict['value'] = val
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            elif metric_type == 'duration':
                # Get duration from HUGO scores
                val = hugo_scores[pid][task_name].get('duration')
                if val is not None:
                    data_dict['value'] = val
                else:
                    continue
            
            # Determine group
            if pid in group_1_highest:
                data_dict['group'] = 'Group 1'
                data_dict['color'] = '#2ca02c'  # Green
            elif pid in group_2_high:
                data_dict['group'] = 'Group 2'
                data_dict['color'] = '#000000'  # Black
            else:
                data_dict['group'] = 'Other'
                data_dict['color'] = '#cccccc'  # Light gray
            
            # Check FLS participation
            data_dict['has_fls'] = pid in fls_data
            
            data_list.append(data_dict)
    
    # Create DataFrame and sort by value (ascending)
    if data_list:
        df = pd.DataFrame(data_list)
        df = df.sort_values('value', ascending=True).reset_index(drop=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(18, 6))
        
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['value'], color=df['color'], edgecolor='black', linewidth=1.2, alpha=0.8)
        
        # Add FLS indicators (asterisks)
        for i, (bar, has_fls) in enumerate(zip(bars, df['has_fls'])):
            if has_fls:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        '*', ha='center', va='bottom', fontsize=14, fontweight='bold', color='red')
        
        ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
        
        if metric_type == 'focus_index':
            ax.set_ylabel('Focus Index Score', fontsize=12, fontweight='bold')
            title = f'Focus Index Scores - {task_name} (HUGO)\nSorted by Score (Ascending)'
        else:
            ax.set_ylabel('Completion Time (seconds)', fontsize=12, fontweight='bold')
            title = f'Task Completion Time - {task_name} (HUGO)\nSorted by Time (Ascending)'
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['PID'], rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', edgecolor='black', label='Group 1: Highest Min Ang. Velocity (n=7)'),
            Patch(facecolor='#000000', edgecolor='black', label='Group 2: High Min Ang. Velocity (n=20)'),
            Patch(facecolor='#cccccc', edgecolor='black', label='Other participants'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
        
        # Add FLS note
        ax.text(0.98, 0.02, '* = Participant completed FLS', transform=ax.transAxes, 
                fontsize=10, ha='right', va='bottom', style='italic')
        
        fig.subplots_adjust(bottom=0.2)
        
        # Save plot
        if metric_type == 'focus_index':
            filename = f'focus_index_{task_name.lower().replace(" ", "_")}.png'
        else:
            filename = f'completion_time_{task_name.lower().replace(" ", "_")}.png'
        
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {filename}")
        plt.close()
        
        return df
    else:
        print(f"⚠️  No data available for {task_name} with metric_type={metric_type}")
        return None

# Generate plots for focus index
print("\n" + "="*100)
print("GENERATING FOCUS INDEX PLOTS")
print("="*100)

ring_tower_focus = create_analysis_plots('Ring Tower Transfer', 'focus_index', exclude_pids=['P177'])
knot_tying_focus = create_analysis_plots('Knot Tying', 'focus_index', exclude_pids=['P177'])

# Generate plots for completion times
print("\n" + "="*100)
print("GENERATING COMPLETION TIME PLOTS")
print("="*100)

ring_tower_time = create_analysis_plots('Ring Tower Transfer', 'duration')
knot_tying_time = create_analysis_plots('Knot Tying', 'duration')

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)
print(f"\nAll outputs saved to: {output_dir}")
print(f"Files created:")
print(f"  - bimanual_dexterity_groupings.json")
print(f"  - focus_index_ring_tower_transfer.png")
print(f"  - focus_index_knot_tying.png")
print(f"  - completion_time_ring_tower_transfer.png")
print(f"  - completion_time_knot_tying.png")
print("\n")
