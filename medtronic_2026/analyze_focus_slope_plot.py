"""
Script to create a slope plot showing focus index changes between Hugo and FlexVR
Only includes participants who completed both platforms
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path



ROOT = Path(__file__).parent
def load_procedure_data(procedure_key):
    """Load metrics for a procedure"""
    paths = {
        'hugo': f'{ROOT}/metrics/eeg_metrics_results.json',
        'flexvr': f'{ROOT}/metrics/flexvr_metrics_results.json'
    }
    
    try:
        with open(paths[procedure_key], 'r') as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Metrics file not found for {procedure_key}")
        return None
    
    return metrics_data

def extract_metrics_all(metrics_data, pid_key):
    """Extract all 4 metrics for all participants (Fourth Arm Cutting task only)"""
    metrics_data_dict = {}
    
    for participant in metrics_data:
        pid = participant.get(pid_key)
        if pid is None:
            continue
        
        # Get metrics from aggregate metrics (Fourth Arm Cutting task)
        agg = participant.get('aggregate_metrics', {})
        task_metrics = agg.get('Fourth Arm Cutting', {})
        
        metrics_data_dict[pid] = {
            'focus_index': task_metrics.get('focus_index'),
            'engagement_index': task_metrics.get('engagement_index'),
            'FAA_index': task_metrics.get('FAA_index'),
            'TLX': task_metrics.get('TLX')
        }
    
    return metrics_data_dict

def extract_focus_indices(metrics_data, pid_key):
    """Extract focus index for all participants (Fourth Arm Cutting task only)"""
    focus_data = {}
    
    for participant in metrics_data:
        pid = participant.get(pid_key)
        if pid is None:
            continue
        
        # Get focus index from aggregate metrics (Fourth Arm Cutting task)
        agg = participant.get('aggregate_metrics', {})
        task_metrics = agg.get('Fourth Arm Cutting', {})
        focus_index = task_metrics.get('focus_index')
        
        if focus_index is not None and not np.isnan(focus_index):
            focus_data[pid] = focus_index
    
    return focus_data

def classify_participants(hugo_focus, flexvr_focus):
    """
    Classify participants into groups (only include those with data on BOTH platforms):
    Group 1: focus_index < 17 on BOTH FlexVR AND Hugo (Fourth Arm Cutting)
    Group 2: Has data on both platforms AND focus_index >= 17 on EITHER Hugo OR FlexVR
    Exclude: Participants missing focus_index on either platform
    """
    group1 = set()
    group2 = set()
    excluded = set()
    
    # Get all participants that have data in both platforms
    all_pids = set(hugo_focus.keys()) & set(flexvr_focus.keys())
    
    for pid in all_pids:
        hugo_focus_val = hugo_focus[pid]
        flexvr_focus_val = flexvr_focus[pid]
        
        if hugo_focus_val < 17 and flexvr_focus_val < 17:
            group1.add(pid)
        elif hugo_focus_val >= 17 or flexvr_focus_val >= 17:
            group2.add(pid)
    
    return group1, group2, excluded

def create_slope_plot(hugo_metrics_dict, flexvr_metrics_dict, metric_name, output_path, group1_pids=None, group2_pids=None):
    """Create a slope plot showing metric changes between platforms, colored by group"""
    
    if group1_pids is None:
        group1_pids = set()
    if group2_pids is None:
        group2_pids = set()
    
    # Get participants with both platforms
    common_pids = set(hugo_metrics_dict.keys()) & set(flexvr_metrics_dict.keys())
    common_pids = sorted(list(common_pids))
    
    print(f"\nCreating slope plot for {metric_name}...")
    
    # Filter to only participants with both metrics
    valid_pids = []
    flexvr_vals = []
    hugo_vals = []
    pid_groups = []  # Track which group each pid belongs to
    
    for pid in common_pids:
        flexvr_val = flexvr_metrics_dict[pid].get(metric_name)
        hugo_val = hugo_metrics_dict[pid].get(metric_name)
        
        if flexvr_val is not None and hugo_val is not None and not np.isnan(flexvr_val) and not np.isnan(hugo_val):
            valid_pids.append(pid)
            flexvr_vals.append(flexvr_val)
            hugo_vals.append(hugo_val)
            
            # Determine group for coloring
            if pid in group1_pids:
                pid_groups.append(1)
            elif pid in group2_pids:
                pid_groups.append(2)
            else:
                pid_groups.append(0)  # Excluded
    
    if len(valid_pids) == 0:
        print(f"  Warning: No valid data for {metric_name}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Set up x-axis
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['FlexVR', 'Hugo'], fontsize=12, fontweight='bold')
    
    # Get y-axis limits with minimal blank space
    all_vals = flexvr_vals + hugo_vals
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range  # Small padding
    y_max = y_max + 0.05 * y_range
    ax.set_ylim(y_min, y_max)
    
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    # Count increases and decreases
    increased = sum(1 for h, f in zip(hugo_vals, flexvr_vals) if h > f)
    decreased = sum(1 for h, f in zip(hugo_vals, flexvr_vals) if h < f)
    no_change = sum(1 for h, f in zip(hugo_vals, flexvr_vals) if h == f)
    
    title = f'{metric_name.replace("_", " ").title()} - FlexVR vs Hugo (Fourth Arm Cutting)\n'
    title += f'Increased: {increased} | Decreased: {decreased} | No change: {no_change}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Plot lines connecting points
    for flexvr_val, hugo_val, pid, group in zip(flexvr_vals, hugo_vals, valid_pids, pid_groups):
        # Color line based on increase/decrease
        if hugo_val > flexvr_val:
            line_color = '#2E75B6'  # Blue for increase
        elif hugo_val < flexvr_val:
            line_color = '#ED7D31'  # Orange for decrease
        else:
            line_color = 'gray'  # Gray for no change
        
        ax.plot([1, 2], [flexvr_val, hugo_val], color=line_color, alpha=0.3, linewidth=1.5, zorder=1)
    
    # Plot FlexVR points - separated by group
    for flexvr_val, pid, group in zip(flexvr_vals, valid_pids, pid_groups):
        if group == 1:
            color = '#D62728'  # Red for Group 1
        elif group == 2:
            color = '#2E75B6'  # Blue for Group 2
        else:
            color = 'gray'  # Gray for excluded
        
        ax.scatter(1, flexvr_val, s=100, color=color, 
                  edgecolor='black', linewidth=1, zorder=3)
    
    # Annotate FlexVR points
    for pid, val, group in zip(valid_pids, flexvr_vals, pid_groups):
        offset_x = -0.08
        ax.annotate(pid, (1 + offset_x, val), fontsize=8, ha='right', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot Hugo points - separated by group
    for hugo_val, pid, group in zip(hugo_vals, valid_pids, pid_groups):
        if group == 1:
            color = '#D62728'  # Red for Group 1
        elif group == 2:
            color = '#2E75B6'  # Blue for Group 2
        else:
            color = 'gray'  # Gray for excluded
        
        ax.scatter(2, hugo_val, s=100, color=color, 
                  edgecolor='black', linewidth=1, zorder=3)
    
    # Annotate Hugo points
    for pid, val, group in zip(valid_pids, hugo_vals, pid_groups):
        offset_x = 0.08
        ax.annotate(pid, (2 + offset_x, val), fontsize=8, ha='left', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62728', edgecolor='black', label=f'Group 1: Focus < 17 both (N={len([g for g in pid_groups if g == 1])})'),
        Patch(facecolor='#2E75B6', edgecolor='black', label=f'Group 2: Focus >= 17 both (N={len([g for g in pid_groups if g == 2])})')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add statistics
    flexvr_array = np.array(flexvr_vals)
    hugo_array = np.array(hugo_vals)
    
    stats_text = f'N = {len(valid_pids)}\n'
    stats_text += f'FlexVR: μ={np.mean(flexvr_array):.2f}, σ={np.std(flexvr_array, ddof=1):.2f}\n'
    stats_text += f'Hugo: μ={np.mean(hugo_array):.2f}, σ={np.std(hugo_array, ddof=1):.2f}'
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("METRIC SLOPE PLOTS - FLEXVR vs HUGO (GROUPED BY FOCUS INDEX)")
    print("="*70)
    
    # Load data
    print("\nLoading metrics data...")
    hugo_metrics = load_procedure_data('hugo')
    if hugo_metrics is None:
        return
    
    flexvr_metrics = load_procedure_data('flexvr')
    if flexvr_metrics is None:
        return
    
    # Extract focus indices for classification
    print("\nExtracting focus indices (Fourth Arm Cutting)...")
    hugo_focus = extract_focus_indices(hugo_metrics, 'sid')
    flexvr_focus = extract_focus_indices(flexvr_metrics, 'pid')
    
    print(f"  Hugo: {len(hugo_focus)} participants")
    print(f"  FlexVR: {len(flexvr_focus)} participants")
    
    # Classify participants
    print("\nClassifying participants...")
    group1, group2, excluded = classify_participants(hugo_focus, flexvr_focus)
    
    print(f"  Group 1 (Focus < 17 both): {len(group1)} participants")
    print(f"  Group 2 (Focus >= 17 both): {len(group2)} participants")
    print(f"  Excluded (mixed or incomplete): {len(excluded)} participants")
    
    # Extract all metrics
    print("\nExtracting metrics...")
    hugo_metrics_dict = extract_metrics_all(hugo_metrics, 'sid')
    flexvr_metrics_dict = extract_metrics_all(flexvr_metrics, 'pid')
    
    # Get common participants
    common_pids = set(hugo_metrics_dict.keys()) & set(flexvr_metrics_dict.keys())
    print(f"  Participants with both: {len(common_pids)}")
    
    if len(common_pids) == 0:
        print("  No participants with data on both platforms!")
        return
    
    # Create output directory
    results_dir = f'{ROOT}/results/focus_index_comparison'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create slope plots for all metrics
    print("\nGenerating slope plots...")
    metrics_to_plot = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']
    
    for metric in metrics_to_plot:
        plot_path = os.path.join(results_dir, f'{metric}_slope_plot.png')
        create_slope_plot(hugo_metrics_dict, flexvr_metrics_dict, metric, plot_path, group1, group2)
    
    # Save participant data
    participant_data = []
    for pid in sorted(common_pids):
        data_entry = {'participant_id': pid}
        
        # Add group classification
        if pid in group1:
            data_entry['group'] = 1
        elif pid in group2:
            data_entry['group'] = 2
        else:
            data_entry['group'] = 0  # Excluded
        
        for metric in metrics_to_plot:
            flexvr_val = flexvr_metrics_dict[pid].get(metric)
            hugo_val = hugo_metrics_dict[pid].get(metric)
            
            if flexvr_val is not None and hugo_val is not None:
                data_entry[f'flexvr_{metric}'] = float(flexvr_val)
                data_entry[f'hugo_{metric}'] = float(hugo_val)
                data_entry[f'{metric}_change'] = float(hugo_val - flexvr_val)
        
        participant_data.append(data_entry)
    
    data_file = os.path.join(results_dir, 'metrics_comparison_data.json')
    with open(data_file, 'w') as f:
        json.dump(participant_data, f, indent=2)
    print(f"  ✓ Saved: metrics_comparison_data.json")
    
    # Save group definitions
    group_info = {
        'group1_criteria': 'Focus index < 17 on BOTH Hugo AND FlexVR (Fourth Arm Cutting)',
        'group1_count': len(group1),
        'group1_pids': sorted(list(group1)),
        'group2_criteria': 'Has data on both platforms AND focus_index >= 17 on EITHER Hugo OR FlexVR (Fourth Arm Cutting)',
        'group2_count': len(group2),
        'group2_pids': sorted(list(group2)),
        'excluded_criteria': 'Missing data on either platform',
        'excluded_count': len(excluded),
        'excluded_pids': sorted(list(excluded))
    }
    
    groups_file = os.path.join(results_dir, 'focus_groups_definitions.json')
    with open(groups_file, 'w') as f:
        json.dump(group_info, f, indent=2)
    print(f"  ✓ Saved: focus_groups_definitions.json")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
