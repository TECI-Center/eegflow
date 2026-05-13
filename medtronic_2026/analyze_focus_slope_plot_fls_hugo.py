"""
Script to create slope plots showing metric changes between FLS and Hugo
Only includes participants who completed both platforms (Circle Cutting for FLS, Fourth Arm Cutting for Hugo)
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
        'fls': f'{ROOT}/metrics/fls_metrics_results.json'
    }
    
    try:
        with open(paths[procedure_key], 'r') as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Metrics file not found for {procedure_key}")
        return None
    
    return metrics_data

def extract_metrics_all(metrics_data_hugo, metrics_data_fls, hugo_pid_key, fls_pid_key):
    """
    Extract all 4 metrics for participants who have data in both platforms
    Uses Fourth Arm Cutting for Hugo and Circle Cutting for FLS
    """
    hugo_metrics_dict = {}
    fls_metrics_dict = {}
    
    # Extract Hugo metrics (Fourth Arm Cutting)
    for participant in metrics_data_hugo:
        pid = participant.get(hugo_pid_key)
        if pid is None:
            continue
        
        agg = participant.get('aggregate_metrics', {})
        task_metrics = agg.get('Fourth Arm Cutting', {})
        
        hugo_metrics_dict[pid] = {
            'focus_index': task_metrics.get('focus_index'),
            'engagement_index': task_metrics.get('engagement_index'),
            'FAA_index': task_metrics.get('FAA_index'),
            'TLX': task_metrics.get('TLX')
        }
    
    # Extract FLS metrics (Circle Cutting)
    for participant in metrics_data_fls:
        pid = participant.get(fls_pid_key)
        if pid is None:
            continue
        
        agg = participant.get('aggregate_metrics', {})
        task_metrics = agg.get('Circle Cutting', {})
        
        fls_metrics_dict[pid] = {
            'focus_index': task_metrics.get('focus_index'),
            'engagement_index': task_metrics.get('engagement_index'),
            'FAA_index': task_metrics.get('FAA_index'),
            'TLX': task_metrics.get('TLX')
        }
    
    return hugo_metrics_dict, fls_metrics_dict

def extract_focus_indices(metrics_data_hugo, metrics_data_fls, hugo_pid_key, fls_pid_key):
    """
    Extract focus index for participants with both platforms
    Uses Fourth Arm Cutting for Hugo and Circle Cutting for FLS
    """
    hugo_focus = {}
    fls_focus = {}
    
    # Extract Hugo focus index
    for participant in metrics_data_hugo:
        pid = participant.get(hugo_pid_key)
        if pid is None:
            continue
        
        agg = participant.get('aggregate_metrics', {})
        task_metrics = agg.get('Fourth Arm Cutting', {})
        focus_index = task_metrics.get('focus_index')
        
        if focus_index is not None and not np.isnan(focus_index):
            hugo_focus[pid] = focus_index
    
    # Extract FLS focus index
    for participant in metrics_data_fls:
        pid = participant.get(fls_pid_key)
        if pid is None:
            continue
        
        agg = participant.get('aggregate_metrics', {})
        task_metrics = agg.get('Circle Cutting', {})
        focus_index = task_metrics.get('focus_index')
        
        if focus_index is not None and not np.isnan(focus_index):
            fls_focus[pid] = focus_index
    
    return hugo_focus, fls_focus

def classify_participants_fls_hugo(fls_focus, hugo_focus):
    """
    Classify participants into groups (only include those with data on BOTH platforms):
    Group 1: focus_index < 17 on BOTH FLS AND Hugo
    Group 2: Has data on both platforms AND focus_index >= 17 on EITHER Hugo OR FLS
    Exclude: Participants missing focus_index on either platform
    """
    group1 = set()
    group2 = set()
    excluded = set()
    
    # Get all participants that have data in both platforms
    all_pids = set(hugo_focus.keys()) & set(fls_focus.keys())
    
    for pid in all_pids:
        hugo_focus_val = hugo_focus[pid]
        fls_focus_val = fls_focus[pid]
        
        if hugo_focus_val < 17 and fls_focus_val < 17:
            group1.add(pid)
        elif hugo_focus_val >= 17 or fls_focus_val >= 17:
            group2.add(pid)
    
    return group1, group2, excluded

def create_slope_plot(fls_metrics_dict, hugo_metrics_dict, metric_name, output_path, group1_pids=None, group2_pids=None):
    """Create a slope plot showing metric changes between FLS and Hugo, colored by group"""
    
    if group1_pids is None:
        group1_pids = set()
    if group2_pids is None:
        group2_pids = set()
    
    # Get participants with both platforms
    common_pids = set(fls_metrics_dict.keys()) & set(hugo_metrics_dict.keys())
    common_pids = sorted(list(common_pids))
    
    print(f"\nCreating slope plot for {metric_name}...")
    
    # Filter to only participants with both metrics
    valid_pids = []
    fls_vals = []
    hugo_vals = []
    pid_groups = []  # Track which group each pid belongs to
    
    for pid in common_pids:
        fls_val = fls_metrics_dict[pid].get(metric_name)
        hugo_val = hugo_metrics_dict[pid].get(metric_name)
        
        if fls_val is not None and hugo_val is not None and not np.isnan(fls_val) and not np.isnan(hugo_val):
            valid_pids.append(pid)
            fls_vals.append(fls_val)
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
    ax.set_xticklabels(['FLS', 'Hugo'], fontsize=12, fontweight='bold')
    
    # Get y-axis limits with minimal blank space
    all_vals = fls_vals + hugo_vals
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_range = y_max - y_min
    y_min = y_min - 0.05 * y_range  # Small padding
    y_max = y_max + 0.05 * y_range
    ax.set_ylim(y_min, y_max)
    
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    # Count increases and decreases
    increased = sum(1 for h, f in zip(hugo_vals, fls_vals) if h > f)
    decreased = sum(1 for h, f in zip(hugo_vals, fls_vals) if h < f)
    no_change = sum(1 for h, f in zip(hugo_vals, fls_vals) if h == f)
    
    title = f'{metric_name.replace("_", " ").title()} - FLS vs Hugo (Circle Cutting vs Fourth Arm Cutting)\n'
    title += f'Increased: {increased} | Decreased: {decreased} | No change: {no_change}'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Plot lines connecting points
    for fls_val, hugo_val, pid, group in zip(fls_vals, hugo_vals, valid_pids, pid_groups):
        # Color line based on increase/decrease
        if hugo_val > fls_val:
            line_color = '#2E75B6'  # Blue for increase
        elif hugo_val < fls_val:
            line_color = '#ED7D31'  # Orange for decrease
        else:
            line_color = 'gray'  # Gray for no change
        
        ax.plot([1, 2], [fls_val, hugo_val], color=line_color, alpha=0.3, linewidth=1.5, zorder=1)
    
    # Plot FLS points - separated by group
    for fls_val, pid, group in zip(fls_vals, valid_pids, pid_groups):
        if group == 1:
            color = '#D62728'  # Red for Group 1
        elif group == 2:
            color = '#2E75B6'  # Blue for Group 2
        else:
            color = 'gray'  # Gray for excluded
        
        ax.scatter(1, fls_val, s=100, color=color, 
                  edgecolor='black', linewidth=1, zorder=3)
    
    # Annotate FLS points
    for pid, val, group in zip(valid_pids, fls_vals, pid_groups):
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
        Patch(facecolor='#2E75B6', edgecolor='black', label=f'Group 2: Focus >= 17 either (N={len([g for g in pid_groups if g == 2])})')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add statistics
    fls_array = np.array(fls_vals)
    hugo_array = np.array(hugo_vals)
    
    stats_text = f'N = {len(valid_pids)}\n'
    stats_text += f'FLS: μ={np.mean(fls_array):.2f}, σ={np.std(fls_array, ddof=1):.2f}\n'
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
    print("METRIC SLOPE PLOTS - FLS vs HUGO (GROUPED BY FOCUS INDEX)")
    print("="*70)
    
    # Load data
    print("\nLoading metrics data...")
    hugo_metrics = load_procedure_data('hugo')
    fls_metrics = load_procedure_data('fls')
    
    if hugo_metrics is None or fls_metrics is None:
        return
    
    # Extract metrics for both platforms
    print("Extracting metrics for common participants...")
    hugo_metrics_dict, fls_metrics_dict = extract_metrics_all(
        hugo_metrics, fls_metrics, 'sid', 'pid'
    )
    
    hugo_focus, fls_focus = extract_focus_indices(
        hugo_metrics, fls_metrics, 'sid', 'pid'
    )
    
    # Classify participants
    print("Classifying participants by focus index...")
    group1, group2, excluded = classify_participants_fls_hugo(fls_focus, hugo_focus)
    
    total_participants = len(set(hugo_focus.keys()) & set(fls_focus.keys()))
    print(f"\nParticipants with data on both platforms: {total_participants}")
    print(f"  Group 1 (Focus < 17 both): {len(group1)}")
    print(f"  Group 2 (Focus >= 17 either): {len(group2)}")
    
    # Create output directory
    results_dir = f'{ROOT}/results/focus_index_comparison_fls_hugo'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate slope plots for each metric
    metrics = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']
    
    print(f"\nGenerating slope plots...")
    for metric in metrics:
        plot_path = os.path.join(results_dir, f'slope_plot_{metric}.png')
        create_slope_plot(fls_metrics_dict, hugo_metrics_dict, metric, plot_path, group1, group2)
    
    # Save group definitions
    group_info = {
        'total_participants': total_participants,
        'group1': {
            'definition': 'Focus Index < 17 on both platforms',
            'count': len(group1),
            'pids': sorted(list(group1))
        },
        'group2': {
            'definition': 'Focus Index >= 17 on either FLS or Hugo',
            'count': len(group2),
            'pids': sorted(list(group2))
        },
        'platforms': 'FLS (Circle Cutting) vs Hugo (Fourth Arm Cutting)'
    }
    
    import json
    group_file = os.path.join(results_dir, 'fls_hugo_focus_groups.json')
    with open(group_file, 'w') as f:
        json.dump(group_info, f, indent=2)
    print(f"  ✓ Group definitions saved: {group_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
