"""
Script to analyze participants by focus index levels across HUGO and FlexVR
Divides participants into two groups:
- Group 1: focus_index < 17 on BOTH FlexVR AND Hugo
- Group 2: Everyone else
"""

import json
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path



ROOT = Path(__file__).parent
def load_procedure_data(procedure_key):
    """Load metrics and scores for a procedure"""
    paths = {
        'hugo': {
            'metrics': f'{ROOT}/metrics/eeg_metrics_results.json',
            'scores': f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json'
        },
        'flexvr': {
            'metrics': f'{ROOT}/metrics/flexvr_metrics_results.json',
            'scores': f'{ROOT}/scores/flexvr_data_using_annotations_scores.json'
        }
    }
    
    try:
        with open(paths[procedure_key]['metrics'], 'r') as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"  Warning: Metrics file not found")
        return None, None
    
    try:
        with open(paths[procedure_key]['scores'], 'r') as f:
            scores_data = json.load(f)
    except FileNotFoundError:
        print(f"  Warning: Scores file not found")
        return None, None
    
    return metrics_data, scores_data

def extract_focus_indices(metrics_data, pid_key):
    """Extract focus index for all participants"""
    focus_data = {}
    
    for participant in metrics_data:
        pid = participant.get(pid_key)
        if pid is None:
            continue
        
        # Get focus index from aggregate metrics (full task)
        agg = participant.get('aggregate_metrics', {})
        full_metrics = agg.get('full', {})
        focus_index = full_metrics.get('focus_index')
        
        if focus_index is not None and not np.isnan(focus_index):
            focus_data[pid] = focus_index
    
    return focus_data

def extract_scores(scores_data):
    """Extract scores for all participants"""
    scores = {}
    for pid, data in scores_data.items():
        score = data.get('score')
        if score is not None and not np.isnan(score):
            scores[pid] = score
    
    return scores

def classify_participants(hugo_focus, flexvr_focus):
    """
    Classify participants into groups (only include those with data on BOTH platforms):
    Group 1: focus_index < 17 on BOTH FlexVR AND Hugo
    Group 2: focus_index >= 17 on BOTH FlexVR AND Hugo
    Exclude: Participants missing focus_index on either platform
    """
    group1 = set()
    group2 = set()
    
    # Get all participants that have data in both platforms
    all_pids = set(hugo_focus.keys()) & set(flexvr_focus.keys())
    
    for pid in all_pids:
        hugo_focus_val = hugo_focus[pid]
        flexvr_focus_val = flexvr_focus[pid]
        
        if hugo_focus_val < 17 and flexvr_focus_val < 17:
            group1.add(pid)
        elif hugo_focus_val >= 17 and flexvr_focus_val >= 17:
            group2.add(pid)
        # Otherwise exclude (mixed scores)
    
    return group1, group2

def create_score_barplot(pids, scores, group1_pids, procedure_name, output_path):
    """Create bar plot for scores, color-coded by group"""
    # Sort by score (ascending)
    sorted_indices = np.argsort(scores)
    sorted_pids = pids[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Color code by group
    colors = ['#D62728' if pid in group1_pids else '#1F77B4' for pid in sorted_pids]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(sorted_pids)), sorted_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xlabel('Participant ID (sorted by score)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{procedure_name} Score', fontsize=12, fontweight='bold')
    
    title = f'{procedure_name} Scores - Focus Index Groups\n'
    title += f'Group 1 (Red): Focus < 17 on BOTH | Group 2 (Blue): Focus >= 17 on BOTH'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # X-axis labels
    ax.set_xticks(range(len(sorted_pids)))
    ax.set_xticklabels(sorted_pids, rotation=90, fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62728', edgecolor='black', label=f'Group 1: Focus < 17 on both (N={len(group1_pids)})'),
        Patch(facecolor='#1F77B4', edgecolor='black', label=f'Group 2: Focus >= 17 on both')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("FOCUS INDEX GROUP ANALYSIS")
    print("="*70)
    
    # Load data
    print("\nLoading HUGO metrics...")
    hugo_metrics, hugo_scores = load_procedure_data('hugo')
    if hugo_metrics is None:
        print("Failed to load HUGO data")
        return
    
    print("Loading FlexVR metrics...")
    flexvr_metrics, flexvr_scores = load_procedure_data('flexvr')
    if flexvr_metrics is None:
        print("Failed to load FlexVR data")
        return
    
    # Extract focus indices
    print("\nExtracting focus indices...")
    hugo_focus = extract_focus_indices(hugo_metrics, 'sid')
    flexvr_focus = extract_focus_indices(flexvr_metrics, 'pid')
    
    print(f"  Hugo focus indices: {len(hugo_focus)} participants")
    print(f"  FlexVR focus indices: {len(flexvr_focus)} participants")
    
    # Extract scores
    hugo_scores_data = extract_scores(hugo_scores)
    flexvr_scores_data = extract_scores(flexvr_scores)
    
    print(f"  Hugo scores: {len(hugo_scores_data)} participants")
    print(f"  FlexVR scores: {len(flexvr_scores_data)} participants")
    
    # Classify participants
    print("\nClassifying participants...")
    group1, group2 = classify_participants(hugo_focus, flexvr_focus)
    
    # Filter groups to only include participants on both platforms
    pids_both_platforms = set(hugo_focus.keys()) & set(flexvr_focus.keys()) & set(hugo_scores_data.keys()) & set(flexvr_scores_data.keys())
    group1 = group1 & pids_both_platforms
    group2 = group2 & pids_both_platforms
    
    print(f"  Group 1 (Focus < 17 on BOTH): {len(group1)} participants")
    print(f"  Group 2 (Everyone else): {len(group2)} participants")
    
    if len(group1) == 0:
        print("  Warning: Group 1 is empty!")
        return
    
    # Create output directory
    results_dir = f'{ROOT}/results/focus_groups'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate Hugo scores bar plot
    print("\nGenerating bar plots...")
    
    # Get common participants with data on BOTH platforms
    pids_both_platforms = set(hugo_focus.keys()) & set(flexvr_focus.keys()) & set(hugo_scores_data.keys()) & set(flexvr_scores_data.keys())
    
    print(f"\nParticipants with data on BOTH platforms: {len(pids_both_platforms)}")
    
    # Filter to only include participants with both platforms
    hugo_pids_with_data = [pid for pid in pids_both_platforms if pid in hugo_scores_data]
    hugo_scores_vals = np.array([hugo_scores_data[pid] for pid in hugo_pids_with_data])
    hugo_pids_arr = np.array(hugo_pids_with_data)
    
    flexvr_pids_with_data = [pid for pid in pids_both_platforms if pid in flexvr_scores_data]
    flexvr_scores_vals = np.array([flexvr_scores_data[pid] for pid in flexvr_pids_with_data])
    flexvr_pids_arr = np.array(flexvr_pids_with_data)
    
    # Create bar plots
    hugo_plot_path = os.path.join(results_dir, 'hugo_scores_by_focus_groups.png')
    create_score_barplot(hugo_pids_arr, hugo_scores_vals, group1, 'HUGO', hugo_plot_path)
    
    flexvr_plot_path = os.path.join(results_dir, 'flexvr_scores_by_focus_groups.png')
    create_score_barplot(flexvr_pids_arr, flexvr_scores_vals, group1, 'FlexVR', flexvr_plot_path)
    
    # Save group definitions
    group_info = {
        'group1_criteria': 'focus_index < 17 on BOTH FlexVR AND Hugo',
        'group1_count': len(group1),
        'group1_pids': sorted(list(group1)),
        'group2_criteria': 'focus_index >= 17 on BOTH FlexVR AND Hugo',
        'group2_count': len(group2),
        'group2_pids': sorted(list(group2)),
        'excluded_count': len(pids_both_platforms) - len(group1) - len(group2),
        'note': 'Excluded: participants with mixed focus scores (one <17, one >=17) or missing data on either platform'
    }
    
    info_file = os.path.join(results_dir, 'focus_groups_definitions.json')
    with open(info_file, 'w') as f:
        json.dump(group_info, f, indent=2)
    print(f"  ✓ Saved: focus_groups_definitions.json")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")

if __name__ == '__main__':
    main()
