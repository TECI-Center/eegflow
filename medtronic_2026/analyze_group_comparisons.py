"""
Script to analyze participants from all platforms divided into three predefined groups
Performs t-tests for all phases/tasks comparing:
- Group 1 vs Group 2
- Group 2 vs Group 3  
- Group 3 vs Group 1

Compares 4 metrics (focus_index, engagement_index, FAA_index, TLX) across all phases
Generates table visualizations and JSON results for each phase
Handles missing participants (just skips them for that platform)
"""

import json
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from pathlib import Path



ROOT = Path(__file__).parent
# Define the three groups (same across all procedures)
GROUPS = {
    'Group1': ['P228', 'P166', 'P211', 'P202', 'P230', 'P118', 'P111', 'P101', 'P180', 'P121'],
    'Group2': ['P146', 'P226', 'P196', 'P175', 'P140', 'P181', 'P115', 'P122', 'P347', 'P189', 'P227', 'P136'],
    'Group3': ['P216', 'P215', 'P148', 'P141', 'P126', 'P206', 'P187', 'P199', 'P106', 'P217', 'P203', 'P224', 'P237', 'P240', 'P154', 'P239', 'P105', 'P218', 'P238', 'P150', 'P219', 'P188', 'P173', 'P209']
}

# Procedure configurations with phases
PROCEDURES = {
    'hugo': {
        'metrics_path': f'{ROOT}/metrics/eeg_metrics_results.json',
        'scores_path': f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json',
        'pid_key': 'sid',
        'phases': ['Fourth Arm Cutting', 'Knot Tying', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Suturing (Railroad Track)', 'full']
    },
    'fls': {
        'metrics_path': f'{ROOT}/metrics/fls_metrics_results.json',
        'scores_path': f'{ROOT}/scores/fls_metrics_scores.json',
        'pid_key': 'pid',
        'phases': ['Circle Cutting', 'Peg Transfer', 'Pen Rose Suturing', 'full']
    },
    'flexvr': {
        'metrics_path': f'{ROOT}/metrics/flexvr_metrics_results.json',
        'scores_path': f'{ROOT}/scores/flexvr_data_using_annotations_scores.json',
        'pid_key': 'pid',
        'phases': ['Fourth Arm Cutting', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Vessel Energy Dissection', 'full']
    }
}

METRICS = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']

def cohen_d(group1_values, group2_values):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1_values), len(group2_values)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1_values, ddof=1), np.var(group2_values, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1_values) - np.mean(group2_values)) / pooled_std if pooled_std > 0 else 0

def load_procedure_data(procedure_key):
    """Load metrics and scores for a procedure"""
    config = PROCEDURES[procedure_key]
    
    try:
        with open(config['metrics_path'], 'r') as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"  Warning: Metrics file not found: {config['metrics_path']}")
        return None, None
    
    try:
        with open(config['scores_path'], 'r') as f:
            scores_data = json.load(f)
    except FileNotFoundError:
        print(f"  Warning: Scores file not found: {config['scores_path']}")
        return None, None
    
    return metrics_data, scores_data

def extract_group_data_all_phases(metrics_data, scores_data, group_pids, pid_key, phases):
    """Extract data for a specific group across all phases"""
    group_metrics_by_phase = {phase: {} for phase in phases}
    group_scores = {}
    
    # Scores data is a dictionary keyed by participant ID
    scores_lookup = scores_data
    
    for participant in metrics_data:
        pid = participant.get(pid_key)
        if pid in group_pids:
            # Extract metrics for each phase
            agg = participant.get('aggregate_metrics', {})
            
            for phase in phases:
                phase_metrics = agg.get(phase, {})
                group_metrics_by_phase[phase][pid] = {
                    'focus_index': phase_metrics.get('focus_index', np.nan),
                    'engagement_index': phase_metrics.get('engagement_index', np.nan),
                    'FAA_index': phase_metrics.get('FAA_index', np.nan),
                    'TLX': phase_metrics.get('TLX', np.nan)
                }
            
            # Extract score (same for all phases)
            group_scores[pid] = scores_lookup.get(pid, {}).get('score', np.nan)
    
    return group_metrics_by_phase, group_scores


def perform_phase_comparison(group1_metrics_phase, group1_scores, group2_metrics_phase, group2_scores, group1_name, group2_name, phase):
    """Perform t-tests comparing two groups for a specific phase"""
    results = {
        'phase': phase,
        'comparison': f'{group1_name} vs {group2_name}',
        'group1_n': len([pid for pid in group1_metrics_phase if not np.isnan(group1_metrics_phase[pid].get('focus_index', np.nan))]),
        'group2_n': len([pid for pid in group2_metrics_phase if not np.isnan(group2_metrics_phase[pid].get('focus_index', np.nan))]),
        'metrics': {}
    }
    
    # Compare each metric
    for metric in METRICS:
        metric1_vals = []
        metric2_vals = []
        
        for pid in group1_metrics_phase:
            val = group1_metrics_phase[pid].get(metric, np.nan)
            if not np.isnan(val):
                metric1_vals.append(val)
        
        for pid in group2_metrics_phase:
            val = group2_metrics_phase[pid].get(metric, np.nan)
            if not np.isnan(val):
                metric2_vals.append(val)
        
        if len(metric1_vals) > 0 and len(metric2_vals) > 0:
            t_stat, p_val = stats.ttest_ind(metric1_vals, metric2_vals)
            d = cohen_d(metric1_vals, metric2_vals)
            results['metrics'][metric] = {
                'group1_mean': float(np.mean(metric1_vals)),
                'group1_std': float(np.std(metric1_vals, ddof=1)) if len(metric1_vals) > 1 else 0,
                'group1_n': len(metric1_vals),
                'group2_mean': float(np.mean(metric2_vals)),
                'group2_std': float(np.std(metric2_vals, ddof=1)) if len(metric2_vals) > 1 else 0,
                'group2_n': len(metric2_vals),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': bool(p_val < 0.05)
            }
    
    return results

def create_consolidated_comparison_table(all_phases_results, procedure_name, comparison_name, output_path, phases):
    """
    Create a consolidated visualization table with all phases
    Shows one table per metric, with rows for each phase
    """
    
    # Increase figure height based on number of phases
    fig_height = 4 + 2.5 * len(phases)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(16, fig_height))
    if len(METRICS) == 1:
        axes = [axes]
    
    # Add more vertical space between subplots
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05, hspace=0.6)
    
    fig.suptitle(f'{procedure_name.upper()} - {comparison_name}', fontsize=16, fontweight='bold', y=0.995)
    
    for metric_idx, metric in enumerate(METRICS):
        ax = axes[metric_idx]
        
        # Prepare table data - one row per phase
        cell_text = []
        cell_colors = []
        
        for phase in phases:
            phase_result = all_phases_results.get(phase)
            
            if phase_result is None or metric not in phase_result['metrics']:
                continue
            
            metric_result = phase_result['metrics'][metric]
            
            row_text = [phase]
            row_colors = ['#E7E6E6']  # Gray for phase column
            
            group1_mean = metric_result['group1_mean']
            group1_n = metric_result['group1_n']
            group2_mean = metric_result['group2_mean']
            group2_n = metric_result['group2_n']
            t_stat = metric_result['t_statistic']
            p_val = metric_result['p_value']
            cohens_d = metric_result['cohens_d']
            is_significant = metric_result['significant']
            
            # Color for significant results
            cell_color = '#B4E7FF' if is_significant else 'white'
            
            # Group 1 columns
            row_text.append(f"{group1_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{group1_n}")
            row_colors.append(cell_color)
            
            # Group 2 columns
            row_text.append(f"{group2_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{group2_n}")
            row_colors.append(cell_color)
            
            # Statistics
            row_text.append(f"t={t_stat:.3f}")
            row_colors.append(cell_color)
            
            p_str = f"p={p_val:.4f}"
            if is_significant:
                p_str += "*"
            row_text.append(p_str)
            row_colors.append(cell_color)
            
            row_text.append(f"d={cohens_d:.3f}")
            row_colors.append(cell_color)
            
            cell_text.append(row_text)
            cell_colors.append(row_colors)
        
        # Extract group names from comparison string
        comp_str = comparison_name
        if ' - ' in comp_str:
            comp_str = comp_str.split(' - ')[0]
        group1_name, group2_name = comp_str.split(' vs ')
        
        # Create table
        col_labels = [
            'Phase/Task',
            f'μ ({group1_name})',
            f'N ({group1_name})',
            f'μ ({group2_name})',
            f'N ({group2_name})',
            't-statistic',
            'p-value',
            "Cohen's d"
        ]
        
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colWidths=[0.18, 0.11, 0.09, 0.11, 0.09, 0.11, 0.11, 0.11]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.8)
        
        # Style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white', size=9)
        
        # Color data cells
        for cell_row in range(len(cell_text)):
            for cell_col in range(len(col_labels)):
                cell = table[(cell_row + 1, cell_col)]
                if cell_col == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    cell.set_facecolor(cell_colors[cell_row][cell_col])
        
        ax.axis('off')
        metric_title = metric.replace("_", " ").title()
        ax.set_title(f'{metric_title} (* = p < 0.05)', fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def analyze_procedure(procedure_key):
    """Analyze all group comparisons for a given procedure across all phases"""
    config = PROCEDURES[procedure_key]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {procedure_key.upper()}")
    print(f"{'='*70}")
    
    # Load data
    print(f"Loading {procedure_key.upper()} data...")
    metrics_data, scores_data = load_procedure_data(procedure_key)
    
    if metrics_data is None or scores_data is None:
        print(f"  Skipping {procedure_key.upper()} due to missing data")
        return None
    
    # Extract group data (all phases)
    group_data = {}
    available_groups = []
    
    for group_name, pids in GROUPS.items():
        metrics_by_phase, scores = extract_group_data_all_phases(metrics_data, scores_data, pids, config['pid_key'], config['phases'])
        if len(scores) > 0:  # Only include groups with at least one participant
            group_data[group_name] = {'metrics_by_phase': metrics_by_phase, 'scores': scores}
            available_groups.append(group_name)
            print(f"  {group_name}: {len(scores)} participants")
        else:
            print(f"  {group_name}: 0 participants (skipping)")
    
    # Perform comparisons for each phase
    comparisons = [
        ('Group1', 'Group2'),
        ('Group2', 'Group3'),
        ('Group3', 'Group1')
    ]
    
    all_results = {comp: {} for comp in ['Group1 vs Group2', 'Group2 vs Group3', 'Group3 vs Group1']}
    
    for group1_name, group2_name in comparisons:
        if group1_name not in available_groups or group2_name not in available_groups:
            print(f"  Skipping {group1_name} vs {group2_name} (missing data)")
            continue
        
        comparison_key = f'{group1_name} vs {group2_name}'
        print(f"\n  {comparison_key}:")
        
        # Collect results for all phases
        for phase in config['phases']:
            result = perform_phase_comparison(
                group_data[group1_name]['metrics_by_phase'][phase],
                group_data[group1_name]['scores'],
                group_data[group2_name]['metrics_by_phase'][phase],
                group_data[group2_name]['scores'],
                group1_name,
                group2_name,
                phase
            )
            
            all_results[comparison_key][phase] = result
            
            # Print summary
            print(f"    {phase} (n={result['group1_n']} vs {result['group2_n']})", end='')
            sig_count = sum(1 for m in METRICS if m in result['metrics'] and result['metrics'][m]['significant'])
            if sig_count > 0:
                print(f" - {sig_count} significant metrics", end='')
            print()
        
        # Generate consolidated visualization
        results_dir = f'{ROOT}/results/group_comparisons/{procedure_key}'
        os.makedirs(results_dir, exist_ok=True)
        
        comparison_safe = f"{group1_name}_vs_{group2_name}"
        viz_path = os.path.join(results_dir, f'{comparison_safe}_consolidated_table.png')
        create_consolidated_comparison_table(
            all_results[comparison_key], 
            procedure_key, 
            comparison_key,
            viz_path,
            config['phases']
        )
    
    # Save results to JSON
    results_dir = f'{ROOT}/results/group_comparisons/{procedure_key}'
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, 'group_comparison_all_phases_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Results saved: {output_file}")
    
    return all_results

def main():
    print("\n" + "="*70)
    print("MULTI-PLATFORM GROUP COMPARISON ANALYSIS")
    print("="*70)
    
    # Analyze each procedure
    for procedure_key in ['hugo', 'fls', 'flexvr']:
        analyze_procedure(procedure_key)
    
    # Save group definitions to top-level results folder
    results_dir = f'{ROOT}/results/group_comparisons'
    os.makedirs(results_dir, exist_ok=True)
    
    groups_file = os.path.join(results_dir, 'group_definitions.json')
    with open(groups_file, 'w') as f:
        json.dump(GROUPS, f, indent=2)
    
    print(f"\n✓ Group definitions saved: {groups_file}")
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
