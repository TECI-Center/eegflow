"""
Script to analyze participants by surgical score levels
Divides participants into low/high scorers and compares EEG metrics across all phases
Generates bar plots for fourth arm cutting task
"""

import json
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from pathlib import Path



ROOT = Path(__file__).parent
# Score thresholds for each procedure (based on quantiles)
THRESHOLDS = {
    'hugo': {'low': 30075.25, 'high': 40246.68},
    'fls': {'low': 30071.92, 'high': 30196.21},
    'flexvr': {'low': 30063.10, 'high': 30178.01}
}

# Procedure configurations
PROCEDURES = {
    'hugo': {
        'metrics_path': f'{ROOT}/metrics/eeg_metrics_results.json',
        'scores_path': f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json',
        'pid_key': 'sid',
        'phases': ['Fourth Arm Cutting', 'Knot Tying', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Suturing (Railroad Track)', 'full'],
        'first_task': 'Fourth Arm Cutting'
    },
    'fls': {
        'metrics_path': f'{ROOT}/metrics/fls_metrics_results.json',
        'scores_path': f'{ROOT}/scores/fls_metrics_scores.json',
        'pid_key': 'pid',
        'phases': ['Circle Cutting', 'Peg Transfer', 'Pen Rose Suturing', 'full'],
        'first_task': 'Circle Cutting'
    },
    'flexvr': {
        'metrics_path': f'{ROOT}/metrics/flexvr_metrics_results.json',
        'scores_path': f'{ROOT}/scores/flexvr_data_using_annotations_scores.json',
        'pid_key': 'pid',
        'phases': ['Fourth Arm Cutting', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Vessel Energy Dissection', 'full'],
        'first_task': 'Fourth Arm Cutting'
    }
}

METRICS = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']

def load_procedure_data(procedure_key):
    """Load metrics and scores for a procedure"""
    config = PROCEDURES[procedure_key]
    
    try:
        with open(config['metrics_path'], 'r') as f:
            metrics_data = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Metrics file not found")
        return None, None
    
    try:
        with open(config['scores_path'], 'r') as f:
            scores_data = json.load(f)
    except FileNotFoundError:
        print(f"  Error: Scores file not found")
        return None, None
    
    return metrics_data, scores_data

def classify_participants(scores_data, low_thresh, high_thresh):
    """Classify participants into low and high scorers"""
    low_scorers = {}
    high_scorers = {}
    medium_scorers = {}
    
    for pid, data in scores_data.items():
        score = data.get('score')
        if score is not None and not np.isnan(score):
            if score < low_thresh:
                low_scorers[pid] = score
            elif score > high_thresh:
                high_scorers[pid] = score
            else:
                medium_scorers[pid] = score
    
    return low_scorers, high_scorers, medium_scorers

def extract_metrics_all_phases(metrics_data, pid_key, phases):
    """Extract all 4 metrics for each phase"""
    metrics_dict = {}
    
    for participant in metrics_data:
        pid = participant.get(pid_key)
        if pid is None:
            continue
        
        metrics_dict[pid] = {}
        agg = participant.get('aggregate_metrics', {})
        
        for phase in phases:
            phase_metrics = agg.get(phase, {})
            metrics_dict[pid][phase] = {
                'focus_index': phase_metrics.get('focus_index'),
                'engagement_index': phase_metrics.get('engagement_index'),
                'FAA_index': phase_metrics.get('FAA_index'),
                'TLX': phase_metrics.get('TLX')
            }
    
    return metrics_dict

def cohen_d(group1_values, group2_values):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1_values), len(group2_values)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1_values, ddof=1), np.var(group2_values, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1_values) - np.mean(group2_values)) / pooled_std if pooled_std > 0 else 0

def perform_ttest_phase(low_metrics, high_metrics, phase):
    """Compare metrics between groups for a phase"""
    results = {
        'phase': phase,
        'low_n': 0,
        'high_n': 0,
        'metrics': {}
    }
    
    for metric in METRICS:
        low_values = []
        high_values = []
        
        for pid in low_metrics:
            val = low_metrics[pid].get(phase, {}).get(metric)
            if val is not None and not np.isnan(val):
                low_values.append(val)
        
        for pid in high_metrics:
            val = high_metrics[pid].get(phase, {}).get(metric)
            if val is not None and not np.isnan(val):
                high_values.append(val)
        
        if len(low_values) > 0 and len(high_values) > 0:
            t_stat, p_val = stats.ttest_ind(low_values, high_values)
            d = cohen_d(low_values, high_values)
            results['metrics'][metric] = {
                'low_mean': float(np.mean(low_values)),
                'low_std': float(np.std(low_values, ddof=1)) if len(low_values) > 1 else 0,
                'low_n': len(low_values),
                'high_mean': float(np.mean(high_values)),
                'high_std': float(np.std(high_values, ddof=1)) if len(high_values) > 1 else 0,
                'high_n': len(high_values),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': bool(p_val < 0.05)
            }
        
        if results['low_n'] == 0:
            results['low_n'] = len(low_values)
        if results['high_n'] == 0:
            results['high_n'] = len(high_values)
    
    return results

def create_barplot(pids, metric_values, metric_name, low_group_pids, high_group_pids, procedure_name, output_path, medium_group_count=0):
    """Create bar plot for a metric, color-coded by group"""
    # Sort by metric value (ascending)
    sorted_indices = np.argsort(metric_values)
    sorted_pids = pids[sorted_indices]
    sorted_values = metric_values[sorted_indices]
    
    # Color code by group
    colors = []
    for pid in sorted_pids:
        if pid in low_group_pids:
            colors.append('#ED7D31')  # Orange for low scorers
        elif pid in high_group_pids:
            colors.append('#70AD47')  # Green for high scorers
        else:
            colors.append('#CCCCCC')  # Gray for medium
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(sorted_pids)), sorted_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xlabel('Participant ID (sorted by metric value)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    title = f'Platform: {procedure_name.upper()} - {metric_name.replace("_", " ").title()} (Fourth Arm Cutting)\n'
    title += f'Low Scorers (Orange) | High Scorers (Green) | Medium (Gray)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # X-axis labels
    ax.set_xticks(range(len(sorted_pids)))
    ax.set_xticklabels(sorted_pids, rotation=90, fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ED7D31', edgecolor='black', label=f'Low Scorers (N={len(low_group_pids)})'),
        Patch(facecolor='#70AD47', edgecolor='black', label=f'High Scorers (N={len(high_group_pids)})'),
        Patch(facecolor='#CCCCCC', edgecolor='black', label=f'Medium (N={medium_group_count})')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def create_consolidated_ttest_table(all_phases_results, procedure_name, output_path, phases, low_count=0, high_count=0, medium_count=0):
    """Create consolidated table for all phases comparing low vs high groups"""
    
    fig_height = 4 + 2.5 * len(phases)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(16, fig_height))
    if len(METRICS) == 1:
        axes = [axes]
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05, hspace=0.6)
    title = f'Platform: {procedure_name.upper()} - Low vs High Score Groups Comparison\n'
    title += f'Low (N={low_count}) vs High (N={high_count}) vs Medium (N={medium_count})'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    
    for metric_idx, metric in enumerate(METRICS):
        ax = axes[metric_idx]
        
        # Prepare table data
        cell_text = []
        cell_colors = []
        
        for phase in phases:
            phase_result = all_phases_results.get(phase)
            
            if phase_result is None or metric not in phase_result.get('metrics', {}):
                continue
            
            metric_result = phase_result['metrics'][metric]
            
            row_text = [phase]
            row_colors = ['#E7E6E6']
            
            low_mean = metric_result['low_mean']
            low_n = metric_result['low_n']
            high_mean = metric_result['high_mean']
            high_n = metric_result['high_n']
            t_stat = metric_result['t_statistic']
            p_val = metric_result['p_value']
            cohens_d = metric_result['cohens_d']
            is_significant = metric_result['significant']
            
            cell_color = '#B4E7FF' if is_significant else 'white'
            
            # Low score columns
            row_text.append(f"{low_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{low_n}")
            row_colors.append(cell_color)
            
            # High score columns
            row_text.append(f"{high_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{high_n}")
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
        
        col_labels = [
            'Phase/Task',
            'μ (Low)',
            'N (Low)',
            'μ (High)',
            'N (High)',
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
    print(f"  ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def analyze_procedure(procedure_key):
    """Analyze a procedure by score groups"""
    config = PROCEDURES[procedure_key]
    thresholds = THRESHOLDS[procedure_key]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {procedure_key.upper()}")
    print(f"{'='*70}")
    print(f"  Thresholds: Low < {thresholds['low']}, High > {thresholds['high']}")
    
    # Load data
    metrics_data, scores_data = load_procedure_data(procedure_key)
    if metrics_data is None or scores_data is None:
        print(f"  Skipping {procedure_key.upper()} due to missing data")
        return
    
    # Classify participants
    low_scorers, high_scorers, medium_scorers = classify_participants(
        scores_data, thresholds['low'], thresholds['high']
    )
    
    print(f"  Low scorers: {len(low_scorers)} participants")
    print(f"  High scorers: {len(high_scorers)} participants")
    print(f"  Medium scorers: {len(medium_scorers)} participants")
    
    if len(low_scorers) == 0 or len(high_scorers) == 0:
        print(f"  Insufficient data in one group for {procedure_key.upper()}")
        return
    
    # Extract metrics
    metrics_dict = extract_metrics_all_phases(metrics_data, config['pid_key'], config['phases'])
    
    # Create output directory
    results_dir = f'{ROOT}/results/score_groups/{procedure_key}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate bar plots for first task
    first_task = config['first_task']
    print(f"\n  Generating bar plots for {first_task}...")
    
    low_pids_with_data = []
    high_pids_with_data = []
    all_pids = []
    all_values = []
    
    for pid in low_scorers:
        if pid in metrics_dict:
            low_pids_with_data.append(pid)
    
    for pid in high_scorers:
        if pid in metrics_dict:
            high_pids_with_data.append(pid)
    
    for metric in METRICS:
        low_vals = []
        high_vals = []
        pids_list = []
        
        for pid in low_pids_with_data:
            val = metrics_dict[pid].get(first_task, {}).get(metric)
            if val is not None and not np.isnan(val):
                low_vals.append(val)
                pids_list.append(pid)
        
        for pid in high_pids_with_data:
            val = metrics_dict[pid].get(first_task, {}).get(metric)
            if val is not None and not np.isnan(val):
                high_vals.append(val)
                pids_list.append(pid)
        
        if low_vals and high_vals:
            all_values_metric = low_vals + high_vals
            plot_path = os.path.join(results_dir, f'barplot_score_{metric}.png')
            medium_count = len([p for p in medium_scorers if p in metrics_dict])
            create_barplot(
                np.array(pids_list),
                np.array(all_values_metric),
                metric,
                set(low_pids_with_data),
                set(high_pids_with_data),
                procedure_key.upper(),
                plot_path,
                medium_group_count=medium_count
            )
    
    # Generate ttests for all phases
    print(f"\n  Performing ttests for all phases...")
    all_results = {}
    
    for phase in config['phases']:
        result = perform_ttest_phase(metrics_dict, metrics_dict, phase)
        
        # Filter to only calculate for low vs high
        low_metrics = {pid: metrics_dict[pid] for pid in low_pids_with_data}
        high_metrics = {pid: metrics_dict[pid] for pid in high_pids_with_data}
        
        result = perform_ttest_phase(low_metrics, high_metrics, phase)
        all_results[phase] = result
        
        sig_count = sum(1 for m in METRICS if m in result.get('metrics', {}) and result['metrics'][m]['significant'])
        print(f"    {phase}: {result['low_n']} low vs {result['high_n']} high, {sig_count} significant metrics")
    
    # Generate consolidated table
    print(f"\n  Generating consolidated comparison table...")
    table_path = os.path.join(results_dir, 'score_groups_consolidated_table.png')
    medium_count = len([p for p in medium_scorers if p in metrics_dict])
    create_consolidated_ttest_table(
        all_results, procedure_key, table_path, config['phases'],
        low_count=len(low_pids_with_data),
        high_count=len(high_pids_with_data),
        medium_count=medium_count
    )
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'score_groups_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Results saved: {results_file}")
    
    # Save group definitions
    group_info = {
        'procedure': procedure_key.upper(),
        'low_scorers_threshold': f"< {thresholds['low']}",
        'low_scorers_count': len(low_pids_with_data),
        'low_scorers_pids': sorted(low_pids_with_data),
        'high_scorers_threshold': f"> {thresholds['high']}",
        'high_scorers_count': len(high_pids_with_data),
        'high_scorers_pids': sorted(high_pids_with_data),
        'medium_scorers_count': len([p for p in medium_scorers if p in metrics_dict]),
        'excluded_from_groups': f"{thresholds['low']} to {thresholds['high']}"
    }
    
    group_file = os.path.join(results_dir, 'score_groups_definitions.json')
    with open(group_file, 'w') as f:
        json.dump(group_info, f, indent=2)
    print(f"  ✓ Group definitions saved: {group_file}")

def main():
    print("\n" + "="*70)
    print("SCORE-BASED GROUP ANALYSIS")
    print("="*70)
    
    # Analyze each procedure
    for procedure_key in ['hugo', 'fls', 'flexvr']:
        analyze_procedure(procedure_key)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {ROOT}/results/score_groups/")

if __name__ == '__main__':
    main()
