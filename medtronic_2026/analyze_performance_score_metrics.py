"""
Script to analyze EEG metrics by surgical performance scores
Compares low scorers (< 30000) vs high scorers (> 40000)
For each platform (HUGO, FLS, FlexVR)
"""

import json
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from pathlib import Path



ROOT = Path(__file__).parent
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
LOW_THRESHOLD = 30000
HIGH_THRESHOLD = 40000

def cohen_d(group1_values, group2_values):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1_values), len(group2_values)
    if n1 == 0 or n2 == 0:
        return 0
    var1 = np.var(group1_values, ddof=1) if n1 > 1 else 0
    var2 = np.var(group2_values, ddof=1) if n2 > 1 else 0
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

def classify_by_score(scores_data):
    """Classify participants by performance score"""
    low_scorers = set()
    high_scorers = set()
    
    for pid, data in scores_data.items():
        score = data.get('score')
        if score is not None and not np.isnan(score):
            if score < LOW_THRESHOLD:
                low_scorers.add(pid)
            elif score > HIGH_THRESHOLD:
                high_scorers.add(pid)
    
    return low_scorers, high_scorers

def extract_metrics_data(metrics_data, pid_key, phases):
    """Extract metrics for all participants"""
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
                'focus_index': phase_metrics.get('focus_index', np.nan),
                'engagement_index': phase_metrics.get('engagement_index', np.nan),
                'FAA_index': phase_metrics.get('FAA_index', np.nan),
                'TLX': phase_metrics.get('TLX', np.nan)
            }
    
    return metrics_dict

def perform_ttests(metrics_data, low_scorers, high_scorers, phases):
    """Perform t-tests comparing low vs high scorers across all phases"""
    results = {}
    
    for phase in phases:
        results[phase] = {
            'phase': phase,
            'comparison': 'Low Scorers (<30000) vs High Scorers (>40000)',
            'low_n': 0,
            'high_n': 0,
            'metrics': {}
        }
        
        # Collect data for each metric
        for metric in METRICS:
            low_vals = []
            high_vals = []
            
            for pid in low_scorers:
                if pid in metrics_data:
                    val = metrics_data[pid][phase].get(metric, np.nan)
                    if not np.isnan(val):
                        low_vals.append(val)
            
            for pid in high_scorers:
                if pid in metrics_data:
                    val = metrics_data[pid][phase].get(metric, np.nan)
                    if not np.isnan(val):
                        high_vals.append(val)
            
            results[phase]['low_n'] = len(low_vals)
            results[phase]['high_n'] = len(high_vals)
            
            if len(low_vals) > 0 and len(high_vals) > 0:
                t_stat, p_val = stats.ttest_ind(low_vals, high_vals)
                d = cohen_d(low_vals, high_vals)
                results[phase]['metrics'][metric] = {
                    'low_mean': float(np.mean(low_vals)),
                    'low_std': float(np.std(low_vals, ddof=1)) if len(low_vals) > 1 else 0,
                    'low_n': len(low_vals),
                    'high_mean': float(np.mean(high_vals)),
                    'high_std': float(np.std(high_vals, ddof=1)) if len(high_vals) > 1 else 0,
                    'high_n': len(high_vals),
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'cohens_d': float(d),
                    'significant': bool(p_val < 0.05)
                }
    
    return results

def create_barplot(pids, metric_values, metric_name, low_scorer_pids, high_scorer_pids, output_path):
    """Create bar plot for a metric, color-coded by score group"""
    # Sort by metric value
    sorted_indices = np.argsort(metric_values)
    sorted_pids = pids[sorted_indices]
    sorted_values = metric_values[sorted_indices]
    
    # Color code by group
    colors = []
    for pid in sorted_pids:
        if pid in low_scorer_pids:
            colors.append('#D62728')  # Red for low scorers
        elif pid in high_scorer_pids:
            colors.append('#2E75B6')  # Blue for high scorers
        else:
            colors.append('gray')  # Gray for middle group
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(sorted_pids)), sorted_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xlabel('Participant ID (sorted by metric value)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    title = f'{metric_name.replace("_", " ").title()}\n'
    title += f'Low Scorers (Red, <30k) vs High Scorers (Blue, >40k) - Fourth Arm Cutting'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # X-axis labels
    ax.set_xticks(range(len(sorted_pids)))
    ax.set_xticklabels(sorted_pids, rotation=90, fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62728', edgecolor='black', label='Low Scorers (<30000)'),
        Patch(facecolor='#2E75B6', edgecolor='black', label='High Scorers (>40000)'),
        Patch(facecolor='gray', edgecolor='black', label='Middle Group (30000-40000)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def create_consolidated_table(all_phases_results, procedure_name, output_path, phases):
    """Create consolidated table for all phases comparing score groups"""
    
    fig_height = 4 + 2.5 * len(phases)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(16, fig_height))
    if len(METRICS) == 1:
        axes = [axes]
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05, hspace=0.6)
    fig.suptitle(f'{procedure_name.upper()} - Performance Score Comparison (Low <30k vs High >40k)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    for metric_idx, metric in enumerate(METRICS):
        ax = axes[metric_idx]
        
        # Prepare table data
        cell_text = []
        cell_colors = []
        
        for phase in phases:
            phase_result = all_phases_results.get(phase)
            
            if phase_result is None or metric not in phase_result['metrics']:
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
            
            # Low scorers columns
            row_text.append(f"{low_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{low_n}")
            row_colors.append(cell_color)
            
            # High scorers columns
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
    """Analyze performance score-based differences for a procedure"""
    config = PROCEDURES[procedure_key]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {procedure_key.upper()}")
    print(f"{'='*70}")
    
    # Load data
    metrics_data_raw, scores_data = load_procedure_data(procedure_key)
    if metrics_data_raw is None or scores_data is None:
        print(f"  Skipping {procedure_key.upper()} due to missing data")
        return None
    
    # Classify by score
    print(f"\n  Classifying by performance score...")
    low_scorers, high_scorers = classify_by_score(scores_data)
    
    print(f"    Low scorers (<{LOW_THRESHOLD}): {len(low_scorers)} participants")
    print(f"    High scorers (>{HIGH_THRESHOLD}): {len(high_scorers)} participants")
    
    if len(low_scorers) == 0 or len(high_scorers) == 0:
        print(f"  Skipping {procedure_key.upper()} - insufficient data in one group")
        return None
    
    # Extract metrics
    metrics_data = extract_metrics_data(metrics_data_raw, config['pid_key'], config['phases'])
    
    # Create output directory
    results_dir = f'{ROOT}/results/performance_score_analysis/{procedure_key}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate bar plots for first task
    first_task = config['first_task']
    print(f"\n  Generating bar plots for {first_task}...")
    
    for metric in METRICS:
        metric_values = []
        pids = []
        
        for pid in metrics_data.keys():
            val = metrics_data[pid][first_task].get(metric, np.nan)
            if not np.isnan(val):
                pids.append(pid)
                metric_values.append(val)
        
        if pids:
            pids_arr = np.array(pids)
            metric_values_arr = np.array(metric_values)
            
            plot_path = os.path.join(results_dir, f'barplot_{metric}_first_task.png')
            create_barplot(pids_arr, metric_values_arr, metric, low_scorers, high_scorers, plot_path)
    
    # Generate ttests for all phases
    print(f"\n  Performing ttests for all phases...")
    all_results = perform_ttests(metrics_data, low_scorers, high_scorers, config['phases'])
    
    for phase in config['phases']:
        result = all_results[phase]
        sig_count = sum(1 for m in METRICS if m in result['metrics'] and result['metrics'][m]['significant'])
        print(f"    {phase}: {result['low_n']} low vs {result['high_n']} high, {sig_count} significant metrics")
    
    # Generate consolidated table
    print(f"\n  Generating consolidated comparison table...")
    table_path = os.path.join(results_dir, 'score_comparison_consolidated_table.png')
    create_consolidated_table(all_results, procedure_key, table_path, config['phases'])
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'score_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Results saved: {results_file}")
    
    # Save group definitions
    group_info = {
        'low_scorers_criteria': f'Score < {LOW_THRESHOLD}',
        'low_scorers_count': len(low_scorers),
        'low_scorers_pids': sorted(list(low_scorers)),
        'high_scorers_criteria': f'Score > {HIGH_THRESHOLD}',
        'high_scorers_count': len(high_scorers),
        'high_scorers_pids': sorted(list(high_scorers))
    }
    
    groups_file = os.path.join(results_dir, 'score_groups_definitions.json')
    with open(groups_file, 'w') as f:
        json.dump(group_info, f, indent=2)
    print(f"  ✓ Group definitions saved: {groups_file}")
    
    return all_results

def main():
    print("\n" + "="*70)
    print("PERFORMANCE SCORE-BASED EEG ANALYSIS")
    print("="*70)
    
    # Analyze each procedure
    for procedure_key in ['hugo', 'fls', 'flexvr']:
        analyze_procedure(procedure_key)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
