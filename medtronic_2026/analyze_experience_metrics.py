"""
Script to analyze EEG metrics by robotic surgery experience level
Compares high experience (>=100 cases) vs low experience (<100 cases)
For each platform (HUGO, FLS, FlexVR)
"""

import json
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from pathlib import Path



ROOT = Path(__file__).parent
# Experience thresholds
HIGH_EXPERIENCE_THRESHOLD = 100

# Procedure configurations with first tasks for bar plots
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

def load_survey_data():
    """Load robotic surgery experience from survey"""
    with open('/Users/calvinperumalla/git/inert_pipe/presurvey_responses.json', 'r') as f:
        data = json.load(f)
    
    # Create lookup by participant ID: PID -> robotic_procedures_career
    experience_lookup = {}
    for entry in data:
        pid = entry.get('participant_id')
        experience = entry.get('robotic_procedures_career')
        if pid and experience is not None:
            # Convert to int (may be stored as string in JSON)
            try:
                experience_lookup[pid] = int(experience)
            except (ValueError, TypeError):
                continue
    
    return experience_lookup

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

def extract_experience_groups(metrics_data, scores_data, experience_lookup, pid_key, phases):
    """Extract data grouped by experience level (including those without experience data)"""
    high_exp_metrics = {phase: {} for phase in phases}
    high_exp_scores = {}
    low_exp_metrics = {phase: {} for phase in phases}
    low_exp_scores = {}
    no_exp_metrics = {phase: {} for phase in phases}
    no_exp_scores = {}
    
    scores_lookup = scores_data
    
    for participant in metrics_data:
        pid = participant.get(pid_key)
        if pid is None:
            continue
        
        # Get experience level
        experience = experience_lookup.get(pid)
        has_experience_data = experience is not None
        
        # Extract metrics for each phase
        agg = participant.get('aggregate_metrics', {})
        
        for phase in phases:
            phase_metrics = agg.get(phase, {})
            metrics_dict = {
                'focus_index': phase_metrics.get('focus_index', np.nan),
                'engagement_index': phase_metrics.get('engagement_index', np.nan),
                'FAA_index': phase_metrics.get('FAA_index', np.nan),
                'TLX': phase_metrics.get('TLX', np.nan)
            }
            
            if has_experience_data:
                is_high_exp = experience >= HIGH_EXPERIENCE_THRESHOLD
                if is_high_exp:
                    high_exp_metrics[phase][pid] = metrics_dict
                else:
                    low_exp_metrics[phase][pid] = metrics_dict
            else:
                no_exp_metrics[phase][pid] = metrics_dict
        
        # Extract score
        score = scores_lookup.get(pid, {}).get('score', np.nan)
        if has_experience_data:
            is_high_exp = experience >= HIGH_EXPERIENCE_THRESHOLD
            if is_high_exp:
                high_exp_scores[pid] = score
            else:
                low_exp_scores[pid] = score
        else:
            no_exp_scores[pid] = score
    
    return (high_exp_metrics, high_exp_scores), (low_exp_metrics, low_exp_scores), (no_exp_metrics, no_exp_scores)

def perform_experience_comparison(high_data, low_data, phase):
    """Perform t-tests comparing experience groups for a phase"""
    high_metrics, high_scores = high_data
    low_metrics, low_scores = low_data
    
    results = {
        'phase': phase,
        'comparison': 'High Experience (>=100) vs Low Experience (<100)',
        'high_n': len([pid for pid in high_metrics[phase] if not np.isnan(high_metrics[phase][pid].get('focus_index', np.nan))]),
        'low_n': len([pid for pid in low_metrics[phase] if not np.isnan(low_metrics[phase][pid].get('focus_index', np.nan))]),
        'metrics': {}
    }
    
    # Compare each metric
    for metric in METRICS:
        metric_high = []
        metric_low = []
        
        for pid in high_metrics[phase]:
            val = high_metrics[phase][pid].get(metric, np.nan)
            if not np.isnan(val):
                metric_high.append(val)
        
        for pid in low_metrics[phase]:
            val = low_metrics[phase][pid].get(metric, np.nan)
            if not np.isnan(val):
                metric_low.append(val)
        
        if len(metric_high) > 0 and len(metric_low) > 0:
            t_stat, p_val = stats.ttest_ind(metric_high, metric_low)
            d = cohen_d(metric_high, metric_low)
            results['metrics'][metric] = {
                'high_mean': float(np.mean(metric_high)),
                'high_std': float(np.std(metric_high, ddof=1)) if len(metric_high) > 1 else 0,
                'high_n': len(metric_high),
                'low_mean': float(np.mean(metric_low)),
                'low_std': float(np.std(metric_low, ddof=1)) if len(metric_low) > 1 else 0,
                'low_n': len(metric_low),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(d),
                'significant': bool(p_val < 0.05)
            }
    
    return results

def create_barplot(pids, metric_values, metric_name, high_exp_pids, low_exp_pids, no_exp_pids, output_path, phase_name=None):
    """Create bar plot for a metric, color-coded by experience level"""
    # Sort by metric value
    sorted_indices = np.argsort(metric_values)
    sorted_pids = pids[sorted_indices]
    sorted_values = metric_values[sorted_indices]
    
    # Color code by experience level
    colors = []
    for pid in sorted_pids:
        if pid in no_exp_pids:
            colors.append('#CCCCCC')  # Grey for no experience data
        elif pid in high_exp_pids:
            colors.append('#2E75B6')  # Blue for high experience
        else:
            colors.append('#ED7D31')  # Orange for low experience
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(sorted_pids)), sorted_values, color=colors, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xlabel('Participant (sorted by metric value)', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    title = f'{metric_name.replace("_", " ").title()}'
    if phase_name:
        title += f'\n{phase_name}'
    title += '\nHigh Experience (Blue, ≥100 cases) vs Low Experience (Orange, <100 cases) vs No Data (Grey)'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # X-axis labels
    ax.set_xticks(range(len(sorted_pids)))
    ax.set_xticklabels(sorted_pids, rotation=90, fontsize=8)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E75B6', edgecolor='black', label=f'High Experience (N={len(high_exp_pids)})'),
        Patch(facecolor='#ED7D31', edgecolor='black', label=f'Low Experience (N={len(low_exp_pids)})'),
        Patch(facecolor='#CCCCCC', edgecolor='black', label=f'No Experience Data (N={len(no_exp_pids)})')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"    ✓ Saved: {os.path.basename(output_path)}")
    plt.close()

def create_consolidated_experience_table(all_phases_results, procedure_name, output_path, phases):
    """Create consolidated table for all phases comparing experience groups"""
    
    fig_height = 4 + 2.5 * len(phases)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(16, fig_height))
    if len(METRICS) == 1:
        axes = [axes]
    
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05, hspace=0.6)
    fig.suptitle(f'{procedure_name.upper()} - Robotic Surgery Experience Comparison', fontsize=16, fontweight='bold', y=0.995)
    
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
            
            high_mean = metric_result['high_mean']
            high_n = metric_result['high_n']
            low_mean = metric_result['low_mean']
            low_n = metric_result['low_n']
            t_stat = metric_result['t_statistic']
            p_val = metric_result['p_value']
            cohens_d = metric_result['cohens_d']
            is_significant = metric_result['significant']
            
            cell_color = '#B4E7FF' if is_significant else 'white'
            
            # High experience columns
            row_text.append(f"{high_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{high_n}")
            row_colors.append(cell_color)
            
            # Low experience columns
            row_text.append(f"{low_mean:.3f}")
            row_colors.append(cell_color)
            row_text.append(f"{low_n}")
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
            'μ (High)',
            'N (High)',
            'μ (Low)',
            'N (Low)',
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

def analyze_procedure(procedure_key, experience_lookup):
    """Analyze experience-based differences for a procedure"""
    config = PROCEDURES[procedure_key]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {procedure_key.upper()}")
    print(f"{'='*70}")
    
    # Load data
    metrics_data, scores_data = load_procedure_data(procedure_key)
    if metrics_data is None or scores_data is None:
        print(f"  Skipping {procedure_key.upper()} due to missing data")
        return None
    
    # Extract experience groups
    high_data, low_data, no_exp_data = extract_experience_groups(
        metrics_data, scores_data, experience_lookup, config['pid_key'], config['phases']
    )
    high_metrics, high_scores = high_data
    low_metrics, low_scores = low_data
    no_exp_metrics, no_exp_scores = no_exp_data
    
    high_count = len(high_scores)
    low_count = len(low_scores)
    no_exp_count = len(no_exp_scores)
    
    print(f"  High Experience (>=100 cases): {high_count} participants")
    print(f"  Low Experience (<100 cases): {low_count} participants")
    print(f"  No Experience Data: {no_exp_count} participants")
    
    if high_count == 0 or low_count == 0:
        print(f"  Skipping {procedure_key.upper()} - insufficient data in high or low group")
        return None
    
    # Create output directory
    results_dir = f'{ROOT}/results/experience_analysis/{procedure_key}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate bar plots for first task
    first_task = config['first_task']
    print(f"\n  Generating bar plots for {first_task}...")
    
    high_pids = set(high_metrics[first_task].keys())
    low_pids = set(low_metrics[first_task].keys())
    no_exp_pids = set(no_exp_metrics[first_task].keys())
    
    for metric in METRICS:
        metric_values_high = []
        metric_values_low = []
        metric_values_no_exp = []
        all_pids = []
        
        for pid in high_metrics[first_task]:
            val = high_metrics[first_task][pid].get(metric, np.nan)
            if not np.isnan(val):
                metric_values_high.append(val)
                all_pids.append(pid)
        
        for pid in low_metrics[first_task]:
            val = low_metrics[first_task][pid].get(metric, np.nan)
            if not np.isnan(val):
                metric_values_low.append(val)
                all_pids.append(pid)
        
        for pid in no_exp_metrics[first_task]:
            val = no_exp_metrics[first_task][pid].get(metric, np.nan)
            if not np.isnan(val):
                metric_values_no_exp.append(val)
                all_pids.append(pid)
        
        if metric_values_high or metric_values_low or metric_values_no_exp:
            all_values = metric_values_high + metric_values_low + metric_values_no_exp
            all_pids_arr = np.array(all_pids)
            all_values_arr = np.array(all_values)
            
            plot_path = os.path.join(results_dir, f'barplot_experience_{metric}.png')
            create_barplot(all_pids_arr, all_values_arr, metric, high_pids, low_pids, no_exp_pids, plot_path, phase_name=first_task)
    
    # Generate ttests for all phases
    print(f"\n  Performing ttests for all phases...")
    all_results = {}
    
    for phase in config['phases']:
        result = perform_experience_comparison(
            (high_metrics, high_scores),
            (low_metrics, low_scores),
            phase
        )
        all_results[phase] = result
        
        # Print summary
        sig_count = sum(1 for m in METRICS if m in result['metrics'] and result['metrics'][m]['significant'])
        print(f"    {phase}: {result['high_n']} high vs {result['low_n']} low, {sig_count} significant metrics")
    
    # Generate consolidated table
    print(f"\n  Generating consolidated comparison table...")
    table_path = os.path.join(results_dir, 'experience_comparison_consolidated_table.png')
    create_consolidated_experience_table(all_results, procedure_key, table_path, config['phases'])
    
    # Save results to JSON
    results_file = os.path.join(results_dir, 'experience_comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  ✓ Results saved: {results_file}")
    
    return all_results

def main():
    print("\n" + "="*70)
    print("ROBOTIC SURGERY EXPERIENCE ANALYSIS")
    print("="*70)
    
    # Load experience data
    print("\nLoading survey data...")
    experience_lookup = load_survey_data()
    print(f"Loaded experience data for {len(experience_lookup)} participants")
    
    # Analyze each procedure
    for procedure_key in ['hugo', 'fls', 'flexvr']:
        analyze_procedure(procedure_key, experience_lookup)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
