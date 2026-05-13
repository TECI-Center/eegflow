"""
Script to analyze EEG metrics by surgeon performance (high vs low scorers)
Supports HUGO, FLS, and FlexVR procedures
1. Read EEG metrics results
2. Read performance scores
3. Divide surgeons into high vs low performers based on procedure-specific thresholds
4. Perform t-tests on all metrics across phases
5. Generate results tables and visualizations
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textwrap import wrap
from pathlib import Path



ROOT = Path(__file__).parent
# Metrics to analyze
METRICS = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']

# Procedure configurations
PROCEDURES = {
    'hugo': {
        'metrics_path': f"{ROOT}/metrics/eeg_metrics_results.json",
        'scores_path': f"{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json",
        'output_table': f"{ROOT}/eeg_metrics_ttest_results_hugo.json",
        'output_image': f"{ROOT}/eeg_metrics_ttest_visualization_hugo.png",
        'threshold': 35000,
        'pid_key': 'sid',
        'phases': ['Fourth Arm Cutting', 'Knot Tying', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Suturing (Railroad Track)', 'full']
    },
    'fls': {
        'metrics_path': f"{ROOT}/metrics/fls_metrics_results.json",
        'scores_path': f"{ROOT}/scores/fls_metrics_scores.json",
        'output_table': f"{ROOT}/eeg_metrics_ttest_results_fls.json",
        'output_image': f"{ROOT}/eeg_metrics_ttest_visualization_fls.png",
        'threshold': 30144,
        'pid_key': 'pid',
        'phases': ['Circle Cutting', 'Peg Transfer', 'Pen Rose Suturing', 'full']
    },
    'flexvr': {
        'metrics_path': f"{ROOT}/metrics/flexvr_metrics_results.json",
        'scores_path': f"{ROOT}/scores/flexvr_data_using_annotations_scores.json",
        'output_table': f"{ROOT}/eeg_metrics_ttest_results_flexvr.json",
        'output_image': f"{ROOT}/eeg_metrics_ttest_visualization_flexvr.png",
        'threshold': 30110,
        'pid_key': 'pid',
        'phases': ['Fourth Arm Cutting', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Vessel Energy Dissection', 'full']
    }
}

def load_data(procedure_key):
    """Load both EEG metrics and scores for specified procedure"""
    config = PROCEDURES[procedure_key]
    
    with open(config['metrics_path'], 'r') as f:
        eeg_data = json.load(f)
    
    with open(config['scores_path'], 'r') as f:
        scores_data = json.load(f)
    
    return eeg_data, scores_data, config

def categorize_surgeons(scores_data, pid_key, threshold):
    """
    Divide surgeons into high and low performers based on threshold
    Returns: dict with 'high' and 'low' lists of surgeon IDs
    """
    high_scorers = []
    low_scorers = []
    
    for pid, data in scores_data.items():
        score = data.get('score', 0) if isinstance(data, dict) else data
        if score >= threshold:
            high_scorers.append(pid)
        else:
            low_scorers.append(pid)
    
    return {
        'high': high_scorers,
        'low': low_scorers
    }

def extract_metrics_by_group(eeg_data, surgeon_groups, config):
    """
    Extract metrics organized by group, phase, and metric type
    Returns: dict with structure {group: {phase: {metric: [values]}}}
    """
    metrics_by_group = {
        'high': {},
        'low': {}
    }
    
    pid_key = config['pid_key']
    phases = config['phases']
    
    # Create lookup dict for EEG data
    eeg_dict = {entry[pid_key]: entry for entry in eeg_data}
    
    for group_name, surgeon_ids in surgeon_groups.items():
        for phase in phases:
            metrics_by_group[group_name][phase] = {metric: [] for metric in METRICS}
            
            for sid in surgeon_ids:
                if sid in eeg_dict:
                    agg_metrics = eeg_dict[sid].get('aggregate_metrics', {})
                    if phase in agg_metrics:
                        for metric in METRICS:
                            value = agg_metrics[phase].get(metric)
                            if value is not None:
                                metrics_by_group[group_name][phase][metric].append(value)
    
    return metrics_by_group

def perform_ttest_analysis(metrics_by_group):
    """
    Perform independent t-tests comparing high vs low scorers
    Returns: dict with t-test results
    Only includes phases where both groups have at least 1 data point
    """
    results = {}
    
    for phase in metrics_by_group['high'].keys():
        results[phase] = {}
        
        for metric in METRICS:
            high_values = metrics_by_group['high'][phase][metric]
            low_values = metrics_by_group['low'][phase][metric]
            
            # Convert to numpy arrays for nanmean/nanstd
            high_array = np.array(high_values)
            low_array = np.array(low_values)
            
            # Count NaN values (convert numpy types to Python int)
            high_nan_count = int(np.sum(np.isnan(high_array)))
            low_nan_count = int(np.sum(np.isnan(low_array)))
            
            high_valid_count = int(len(high_array) - high_nan_count)
            low_valid_count = int(len(low_array) - low_nan_count)
            
            # Calculate percentages
            high_nan_pct = (high_nan_count / len(high_array) * 100) if len(high_array) > 0 else 0
            low_nan_pct = (low_nan_count / len(low_array) * 100) if len(low_array) > 0 else 0
            
            # Only perform t-test if both groups have valid data
            if high_valid_count > 0 and low_valid_count > 0:
                # Use nanmean and nanstd to ignore NaN values
                high_mean = float(np.nanmean(high_array))
                low_mean = float(np.nanmean(low_array))
                high_std = float(np.nanstd(high_array))
                low_std = float(np.nanstd(low_array))
                
                # Perform t-test (only on valid values)
                t_stat, p_value = stats.ttest_ind(high_array[~np.isnan(high_array)], 
                                                   low_array[~np.isnan(low_array)])
                
                results[phase][metric] = {
                    'high_mean': high_mean,
                    'low_mean': low_mean,
                    'high_std': high_std,
                    'low_std': low_std,
                    'high_n': int(len(high_array)),
                    'high_valid_n': int(high_valid_count),
                    'high_nan_count': int(high_nan_count),
                    'high_nan_pct': float(high_nan_pct),
                    'low_n': int(len(low_array)),
                    'low_valid_n': int(low_valid_count),
                    'low_nan_count': int(low_nan_count),
                    'low_nan_pct': float(low_nan_pct),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
            else:
                # Mark as skipped if no valid data for either group
                results[phase][metric] = {
                    'high_n': int(len(high_array)),
                    'high_valid_n': int(high_valid_count),
                    'high_nan_count': int(high_nan_count),
                    'high_nan_pct': float(high_nan_pct),
                    'low_n': int(len(low_array)),
                    'low_valid_n': int(low_valid_count),
                    'low_nan_count': int(low_nan_count),
                    'low_nan_pct': float(low_nan_pct),
                    'note': 'Insufficient valid data for t-test'
                }
    
    return results

def save_results_json(results, output_path):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def create_visualization(results, surgeon_groups, output_path, procedure_name, config):
    """
    Create a visualization table with:
    - Blue shading for significant p-values (p < 0.05)
    - Means and N for each group (no p-values in high scorers column)
    - Effect sizes and p-values shown for comparison
    """
    # Filter to relevant phases (sort by procedure phases)
    phases = config['phases']
    
    # Create figure with proper margins
    fig = plt.figure(figsize=(16, 14))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.05)
    
    axes = []
    for metric_idx in range(len(METRICS)):
        ax = fig.add_subplot(len(METRICS), 1, metric_idx + 1)
        axes.append(ax)
    
    for metric_idx, metric in enumerate(METRICS):
        ax = axes[metric_idx]
        
        # Prepare data for table
        cell_text = []
        cell_colors = []
        
        for phase in phases:
            row_text = [phase]
            row_colors = ['#E7E6E6']  # Gray for phase column
            
            phase_result = results[phase].get(metric)
            
            if phase_result is None:
                # Skip row if no data available for this phase
                continue
            else:
                # Check if result has the full data (has t_statistic)
                if 't_statistic' in phase_result:
                    high_mean = phase_result['high_mean']
                    low_mean = phase_result['low_mean']
                    p_value = phase_result['p_value']
                    high_n = phase_result['high_n']
                    low_n = phase_result['low_n']
                    high_nan_pct = phase_result['high_nan_pct']
                    low_nan_pct = phase_result['low_nan_pct']
                    is_significant = phase_result['significant']
                    
                    # High scorers sub-columns: Mean, N, NaN%
                    row_text.append(f"{high_mean:.2f}")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    row_text.append(f"{high_n}")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    row_text.append(f"{high_nan_pct:.0f}%")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    # Low scorers sub-columns: Mean, N, NaN%
                    row_text.append(f"{low_mean:.2f}")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    row_text.append(f"{low_n}")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    row_text.append(f"{low_nan_pct:.0f}%")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    # T-statistic
                    t_stat = phase_result['t_statistic']
                    row_text.append(f"t={t_stat:.2f}")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    # P-value with significance indicator
                    p_str = f"p={p_value:.3f}"
                    if is_significant:
                        p_str += "*"
                    row_text.append(p_str)
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (phase_result['high_std']**2 + phase_result['low_std']**2) / 2
                    )
                    cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0
                    row_text.append(f"d={cohens_d:.2f}")
                    row_colors.append('#B4E7FF' if is_significant else 'white')
                else:
                    # Insufficient data case
                    high_n = phase_result['high_n']
                    low_n = phase_result['low_n']
                    high_nan_pct = phase_result['high_nan_pct']
                    low_nan_pct = phase_result['low_nan_pct']
                    
                    # High scorers sub-columns
                    row_text.append("N/A")
                    row_colors.append('#FFE6E6')
                    row_text.append(f"{high_n}")
                    row_colors.append('#FFE6E6')
                    row_text.append(f"{high_nan_pct:.0f}%")
                    row_colors.append('#FFE6E6')
                    
                    # Low scorers sub-columns
                    row_text.append("N/A")
                    row_colors.append('#FFE6E6')
                    row_text.append(f"{low_n}")
                    row_colors.append('#FFE6E6')
                    row_text.append(f"{low_nan_pct:.0f}%")
                    row_colors.append('#FFE6E6')
                    
                    row_text.append("N/A")
                    row_colors.append('#FFE6E6')
                    
                    row_text.append("N/A")
                    row_colors.append('#FFE6E6')
                    
                    row_text.append("N/A")
                    row_colors.append('#FFE6E6')
            
            cell_text.append(row_text)
            cell_colors.append(row_colors)
        
        # Flatten color list for matplotlib
        flat_colors = []
        for row_colors_list in cell_colors:
            flat_colors.extend(row_colors_list)
        
        # Create table
        col_labels = ['Phase', 'μ (H)', 'N (H)', 'NaN% (H)', 'μ (L)', 'N (L)', 'NaN% (L)', 't-stat', 'p-value', "Cohen's d"]
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            colWidths=[0.12, 0.09, 0.08, 0.09, 0.09, 0.08, 0.09, 0.10, 0.10, 0.10]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.0)
        
        # Style header
        for i in range(10):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white', size=9)
        
        # Color cells based on significance
        for cell_row in range(len(cell_text)):
            for cell_col in range(10):
                cell = table[(cell_row + 1, cell_col)]
                if cell_col == 0:
                    cell.set_facecolor('#E7E6E6')
                else:
                    # Map to flattened colors
                    flat_idx = cell_row * 10 + cell_col
                    if flat_idx < len(flat_colors):
                        cell.set_facecolor(flat_colors[flat_idx])
        
        ax.axis('off')
        metric_title = metric.replace("_", " ").title()
        ax.set_title(f'{metric_title} (High vs Low Performers, * = p < 0.05)', 
                    fontsize=11, fontweight='bold', pad=15)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    plt.close()

def print_summary(procedure_name, surgeon_groups, results, config):
    """Print summary of results"""
    threshold = config['threshold']
    print("\n" + "="*70)
    print(f"{procedure_name.upper()} SURGEON PERFORMANCE ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nHigh Scorers (≥{threshold:,}): {len(surgeon_groups['high'])} surgeons")
    print(f"Low Scorers (<{threshold:,}): {len(surgeon_groups['low'])} surgeons")
    print(f"Total analyzed: {len(surgeon_groups['high']) + len(surgeon_groups['low'])} surgeons")
    
    print("\n" + "-"*70)
    print("SIGNIFICANT FINDINGS (p < 0.05)")
    print("-"*70)
    
    significant_count = 0
    for phase in results:
        for metric in METRICS:
            result = results[phase].get(metric)
            if result and 'significant' in result and result['significant']:
                significant_count += 1
                high_mean = result['high_mean']
                low_mean = result['low_mean']
                p_value = result['p_value']
                high_n = result['high_n']
                low_n = result['low_n']
                high_nan_pct = result['high_nan_pct']
                low_nan_pct = result['low_nan_pct']
                high_valid_n = result['high_valid_n']
                low_valid_n = result['low_valid_n']
                
                print(f"\n{phase} - {metric.replace('_', ' ').title()} (p={p_value:.4f})")
                print(f"  High scorers: mean={high_mean:.4f}, N={high_n} (valid={high_valid_n}, NaN={high_nan_pct:.1f}%)")
                print(f"  Low scorers:  mean={low_mean:.4f}, N={low_n} (valid={low_valid_n}, NaN={low_nan_pct:.1f}%)")
                print(f"  t-statistic: {result['t_statistic']:.4f}")
    
    if significant_count == 0:
        print("\nNo significant differences found (p < 0.05)")
    else:
        print(f"\nTotal significant findings: {significant_count}")
    
    print("\n" + "="*70)

def process_procedure(procedure_key):
    """Process a single procedure (HUGO, FLS, or FlexVR)"""
    procedure_name = procedure_key.upper()
    config = PROCEDURES[procedure_key]
    
    print(f"\n{'='*70}")
    print(f"PROCESSING {procedure_name}")
    print(f"{'='*70}")
    
    print("Loading data...")
    eeg_data, scores_data, config = load_data(procedure_key)
    
    # Filter EEG data to only participants with scores
    pid_key = config['pid_key']
    eeg_pids = set(entry[pid_key] for entry in eeg_data)
    score_pids = set(scores_data.keys())
    missing_pids = eeg_pids - score_pids
    
    if missing_pids:
        print(f"\nNote: {len(missing_pids)} participant(s) in EEG data but missing from scores")
        print(f"These participants will be excluded from analysis.")
        eeg_data = [entry for entry in eeg_data if entry[pid_key] in score_pids]
    
    print(f"\nFinal sample size: {len(eeg_data)} participants with both EEG and score data")
    
    print("Categorizing participants...")
    surgeon_groups = categorize_surgeons(scores_data, pid_key, config['threshold'])
    print(f"  High scorers: {len(surgeon_groups['high'])}")
    print(f"  Low scorers: {len(surgeon_groups['low'])}")
    
    print("Extracting metrics by group...")
    metrics_by_group = extract_metrics_by_group(eeg_data, surgeon_groups, config)
    
    print("Performing t-tests...")
    results = perform_ttest_analysis(metrics_by_group)
    
    print("Saving results...")
    save_results_json(results, config['output_table'])
    
    print("Creating visualization...")
    create_visualization(results, surgeon_groups, config['output_image'], procedure_name, config)
    
    print_summary(procedure_name, surgeon_groups, results, config)

def main():
    """Process all procedures"""
    print("\n" + "="*70)
    print("EEG METRICS BY PERFORMANCE ANALYSIS - ALL PROCEDURES")
    print("="*70)
    
    for procedure_key in ['hugo', 'fls', 'flexvr']:
        try:
            process_procedure(procedure_key)
        except Exception as e:
            print(f"\nError processing {procedure_key.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL PROCEDURES COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
