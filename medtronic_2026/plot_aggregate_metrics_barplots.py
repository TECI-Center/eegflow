import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path



ROOT = Path(__file__).parent
# Configuration for different procedure types
PROCEDURES = {
    'hugo': {
        'metrics_file': f'{ROOT}/metrics/eeg_metrics_results.json',
        'scores_file': f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json',
        'output_dir': 'metric_barplots_hugo',
        'pid_key': 'sid',
        'score_threshold': 35000,
        'threshold_label': 'High/Low Scorer Threshold (35,000)'
    },
    'fls': {
        'metrics_file': f'{ROOT}/metrics/fls_metrics_results.json',
        'scores_file': f'{ROOT}/scores/fls_metrics_scores.json',
        'output_dir': 'metric_barplots_fls',
        'pid_key': 'pid',
        'score_threshold': 30144,
        'threshold_label': 'High/Low Scorer Threshold (30,144)'
    },
    'flexvr': {
        'metrics_file': f'{ROOT}/metrics/flexvr_metrics_results.json',
        'scores_file': f'{ROOT}/scores/flexvr_data_using_annotations_scores.json',
        'output_dir': 'metric_barplots_flexvr',
        'pid_key': 'pid',
        'score_threshold': 30110,
        'threshold_label': 'High/Low Scorer Threshold (30,110)'
    }
}

def create_metric_barplot(pids, metric_values, scores, metric_name, filename, output_dir, threshold=None, phase_label=None):
    """Create a sorted bar plot with scores on top of bars, color-coded by score threshold"""
    
    # Sort by metric value
    sorted_indices = np.argsort(metric_values)
    sorted_pids = pids[sorted_indices]
    sorted_values = metric_values[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Color bars based on score threshold if provided
    if threshold is not None:
        colors = ['#388E3C' if score >= threshold else '#D32F2F' for score in sorted_scores]
    else:
        colors = ['#1976D2'] * len(sorted_pids)
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_pids)), sorted_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add score labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:,.0f}',
                ha='center', va='bottom', fontsize=7, rotation=0)
    
    # Customize plot
    ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
    title = f'{metric_name} by Participant (Sorted)\nScores displayed on top of bars'
    if phase_label:
        title = f'{metric_name} by Participant (Sorted) - {phase_label}\nScores displayed on top of bars'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sorted_pids)))
    ax.set_xticklabels(sorted_pids, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend if threshold is applied
    if threshold is not None:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#388E3C', alpha=0.8, edgecolor='black', label='High Scorer'),
                          Patch(facecolor='#D32F2F', alpha=0.8, edgecolor='black', label='Low Scorer')]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def create_scores_barplot(pids, scores, filename, output_dir, threshold=None, threshold_label=None):
    """Create a sorted bar plot of scores"""
    
    # Sort by score
    sorted_indices = np.argsort(scores)
    sorted_pids = pids[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Color bars based on threshold if provided
    if threshold is not None:
        colors = ['#D32F2F' if score < threshold else '#388E3C' for score in sorted_scores]
    else:
        colors = ['#1976D2'] * len(sorted_scores)
    
    # Create bar plot
    bars = ax.bar(range(len(sorted_pids)), sorted_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add horizontal line at threshold if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=threshold_label)
        ax.legend(loc='upper left', fontsize=10)
    
    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:,.0f}',
                ha='center', va='bottom', fontsize=7, rotation=0)
    
    # Customize plot
    ax.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Scores by Participant (Sorted)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(sorted_pids)))
    ax.set_xticklabels(sorted_pids, rotation=45, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

def main(procedure_type):
    """Main function to generate bar plots for specified procedure type"""
    
    if procedure_type not in PROCEDURES:
        print(f"Error: Unknown procedure type '{procedure_type}'")
        print(f"Supported types: {', '.join(PROCEDURES.keys())}")
        sys.exit(1)
    
    config = PROCEDURES[procedure_type]
    output_dir = config['output_dir']
    pid_key = config['pid_key']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Creating plots in '{output_dir}/' directory...")
    print(f"Procedure: {procedure_type.upper()}")
    print(f"{'='*60}\n")
    
    # Load metrics data
    if not os.path.exists(config['metrics_file']):
        print(f"Error: Metrics file not found: {config['metrics_file']}")
        sys.exit(1)
    
    with open(config['metrics_file']) as f:
        metrics_data = json.load(f)
    
    # Load scores
    if not os.path.exists(config['scores_file']):
        print(f"Error: Scores file not found: {config['scores_file']}")
        sys.exit(1)
    
    with open(config['scores_file']) as f:
        scores_data = json.load(f)

    # Determine which aggregate phase to use
    # Use 'Fourth Arm Cutting' for HUGO and FlexVR, 'full' for FLS (doesn't have Fourth Arm Cutting)
    aggregate_phase = 'Fourth Arm Cutting' if procedure_type in ['hugo', 'flexvr'] else 'full'
    phase_label = 'Fourth Arm Cutting Phase' if procedure_type in ['hugo', 'flexvr'] else 'Full Aggregate'

    # Extract aggregate metrics and scores
    pids = []
    focus_values = []
    engagement_values = []
    faa_values = []
    tlx_values = []
    scores = []
    
    for item in metrics_data:
        pid = item[pid_key]
        
        # Skip P177 (bad data)
        if pid == 'P177':
            continue
        
        # Get aggregate metrics (using specified phase)
        if 'aggregate_metrics' in item and aggregate_phase in item['aggregate_metrics']:
            agg = item['aggregate_metrics'][aggregate_phase]
            
            # Get score
            if pid in scores_data:
                score = scores_data[pid]['score']
                
                pids.append(pid)
                focus_values.append(agg['focus_index'])
                engagement_values.append(agg['engagement_index'])
                faa_values.append(agg['FAA_index'])
                tlx_values.append(agg['TLX'])
                scores.append(score)
    
    # Convert to numpy arrays
    pids = np.array(pids)
    focus_values = np.array(focus_values)
    engagement_values = np.array(engagement_values)
    faa_values = np.array(faa_values)
    tlx_values = np.array(tlx_values)
    scores = np.array(scores)
    
    print(f"Loaded data for {len(pids)} participants\n")
    
    # Create all plots
    print("Generating bar plots...")
    create_metric_barplot(pids, focus_values, scores, 'Focus Index (Average)', '01_focus_index_barplot.png', output_dir, threshold=config['score_threshold'], phase_label=phase_label)
    create_metric_barplot(pids, engagement_values, scores, 'Engagement Index (Average)', '02_engagement_index_barplot.png', output_dir, threshold=config['score_threshold'], phase_label=phase_label)
    create_metric_barplot(pids, faa_values, scores, 'FAA Index (Average)', '03_faa_index_barplot.png', output_dir, threshold=config['score_threshold'], phase_label=phase_label)
    create_metric_barplot(pids, tlx_values, scores, 'Task Load Index - TLX (Average)', '04_tlx_barplot.png', output_dir, threshold=config['score_threshold'], phase_label=phase_label)
    create_scores_barplot(pids, scores, '05_scores_barplot.png', output_dir, 
                         threshold=config['score_threshold'], 
                         threshold_label=config['threshold_label'])
    
    print("\n" + "="*60)
    print(f"All plots saved to '{output_dir}/' directory!")
    print("="*60)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Focus Index - Min: {np.min(focus_values):.4f}, Max: {np.max(focus_values):.4f}, Mean: {np.mean(focus_values):.4f}")
    print(f"Engagement Index - Min: {np.min(engagement_values):.4f}, Max: {np.max(engagement_values):.4f}, Mean: {np.mean(engagement_values):.4f}")
    print(f"FAA Index - Min: {np.min(faa_values):.4f}, Max: {np.max(faa_values):.4f}, Mean: {np.mean(faa_values):.4f}")
    print(f"TLX - Min: {np.min(tlx_values):.4f}, Max: {np.max(tlx_values):.4f}, Mean: {np.mean(tlx_values):.4f}")
    print(f"Scores - Min: {np.min(scores):,.0f}, Max: {np.max(scores):,.0f}, Mean: {np.mean(scores):,.0f}")
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_aggregate_metrics_barplots.py [hugo|fls|flexvr]")
        sys.exit(1)
    
    procedure_type = sys.argv[1].lower()
    main(procedure_type)
