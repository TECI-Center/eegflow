#!/usr/bin/env python3
"""
Compare HUGO vs FlexVR EEG metrics with color-coded scoring.

Creates 4 scatter plots comparing aggregate metrics between HUGO and FlexVR,
with color coding based on performance level on both procedures.

Color scheme:
- Red: Low score on both HUGO and FlexVR
- Yellow: High FlexVR, Low HUGO
- Blue: High HUGO, Low FlexVR
- Green: High score on both
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ===== CONFIGURATION =====
HUGO_METRICS_FILE = 'metrics/eeg_metrics_results.json'
HUGO_SCORES_FILE = 'scores/medtronic_hugo_metrics_NORMALIZED_scores.json'

FLEXVR_METRICS_FILE = 'metrics/flexvr_metrics_results.json'
FLEXVR_SCORES_FILE = 'scores/flexvr_data_using_annotations_scores.json'

HUGO_THRESHOLD = 35000
FLEXVR_THRESHOLD = 30110

OUTPUT_DIR = 'metric_scatter_comparisons_hugo_flexvr'

# Metrics to plot
METRICS = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']
METRIC_LABELS = {
    'focus_index': 'Focus Index',
    'engagement_index': 'Engagement Index',
    'FAA_index': 'FAA Index',
    'TLX': 'Task Load Index (TLX)'
}

# ========================

def load_data():
    """Load metrics and scores for both HUGO and FlexVR."""
    # Load HUGO data
    with open(HUGO_METRICS_FILE) as f:
        hugo_metrics = json.load(f)
    with open(HUGO_SCORES_FILE) as f:
        hugo_scores = json.load(f)
    
    # Load FlexVR data
    with open(FLEXVR_METRICS_FILE) as f:
        flexvr_metrics = json.load(f)
    with open(FLEXVR_SCORES_FILE) as f:
        flexvr_scores = json.load(f)
    
    return hugo_metrics, hugo_scores, flexvr_metrics, flexvr_scores


def build_metric_dicts(hugo_metrics, flexvr_metrics):
    """Build dictionaries mapping participant ID to Fourth Arm Cutting aggregate metrics."""
    hugo_dict = {}
    for item in hugo_metrics:
        sid = item['sid']
        if 'aggregate_metrics' in item and 'Fourth Arm Cutting' in item['aggregate_metrics']:
            hugo_dict[sid] = item['aggregate_metrics']['Fourth Arm Cutting']
    
    flexvr_dict = {}
    for item in flexvr_metrics:
        pid = item['pid']
        if 'aggregate_metrics' in item and 'Fourth Arm Cutting' in item['aggregate_metrics']:
            flexvr_dict[pid] = item['aggregate_metrics']['Fourth Arm Cutting']
    
    return hugo_dict, flexvr_dict


def get_color(hugo_score, flexvr_score, hugo_threshold, flexvr_threshold):
    """
    Determine color based on performance levels.
    
    Red: Low on both
    Yellow: High FlexVR, Low HUGO
    Blue: High HUGO, Low FlexVR
    Green: High on both
    """
    hugo_high = hugo_score >= hugo_threshold
    flexvr_high = flexvr_score >= flexvr_threshold
    
    if hugo_high and flexvr_high:
        return '#2ECC40', 'High Both'  # Green
    elif hugo_high and not flexvr_high:
        return '#1976D2', 'High HUGO'  # Blue
    elif not hugo_high and flexvr_high:
        return '#FFD700', 'High FlexVR'  # Yellow
    else:
        return '#E74C3C', 'Low Both'  # Red


def create_scatter_plot(metric_name, hugo_data, flexvr_data, hugo_scores, flexvr_scores, output_dir):
    """Create a scatter plot for a specific metric."""
    # Collect data for common participants
    hugo_vals = []
    flexvr_vals = []
    colors = []
    labels = []
    
    # Exclude P177 (known bad data)
    EXCLUDED_PIDS = {'P177'}
    
    for pid in hugo_data.keys():
        if pid in EXCLUDED_PIDS:
            continue
        if pid in flexvr_data and pid in hugo_scores and pid in flexvr_scores:
            hugo_score = hugo_scores[pid]['score']
            flexvr_score = flexvr_scores[pid]['score']
            
            hugo_metric = hugo_data[pid].get(metric_name)
            flexvr_metric = flexvr_data[pid].get(metric_name)
            
            # Skip if missing data
            if hugo_metric is None or flexvr_metric is None:
                continue
            if np.isnan(hugo_metric) or np.isnan(flexvr_metric):
                continue
            
            hugo_vals.append(hugo_metric)
            flexvr_vals.append(flexvr_metric)
            color, color_label = get_color(hugo_score, flexvr_score, HUGO_THRESHOLD, FLEXVR_THRESHOLD)
            colors.append(color)
            labels.append(pid)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points with annotations
    scatter = ax.scatter(hugo_vals, flexvr_vals, c=colors, s=40, alpha=0.7, edgecolors='black', linewidth=1.2)
    
    # Annotate with participant IDs
    for i, pid in enumerate(labels):
        # Remove 'P' prefix to save space
        pid_label = pid.lstrip('Pp') if pid else pid
        ax.annotate(pid_label, (hugo_vals[i], flexvr_vals[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7)
    
    # Labels and title
    ax.set_xlabel(f'HUGO {METRIC_LABELS[metric_name]}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'FlexVR {METRIC_LABELS[metric_name]}', fontsize=12, fontweight='bold')
    ax.set_title(f'HUGO vs FlexVR: {METRIC_LABELS[metric_name]}\n(Fourth Arm Cutting Phase)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC40', edgecolor='black', label='High Both'),
        Patch(facecolor='#1976D2', edgecolor='black', label='High HUGO'),
        Patch(facecolor='#FFD700', edgecolor='black', label='High FlexVR'),
        Patch(facecolor='#E74C3C', edgecolor='black', label='Low Both'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    # Add statistics text
    n_points = len(hugo_vals)
    textstr = f'N = {n_points} participants'
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f'{metric_name}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    return len(hugo_vals)


def main():
    """Main execution."""
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"\nCreating comparison plots in: {output_dir}/\n")
    
    # Load data
    print("Loading metrics and scores...")
    hugo_metrics, hugo_scores, flexvr_metrics, flexvr_scores = load_data()
    
    # Build dictionaries
    hugo_dict, flexvr_dict = build_metric_dicts(hugo_metrics, flexvr_metrics)
    
    print(f"HUGO: {len(hugo_dict)} participants with full aggregate metrics")
    print(f"FlexVR: {len(flexvr_dict)} participants with full aggregate metrics")
    
    # Find common participants
    common_pids = set(hugo_dict.keys()) & set(flexvr_dict.keys())
    common_with_scores = common_pids & set(hugo_scores.keys()) & set(flexvr_scores.keys())
    print(f"Common participants with both metrics and scores: {len(common_with_scores)}\n")
    
    # Create scatter plots for each metric
    print("Creating scatter plots...")
    for metric in METRICS:
        n_points = create_scatter_plot(metric, hugo_dict, flexvr_dict, 
                                      hugo_scores, flexvr_scores, output_dir)
        print(f"  {METRIC_LABELS[metric]}: {n_points} data points")
    
    print(f"\n✅ All plots saved to {output_dir}/")
    print("\nColor Legend:")
    print("  🟢 Green  : High score on both HUGO and FlexVR")
    print("  � Blue   : High HUGO, Low FlexVR")
    print("  🟡 Yellow : High FlexVR, Low HUGO")
    print("  🔴 Red    : Low score on both HUGO and FlexVR")
    print(f"\nThresholds: HUGO={HUGO_THRESHOLD:,}, FlexVR={FLEXVR_THRESHOLD:,}")


if __name__ == '__main__':
    main()
