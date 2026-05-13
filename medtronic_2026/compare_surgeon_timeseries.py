"""
Compare time series metrics between a high scorer and low scorer surgeon.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import os

# Configuration
HIGH_THRESHOLD = 45000
LOW_THRESHOLD = 20000
OUTPUT_FOLDER = 'surgeon_comparisons'
ANNOTATIONS_PATH = "/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data_NORMALIZED.json"

# Time series metrics to compare
TIME_SERIES_METRICS = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']

def load_data():
    """Load metrics results and HUGO scores"""
    with open('eeg_metrics_results.json') as f:
        metrics_data = json.load(f)
    
    with open('medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
        hugo_scores = json.load(f)
    
    return metrics_data, hugo_scores

def find_surgeons(metrics_data, hugo_scores):
    """Find a high scorer and use P150 as low scorer"""
    # Create mapping from sid to metrics data
    metrics_by_sid = {item['sid']: item for item in metrics_data}
    
    high_scorers = []
    
    for sid, score_data in hugo_scores.items():
        score = score_data.get('score', 0)
        
        if sid in metrics_by_sid and score >= HIGH_THRESHOLD:
            high_scorers.append((sid, score, metrics_by_sid[sid]))
    
    # Pick first high scorer and use P150 as low scorer
    if not high_scorers:
        print(f"Not enough high scorers (>={HIGH_THRESHOLD})")
        return None, None
    
    if 'P150' not in metrics_by_sid or 'P150' not in hugo_scores:
        print("P150 not found in data")
        return None, None
    
    high_sid, high_score, high_metrics = high_scorers[0]
    low_sid = 'P150'
    low_score = hugo_scores['P150'].get('score', 0)
    low_metrics = metrics_by_sid['P150']
    
    return (high_sid, high_score, high_metrics), (low_sid, low_score, low_metrics)

def load_annotations():
    """Load phase annotations"""
    with open(ANNOTATIONS_PATH) as f:
        annotations_list = json.load(f)
    
    # Convert to dict keyed by surgeon ID
    annotations_dict = {}
    for entry in annotations_list:
        sid = entry['pid']
        annotations_dict[sid] = entry['annotations']
    
    return annotations_dict

def convert_timestamps_to_seconds(timestamps):
    """Convert ISO format timestamps to seconds elapsed from start"""
    if not timestamps:
        return []
    
    # Parse first timestamp
    start_time = datetime.fromisoformat(timestamps[0])
    start_unix = int(start_time.timestamp())
    
    # Convert all timestamps to seconds from start
    seconds = []
    for ts_str in timestamps:
        ts = datetime.fromisoformat(ts_str)
        ts_unix = int(ts.timestamp())
        elapsed = ts_unix - start_unix
        seconds.append(elapsed)
    
    return seconds

def get_phase_boundaries(surgeon_id, annotations):
    """Get phase boundaries in seconds for this surgeon"""
    if surgeon_id not in annotations:
        return {}
    
    surgeon_annot = annotations[surgeon_id]
    
    # Get all phase boundaries to find minimum unix time
    all_boundaries = []
    for phase_name, phase_times in surgeon_annot.items():
        if isinstance(phase_times, list) and len(phase_times) >= 2:
            all_boundaries.extend([phase_times[0], phase_times[1]])
    
    if not all_boundaries:
        return {}
    
    min_unix = min(all_boundaries)
    
    phases_in_seconds = {}
    for phase_name, phase_times in surgeon_annot.items():
        if isinstance(phase_times, list) and len(phase_times) >= 2:
            start_unix, end_unix = phase_times[0], phase_times[1]
            start_sec = start_unix - min_unix
            end_sec = end_unix - min_unix
            phases_in_seconds[phase_name] = (start_sec, end_sec)
    
    return phases_in_seconds

def create_comparison_plots(high_surgeon, low_surgeon, annotations):
    """Create comparison plots for time series metrics"""
    high_sid, high_score, high_metrics = high_surgeon
    low_sid, low_score, low_metrics = low_surgeon
    
    # Convert timestamps to seconds
    high_seconds = convert_timestamps_to_seconds(high_metrics['time_series']['time'])
    low_seconds = convert_timestamps_to_seconds(low_metrics['time_series']['time'])
    
    # Get phase boundaries
    high_phases = get_phase_boundaries(high_sid, annotations)
    low_phases = get_phase_boundaries(low_sid, annotations)
    
    # Calculate total durations
    high_duration = high_seconds[-1] if high_seconds else 0
    low_duration = low_seconds[-1] if low_seconds else 0
    
    # Create a figure for each metric
    for metric in TIME_SERIES_METRICS:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'{metric.replace("_", " ").title()} Comparison', 
                     fontsize=14, fontweight='bold')
        
        # High scorer plot
        high_values = high_metrics['time_series'][metric]
        axes[0].plot(high_seconds, high_values, linewidth=1.5, color='#2E7D32')
        axes[0].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        axes[0].set_title(f'High Scorer: {high_sid} (Score: {high_score:,}) | Duration: {high_duration:.0f}s', 
                         fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add phase annotations for high scorer
        for idx, (phase_name, (start, end)) in enumerate(high_phases.items()):
            axes[0].axvspan(start, end, alpha=0.2, color='blue')
            # Add phase label near the end, staggered vertically
            label_x = start + (end - start) * 0.85
            label_y = axes[0].get_ylim()[1] * (0.92 if idx % 2 == 0 else 0.85)
            axes[0].text(label_x, label_y, phase_name, 
                        ha='left', fontsize=8, rotation=0, color='darkblue', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Low scorer plot
        low_values = low_metrics['time_series'][metric]
        axes[1].plot(low_seconds, low_values, linewidth=1.5, color='#D32F2F')
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        axes[1].set_title(f'Low Scorer: {low_sid} (Score: {low_score:,}) | Duration: {low_duration:.0f}s', 
                         fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add phase annotations for low scorer
        for idx, (phase_name, (start, end)) in enumerate(low_phases.items()):
            axes[1].axvspan(start, end, alpha=0.2, color='blue')
            # Add phase label near the end, staggered vertically
            label_x = start + (end - start) * 0.85
            label_y = axes[1].get_ylim()[1] * (0.92 if idx % 2 == 0 else 0.85)
            axes[1].text(label_x, label_y, phase_name, 
                        ha='left', fontsize=8, rotation=0, color='darkblue',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Scale Y axes the same for comparison
        all_values = [v for v in high_values if v is not None and not np.isnan(v)]
        all_values.extend([v for v in low_values if v is not None and not np.isnan(v)])
        
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.5
            axes[0].set_ylim(y_min - y_margin, y_max + y_margin)
            axes[1].set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Add phase legend
        phase_patch = mpatches.Patch(alpha=0.2, color='blue', label='EEG Phases')
        axes[1].legend(handles=[phase_patch], loc='upper right')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        filename = f'{OUTPUT_FOLDER}/{metric}_comparison_{high_sid}_vs_{low_sid}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def main():
    print("Loading data...")
    metrics_data, hugo_scores = load_data()
    
    print("Finding surgeons...")
    high_surgeon, low_surgeon = find_surgeons(metrics_data, hugo_scores)
    
    if not high_surgeon or not low_surgeon:
        print("Could not find suitable surgeons for comparison")
        return
    
    high_sid, high_score, _ = high_surgeon
    low_sid, low_score, _ = low_surgeon
    print(f"High scorer: {high_sid} (score: {high_score:,})")
    print(f"Low scorer: {low_sid} (score: {low_score:,})")
    
    print("Loading annotations...")
    annotations = load_annotations()
    
    print("Creating comparison plots...")
    create_comparison_plots(high_surgeon, low_surgeon, annotations)
    
    print(f"\nAll plots saved to '{OUTPUT_FOLDER}/' folder")

if __name__ == '__main__':
    main()
