"""
Script to print actual EEG metric values (arrays) for each group by phase/metric
"""

import json
import numpy as np

def print_metric_values():
    """Load and print actual metric values in arrays by group"""
    
    # Load EEG metrics data
    with open('eeg_metrics_results.json', 'r') as f:
        eeg_data = json.load(f)
    
    # Load HUGO scores to categorize surgeons
    with open('medtronic_hugo_metrics_NORMALIZED_scores.json', 'r') as f:
        hugo_scores = json.load(f)
    
    # Categorize surgeons
    high_scorers = set()
    low_scorers = set()
    
    for sid, data in hugo_scores.items():
        score = data.get('score', 0)
        if score >= 35000:
            high_scorers.add(sid)
        else:
            low_scorers.add(sid)
    
    # Create lookup for EEG data
    eeg_dict = {entry['sid']: entry for entry in eeg_data}
    
    metrics = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']
    metric_names = {
        'focus_index': 'Focus Index',
        'engagement_index': 'Engagement Index',
        'FAA_index': 'FAA Index',
        'TLX': 'TLX (Task Load Index)'
    }
    
    # Get all phases (exclude calibration)
    all_phases = set()
    for entry in eeg_data:
        agg_metrics = entry.get('aggregate_metrics', {})
        for phase in agg_metrics.keys():
            if phase != 'calibration':
                all_phases.add(phase)
    
    phases = sorted(all_phases)
    
    print("\n" + "="*100)
    print("EEG METRIC VALUES BY GROUP AND PHASE")
    print("="*100)
    
    for phase in phases:
        print(f"\n{'─'*100}")
        print(f"PHASE: {phase.upper()}")
        print(f"{'─'*100}")
        
        for metric in metrics:
            print(f"\n{metric_names[metric]}:")
            
            # Collect values for high and low scorers
            high_values = []
            low_values = []
            
            for sid in high_scorers:
                if sid in eeg_dict:
                    agg_metrics = eeg_dict[sid].get('aggregate_metrics', {})
                    if phase in agg_metrics:
                        value = agg_metrics[phase].get(metric)
                        if value is not None:
                            high_values.append(value)
            
            for sid in low_scorers:
                if sid in eeg_dict:
                    agg_metrics = eeg_dict[sid].get('aggregate_metrics', {})
                    if phase in agg_metrics:
                        value = agg_metrics[phase].get(metric)
                        if value is not None:
                            low_values.append(value)
            
            # Print high scorers
            print(f"  High Scorers (N={len(high_values)}):")
            if high_values:
                high_values_str = ", ".join([f"{v:.6f}" for v in high_values])
                print(f"    [{high_values_str}]")
            else:
                print(f"    []")
            
            # Print low scorers
            print(f"  Low Scorers (N={len(low_values)}):")
            if low_values:
                low_values_str = ", ".join([f"{v:.6f}" for v in low_values])
                print(f"    [{low_values_str}]")
            else:
                print(f"    []")
            
            # Print summary stats if data exists
            if high_values or low_values:
                if high_values:
                    print(f"  High Scorers: mean={np.mean(high_values):.6f}, std={np.std(high_values):.6f}, min={np.min(high_values):.6f}, max={np.max(high_values):.6f}")
                if low_values:
                    print(f"  Low Scorers: mean={np.mean(low_values):.6f}, std={np.std(low_values):.6f}, min={np.min(low_values):.6f}, max={np.max(low_values):.6f}")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    print_metric_values()
