"""
Script to print EEG metrics results in an easy-to-read format
"""

import json

def print_metrics_summary():
    """Load and pretty-print the t-test results"""
    
    with open('eeg_metrics_ttest_results.json', 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*90)
    print("EEG METRICS COMPARISON: HIGH vs LOW PERFORMERS")
    print("="*90)
    
    metrics = ['focus_index', 'engagement_index', 'FAA_index', 'TLX']
    metric_names = {
        'focus_index': 'Focus Index',
        'engagement_index': 'Engagement Index',
        'FAA_index': 'FAA Index',
        'TLX': 'TLX (Task Load Index)'
    }
    
    for phase in results.keys():
        if phase == 'calibration':
            continue
            
        print(f"\n{'─'*90}")
        print(f"PHASE: {phase.upper()}")
        print(f"{'─'*90}")
        
        for metric in metrics:
            data = results[phase][metric]
            
            if data is None:
                print(f"\n  {metric_names[metric]}: NO DATA")
                continue
            
            high_mean = data['high_mean']
            low_mean = data['low_mean']
            high_n = data['high_n']
            low_n = data['low_n']
            p_value = data['p_value']
            significant = data['significant']
            
            # Check if data is valid (not NaN)
            if high_mean != high_mean or low_mean != low_mean:  # NaN check
                print(f"\n  {metric_names[metric]}: INCOMPLETE DATA")
                print(f"    High Scorers (N={high_n}): Data unavailable")
                print(f"    Low Scorers (N={low_n}): Data unavailable")
                continue
            
            sig_marker = " ⭐ SIGNIFICANT" if significant else ""
            print(f"\n  {metric_names[metric]}{sig_marker}")
            print(f"    High Scorers (N={high_n:2d}): mean = {high_mean:9.4f} ± {data['high_std']:7.4f}")
            print(f"    Low Scorers  (N={low_n:2d}): mean = {low_mean:9.4f} ± {data['low_std']:7.4f}")
            print(f"    t-statistic = {data['t_statistic']:7.4f}  |  p-value = {p_value:.6f}")
            print(f"    Cohen's d = {(high_mean - low_mean) / ((data['high_std']**2 + data['low_std']**2)**0.5 * 0.5**0.5):7.4f}")
    
    print("\n" + "="*90)
    print("LEGEND:")
    print("  ⭐ = Statistically significant (p < 0.05)")
    print("  N = Number of surgeons with data for that phase")
    print("  ± = Standard deviation")
    print("="*90 + "\n")

if __name__ == "__main__":
    print_metrics_summary()
