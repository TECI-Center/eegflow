"""
Script to find a PID with NaN aggregate metric and count NaNs in corresponding time series
"""

import json
import math

def count_nan_in_timeseries():
    """Find a PID with NaN aggregate metric and check time series NaNs"""
    
    # Load the data
    with open('eeg_metrics_results.json', 'r') as f:
        eeg_data = json.load(f)
    
    with open('eeg_metrics_ttest_results.json', 'r') as f:
        ttest_results = json.load(f)
    
    # Load HUGO scores to categorize
    with open('medtronic_hugo_metrics_NORMALIZED_scores.json', 'r') as f:
        hugo_scores = json.load(f)
    
    print("\n" + "="*100)
    print("ANALYZING NaN VALUES IN AGGREGATE METRICS AND CORRESPONDING TIME SERIES")
    print("="*100)
    
    # Create EEG data lookup
    eeg_dict = {entry['sid']: entry for entry in eeg_data}
    
    # Categorize surgeons
    high_scorers = set()
    low_scorers = set()
    
    for sid, score_data in hugo_scores.items():
        score = score_data.get('score', 0)
        if score >= 35000:
            high_scorers.add(sid)
        else:
            low_scorers.add(sid)
    
    found_any = False
    
    # Look for a case where aggregate has NaN
    for phase, metrics in ttest_results.items():
        for metric, data in metrics.items():
            if data is not None:
                high_mean = data['high_mean']
                low_mean = data['low_mean']
                
                # Check if either group has NaN
                if (high_mean != high_mean) or (low_mean != low_mean):  # NaN check
                    print(f"\n{'─'*100}")
                    print(f"Found NaN in aggregate metric:")
                    print(f"  Phase: {phase}")
                    print(f"  Metric: {metric}")
                    
                    group_to_check = high_scorers if (high_mean != high_mean) else low_scorers
                    group_name = "HIGH" if (high_mean != high_mean) else "LOW"
                    
                    # Find surgeons in the group and check which don't have data for this phase
                    surgeons_missing_phase = []
                    surgeons_with_phase_data = {}
                    
                    for sid in sorted(group_to_check):
                        if sid in eeg_dict:
                            agg_metrics = eeg_dict[sid].get('aggregate_metrics', {})
                            if phase not in agg_metrics:
                                surgeons_missing_phase.append(sid)
                            else:
                                surgeons_with_phase_data[sid] = agg_metrics[phase].get(metric)
                    
                    print(f"  Affected Group: {group_name} SCORERS")
                    
                    if surgeons_missing_phase:
                        print(f"\n  Surgeons with NO data for '{phase}': {len(surgeons_missing_phase)}")
                        print(f"  IDs: {', '.join(surgeons_missing_phase[:10])}")
                        if len(surgeons_missing_phase) > 10:
                            print(f"       ... and {len(surgeons_missing_phase) - 10} more")
                    
                    if surgeons_with_phase_data:
                        print(f"\n  Surgeons WITH phase data: {len(surgeons_with_phase_data)}")
                        
                        # Check their time series for NaN counts
                        for sid in list(surgeons_with_phase_data.keys())[:3]:  # Show first 3
                            print(f"\n    Example Surgeon: {sid}")
                            
                            if sid in eeg_dict:
                                time_series = eeg_dict[sid].get('time_series', {})
                                
                                if metric in time_series:
                                    ts_data = time_series[metric]
                                    
                                    # Count NaN values
                                    nan_count = 0
                                    total_count = len(ts_data)
                                    
                                    if isinstance(ts_data, list):
                                        for val in ts_data:
                                            if isinstance(val, float):
                                                if math.isnan(val):
                                                    nan_count += 1
                                            elif isinstance(val, str):
                                                if val.lower() == 'nan':
                                                    nan_count += 1
                                    
                                    print(f"      Metric: {metric}")
                                    print(f"      Time series data points: {total_count}")
                                    print(f"      NaN values in time series: {nan_count}")
                                    if total_count > 0:
                                        print(f"      Percentage NaN: {(nan_count/total_count)*100:.2f}%")
                                    
                                    if nan_count > 20:
                                        print(f"      ⚠️  High proportion of NaN values in time series!")
                    
                    found_any = True
                    
                    # Only show the first example
                    if found_any:
                        break
        
        if found_any:
            break
    
    if not found_any:
        print("\nNo NaN values found in aggregate metrics.")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    count_nan_in_timeseries()
