"""
Script to debug what data is returned when filtering by phase boundaries
"""

import json
import numpy as np
from datetime import datetime

def debug_phase_filtering():
    """Check what data is actually returned when filtering by phase"""
    
    with open('eeg_metrics_results.json', 'r') as f:
        eeg_data = json.load(f)
    
    with open('/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json', 'r') as f:
        annotations = json.load(f)
    
    eeg_dict = {entry['sid']: entry for entry in eeg_data}
    annotations_dict = {entry['pid']: entry for entry in annotations}
    
    print("\n" + "="*100)
    print("DEBUGGING PHASE FILTERING LOGIC")
    print("="*100)
    
    # Test with P105 for Fourth Arm Cutting
    test_sid = 'P105'
    test_phase = 'Fourth Arm Cutting'
    test_metric = 'focus_index'
    
    if test_sid in eeg_dict and test_sid in annotations_dict:
        eeg_entry = eeg_dict[test_sid]
        anno_entry = annotations_dict[test_sid]
        
        print(f"\nTesting: {test_sid} - Phase: {test_phase}")
        print(f"{'─'*100}")
        
        # Get phase boundaries
        phase_bounds = anno_entry['annotations'].get(test_phase)
        if phase_bounds:
            start_epoch = phase_bounds[0]
            end_epoch = phase_bounds[1]
            
            print(f"Phase boundaries:")
            print(f"  Start: {start_epoch} ({datetime.fromtimestamp(start_epoch)})")
            print(f"  End: {end_epoch} ({datetime.fromtimestamp(end_epoch)})")
            
            # Get time series
            time_series = eeg_entry['time_series']
            times = time_series['time']
            metric_data = time_series[test_metric]
            
            print(f"\nRaw time series:")
            print(f"  Total data points: {len(metric_data)}")
            print(f"  First time: {times[0]}")
            print(f"  Last time: {times[-1]}")
            print(f"  First {test_metric}: {metric_data[0]}")
            print(f"  Last {test_metric}: {metric_data[-1]}")
            
            # Manually do the filtering
            filtered_indices = []
            filtered_times = []
            filtered_values = []
            
            for i, time_str in enumerate(times):
                # Parse ISO time
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                epoch = dt.timestamp()
                
                if start_epoch <= epoch <= end_epoch:
                    filtered_indices.append(i)
                    filtered_times.append(time_str)
                    filtered_values.append(metric_data[i])
            
            print(f"\nFiltered results (within phase boundaries):")
            print(f"  Data points in phase: {len(filtered_values)}")
            
            if filtered_values:
                print(f"  First value: {filtered_values[0]}")
                print(f"  Last value: {filtered_values[-1]}")
                
                # Check for NaN
                nan_count = 0
                for v in filtered_values:
                    if isinstance(v, float) and np.isnan(v):
                        nan_count += 1
                
                print(f"  NaN values in filtered data: {nan_count}")
                
                # Calculate mean
                filtered_array = np.array(filtered_values)
                mean_val = np.mean(filtered_array)
                
                print(f"\n  Mean of filtered values: {mean_val}")
                
                if np.isnan(mean_val):
                    print(f"  ⚠️  Mean is NaN! Checking why...")
                    print(f"      All values are NaN: {np.all(np.isnan(filtered_array))}")
                    print(f"      Filtered array: {filtered_array[:10]}")  # Show first 10
            else:
                print(f"  ⚠️  NO DATA POINTS FOUND IN PHASE BOUNDARIES!")
                print(f"  This would result in NaN mean.")
        else:
            print(f"  Phase '{test_phase}' not found in annotations")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    debug_phase_filtering()
