import json
from datetime import datetime
from pathlib import Path



ROOT = Path(__file__).parent
# Load the EEG metrics data
with open(f'{ROOT}/metrics/eeg_metrics_results.json') as f:
    eeg_data = json.load(f)

# Find P173
p173_data = None
for entry in eeg_data:
    if entry.get('sid') == 'P173':
        p173_data = entry
        break

if p173_data:
    print("\n" + "="*100)
    print("PARTICIPANT P173 - DATA OVERVIEW")
    print("="*100)
    
    # Get time series data
    time_series = p173_data.get('time_series', {})
    times = time_series.get('time', [])
    focus_index = time_series.get('focus_index', [])
    engagement_index = time_series.get('engagement_index', [])
    faa_index = time_series.get('FAA_index', [])
    tlx = time_series.get('TLX', [])
    
    print(f"\n📊 TIME SERIES DATA POINTS:")
    print(f"  - Total time points: {len(times)}")
    print(f"  - Focus Index values: {len(focus_index)}")
    print(f"  - Engagement Index values: {len(engagement_index)}")
    print(f"  - FAA Index values: {len(faa_index)}")
    print(f"  - TLX values: {len(tlx)}")
    
    # Calculate total data points
    total_metric_points = len(focus_index) + len(engagement_index) + len(faa_index) + len(tlx)
    print(f"\n📈 TOTAL METRIC DATA POINTS: {total_metric_points}")
    
    # Get time range
    if times:
        start_time = times[0]
        stop_time = times[-1]
        
        # Parse times
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        stop_dt = datetime.fromisoformat(stop_time.replace('Z', '+00:00'))
        duration = stop_dt - start_dt
        
        print(f"\n⏱️  TIME RANGE:")
        print(f"  - Start: {start_time}")
        print(f"  - Stop:  {stop_time}")
        print(f"  - Duration: {duration}")
        print(f"  - Duration in seconds: {duration.total_seconds()} seconds")
        print(f"  - Duration in minutes: {duration.total_seconds() / 60:.2f} minutes")
    
    # Get aggregated metrics if available
    print(f"\n📋 AGGREGATED METRICS (if available):")
    if 'metrics_by_phase' in p173_data:
        phases = p173_data['metrics_by_phase']
        for phase, metrics in phases.items():
            print(f"\n  {phase}:")
            for metric_name, metric_value in metrics.items():
                if metric_value is not None and not isinstance(metric_value, dict):
                    print(f"    - {metric_name}: {metric_value}")
    else:
        print("  No phase-based metrics found")
    
    # Sample of raw data
    print(f"\n📊 SAMPLE OF RAW DATA (first 5 points):")
    for i in range(min(5, len(times))):
        print(f"\n  Point {i+1}:")
        print(f"    - Time: {times[i]}")
        print(f"    - Focus Index: {focus_index[i] if i < len(focus_index) else 'N/A'}")
        print(f"    - Engagement Index: {engagement_index[i] if i < len(engagement_index) else 'N/A'}")
        print(f"    - FAA Index: {faa_index[i] if i < len(faa_index) else 'N/A'}")
        print(f"    - TLX: {tlx[i] if i < len(tlx) else 'N/A'}")
    
    print("\n" + "="*100 + "\n")
else:
    print("P173 not found in EEG metrics data")
