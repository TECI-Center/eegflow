"""
Script to check if EEG timestamps match annotation timestamps for phases
"""

import json
from datetime import datetime

def check_timestamp_alignment():
    """Check if EEG data timestamps overlap with phase annotation times"""
    
    with open('eeg_metrics_results.json', 'r') as f:
        eeg_data = json.load(f)
    
    with open('/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data.json', 'r') as f:
        annotations = json.load(f)
    
    # Create lookup dicts
    eeg_dict = {entry['sid']: entry for entry in eeg_data}
    annotations_dict = {entry['pid']: entry for entry in annotations}
    
    print("\n" + "="*100)
    print("CHECKING TIMESTAMP ALIGNMENT BETWEEN EEG DATA AND PHASE ANNOTATIONS")
    print("="*100)
    
    # Check an example surgeon
    test_sid = 'P105'
    
    if test_sid in eeg_dict and test_sid in annotations_dict:
        eeg_entry = eeg_dict[test_sid]
        anno_entry = annotations_dict[test_sid]
        
        print(f"\n{test_sid} Analysis:")
        print(f"{'─'*100}")
        
        # Get EEG time bounds
        eeg_times = eeg_entry['time_series']['time']
        if eeg_times:
            first_time_str = eeg_times[0]
            last_time_str = eeg_times[-1]
            
            # Parse ISO format times
            first_time = datetime.fromisoformat(first_time_str.replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(last_time_str.replace('Z', '+00:00'))
            first_epoch = first_time.timestamp()
            last_epoch = last_time.timestamp()
            
            print(f"\nEEG Data Time Range:")
            print(f"  First timestamp: {first_time_str} (epoch: {first_epoch})")
            print(f"  Last timestamp: {last_time_str} (epoch: {last_epoch})")
            print(f"  Duration: {(last_epoch - first_epoch):.0f} seconds ({(last_epoch - first_epoch)/60:.1f} minutes)")
        
        print(f"\nAnnotation Phases:")
        anno_phases = anno_entry['annotations']
        
        for phase_name, time_bounds in anno_phases.items():
            start_epoch = time_bounds[0]
            end_epoch = time_bounds[1]
            
            # Check overlap
            overlaps = not (end_epoch < first_epoch or start_epoch > last_epoch)
            
            print(f"\n  {phase_name}:")
            print(f"    Start epoch: {start_epoch} -> {datetime.fromtimestamp(start_epoch)}")
            print(f"    End epoch: {end_epoch} -> {datetime.fromtimestamp(end_epoch)}")
            print(f"    Duration: {end_epoch - start_epoch} seconds")
            print(f"    Overlaps with EEG data: {'✓ YES' if overlaps else '✗ NO'}")
            
            if overlaps:
                # Calculate overlap
                overlap_start = max(start_epoch, first_epoch)
                overlap_end = min(end_epoch, last_epoch)
                overlap_duration = overlap_end - overlap_start
                print(f"    Overlap duration: {overlap_duration:.0f} seconds ({overlap_duration/60:.1f} minutes)")
            else:
                if end_epoch < first_epoch:
                    gap = first_epoch - end_epoch
                    print(f"    ⚠️  Phase ends {gap:.0f}s BEFORE EEG starts")
                else:
                    gap = start_epoch - last_epoch
                    print(f"    ⚠️  Phase starts {gap:.0f}s AFTER EEG ends")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    check_timestamp_alignment()
