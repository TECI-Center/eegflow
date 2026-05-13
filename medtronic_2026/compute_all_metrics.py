"""
Script to compute all EEG metrics for all surgeon participants
Reads EEG files, computes time series and aggregate metrics per phase
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from musereader import musereader
from museanalyzer import musemetrics
from config import config


ROOT = Path(__file__).parent
# Paths
EEG_BASE_PATH = "/Users/calvinperumalla/datasets/medtronic/eeg/ACS_2024_database"
ANNOTATIONS_PATH = "/Users/calvinperumalla/git/inert_pipe/annotations_json/medtronic_hugo_data_NORMALIZED.json"
OUTPUT_PATH = f"{ROOT}/eeg_metrics_results.json"

def load_annotations(annotations_file):
    """Load annotations JSON and create a lookup dictionary by surgeon ID"""
    with open(annotations_file, 'r') as f:
        annotations_list = json.load(f)
    
    annotations_dict = {}
    for entry in annotations_list:
        sid = entry['pid']
        annotations_dict[sid] = entry['annotations']
    
    return annotations_dict

def get_surgeon_dirs():
    """Get all surgeon directories sorted by surgeon ID"""
    surgeon_dirs = []
    for item in sorted(os.listdir(EEG_BASE_PATH)):
        if item.startswith('P') and item[1:].isdigit():
            full_path = os.path.join(EEG_BASE_PATH, item)
            if os.path.isdir(full_path):
                surgeon_dirs.append((item, full_path))
    return surgeon_dirs

def convert_timestamps_to_serializable(obj):
    """Recursively convert numpy types and timestamps to JSON-serializable types"""
    if isinstance(obj, dict):
        return {k: convert_timestamps_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_timestamps_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def process_surgeon(sid, eeg_file_path, annotations):
    """
    Process a single surgeon:
    1. Read EEG data
    2. Compute time series metrics
    3. Compute aggregate metrics per phase
    
    Returns a dictionary with results or None if processing fails
    """
    try:
        print(f"Processing {sid}...")
        
        # Read EEG file
        if not os.path.exists(eeg_file_path):
            print(f"  Error: EEG file not found at {eeg_file_path}")
            return None
        
        muse = musereader(eeg_file_path, pid=sid, conifg=config)
        
        # Initialize metrics calculator
        metrics_calc = musemetrics(annotations, muse, sid, config=config)
        
        # Compute time series metrics
        metrics_calc.calculate_time_series()
        time_series = metrics_calc.metric_time_series
        
        # Compute aggregate metrics per phase
        aggregate_metrics = metrics_calc.compute_metrics()
        
        # Prepare results
        result = {
            'sid': sid,
            'time_series': {
                'time': time_series['time'].tolist() if isinstance(time_series['time'], np.ndarray) else time_series['time'],
                'focus_index': time_series['focus_index'][1].tolist() if isinstance(time_series['focus_index'][1], np.ndarray) else time_series['focus_index'][1],
                'engagement_index': time_series['engagement_index'][1].tolist() if isinstance(time_series['engagement_index'][1], np.ndarray) else time_series['engagement_index'][1],
                'FAA_index': time_series['FAA_index'][1].tolist() if isinstance(time_series['FAA_index'][1], np.ndarray) else time_series['FAA_index'][1],
                'TLX': time_series['TLX'][1].tolist() if isinstance(time_series['TLX'][1], np.ndarray) else time_series['TLX'][1]
            },
            'aggregate_metrics': convert_timestamps_to_serializable(aggregate_metrics)
        }
        
        # Convert entire result to JSON-serializable format
        result = convert_timestamps_to_serializable(result)
        
        print(f"  Successfully processed {sid}")
        print(f"  Phases: {list(aggregate_metrics.keys())}")
        return result
        
    except Exception as e:
        print(f"  Error processing {sid}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to process all surgeons and save results"""
    
    print("Loading annotations...")
    annotations = load_annotations(ANNOTATIONS_PATH)
    
    print("Getting surgeon directories...")
    surgeon_dirs = get_surgeon_dirs()
    print(f"Found {len(surgeon_dirs)} surgeon directories\n")
    
    results = []
    successful_count = 0
    failed_count = 0
    
    for sid, surgeon_dir in surgeon_dirs:
        eeg_file = os.path.join(surgeon_dir, "eeg", f"{sid}.txt")
        
        # Check if surgeon has annotations
        if sid not in annotations:
            print(f"Skipping {sid} - no annotations found")
            continue
        
        # Process surgeon
        result = process_surgeon(sid, eeg_file, annotations[sid])
        
        if result:
            results.append(result)
            successful_count += 1
        else:
            failed_count += 1
    
    # Save results to JSON
    print(f"\nSaving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"Successfully processed: {successful_count} surgeons")
    print(f"Failed: {failed_count} surgeons")
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
