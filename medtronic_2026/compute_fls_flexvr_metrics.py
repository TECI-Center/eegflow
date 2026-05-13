"""
Script to compute EEG metrics for FLS and FlexVR participants
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
# Configuration for different procedure types
PROCEDURES = {
    'fls': {
        'eeg_base_path': "/Users/calvinperumalla/datasets/medtronic/E1_EEG_fls",
        'annotations_path': "/Users/calvinperumalla/git/inert_pipe/annotations_json/fls_data.json",
        'scores_path': str(ROOT / 'scores/fls_metrics_scores.json'),
        'output_path': f"{ROOT}/fls_metrics_results.json"
    },
    'flexvr': {
        'eeg_base_path': "/Users/calvinperumalla/datasets/medtronic/A1_EEG_flexvr",
        'annotations_path': "/Users/calvinperumalla/git/inert_pipe/annotations_json/flexvr_data.json",
        'scores_path': str(ROOT / 'scores/flexvr_metrics_scores.json'),
        'output_path': f"{ROOT}/flexvr_metrics_results.json"
    }
}

# PIDs to filter out (bad data)
EXCLUDED_PIDS = {'P177'}

def load_annotations(annotations_file):
    """Load annotations JSON and create a lookup dictionary by participant ID"""
    with open(annotations_file, 'r') as f:
        annotations_list = json.load(f)
    
    annotations_dict = {}
    for entry in annotations_list:
        # Handle both 'pid' and 'id' keys
        pid = entry.get('pid') or entry.get('id')
        if pid:
            annotations_dict[pid] = entry['annotations']
    
    return annotations_dict

def load_scores(scores_file):
    """Load scores JSON and create a lookup dictionary by participant ID"""
    with open(scores_file, 'r') as f:
        scores_data = json.load(f)
    return scores_data

def get_participant_dirs(base_path):
    """Get all participant directories sorted by participant ID"""
    participant_dirs = []
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return []
    
    for item in sorted(os.listdir(base_path)):
        if item.startswith('P') and item[1:].isdigit():
            # Skip excluded PIDs
            if item in EXCLUDED_PIDS:
                print(f"Skipping {item} (excluded)")
                continue
            
            full_path = os.path.join(base_path, item)
            if os.path.isdir(full_path):
                participant_dirs.append((item, full_path))
    
    return participant_dirs

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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def process_participant(pid, eeg_file_path, annotations):
    """
    Process a single participant:
    1. Read EEG data
    2. Compute time series metrics
    3. Compute aggregate metrics per phase
    
    Returns a dictionary with results or None if processing fails
    """
    try:
        print(f"Processing {pid}...")
        
        # Read EEG file
        if not os.path.exists(eeg_file_path):
            print(f"  Error: EEG file not found at {eeg_file_path}")
            return None
        
        muse = musereader(eeg_file_path, pid=pid, conifg=config)
        
        # Initialize metrics calculator
        metrics_calc = musemetrics(annotations, muse, pid, config=config)
        
        # Compute time series metrics
        metrics_calc.calculate_time_series()
        time_series = metrics_calc.metric_time_series
        
        # Compute aggregate metrics per phase
        aggregate_metrics = metrics_calc.compute_metrics()
        
        # Prepare results
        result = {
            'pid': pid,
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
        
        print(f"  Successfully processed {pid}")
        print(f"  Phases: {list(aggregate_metrics.keys())}")
        return result
        
    except Exception as e:
        print(f"  Error processing {pid}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main(procedure_type):
    """Main function to process all participants and save results"""
    
    if procedure_type not in PROCEDURES:
        print(f"Error: Unknown procedure type '{procedure_type}'")
        print(f"Supported types: {', '.join(PROCEDURES.keys())}")
        sys.exit(1)
    
    config_dict = PROCEDURES[procedure_type]
    
    print(f"\n{'='*60}")
    print(f"Processing {procedure_type.upper()} Data")
    print(f"{'='*60}\n")
    
    print(f"Loading annotations from {config_dict['annotations_path']}...")
    if not os.path.exists(config_dict['annotations_path']):
        print(f"Error: Annotations file not found: {config_dict['annotations_path']}")
        sys.exit(1)
    annotations = load_annotations(config_dict['annotations_path'])
    print(f"  Loaded {len(annotations)} annotation entries\n")
    
    print(f"Loading scores from {config_dict['scores_path']}...")
    if not os.path.exists(config_dict['scores_path']):
        print(f"Error: Scores file not found: {config_dict['scores_path']}")
        sys.exit(1)
    scores = load_scores(config_dict['scores_path'])
    print(f"  Loaded {len(scores)} score entries\n")
    
    print(f"Getting participant directories from {config_dict['eeg_base_path']}...")
    participant_dirs = get_participant_dirs(config_dict['eeg_base_path'])
    print(f"  Found {len(participant_dirs)} participant directories\n")
    
    results = []
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for pid, participant_dir in participant_dirs:
        eeg_file = os.path.join(participant_dir, "eeg", f"{pid}.txt")
        
        # Check if participant has annotations
        if pid not in annotations:
            print(f"Skipping {pid} - no annotations found")
            skipped_count += 1
            continue
        
        # Process participant
        result = process_participant(pid, eeg_file, annotations[pid])
        
        if result:
            results.append(result)
            successful_count += 1
        else:
            failed_count += 1
    
    # Save results to JSON
    print(f"\nSaving results to {config_dict['output_path']}...")
    with open(config_dict['output_path'], 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"Successfully processed: {successful_count} participants")
    print(f"Failed: {failed_count} participants")
    print(f"Skipped: {skipped_count} participants")
    print(f"Results saved to: {config_dict['output_path']}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_fls_flexvr_metrics.py [fls|flexvr]")
        sys.exit(1)
    
    procedure_type = sys.argv[1].lower()
    main(procedure_type)
