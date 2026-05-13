"""
Print aggregate focus index scores for low scorers in Knot Tying phase
"""

import json
import numpy as np

# Configuration
LOW_THRESHOLD = 35000  # Threshold for low scorers

def main():
    # Load metrics and scores
    with open('eeg_metrics_results.json') as f:
        metrics_data = json.load(f)
    
    with open('medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
        hugo_scores = json.load(f)
    
    # Create mapping
    metrics_by_sid = {item['sid']: item for item in metrics_data}
    
    # Collect low scorers with Knot Tying data
    low_scorers_data = []
    
    for sid, score_data in hugo_scores.items():
        score = score_data.get('score', 0)
        
        if score < LOW_THRESHOLD and sid in metrics_by_sid:
            metrics = metrics_by_sid[sid]
            agg_metrics = metrics.get('aggregate_metrics', {})
            
            if 'Knot Tying' in agg_metrics:
                phase_metrics = agg_metrics['Knot Tying']
                focus_index = phase_metrics.get('focus_index')
                
                if focus_index is not None:
                    low_scorers_data.append({
                        'sid': sid,
                        'score': score,
                        'focus_index': focus_index
                    })
    
    # Sort by HUGO score (descending)
    low_scorers_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Print results
    print("\n" + "="*70)
    print("KNOT TYING - FOCUS INDEX SCORES (Low Scorers)")
    print("="*70)
    print(f"Low Scorer Threshold: < {LOW_THRESHOLD}")
    print(f"Total Low Scorers with Knot Tying data: {len(low_scorers_data)}\n")
    
    print(f"{'SID':<8} {'HUGO Score':<15} {'Focus Index':<15}")
    print("-"*70)
    
    for data in low_scorers_data:
        print(f"{data['sid']:<8} {data['score']:>14,.2f} {data['focus_index']:>14.4f}")
    
    print("-"*70)
    if low_scorers_data:
        focus_values = np.array([d['focus_index'] for d in low_scorers_data])
        valid_values = focus_values[~np.isnan(focus_values)]
        nan_count = np.sum(np.isnan(focus_values))
        
        print(f"Mean Focus Index: {np.nanmean(focus_values):.4f}")
        print(f"Min Focus Index:  {np.nanmin(focus_values):.4f}")
        print(f"Max Focus Index:  {np.nanmax(focus_values):.4f}")
        if nan_count > 0:
            print(f"NaN count: {nan_count} ({100*nan_count/len(focus_values):.1f}%)")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
