import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path



ROOT = Path(__file__).parent
# Load the ttest results
with open(f'{ROOT}/results/eeg_metrics_ttest_results_hugo.json') as f:
    results = json.load(f)

# Create a comprehensive table visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 1, hspace=0.4)

phases_to_display = ['Fourth Arm Cutting', 'Knot Tying', 'Ring Tower Transfer']
colors_table = ['#f0f0f0', '#ffffff']

for idx, phase in enumerate(phases_to_display):
    ax = fig.add_subplot(gs[idx])
    ax.axis('tight')
    ax.axis('off')
    
    phase_data = results[phase]
    
    # Prepare table data
    headers = ['Metric', 'μ (H)', 'N (H)', 'NaN% (H)', 'μ (L)', 'N (L)', 'NaN% (L)', 't-stat', 'p-value', "Cohen's d"]
    
    table_data = []
    for metric, metric_data in phase_data.items():
        row = [
            metric,
            f"{metric_data['high_mean']:.2f}",
            str(metric_data['high_valid_n']),
            f"{metric_data['high_nan_pct']:.0f}%",
            f"{metric_data['low_mean']:.2f}",
            str(metric_data['low_valid_n']),
            f"{metric_data['low_nan_pct']:.0f}%",
            f"{metric_data['t_statistic']:.3f}",
            f"{metric_data['p_value']:.3f}" + ("*" if metric_data['p_value'] < 0.05 else ""),
            f"{metric_data.get('significant', False)}"  # Placeholder for Cohen's d
        ]
        table_data.append(row)
    
    # Extract Cohen's d values if available in another file or calculate
    for i, metric in enumerate(phase_data.keys()):
        # You would need to calculate or extract Cohen's d
        # For now, we'll calculate it from the data
        metric_info = phase_data[metric]
        n1 = metric_info['high_valid_n']
        n2 = metric_info['low_valid_n']
        mean1 = metric_info['high_mean']
        mean2 = metric_info['low_mean']
        std1 = metric_info['high_std']
        std2 = metric_info['low_std']
        
        if n1 > 1 and n2 > 1:
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        table_data[i][-1] = f"{cohens_d:.2f}"
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            
            # Highlight significant p-values
            if j == 8 and float(table_data[i-1][j].rstrip('*')) < 0.05:
                table[(i, j)].set_facecolor('#FFFF99')
                table[(i, j)].set_text_props(weight='bold')
            
            table[(i, j)].set_text_props(ha='center', va='center')
    
    # Add title
    ax.text(0.5, 1.15, f'{phase} - EEG Metrics Comparison\n(High vs Low Performers)',
            ha='center', va='top', fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    # Add note about significance
    ax.text(0.02, -0.15, '* p < 0.05 (significant)', ha='left', va='top', 
            fontsize=8, style='italic', transform=ax.transAxes)

plt.suptitle('EEG Metrics T-Test Results - HUGO Studies\nHigh vs Low Performers', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(f'{ROOT}/results/eeg_metrics_ttest_visualization_hugo.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: /Users/calvinperumalla/git/eegflow/results/eeg_metrics_ttest_visualization_hugo.png")

# Also create a version with all phases on one multi-page style
fig = plt.figure(figsize=(18, 14))

all_phases = list(results.keys())
n_phases = len(all_phases)
n_cols = 2
n_rows = (n_phases + n_cols - 1) // n_cols

plt.suptitle('EEG Metrics T-Test Results - All HUGO Phases\nHigh vs Low Performers', 
             fontsize=16, fontweight='bold')

for idx, phase in enumerate(all_phases):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    ax.axis('tight')
    ax.axis('off')
    
    phase_data = results[phase]
    
    # Prepare table data
    headers = ['Metric', 'μ (H)', 'N (H)', 'NaN%', 'μ (L)', 'N (L)', 'NaN%', 't-stat', 'p-val', "d"]
    
    table_data = []
    for metric, metric_data in phase_data.items():
        # Calculate Cohen's d
        n1 = metric_data['high_valid_n']
        n2 = metric_data['low_valid_n']
        mean1 = metric_data['high_mean']
        mean2 = metric_data['low_mean']
        std1 = metric_data['high_std']
        std2 = metric_data['low_std']
        
        if n1 > 1 and n2 > 1:
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        p_val = metric_data['p_value']
        p_str = f"{p_val:.3f}" + ("*" if p_val < 0.05 else "")
        
        row = [
            metric[:10],  # Shorten metric name
            f"{mean1:.1f}",
            str(n1),
            f"{metric_data['high_nan_pct']:.0f}",
            f"{mean2:.1f}",
            str(n2),
            f"{metric_data['low_nan_pct']:.0f}",
            f"{metric_data['t_statistic']:.2f}",
            p_str,
            f"{cohens_d:.2f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=7)
    
    # Style rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            
            # Highlight significant p-values
            if j == 8 and '*' in table_data[i-1][j]:
                table[(i, j)].set_facecolor('#FFFF99')
                table[(i, j)].set_text_props(weight='bold')
            
            table[(i, j)].set_text_props(ha='center', va='center', fontsize=7)
    
    # Add title
    ax.text(0.5, 1.10, phase, ha='center', va='top', fontsize=10, 
            fontweight='bold', transform=ax.transAxes)

# Add legend
fig.text(0.5, 0.02, '* p < 0.05 (significant) | H = High Performers | L = Low Performers | μ = Mean | d = Cohen\'s d',
         ha='center', fontsize=9, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.98])
plt.savefig(f'{ROOT}/results/eeg_metrics_ttest_all_phases_hugo.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Saved: /Users/calvinperumalla/git/eegflow/results/eeg_metrics_ttest_all_phases_hugo.png")

print("\n✅ Table visualizations complete!")
