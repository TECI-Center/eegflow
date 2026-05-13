import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


ROOT = Path(__file__).parent
# Load group definitions
with open(f'{ROOT}/results/group_comparisons/group_definitions.json') as f:
    groups = json.load(f)

# Load presurvey data
with open('/Users/calvinperumalla/git/inert_pipe/presurvey_responses.json') as f:
    presurvey_data = json.load(f)

# Create a lookup dictionary
presurvey_dict = {entry['participant_id']: entry for entry in presurvey_data}

# Function to parse numeric values from strings
def parse_numeric(value):
    """Convert various string formats to numeric value"""
    if value is None or value == '':
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value).strip()
    
    # Handle "> X" format
    if value_str.startswith('>'):
        try:
            return float(value_str[1:].strip())
        except:
            return 0
    
    # Handle "X+" format
    if value_str.endswith('+'):
        try:
            return float(value_str[:-1].strip())
        except:
            return 0
    
    # Handle "X-Y" format (range) - take the lower value
    if '-' in value_str and not value_str.startswith('-'):
        try:
            parts = value_str.split('-')
            return float(parts[0].strip())
        except:
            return 0
    
    # Try direct conversion
    try:
        return float(value_str)
    except:
        return 0

# Create output directory
output_dir = ROOT / 'results/arm_cutting_analysis'
output_dir.mkdir(exist_ok=True)

# Process each group
for group_name in ['Group1', 'Group2', 'Group3']:
    pids = groups[group_name]
    
    # Collect data
    data = []
    missing_pids = []
    for pid in pids:
        if pid in presurvey_dict:
            entry = presurvey_dict[pid]
            robotic = parse_numeric(entry.get('robotic_procedures_career', 0))
            laparoscopic = parse_numeric(entry.get('laparoscopic_procedures_career', 0))
            
            data.append({
                'pid': pid,
                'robotic': robotic,
                'laparoscopic': laparoscopic
            })
        else:
            missing_pids.append(pid)
    
    # Print missing participants
    if missing_pids:
        print(f"{group_name} - Missing presurvey data: {', '.join(missing_pids)}")
    
    # Sort by laparoscopic cases (increasing order)
    data.sort(key=lambda x: x['laparoscopic'])
    
    # Extract sorted values
    pids_sorted = [d['pid'] for d in data]
    robotic_sorted = [d['robotic'] for d in data]
    laparoscopic_sorted = [d['laparoscopic'] for d in data]
    
    print(f"\n{group_name} (sorted by laparoscopic cases):")
    print(f"  Included: {len(data)} participants, Missing: {len(missing_pids)} participants")
    for d in data:
        print(f"  {d['pid']}: Robotic={int(d['robotic'])}, Laparoscopic={int(d['laparoscopic'])}")
    
    # Create plot with dynamic figure width based on number of participants
    fig_width = max(10, len(pids_sorted) * 0.9)  # ~0.9 inches per participant, minimum 10
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    
    # Create x-axis positions with consistent spacing
    x_positions = np.arange(len(pids_sorted))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x_positions - width/2, robotic_sorted, width, label='Robotic Cases', 
                   color='#ff7f0e', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x_positions + width/2, laparoscopic_sorted, width, label='Laparoscopic Cases', 
                   color='#1f77b4', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Participant ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Cases (Career)', fontsize=13, fontweight='bold')
    ax.set_title(f'{group_name}: Career Surgical Experience (Sorted by Laparoscopic Cases)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(pids_sorted, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis limit to 6000
    ax.set_ylim(0, 6000)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=8)
    
    fig.subplots_adjust(bottom=0.25, top=0.92)
    filename = f'group{group_name[-1]}_surgical_experience_sorted_bylap.png'
    plt.savefig(output_dir / filename, dpi=300)
    print(f"✅ Saved: {filename}")
    plt.close()

print(f"\n✅ All plots saved to: {output_dir}")
