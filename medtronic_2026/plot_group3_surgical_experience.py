import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


ROOT = Path(__file__).parent
# Group3 PIDs
group3_pids = [
    "P216", "P215", "P148", "P141", "P126", "P206", "P187", "P199", 
    "P106", "P217", "P203", "P224", "P237", "P240", "P154", "P239", 
    "P105", "P218", "P238", "P150", "P219", "P188", "P173", "P209"
]

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

# Collect data for Group3 participants
robotic_cases = []
laparoscopic_cases = []
pids_found = []

for pid in group3_pids:
    if pid in presurvey_dict:
        entry = presurvey_dict[pid]
        robotic = parse_numeric(entry.get('robotic_procedures_career', 0))
        laparoscopic = parse_numeric(entry.get('laparoscopic_procedures_career', 0))
        
        robotic_cases.append(robotic)
        laparoscopic_cases.append(laparoscopic)
        pids_found.append(pid)
        
        print(f"{pid}: Robotic={int(robotic)}, Laparoscopic={int(laparoscopic)}")
    else:
        print(f"{pid}: NOT FOUND in presurvey")

print(f"\nFound {len(pids_found)} of {len(group3_pids)} participants in presurvey")

# Create output directory
output_dir = ROOT / 'results/arm_cutting_analysis'
output_dir.mkdir(exist_ok=True)

# Create subplots for better visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Grouped bar plot (side-by-side)
x = np.arange(len(pids_found))
width = 0.35

ax1 = axes[0]
bars1 = ax1.bar(x - width/2, robotic_cases, width, label='Robotic Cases', color='#1f77b4', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, laparoscopic_cases, width, label='Laparoscopic Cases', color='#ff7f0e', edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Cases (Career)', fontsize=12, fontweight='bold')
ax1.set_title('Group3: Career Surgical Experience - Grouped View', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(pids_found, rotation=45, ha='right')
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)

# Plot 2: Stacked bar plot
ax2 = axes[1]
bars_robotic = ax2.bar(x, robotic_cases, label='Robotic Cases', color='#1f77b4', edgecolor='black', linewidth=1.5)
bars_lap = ax2.bar(x, laparoscopic_cases, bottom=robotic_cases, label='Laparoscopic Cases', color='#ff7f0e', edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Participant ID', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Cases (Career)', fontsize=12, fontweight='bold')
ax2.set_title('Group3: Career Surgical Experience - Stacked View', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(pids_found, rotation=45, ha='right')
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (pid, rob, lap) in enumerate(zip(pids_found, robotic_cases, laparoscopic_cases)):
    if rob > 0:
        ax2.text(i, rob/2, f'{int(rob)}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    if lap > 0:
        ax2.text(i, rob + lap/2, f'{int(lap)}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig(output_dir / 'group3_surgical_experience.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved: group3_surgical_experience.png")
plt.close()

# Create a summary statistics plot
fig, ax = plt.subplots(figsize=(10, 6))

# Summary statistics
categories = ['Robotic Cases', 'Laparoscopic Cases']
mean_values = [np.mean(robotic_cases), np.mean(laparoscopic_cases)]
median_values = [np.median(robotic_cases), np.median(laparoscopic_cases)]
max_values = [np.max(robotic_cases), np.max(laparoscopic_cases)]
min_values = [np.min(robotic_cases), np.min(laparoscopic_cases)]

x_pos = np.arange(len(categories))
width = 0.2

ax.bar(x_pos - 1.5*width, mean_values, width, label='Mean', color='#1f77b4', edgecolor='black', linewidth=1.5)
ax.bar(x_pos - 0.5*width, median_values, width, label='Median', color='#ff7f0e', edgecolor='black', linewidth=1.5)
ax.bar(x_pos + 0.5*width, max_values, width, label='Max', color='#2ca02c', edgecolor='black', linewidth=1.5)
ax.bar(x_pos + 1.5*width, min_values, width, label='Min', color='#d62728', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
ax.set_title('Group3: Summary Statistics of Surgical Experience', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for i, (cat, mean_val, med_val, max_val, min_val) in enumerate(zip(categories, mean_values, median_values, max_values, min_values)):
    ax.text(i - 1.5*width, mean_val, f'{mean_val:.0f}', ha='center', va='bottom', fontsize=9)
    ax.text(i - 0.5*width, med_val, f'{med_val:.0f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + 0.5*width, max_val, f'{int(max_val)}', ha='center', va='bottom', fontsize=9)
    ax.text(i + 1.5*width, min_val, f'{int(min_val)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'group3_surgical_experience_summary.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: group3_surgical_experience_summary.png")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS FOR GROUP3")
print("="*60)
print(f"\nRobotic Cases:")
print(f"  Mean: {np.mean(robotic_cases):.1f}")
print(f"  Median: {np.median(robotic_cases):.1f}")
print(f"  Min: {np.min(robotic_cases)}")
print(f"  Max: {np.max(robotic_cases)}")
print(f"  Total: {np.sum(robotic_cases)}")

print(f"\nLaparoscopic Cases:")
print(f"  Mean: {np.mean(laparoscopic_cases):.1f}")
print(f"  Median: {np.median(laparoscopic_cases):.1f}")
print(f"  Min: {np.min(laparoscopic_cases)}")
print(f"  Max: {np.max(laparoscopic_cases)}")
print(f"  Total: {np.sum(laparoscopic_cases)}")

print(f"\n✅ All visualizations saved to: {output_dir}")
