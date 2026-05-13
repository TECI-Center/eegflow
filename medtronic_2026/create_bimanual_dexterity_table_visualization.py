import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path



ROOT = Path(__file__).parent
# Load the CSV
csv_path = f'{ROOT}/results/arm_cutting_analysis/bimanual_dexterity_statistics.csv'
df = pd.read_csv(csv_path)

# Convert numeric columns to proper formatting
df_display = df.copy()

# Format numerical columns for display
numeric_cols = ['μ (H)', 'μ (L)', 't-stat', 'p-value', "Cohen's d"]
for col in numeric_cols:
    if col == 'p-value':
        df_display[col] = df[col].apply(lambda x: f"{x:.3f}" + ("*" if x < 0.05 else ""))
    elif col == "Cohen's d":
        df_display[col] = df[col].apply(lambda x: f"{x:.2f}")
    else:
        df_display[col] = df[col].apply(lambda x: f"{x:.2f}")

# Keep integer columns as is
df_display['N (H)'] = df['N (H)'].astype(int)
df_display['N (L)'] = df['N (L)'].astype(int)
df_display['NaN% (H)'] = df['NaN% (H)'].astype(int).astype(str) + '%'
df_display['NaN% (L)'] = df['NaN% (L)'].astype(int).astype(str) + '%'

# Create figure with table
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=df_display.values, 
                colLabels=df_display.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.12, 0.10, 0.08, 0.10, 0.10, 0.08, 0.10, 0.10, 0.10, 0.10])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.8)

# Style header
for i in range(len(df_display.columns)):
    table[(0, i)].set_facecolor('#2E5090')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    table[(0, i)].set_height(0.08)

# Style rows with alternating colors
for i in range(1, len(df_display) + 1):
    for j in range(len(df_display.columns)):
        # Alternating row colors
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E8F0F8')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        
        # Highlight significant p-values
        if j == 8:  # p-value column
            if '*' in df_display.iloc[i-1, j]:
                table[(i, j)].set_facecolor('#FFFACD')
                table[(i, j)].set_text_props(weight='bold', fontsize=11)
        
        table[(i, j)].set_text_props(ha='center', va='center', fontsize=11)
        table[(i, j)].set_height(0.06)

# Add title
plt.text(0.5, 1.08, 'Bimanual Dexterity Groups - EEG Metrics Comparison\nFourth Arm Cutting Task on HUGO',
         ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax.transAxes)

# Add legend/notes
notes_text = ('H = Highest Min Angular Velocity (n=7) | L = High Min Angular Velocity (n=19)\n'
              '* p < 0.05 (significant difference) | μ = Mean | Cohen\'s d = Effect size')
plt.text(0.5, -0.12, notes_text,
         ha='center', va='top', fontsize=10, style='italic', transform=ax.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Save figure
output_path = f'{ROOT}/results/arm_cutting_analysis/bimanual_dexterity_statistics.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_path}")

# Also create a summary statistics visualization
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Bimanual Dexterity Groups - EEG Metrics Summary\nFourth Arm Cutting Task', 
              fontsize=14, fontweight='bold')

metrics = df['Metric'].tolist()
colors = ['#2E5090', '#E8744B']  # Blue for H, Orange for L

for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    row = df[df['Metric'] == metric].iloc[0]
    
    groups = ['Highest', 'High']
    means = [row['μ (H)'], row['μ (L)']]
    x = np.arange(len(groups))
    width = 0.6
    
    bars = ax.bar(x, means, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Mean Value', fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add sample size info
    n_h = int(row['N (H)'])
    n_l = int(row['N (L)'])
    p_val = row['p-value']
    cohens_d = row["Cohen's d"]
    
    info_text = f'n(H)={n_h}, n(L)={n_l}\np-value={p_val:.3f}, Cohen\'s d={cohens_d:.2f}'
    ax.text(0.98, 0.97, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
output_path2 = f'{ROOT}/results/arm_cutting_analysis/bimanual_dexterity_bar_summary.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {output_path2}")

print("\n✅ Bimanual dexterity visualizations complete!")
