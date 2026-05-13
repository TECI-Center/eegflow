import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



ROOT = Path(__file__).parent
# Load the CSV
csv_path = f'{ROOT}/results/arm_cutting_analysis/bimanual_dexterity_statistics.csv'
df = pd.read_csv(csv_path)

# Format the dataframe for display
df_display = df.copy()

# Round numeric columns
df_display['μ (H)'] = df['μ (H)'].round(2)
df_display['μ (L)'] = df['μ (L)'].round(2)
df_display['t-stat'] = df['t-stat'].round(3)
df_display['p-value'] = df['p-value'].round(3)
df_display["Cohen's d"] = df["Cohen's d"].round(2)

# Convert to strings for table display
df_display['N (H)'] = df['N (H)'].astype(int).astype(str)
df_display['N (L)'] = df['N (L)'].astype(int).astype(str)
df_display['NaN% (H)'] = df['NaN% (H)'].astype(int).astype(str)
df_display['NaN% (L)'] = df['NaN% (L)'].astype(int).astype(str)

# Round means for display
df_display['μ (H)'] = df_display['μ (H)'].astype(str)
df_display['μ (L)'] = df_display['μ (L)'].astype(str)
df_display['t-stat'] = df_display['t-stat'].astype(str)
df_display['p-value'] = df_display['p-value'].astype(str)
df_display["Cohen's d"] = df_display["Cohen's d"].astype(str)

# Create figure
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=df_display.values, 
                colLabels=df_display.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.10, 0.09, 0.10, 0.10, 0.09, 0.10, 0.10, 0.10, 0.10])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(df_display.columns)):
    table[(0, i)].set_facecolor('#2E5090')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(df_display) + 1):
    for j in range(len(df_display.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F5F5F5')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')
        table[(i, j)].set_text_props(ha='center', va='center')

plt.savefig(f'{ROOT}/results/arm_cutting_analysis/bimanual_dexterity_statistics.png', 
            dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print("✅ Saved: /Users/calvinperumalla/git/eegflow/results/arm_cutting_analysis/bimanual_dexterity_statistics.png")
