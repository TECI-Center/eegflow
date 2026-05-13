import json
import pandas as pd
from pathlib import Path



ROOT = Path(__file__).parent
# Load FLS metrics
with open(f'{ROOT}/scores/fls_metrics_scores.json') as f:
    fls_data = json.load(f)

# Extract data
data_list = []
for pid, metrics in fls_data.items():
    row = {'PID': pid}
    
    # Extract individual task times
    for task in ['Circle Cutting', 'Peg Transfer', 'Pen Rose Suturing']:
        if task in metrics:
            duration = metrics[task].get('duration', None)
            row[task] = duration
        else:
            row[task] = None
    
    data_list.append(row)

# Create DataFrame
df = pd.DataFrame(data_list)

# Calculate overall time (sum of all tasks, ignoring None values)
df['Overall Time'] = df[['Circle Cutting', 'Peg Transfer', 'Pen Rose Suturing']].sum(axis=1, skipna=True)

# Sort by overall time (fastest first)
df = df.sort_values('Overall Time', ascending=True, na_position='last').reset_index(drop=True)

# Format time columns to show minutes and seconds
def format_time(seconds):
    if pd.isna(seconds):
        return 'N/A'
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

# Display the table
print("\n" + "="*120)
print("FLS TASK COMPLETION TIMES - SORTED BY FASTEST OVERALL")
print("="*120)

# Create display dataframe
df_display = df.copy()
for col in ['Circle Cutting', 'Peg Transfer', 'Pen Rose Suturing', 'Overall Time']:
    df_display[col] = df[col].apply(format_time)

# Print with pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', None)

print("\n")
print(df_display.to_string(index=False))
print("\n" + "="*120)
print(f"Total FLS participants: {len(df)}")
print("="*120 + "\n")

# Also save as CSV
df_display.to_csv(f'{ROOT}/results/fls_completion_times.csv', index=False)
print("✅ Saved to: /Users/calvinperumalla/git/eegflow/results/fls_completion_times.csv")
