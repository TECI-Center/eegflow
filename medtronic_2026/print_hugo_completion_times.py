import json
import pandas as pd
from pathlib import Path



ROOT = Path(__file__).parent
# Load HUGO metrics
with open(f'{ROOT}/scores/medtronic_hugo_metrics_NORMALIZED_scores.json') as f:
    hugo_data = json.load(f)

# Load presurvey data
with open('/Users/calvinperumalla/git/inert_pipe/presurvey_responses.json') as f:
    presurvey_data = json.load(f)

# Create lookup dict for presurvey data
presurvey_dict = {entry['participant_id']: entry for entry in presurvey_data}

# Extract data
data_list = []
tasks = ['Fourth Arm Cutting', 'Knot Tying', 'Puzzle Piece Dissection', 'Ring Tower Transfer', 'Suturing (Railroad Track)']

for pid, metrics in hugo_data.items():
    row = {'PID': pid}
    
    # Extract individual task times
    for task in tasks:
        if task in metrics:
            duration = metrics[task].get('duration', None)
            row[task] = duration
        else:
            row[task] = None
    
    # Extract robotic cases from presurvey
    if pid in presurvey_dict:
        robotic_cases = presurvey_dict[pid].get('robotic_procedures_career', 'N/A')
        row['Robotic Cases (Career)'] = robotic_cases
    else:
        row['Robotic Cases (Career)'] = 'N/A'
    
    data_list.append(row)

# Create DataFrame
df = pd.DataFrame(data_list)

# Filter to only participants with all 5 tasks completed (no None values)
df = df.dropna(subset=tasks)

# Calculate overall time (sum of all tasks)
df['Overall Time'] = df[tasks].sum(axis=1)

# Sort by overall time (fastest first)
df = df.sort_values('Overall Time', ascending=True).reset_index(drop=True)

# Format time columns to show minutes and seconds
def format_time(seconds):
    if pd.isna(seconds) or seconds == 0:
        return 'N/A'
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s"

# Display the table
print("\n" + "="*200)
print("HUGO TASK COMPLETION TIMES (Completed All 5 Tasks) - SORTED BY FASTEST OVERALL")
print("="*200)

# Create display dataframe
df_display = df.copy()
for col in tasks + ['Overall Time']:
    df_display[col] = df[col].apply(format_time)

# Keep robotic cases as-is
df_display['Robotic Cases (Career)'] = df['Robotic Cases (Career)']

# Print with pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

print("\n")
print(df_display.to_string(index=False))
print("\n" + "="*200)
print(f"Total HUGO participants (with all 5 tasks completed): {len(df)}")
print("="*200 + "\n")

# Also save as CSV
df_display.to_csv(f'{ROOT}/results/hugo_completion_times.csv', index=False)
print("✅ Saved to: /Users/calvinperumalla/git/eegflow/results/hugo_completion_times.csv")
