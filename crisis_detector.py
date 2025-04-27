import pandas as pd

# Load the dataset
df = pd.read_excel('Dataset4_EWS.xlsx')

# Find crisis periods
crisis_periods = []
current_start = None

for i in range(len(df)):
    # Start of a crisis period
    if df['Y'].iloc[i] == 1 and (i == 0 or df['Y'].iloc[i-1] == 0):
        current_start = df['Data'].iloc[i]
    
    # End of a crisis period
    elif df['Y'].iloc[i] == 0 and i > 0 and df['Y'].iloc[i-1] == 1 and current_start is not None:
        crisis_periods.append((current_start, df['Data'].iloc[i-1]))
        current_start = None

# Handle the case where the last period in the dataset is a crisis
if current_start is not None:
    crisis_periods.append((current_start, df['Data'].iloc[-1]))

# Print crisis periods
print('Crisis periods detected:')
for i, (start, end) in enumerate(crisis_periods):
    print(f'Crisis {i+1}: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}')
    duration = (end - start).days
    print(f'  Duration: {duration} days')

print(f'\nTotal number of crisis periods: {len(crisis_periods)}')

# Export crisis periods to a CSV file
crisis_df = pd.DataFrame({
    'CrisisNumber': range(1, len(crisis_periods) + 1),
    'StartDate': [start for start, _ in crisis_periods],
    'EndDate': [end for _, end in crisis_periods],
    'Duration': [(end - start).days for start, end in crisis_periods]
})

crisis_df.to_csv('crisis_periods.csv', index=False)
print("Crisis periods exported to 'crisis_periods.csv'")

# Update context file with crisis periods
with open('context.md', 'a') as f:
    f.write("\n\n## Crisis Periods Detected\n")
    for i, (start, end) in enumerate(crisis_periods):
        duration = (end - start).days
        f.write(f"- **Crisis {i+1}**: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({duration} days)\n")
    f.write(f"\nTotal number of crisis periods: {len(crisis_periods)}\n")

print("Crisis periods added to context.md") 