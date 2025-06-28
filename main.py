# Entry point for Premier League betting ML pipeline
import pandas as pd
import os

# === Step 1: Load CSV ===
data_path = os.path.join("Data", "epl_2023.csv")

try:
    df = pd.read_csv(data_path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Could not find the file at", data_path)
    exit()

# === Step 2: Basic Cleaning ===
# Drop rows with missing values in critical columns
critical_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
df.dropna(subset=critical_cols, inplace=True)

# === Step 3: Feature Engineering ===
# Goal Difference
df['GoalDiff'] = df['FTHG'] - df['FTAG']

# Outcome Label for ML classification
def label_result(result):
    if result == 'H':
        return 'Win'
    elif result == 'A':
        return 'Loss'
    else:
        return 'Draw'

df['MatchResult'] = df['FTR'].apply(label_result)

# === Step 4: Preview Cleaned Data ===
print("\n🎯 Cleaned & Enriched Data Sample:")
print(df.head())

# Optional: Save cleaned data
df.to_csv("Data/epl_2023_cleaned.csv", index=False)
print("\n💾 Cleaned data saved to Data/epl_2023_cleaned.csv")
