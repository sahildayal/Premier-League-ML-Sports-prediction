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

# === Step 5: Convert Odds to Implied Probabilities ===
# Prevent division by zero
df['Prob_H'] = 1 / df['B365H'].replace(0, pd.NA)
df['Prob_D'] = 1 / df['B365D'].replace(0, pd.NA)
df['Prob_A'] = 1 / df['B365A'].replace(0, pd.NA)

# Normalize so the three probabilities sum to 1 (remove bookmaker margin)
prob_sum = df['Prob_H'] + df['Prob_D'] + df['Prob_A']
df['Prob_H'] /= prob_sum
df['Prob_D'] /= prob_sum
df['Prob_A'] /= prob_sum

# Preview probabilities
print("\n🧠 Implied Probabilities Sample:")
print(df[['HomeTeam', 'AwayTeam', 'Prob_H', 'Prob_D', 'Prob_A']])

# === Step 4: Preview Cleaned Data ===
print("\n🎯 Cleaned & Enriched Data Sample:")
print(df.head())

# Optional: Save cleaned data
df.to_csv("Data/epl_2023_cleaned.csv", index=False)
print("\n💾 Cleaned data saved to Data/epl_2023_cleaned.csv")
