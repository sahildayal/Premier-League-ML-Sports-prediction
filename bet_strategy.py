import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# === Step 1: Load Cleaned Data ===
df = pd.read_csv("Data/epl_full_cleaned.csv")

# === Step 2: Load Trained Model ===
# For now, we’ll just retrain it here (but you can save and load with joblib later)
features = ['GoalDiff', 'Prob_H', 'Prob_D', 'Prob_A']
X = df[features]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['MatchResult'])

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X, y)

# === Step 3: Predict Probabilities ===
df[['Pred_Win', 'Pred_Draw', 'Pred_Loss']] = model.predict_proba(X)

# === Step 4: Calculate Expected Value (EV) ===
df['EV_Win'] = (df['Pred_Win'] * df['B365H']) - 1
df['EV_Draw'] = (df['Pred_Draw'] * df['B365D']) - 1
df['EV_Loss'] = (df['Pred_Loss'] * df['B365A']) - 1

# === Step 5: Simulate Bets ===
threshold = 0.00  # Only bet if EV > 5%
unit_bet = 100    # Flat betting strategy (100 units)

def simulate_bet(ev, outcome, predicted_class, odds):
    if ev > threshold and outcome == predicted_class:
        return (odds - 1) * unit_bet  # Profit
    elif ev > threshold:
        return -unit_bet  # Loss
    return 0  # No bet

df['Bet_Profit'] = 0

# Determine predicted outcome
predicted_classes = model.predict(X)
decoded_preds = label_encoder.inverse_transform(predicted_classes)

for i in range(len(df)):
    pred = decoded_preds[i]
    actual = df.iloc[i]['MatchResult']

    if pred == 'Win':
        ev = df.iloc[i]['EV_Win']
        odds = df.iloc[i]['B365H']
    elif pred == 'Draw':
        ev = df.iloc[i]['EV_Draw']
        odds = df.iloc[i]['B365D']
    else:
        ev = df.iloc[i]['EV_Loss']
        odds = df.iloc[i]['B365A']

    df.at[i, 'Bet_Profit'] = simulate_bet(ev, actual, pred, odds)

# === Step 6: Report Results ===
total_bets = df[df['Bet_Profit'] != 0].shape[0]
total_profit = df['Bet_Profit'].sum()
roi = (total_profit / (total_bets * unit_bet)) * 100 if total_bets > 0 else 0

print(f"📊 Total Bets Placed: {total_bets}")
print(f"💰 Total Profit: {total_profit:.2f} units")
print(f"📈 ROI: {roi:.2f}%")
