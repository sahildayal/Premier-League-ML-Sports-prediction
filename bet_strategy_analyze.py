import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === Step 1: Load Cleaned Data ===
df = pd.read_csv("Data/epl_full_cleaned.csv")

# === Step 2: Feature and Label Setup ===
features = ['GoalDiff', 'Prob_H', 'Prob_D', 'Prob_A']
X = df[features]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['MatchResult'])

# === Step 3: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Step 4: Train Model ===
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# === Step 5: Predict on Test Data ===
df_test = df.iloc[X_test.index].copy()
df_test[['Pred_Win', 'Pred_Draw', 'Pred_Loss']] = model.predict_proba(X_test)

# === Step 6: Calculate EVs ===
df_test['EV_Win'] = (df_test['Pred_Win'] * df_test['B365H']) - 1
df_test['EV_Draw'] = (df_test['Pred_Draw'] * df_test['B365D']) - 1
df_test['EV_Loss'] = (df_test['Pred_Loss'] * df_test['B365A']) - 1

# === Step 7: Simulate Bets on Test Set ===
threshold = 0.00
unit_bet = 100

predicted_classes = model.predict(X_test)
decoded_preds = label_encoder.inverse_transform(predicted_classes)
true_classes = label_encoder.inverse_transform(y_test)

bet_logs = []
total_profit = 0
total_bets = 0
cumulative_profit = []

for i in range(len(df_test)):
    pred = decoded_preds[i]
    actual = true_classes[i]

    if pred == 'Win':
        ev = df_test.iloc[i]['EV_Win']
        odds = df_test.iloc[i]['B365H']
    elif pred == 'Draw':
        ev = df_test.iloc[i]['EV_Draw']
        odds = df_test.iloc[i]['B365D']
    else:
        ev = df_test.iloc[i]['EV_Loss']
        odds = df_test.iloc[i]['B365A']

    if ev > threshold:
        total_bets += 1
        profit = (odds - 1) * unit_bet if pred == actual else -unit_bet
        total_profit += profit
        cumulative_profit.append(total_profit)

        bet_logs.append({
            "Match": f"{df_test.iloc[i]['HomeTeam']} vs {df_test.iloc[i]['AwayTeam']}",
            "Prediction": pred,
            "Actual": actual,
            "EV": round(ev, 2),
            "Odds": round(odds, 2),
            "Profit": round(profit, 2)
        })

# === Step 8: Save Bet Log to CSV ===
bet_log_df = pd.DataFrame(bet_logs)
bet_log_df.to_csv("Data/bet_log.csv", index=False)

# === Step 9: Report Results ===
roi = (total_profit / (total_bets * unit_bet)) * 100 if total_bets > 0 else 0
print(f"📊 Total Bets Placed: {total_bets}")
print(f"💰 Total Profit: {total_profit:.2f} units")
print(f"📈 ROI: {roi:.2f}%")

print("\n🧾 Bet Log Sample:")
print(bet_log_df.head())

# === Step 10: Plot Cumulative Profit ===
plt.figure(figsize=(10, 5))
plt.plot(cumulative_profit, marker='o')
plt.title("Cumulative Profit Over Bets")
plt.xlabel("Bet Number")
plt.ylabel("Profit (Units)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Data/cumulative_profit.png")
print("\n📊 Chart saved to Data/cumulative_profit.png")
