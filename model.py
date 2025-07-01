from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# === Step 1: Load Cleaned Data ===
df = pd.read_csv("Data/epl_full_cleaned.csv")

# === Step 2: Feature Selection ===
features = ['GoalDiff', 'Prob_H', 'Prob_D', 'Prob_A']
X = df[features]

# === Step 3: Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['MatchResult'])  # Win → 2, Draw → 0, Loss → 1 (for example)

# === Step 4: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === Step 5: Train Model ===
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# === Step 6: Evaluate ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {acc:.2f}")
print("\n🧾 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
