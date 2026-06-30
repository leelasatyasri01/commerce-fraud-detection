import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("fraud_dataset.csv")

# Features and Target
X = df.drop("Is_Fraud", axis=1)
y = df["Is_Fraud"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f"Model Accuracy: {accuracy:.2f}")

# Save Model
joblib.dump(model, "model.pkl")

print("✅ model.pkl saved successfully!")