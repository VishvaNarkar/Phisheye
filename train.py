import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("sample_data/Phishing_Email.csv")

# Features (drop 'url' because it's just text, not numeric)
X = df.drop(columns=["status", "url"])
y = df["status"]

# Encode target labels (legitimate=0, phishing=1)
y = y.map({"legitimate": 0, "phishing": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "phishing_rf_model.pkl")
print("✅ Model saved as phishing_rf_model.pkl")
