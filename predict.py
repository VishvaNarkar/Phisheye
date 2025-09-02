# predict.py
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("phishing_rf_model.pkl")

# Get the expected feature names from the trained model
expected_features = model.feature_names_in_

# --- Example: Create a new sample (you can later replace with dynamic input) ---
# NOTE: Use realistic dummy values; order does not matter since we will align automatically
sample_data = {
    "abnormal_subdomain": 0,
    "avg_word_host": 10,
    "avg_word_path": 5,
    "avg_words_raw": 15,
    "brand_in_path": 0,
    "brand_in_subdomain": 0,
    "char_repeat": 2,
    "domain_in_title": 1,
    "dot_count": 3,
    "embedded_domain": 0,
    "entropy_url": 4.2,
    "has_ip": 0,
    "https_in_path": 0,
    "longest_word_host": 8,
    "longest_word_path": 12,
    "phish_hints": 1,
    "query_length": 20,
    "slashes_url": 4,
    "subdir_count": 2,
    "url_length": 75,
    "www_count": 1,
}

# --- Ensure alignment with training features ---
# Fill in any missing features with 0, drop extras
aligned_sample = {f: sample_data.get(f, 0) for f in expected_features}

# Convert to DataFrame with correct column order
sample_df = pd.DataFrame([aligned_sample], columns=expected_features)

# Predict
prediction = model.predict(sample_df)[0]
proba = model.predict_proba(sample_df)[0]

print("✅ Prediction:", "Phishing" if prediction == 1 else "Legitimate")
print("🔢 Probability:", proba)
