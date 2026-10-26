from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load training records only.
train_data = pd.read_csv("data/train_data.csv")

# Inputs: all columns except the correct answer.
X_train = train_data.drop(columns=["spam"])

# Target: the correct answer for each record.
y_train = train_data["spam"]

# Keep scaling and classification together in one pipeline.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
    )),
])

# Learn scaling parameters and model coefficients from training data.
model.fit(X_train, y_train)

# Save the complete pipeline for evaluation and the web app.
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/logistic_regression.joblib")

print("Logistic Regression training completed.")
print("Training records:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", model.classes_)
print("Saved: model/logistic_regression.joblib")
