from pathlib import Path

import joblib
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load the same training records used for the other models.
train_data = pd.read_csv("data/train_data.csv")

# Separate input features from the correct answers.
X_train = train_data.drop(columns=["spam"])
y_train = train_data["spam"]

# Create Gaussian Naive Bayes with default settings.
model = GaussianNB()

# Learn class probabilities and feature statistics.
model.fit(X_train, y_train)

# Save the trained model.
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/naive_bayes.joblib")

print("Gaussian Naive Bayes training completed.")
print("Training records:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", model.classes_)
print("Saved: model/naive_bayes.joblib")
