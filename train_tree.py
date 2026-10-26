from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the same training records used for Logistic Regression.
train_data = pd.read_csv("data/train_data.csv")

# Separate the input features from the correct answers.
X_train = train_data.drop(columns=["spam"])
y_train = train_data["spam"]

# Create a baseline Decision Tree.
model = DecisionTreeClassifier(random_state=42)

# Learn classification rules from the training records.
model.fit(X_train, y_train)

# Save the trained model.
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/decision_tree.joblib")

print("Decision Tree training completed.")
print("Training records:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", model.classes_)
print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())
print("Saved: model/decision_tree.joblib")
