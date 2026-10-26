from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the same training records used for the other models.
train_data = pd.read_csv("data/train_data.csv")

# Separate input features from the correct answers.
X_train = train_data.drop(columns=["spam"])
y_train = train_data["spam"]

# Scale features and train an SVM with an RBF kernel.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=42,
    )),
])

# Learn preprocessing settings and the classification boundary.
model.fit(X_train, y_train)

# Save the complete pipeline.
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/svm.joblib")

print("SVM training completed.")
print("Training records:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", model.classes_)
print("Saved: model/svm.joblib")
