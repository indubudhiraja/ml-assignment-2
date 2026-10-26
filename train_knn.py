from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the same training records used for the other models.
train_data = pd.read_csv("data/train_data.csv")

# Separate input features and correct answers.
X_train = train_data.drop(columns=["spam"])
y_train = train_data["spam"]

# Scale features, then use the five nearest training records.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

# Learn scaling settings and store the training examples.
model.fit(X_train, y_train)

# Save the complete pipeline.
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/knn.joblib")

print("KNN training completed.")
print("Training records:", len(X_train))
print("Input features:", X_train.shape[1])
print("Classes:", model.classes_)
print("Number of neighbors:", 5)
print("Saved: model/knn.joblib")
