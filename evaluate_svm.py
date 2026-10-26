import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Load the held-out test records.
test_data = pd.read_csv("test_data.csv")
X_test = test_data.drop(columns=["spam"])
y_test = test_data["spam"]

# Load the trained pipeline, including its saved scaler.
model = joblib.load("model/svm.joblib")

# Predict class labels: 0 = not spam, 1 = spam.
predictions = model.predict(X_test)

# AUC needs scores, so obtain the probability of the spam class.
spam_column = list(model.classes_).index(1)
spam_probabilities = model.predict_proba(X_test)[:, spam_column]

results = {
    "Model": "SVM",
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC": roc_auc_score(y_test, spam_probabilities),
    "Precision": precision_score(
        y_test, predictions, pos_label=1, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, pos_label=1, zero_division=0
    ),
    "F1": f1_score(
        y_test, predictions, pos_label=1, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_test, predictions),
}

results_table = pd.DataFrame([results])

print(results_table.to_string(index=False, float_format="%.4f"))

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

print("\nConfusion matrix:")
print("Rows = actual class; columns = predicted class")
print("Class order: 0 = not spam, 1 = spam")
print(matrix)

results_table.to_csv("svm_metrics.csv", index=False)
print("\nSaved: svm_metrics.csv")
