from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve files relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent

# Configure the browser tab and page layout.
st.set_page_config(
    page_title="Spam Classification Explorer",
    page_icon="📧",
    layout="wide",
)

st.title("Spam Classification Explorer")
st.write(
    "Compare five machine learning models that classify "
    "emails as spam or not spam using numeric email features."
)

# Map readable model names to saved model files.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}

selected_model = st.selectbox(
    "Choose a classification model",
    options=list(MODEL_FILES),
)

st.write("Selected model:", selected_model)

st.subheader("Original test-set comparison")
st.caption(
    "Previously calculated results on the fixed "
    "921-record test set. These are not upload-based results."
)

comparison_path = PROJECT_DIR / "model_comparison.csv"

if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)

    metric_columns = [
        "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
    ]

    st.dataframe(
        comparison.style.format({
            column: "{:.4f}" for column in metric_columns
        }),
        hide_index=True,
    )
else:
    st.error(
        "model_comparison.csv is missing. "
        "Run compare_models.py first."
    )
st.subheader("Upload test data")
st.write(
    "Upload a CSV containing the 57 input features and "
    "the spam target column: 0 = not spam, 1 = spam."
)

# Provide the original test file as a downloadable example.
sample_path = PROJECT_DIR / "test_data.csv"

if not sample_path.exists():
    st.error("The sample test_data.csv file is missing.")
    st.stop()

st.download_button(
    label="Download sample test CSV",
    data=sample_path.read_bytes(),
    file_name="test_data.csv",
    mime="text/csv",
)

uploaded_file = st.file_uploader(
    "Choose a labelled test CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a test CSV to continue.")
    st.stop()

try:
    uploaded_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV: {error}")
    st.stop()

# Use the original test file to identify the required columns.
expected_columns = pd.read_csv(sample_path, nrows=0).columns.tolist()

missing_columns = sorted(
    set(expected_columns) - set(uploaded_data.columns)
)
extra_columns = sorted(
    set(uploaded_data.columns) - set(expected_columns)
)

if missing_columns or extra_columns:
    st.error("The uploaded CSV columns do not match the expected format.")
    if missing_columns:
        st.write("Missing columns:", missing_columns)
    if extra_columns:
        st.write("Unexpected columns:", extra_columns)
    st.stop()

if uploaded_data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

# Restore the same column order used during training.
uploaded_data = uploaded_data[expected_columns]

try:
    uploaded_data = uploaded_data.apply(
        pd.to_numeric, errors="raise"
    )
except (ValueError, TypeError):
    st.error("All feature values and target labels must be numeric.")
    st.stop()

# Reject missing values and positive or negative infinity.
import numpy as np

if not np.isfinite(uploaded_data.to_numpy(dtype=float)).all():
    st.error("The CSV contains missing or infinite values.")
    st.stop()

if not uploaded_data["spam"].isin([0, 1]).all():
    st.error("The spam column must contain only 0 or 1.")
    st.stop()

X_uploaded = uploaded_data.drop(columns=["spam"])
y_uploaded = uploaded_data["spam"].astype(int)

st.success(
    f"Valid CSV: {len(uploaded_data)} records and "
    f"{X_uploaded.shape[1]} input features."
)

st.dataframe(uploaded_data.head(), hide_index=True)
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# Cache saved models so dropdown changes do not repeatedly load files.
@st.cache_resource
def load_saved_model(model_path):
    return joblib.load(model_path)

model_path = PROJECT_DIR / "model" / MODEL_FILES[selected_model]

if not model_path.exists():
    st.error(f"Saved model file is missing: {model_path.name}")
    st.stop()

try:
    model = load_saved_model(str(model_path))

    # Match the feature order expected by the saved model.
    expected_features = list(model.feature_names_in_)
    model_inputs = X_uploaded[expected_features]

    predictions = model.predict(model_inputs)

    # Find the probability column for spam, rather than assuming its position.
    spam_column = list(model.classes_).index(1)
    spam_probabilities = model.predict_proba(
        model_inputs
    )[:, spam_column]

except Exception as error:
    st.error(f"Could not evaluate the selected model: {error}")
    st.stop()

# AUC requires both classes in the uploaded target column.
auc = None
if y_uploaded.nunique() == 2:
    auc = roc_auc_score(y_uploaded, spam_probabilities)
else:
    st.warning(
        "AUC is unavailable because the uploaded target "
        "contains only one class."
    )

metrics = {
    "Accuracy": accuracy_score(y_uploaded, predictions),
    "AUC": auc,
    "Precision": precision_score(
        y_uploaded, predictions, pos_label=1, zero_division=0
    ),
    "Recall": recall_score(
        y_uploaded, predictions, pos_label=1, zero_division=0
    ),
    "F1": f1_score(
        y_uploaded, predictions, pos_label=1, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_uploaded, predictions),
}

st.subheader(f"Uploaded-data results: {selected_model}")
st.caption(
    "Spam (1) is the positive class. Results are calculated "
    "from your uploaded CSV using the saved model."
)

# Display three metrics per row.
metric_items = list(metrics.items())

for start in range(0, len(metric_items), 3):
    columns = st.columns(3)

    for column, (name, value) in zip(
        columns, metric_items[start:start + 3]
    ):
        column.metric(
            label=name,
            value="N/A" if value is None else f"{value:.4f}",
        )
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.subheader(f"Confusion matrix: {selected_model}")

matrix = confusion_matrix(
    y_uploaded,
    predictions,
    labels=[0, 1],
)

fig, ax = plt.subplots(figsize=(6, 4))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=["Not spam (0)", "Spam (1)"],
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("Predictions on uploaded test data")
fig.tight_layout()

st.pyplot(fig, width=550)
plt.close(fig)

# With labels=[0, 1], the cells are ordered TN, FP, FN, TP.
tn, fp, fn, tp = matrix.ravel()

st.write(f"Correctly classified non-spam emails: **{tn}**")
st.write(f"Non-spam emails incorrectly flagged as spam: **{fp}**")
st.write(f"Spam emails missed: **{fn}**")
st.write(f"Correctly detected spam emails: **{tp}**")

st.caption(
    "Rows show actual classes. Columns show predicted classes."
)

