# Spam Classification Explorer

## 1. Problem Statement

[Explain what the project predicts, why spam classification is useful,
and why both missed spam and incorrectly blocked legitimate emails matter.]

## 2. Dataset Description

- Dataset: UCI Spambase
- Source: https://archive.ics.uci.edu/dataset/94/spambase
- Task: Binary classification
- Records: 4,601
- Input features: 57 numeric features
- Target column: spam
- Target labels: 0 = not spam, 1 = spam
- Non-spam records: 2,788
- Spam records: 1,813
- Missing values found: 0
- Training records: 3,680
- Test records: 921
- Split: 80% training and 20% testing, stratified by target
- Random seed for splitting: 42

Features describe word frequencies, character frequencies, and
sequences of capital letters. The dataset contains numeric email
measurements rather than raw email text.

Dataset citation:
Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999).
Spambase [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C53G6X

License: Creative Commons Attribution 4.0 (CC BY 4.0).

## 3. GitHub Repository

https://github.com/indubudhiraja/ml-assignment-2

## 4. Models and Evaluation

All five models use the same training and test split.
The target column is excluded from the input features.

StandardScaler is used for Logistic Regression, KNN, and SVM.
Each scaler is fitted only on training data and saved with its
classifier in a pipeline.

Spam (1) is the positive class for precision, recall, and F1.
AUC is ROC AUC calculated using predicted spam probabilities.
Results below are from the held-out test set.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9110 | 0.9078 | 0.8828 | 0.8926 | 0.8877 | 0.8140 |
| KNN | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Gaussian Naive Bayes | 0.8339 | 0.9449 | 0.7178 | 0.9532 | 0.8189 | 0.6941 |
| SVM | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

## 5. Performance Observations

| Model | Observation |
|---|---|
| Logistic Regression | [Your observation] |
| Decision Tree | [Your observation] |
| KNN | [Your observation] |
| Gaussian Naive Bayes | [Your observation] |
| SVM | [Your observation] |
| Overall winner | [Your choice and justification] |

## 6. Run the App Locally

With a Python environment activated, run:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

Upload test_data.csv from the repository's root folder.
It contains all 57 input features and the spam target column.

Select a model to view metrics and its confusion matrix.
The app loads saved models and does not retrain on uploaded data.

## 7. Live Streamlit App

[Open the live Streamlit app](https://indu-ml-assignment-2.streamlit.app/)
