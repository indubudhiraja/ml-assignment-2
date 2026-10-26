import pandas as pd

# Each file contains the evaluation results for one model.
metric_files = [
    "logistic_metrics.csv",
    "tree_metrics.csv",
    "knn_metrics.csv",
    "naive_bayes_metrics.csv",
    "svm_metrics.csv",
]

# Read each file and combine the rows into one table.
tables = [pd.read_csv(filename) for filename in metric_files]
comparison = pd.concat(tables, ignore_index=True)

# Check that all five models are present.
if len(comparison) != 5 or comparison["Model"].nunique() != 5:
    raise ValueError("Expected exactly five different models.")

# Save full precision; round only the displayed values.
comparison.to_csv("model_comparison.csv", index=False)

print(comparison.to_string(index=False, float_format="%.4f"))
print("\nSaved: model_comparison.csv")
