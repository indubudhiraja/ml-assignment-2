from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Read feature names, ignoring documentation comments.
names_text = Path("data/spambase.names").read_text()

feature_names = [
    line.split(":")[0].strip()
    for line in names_text.splitlines()
    if line.strip()
    and not line.lstrip().startswith("|")
    and ":" in line
]

# Stop if the names file does not contain the expected features.
if len(feature_names) != 57:
    raise ValueError(
        f"Expected 57 feature names, found {len(feature_names)}"
    )

# Add the target name after the 57 input feature names.
data = pd.read_csv(
    "data/spambase.data",
    header=None,
    names=feature_names + ["spam"],
)

# Preserve approximately the same class proportions in both sets.
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["spam"],
)

# Save raw records without adding a row-number column.
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("Training records:", len(train_data))
print("Test records:", len(test_data))
print("Input features:", len(feature_names))

print("\nTraining class counts:")
print(train_data["spam"].value_counts().sort_index())

print("\nTest class counts:")
print(test_data["spam"].value_counts().sort_index())

print("\nSaved: data/train_data.csv")
print("Saved: test_data.csv")
