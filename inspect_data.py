import pandas as pd

# The data file contains comma-separated values without column headings.
data = pd.read_csv("data/spambase.data", header=None)

# All columns except the last are input features.
X = data.iloc[:, :-1]

# The last column is the target: 0 = not spam, 1 = spam.
y = data.iloc[:, -1]

print("Number of records:", len(data))
print("Number of input features:", X.shape[1])
print("Missing values:", data.isna().sum().sum())

print("\nTarget class counts (0 = not spam, 1 = spam):")
print(y.value_counts().sort_index())

print("\nFirst five records:")
print(data.head())
